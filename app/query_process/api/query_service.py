import asyncio
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import shutil
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.clients.mongo_history_utils import get_recent_messages, clear_history, list_chat_sessions
from app.core.logger import logger
from app.query_process.agent.state import create_query_default_state

from app.utils.path_util import PROJECT_ROOT
from app.utils.asyncio_utils import install_asyncio_noise_filter
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
# from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import kb_query_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    启动时安装 asyncio 噪音过滤器。

    背景：SSE 长连接被浏览器主动断开（刷新页面 / 关标签页）时，Windows 的 Proactor
    transport 会抛 ConnectionResetError(WinError 10054)，asyncio 默认处理器会打印
    `ERROR:asyncio:Exception in callback _ProactorBasePipeTransport._call_connection_lost` 堆栈。
    这不是业务错误（SSE 通道自身已做断连清理），这里统一降级为 DEBUG 日志。
    """
    install_asyncio_noise_filter(asyncio.get_running_loop())
    yield


# 定义fastapi对象
app = FastAPI(title="query service", description="听书智库检索问答服务！", lifespan=lifespan)

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# 挂载静态资源目录（头像 / 图标等），前端以 /assets/xxx 访问
app.mount("/assets", StaticFiles(directory=str(PROJECT_ROOT / "assets")), name="assets")

# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")
    audio_url: str = Field(None, description="语音提问的音频URL（可选，随消息存档供刷新后回放）")

# 访问query.html
@app.get("/query.html")
def query_page():
    # 获取query.html的路径
    query_file_path = Path(__file__).parent.parent / "page" / "query.html"
    # 判断query_file_path是否存在
    if not query_file_path.exists():
        raise HTTPException(status_code=404, detail="query.html页面不存在")
    return FileResponse(str(query_file_path))

# 创建后台任务，通过图对象处理以后的问题query
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True, audio_url: str = ""):
    # 创建初始化状态（audio_url 随状态流转，最终写进历史记录，供刷新后回放语音）
    init_state = create_query_default_state(
        session_id=session_id,
        original_query=user_query,
        is_stream=is_stream,
        audio_url=audio_url or "",
    )
    try:
        # 执行图对象
        kb_query_app.invoke(init_state)
        # 更新当前后台任务的状态为已完成
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        logger.error(f"{session_id}后台任务失败，{e}")
        # 更新当前后台任务的状态为失败
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        # 判断是否是流式调用
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
    finally:
        # 链路已经结束，主动通知 SSE 生成器关闭连接。
        # 否则服务端会一直等客户端先断开：浏览器刷新/关页时会自行断开，
        # 但用 curl 等非浏览器客户端调试时，连接会一直挂着直到超时。
        if is_stream:
            push_to_session(session_id, SSEEvent.CLOSE, {})

# 处理用户的问题
@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    # 获取session_id,user_query,is_stream
    session_id = request.session_id or str(uuid.uuid4())
    user_query = request.query
    is_stream = request.is_stream
    # 判断是否为流式调用
    if is_stream:
        # 创建队列
        create_sse_queue(session_id)
    # 修改当前后台任务的状态为运行中
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    # 判断是否为流式调用
    if is_stream:
        # 执行后台任务
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream, request.audio_url)
        return {
            "message": "结果正在处理中...",
            "session_id": session_id
        }
    else:
        run_query_graph(session_id, user_query, is_stream, request.audio_url)
        answer = get_task_result(session_id, "answer", "")
        return {
            "message": "处理完成！",
            "session_id": session_id,
            "answer": answer,
            "done_list": []
        }

# 音频提问：上传音频文件，后端转写为文本后返回给前端（前端再走常规 /query 检索）
# 说明：定义为同步 def，FastAPI 会自动丢进线程池执行，避免转写阻塞事件循环。
@app.post("/query_audio")
def query_audio(file: UploadFile = File(...)):
    # 校验扩展名，只接受音频格式
    filename = file.filename or ""
    if not filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg")):
        raise HTTPException(status_code=400, detail="仅支持音频文件（mp3/wav/m4a/flac/aac/ogg）")
    # 保存音频到临时目录（不进正式导入目录）
    audio_dir = PROJECT_ROOT / "output" / "audio_query"
    audio_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix or ".mp3"
    audio_path = audio_dir / f"{uuid.uuid4().hex}{suffix}"
    with audio_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"收到音频提问：{filename} -> {audio_path}")

    # 同步转写（SenseVoice，本地，短音频约秒级）
    from app.lm.asr_utils import transcribe_audio
    text = transcribe_audio(str(audio_path))
    if not text.strip():
        raise HTTPException(status_code=422, detail="音频转写结果为空，请确认音频内容清晰可辨")

    # 音频同样存档到 MinIO 语音专用桶（与录音路径一致），刷新页面后可回放；失败不阻断问答
    audio_url = ""
    try:
        from app.clients.minio_utils import upload_audio_file
        day = datetime.now().strftime("%Y-%m-%d")
        ct_map = {
            ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
            ".flac": "audio/flac", ".aac": "audio/aac", ".ogg": "audio/ogg",
        }
        audio_url = upload_audio_file(
            str(audio_path),
            f"speak_content/{day}/{audio_path.name}",
            content_type=ct_map.get(suffix, "audio/mpeg"),
        )
    except Exception as e:
        logger.warning(f"音频上传 MinIO 失败（不影响本次问答，该语音刷新后不可回放）：{e}")

    return {"query": text, "filename": filename, "audio_url": audio_url}

# 语音提问（浏览器实时录音）：接收前端 MediaRecorder 录制的音频，统一转 mp3 存档后转写
# 说明：录音默认落盘到项目 mp3/speak_content/（测试素材目录，已在 .gitignore 中忽略）；
#      浏览器录制格式通常是 webm/ogg，这里用 ffmpeg 统一转成 mp3（转码失败则保留原格式）。
@app.post("/record_audio")
def record_audio(file: UploadFile = File(...)):
    # 校验扩展名，只接受音频/录音格式
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower() or ".webm"
    if suffix not in (".webm", ".ogg", ".mp3", ".wav", ".m4a", ".aac", ".flac"):
        raise HTTPException(status_code=400, detail="不支持的录音格式（webm/ogg/mp3/wav/m4a/aac/flac）")

    # 落盘目录：项目 mp3/speak_content/
    save_dir = PROJECT_ROOT / "mp3" / "speak_content"
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = save_dir / f"speak_{stamp}{suffix}"
    with raw_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"收到录音提问：{filename} -> {raw_path}")

    # 导入 asr_utils 会顺带兜底 ffmpeg 的 PATH，随后用 ffmpeg 统一转 mp3 存档
    from app.lm.asr_utils import transcribe_audio
    final_path = raw_path
    if suffix != ".mp3":
        mp3_path = raw_path.with_suffix(".mp3")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(raw_path), "-ar", "16000", "-ac", "1", str(mp3_path)],
                check=True, capture_output=True,
            )
            raw_path.unlink(missing_ok=True)
            final_path = mp3_path
            logger.info(f"录音已转 mp3 存档：{mp3_path.name}")
        except Exception as e:
            logger.warning(f"ffmpeg 转 mp3 失败，保留原始格式 {suffix}：{e}")

    # 同步转写（SenseVoice 本地，短音频秒级）
    text = transcribe_audio(str(final_path))
    if not text.strip():
        raise HTTPException(status_code=422, detail="录音转写结果为空，请确认录音清晰、非静音")

    # 把录音存档到 MinIO 语音专用桶，拿到可长期访问的 URL —— 刷新页面后语音条仍能回放。
    # 上传失败不阻断本次问答，只是这条语音刷新后不可回放（降级为空串）。
    audio_url = ""
    try:
        from app.clients.minio_utils import upload_audio_file
        day = datetime.now().strftime("%Y-%m-%d")
        audio_url = upload_audio_file(
            str(final_path),
            f"speak_content/{day}/{final_path.name}",
            content_type="audio/mpeg",
        )
    except Exception as e:
        logger.warning(f"录音上传 MinIO 失败（不影响本次问答，该语音刷新后不可回放）：{e}")

    return {
        "query": text,
        "filename": final_path.name,
        "saved_path": str(final_path),
        "audio_url": audio_url,
    }

# 创建处理sse请求的路径处理函数
@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# 健康检查
@app.get("/health")
def health():
    return {"ok": True}

# 查询最近的10条历史对话
# 会话列表：供前端左侧会话栏展示与切换
@app.get("/sessions")
async def sessions(limit: int = 50):
    """列出最近会话（标题 / 最后活动时间 / 提问条数），按最后活动倒序。"""
    try:
        return {"items": list_chat_sessions(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"sessions error: {e}")


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 10):
    try:
        history_list = get_recent_messages(session_id, limit)
        for history in history_list:
            history["_id"] = str(history["_id"])
        return {"session_id": session_id, "items": history_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")

# 清空历史对话
@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    count =  clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}

if __name__ == "__main__":
    # 说明：直接传入 app 对象而非 "模块:变量" 字符串，
    # 保证通过 `uv run python -m app.query_process.api.query_service` 也能正常启动。
    uvicorn.run(app, host="127.0.0.1", port=9091)

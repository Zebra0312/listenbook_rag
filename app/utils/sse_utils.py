import json
import queue
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request

from app.core.logger import logger


class SSEEvent:
    READY = "ready"         # 连接建立
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案
    ERROR = "error"         # 错误信息
    CLOSE = "__close__"     # 关闭连接信号


# 全局 SSE 会话队列存储
# Key: session_id, Value: queue.Queue
_session_stream: Dict[str, queue.Queue] = {}

def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取指定 session 的队列"""
    return _session_stream.get(session_id)

def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建并注册一个新的 SSE 队列"""
    logger.debug(f"[SSE] 创建会话队列: {session_id}")
    q = queue.Queue()
    _session_stream[session_id] = q
    return q

def remove_sse_queue(session_id: str):
    """移除指定 session 的队列"""
    logger.debug(f"[SSE] 移除会话队列: {session_id}")
    _session_stream.pop(session_id, None)

def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    # print(f"[SSE] Packing event: {event}, payload: {payload[:50]}...")
    return f"event: {event}\ndata: {payload}\n\n"

def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """
    通过 session_id 推送事件
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        # print(f"[SSE] Pushing to session {session_id}: {event}")
        stream_queue.put({"event": event, "data": data})
    else:
        # 客户端断开后队列已被清理，后续推送必然落空：属正常情况，降为 DEBUG 避免刷屏
        logger.debug(f"[SSE] 会话队列不存在，跳过推送（客户端可能已断开）: {session_id} / {event}")

async def sse_generator(session_id: str, request: Request):
    """
    SSE 生成器，用于 FastAPI 的 StreamingResponse
    """
    logger.debug(f"[SSE] 建立连接: {session_id}")
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        # 如果没有对应的队列，直接结束
        logger.warning(f"[SSE] 未找到会话队列，连接直接结束: {session_id}")
        return

    loop = asyncio.get_running_loop()
    try:
        # 发送连接建立信号
        logger.debug(f"[SSE] 发送 ready 事件: {session_id}")
        yield _sse_pack("ready", {})

        while True:
            # 若客户端断开，尽快退出
            if await request.is_disconnected():
                logger.debug(f"[SSE] 客户端断开: {session_id}")

                break

            try:
                # 使用 run_in_executor 避免阻塞 async 事件循环
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                # print(f"[SSE] Queue empty for {session_id}, waiting...")
                continue

            event = msg.get("event")
            data = msg.get("data")
            
            # print(f"[SSE] Yielding event {event} for {session_id}")

            # 特殊关闭事件
            if event == "__close__":
                logger.debug(f"[SSE] 收到关闭信号: {session_id}")
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        # 客户端刷新页面/关标签页时会走到这里（Windows 上伴随 WinError 10054），属正常断连
        logger.debug(f"[SSE] 客户端断开(Cancelled/Reset/Pipe): {session_id}")
        # 生成器被取消/对端断开：静默退出
        return
    except Exception as e:
        logger.error(f"[SSE] 生成器异常 {session_id}: {e}")
    finally:
        logger.debug(f"[SSE] 连接结束: {session_id}")
        # 清理资源
        remove_sse_queue(session_id)
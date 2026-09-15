import os

from app.core.logger import logger, node_log, step_log
from app.lm.asr_utils import transcribe_audio
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 支持的音频输入后缀（用户提问时上传的音频文件格式）
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg")


@step_log("step_1_detect_audio")
def step_1_detect_audio(original_query: str):
    """
    步骤1：判断用户输入是否为音频文件路径
    :param original_query: 用户原始输入
    :return: (是否音频, 清洗后的输入)
    """
    query = (original_query or "").strip()
    if not query:
        return False, query
    # 后缀命中且文件确实存在，才视为音频输入（避免误判普通文本）
    is_audio = query.lower().endswith(AUDIO_EXTENSIONS) and os.path.exists(query)
    return is_audio, query


@step_log("step_2_transcribe_query")
def step_2_transcribe_query(audio_path: str) -> str:
    """
    步骤2：把音频提问转写为文本
    :param audio_path: 音频文件路径
    :return: 转写文本
    """
    return transcribe_audio(audio_path)


@node_log("node_query_intent")
def node_query_intent(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 输入意图识别 (node_query_intent)
    为什么叫这个名字: 作为检索链路的入口，判断用户输入是一段文本问题、还是一个音频文件；
        若为音频文件，则复用与导入侧相同的语音识别工具先转写成文本，再继续走既有检索流程。
    实现内容:
        1. 判断 original_query 是否为音频文件路径（后缀 + 文件存在性双重校验）
        2. 音频输入 → 转写为文本 → 覆盖 original_query
        3. 文本输入 → 直接透传，不改变原有逻辑
    """
    # 记录节点的运行状态
    add_running_task(state["session_id"], "node_query_intent")

    original_query = state.get("original_query") or ""
    # 步骤1：判断输入是否音频
    is_audio, query = step_1_detect_audio(original_query)

    if is_audio:
        # 步骤2：音频转写，覆盖 original_query，后续检索流程完全复用
        logger.info(f"检测到音频输入，开始转写：{query}")
        text = step_2_transcribe_query(query)
        if not text.strip():
            logger.warning("音频转写结果为空，保持原始输入不变")
        else:
            logger.info(f"音频转写完成，作为查询文本继续检索（{len(text)} 字）")
            state["original_query"] = text
    else:
        logger.debug("输入为普通文本，直接透传进入既有检索流程")

    # 记录节点的完成状态
    add_done_task(state["session_id"], "node_query_intent")
    return state


if __name__ == "__main__":
    from app.query_process.agent.state import create_query_default_state

    logger.info("===== 开始 node_query_intent 节点单元测试 =====")
    # 测试1：普通文本
    s1 = create_query_default_state(session_id="t1", original_query="《三体》适合谁听？")
    r1 = node_query_intent(s1)
    print("文本输入 ->", r1.get("original_query"))

    # 测试2：音频文件（不存在，应透传）
    s2 = create_query_default_state(session_id="t2", original_query="output/不存在.mp3")
    r2 = node_query_intent(s2)
    print("不存在的音频 ->", r2.get("original_query"))
    logger.info("===== 结束 node_query_intent 节点单元测试 =====")

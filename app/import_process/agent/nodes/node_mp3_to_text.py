import os
from pathlib import Path

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.asr_utils import transcribe_audio
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_running_task, add_done_task


@step_log("step_1_validate_mp3")
def step_1_validate_mp3(state: ImportGraphState):
    """
    步骤1：校验 mp3 路径与输出目录
    :param state: 流程状态字典，包含 mp3_path、local_dir
    :return: (mp3_path, local_dir)
    """
    # 获取 mp3 文件路径
    mp3_path = state.get("mp3_path")
    # 判断 mp3_path 是否为空、文件是否存在
    if not mp3_path or not os.path.exists(mp3_path):
        logger.error(f"mp3 路径为空或文件不存在：{mp3_path}")
        raise FileNotFoundError(f"mp3 文件不存在：{mp3_path}")

    # 输出目录：默认取项目 output 目录，与 pdf/md 处理产物保持一致
    local_dir = state.get("local_dir") or str(PROJECT_ROOT / "output")
    os.makedirs(local_dir, exist_ok=True)
    return mp3_path, local_dir


@step_log("step_2_transcribe")
def step_2_transcribe(mp3_path: str) -> str:
    """
    步骤2：调用语音识别工具把音频转写为纯文本
    :param mp3_path: 音频文件路径
    :return: 带标点的纯文本
    """
    text = transcribe_audio(mp3_path)
    # 转写结果为空（如纯静音/识别失败），直接报错，避免把空内容送入下游切分
    if not text.strip():
        logger.error(f"音频转写结果为空：{mp3_path}")
        raise RuntimeError(f"音频转写结果为空：{mp3_path}")
    return text


@step_log("step_3_write_md")
def step_3_write_md(text: str, local_dir: str, file_title: str) -> str:
    """
    步骤3：把转写文本落盘为 .md 文件（供追溯 source_path 与备份）
    :param text: 转写文本
    :param local_dir: 输出目录
    :param file_title: 文件标题（去后缀）
    :return: md 文件路径
    """
    md_path = str(Path(local_dir) / f"{file_title}.md")
    Path(md_path).write_text(text, encoding="utf-8")
    logger.info(f"转写文本已落盘：{md_path}（{len(text)} 字）")
    return md_path


@node_log("node_mp3_to_text")
def node_mp3_to_text(state: ImportGraphState) -> ImportGraphState:
    """
    节点: MP3 音频转文本 (node_mp3_to_text)
    为什么叫这个名字: 承接 node_entry 识别出的 mp3 文件，用语音识别模型转成纯文本，
        作为 Markdown 内容进入下游的文档切分节点——切分及其之后的全部流程完全复用，不做改动。
    实现内容:
        1. 校验 mp3 路径与输出目录
        2. 调用 SenseVoice 把音频转写为带标点的纯文本
        3. 转写结果写入 md_content，并落盘为 .md（供 source_path 追溯）
    """
    # 记录节点的运行状态
    add_running_task(state["task_id"], "node_mp3_to_text")

    # 步骤1：校验路径与输出目录
    mp3_path, local_dir = step_1_validate_mp3(state)
    # 步骤2：语音转文本
    text = step_2_transcribe(mp3_path)

    # 文件标题（node_entry 已设置；这里兜底取文件名去后缀）
    file_title = state.get("file_title") or Path(mp3_path).stem
    state["file_title"] = file_title

    # 步骤3：落盘 + 写入 state，供下游 node_document_split 直接消费
    md_path = step_3_write_md(text, local_dir, file_title)
    state["md_path"] = md_path
    state["md_content"] = text

    # 记录节点的完成状态
    add_done_task(state["task_id"], "node_mp3_to_text")
    return state


if __name__ == "__main__":
    from app.import_process.agent.state import create_default_state

    logger.info("===== 开始 node_mp3_to_text 节点单元测试 =====")
    test_state = create_default_state(
        task_id="test_mp3_001",
        mp3_path="output/三体_有声书.mp3",
        file_title="三体_有声书",
    )
    result = node_mp3_to_text(test_state)
    print("转写字数:", len(result.get("md_content", "")))
    print("md_path:", result.get("md_path"))
    print("内容预览:", result.get("md_content", "")[:200])
    logger.info("===== 结束 node_mp3_to_text 节点单元测试 =====")

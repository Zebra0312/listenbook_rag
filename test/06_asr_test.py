"""
06 · 语音识别（MP3 转文本）测试

作用：验证 FunASR SenseVoice 本地语音识别工具是否可用，直接调用转写工具跑一次真实转写。

前置条件：
    - 已下载 SenseVoice 与 FSMN-VAD 模型到 D:/ai_models（配置在 .env 的
      SENSEVOICE_MODEL_PATH / FSMN_VAD_MODEL_PATH）；
    - funasr 已安装；ffmpeg 可用（工具模块内已做 PATH 兜底）。

用法：
    cd listenbook_rag
    uv run python test/06_asr_test.py                          # 用 output/ 下的示例音频
    uv run python test/06_asr_test.py D:/path/to/音频.mp3       # 指定音频
    uv run python test/06_asr_test.py --dry-run                # 只检查配置，不加载模型
"""
import argparse
import os
import sys
import time
from pathlib import Path

# 保证可以直接 `python test/06_asr_test.py` 运行（无需手动设置 PYTHONPATH）
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from app.conf.asr_config import asr_config
from app.core.logger import logger
from app.lm.asr_utils import transcribe_audio

LINE = "-" * 70


def check_env() -> bool:
    """校验 ASR 所需配置是否齐全，返回是否通过"""
    problems = []
    if not asr_config.sensevoice_path or not os.path.exists(asr_config.sensevoice_path):
        problems.append("SenseVoice 模型路径未配置或不存在（SENSEVOICE_MODEL_PATH）")
    if not asr_config.fsmn_vad_path or not os.path.exists(asr_config.fsmn_vad_path):
        problems.append("FSMN-VAD 模型路径未配置或不存在（FSMN_VAD_MODEL_PATH）")
    if problems:
        for p in problems:
            logger.error(p)
        return False

    logger.success("ASR 配置校验通过")
    logger.info(f"  SenseVoice : {asr_config.sensevoice_path}")
    logger.info(f"  FSMN-VAD   : {asr_config.fsmn_vad_path}")
    logger.info(f"  设备 / 语言 : {asr_config.asr_device} / {asr_config.asr_language}")
    return True


def run_transcribe(audio_path: str) -> bool:
    """转写单个音频并打印结果，返回是否成功"""
    logger.info(f"测试音频: {audio_path}")
    t0 = time.time()
    text = transcribe_audio(audio_path)
    cost = time.time() - t0
    logger.info(f"耗时: {cost:.1f} 秒")
    logger.info(f"字数: {len(text)}")
    logger.info(f"转写结果:\n{text}")
    return bool(text.strip())


def pick_test_file(arg_path):
    """选择测试音频：优先用命令行传入的，其次在 mp3/ 下自动查找"""
    if arg_path and os.path.exists(arg_path):
        return arg_path
    # 未指定时，在 mp3/query/、mp3/import/、output/ 下找第一个音频文件
    for d in (PROJECT_ROOT_DIR / "mp3" / "query",
              PROJECT_ROOT_DIR / "mp3" / "import",
              PROJECT_ROOT_DIR / "output"):
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"):
                return str(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="听书智库 · 语音识别（MP3 转文本）测试")
    parser.add_argument("audio", nargs="?", default=None, help="音频文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置，不加载模型")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("06 · 语音识别测试 开始")
    logger.info("=" * 70)

    if not check_env():
        sys.exit(1)

    if args.dry_run:
        logger.info("--dry-run：仅校验配置，跳过模型加载与转写")
        logger.success("06 配置校验通过")
        return

    audio = pick_test_file(args.audio)
    if not audio:
        logger.error("未找到测试音频。请传入一个音频路径，或用 TTS 生成一个示例音频放到 output/ 下")
        sys.exit(1)

    try:
        ok = run_transcribe(audio)
    except Exception as e:
        logger.error(f"转写失败：{e}")
        sys.exit(1)

    if ok:
        logger.success("✅ 06 测试通过：音频转文本正常")
    else:
        logger.error("06 测试失败：转写结果为空")
        sys.exit(1)


if __name__ == "__main__":
    main()

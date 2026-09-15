import os
import shutil

# 确保 ffmpeg 在 PATH（funasr 读取 mp3 依赖 ffmpeg 后端解码）。
# 说明：ffmpeg 是可执行程序而非 Python 包，位于 D:\ai_models\tools\ffmpeg\bin 并已加入用户 PATH，
#      但某些长驻进程（如后台服务冷启动时）PATH 可能是旧快照。这里做兜底：找不到就临时补上，
#      保证服务端加载 mp3 不因 ffmpeg 缺失而失败。
_FFMPEG_FALLBACK = r"D:\ai_models\tools\ffmpeg\bin"
if shutil.which("ffmpeg") is None and os.path.isdir(_FFMPEG_FALLBACK):
    os.environ["PATH"] = _FFMPEG_FALLBACK + ";" + os.environ.get("PATH", "")

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from app.core.logger import logger
from app.conf.asr_config import asr_config

# 模型单例对象，避免重复初始化
_asr_model = None


def get_asr_model():
    """
    获取 SenseVoice 语音识别模型单例对象，自动加载环境变量配置
    :return: 初始化完成的 AutoModel 实例
    :raise: 模型路径未配置或初始化失败时抛出异常
    """
    global _asr_model
    # 单例模式：已初始化则直接返回，避免重复加载模型（加载耗时较长）
    if _asr_model is not None:
        logger.debug("SenseVoice 模型单例已存在，直接返回实例")
        return _asr_model

    # 从环境变量加载配置
    model_path = asr_config.sensevoice_path
    vad_path = asr_config.fsmn_vad_path
    device = asr_config.asr_device or "cpu"

    # 主模型路径为空时直接报错，避免 AutoModel 去 ModelScope 在线下载
    if not model_path:
        logger.error("未配置 SenseVoice 模型路径（SENSEVOICE_MODEL_PATH），无法初始化语音识别模型")
        raise ValueError("缺少 SENSEVOICE_MODEL_PATH 配置，请在 .env 中补全")

    logger.info(
        "开始初始化 SenseVoice 语音识别模型",
        extra={"model_path": model_path, "vad_path": vad_path, "device": device}
    )

    try:
        # 初始化模型：主模型 + VAD（长音频切分）+ 纯 CPU 推理
        _asr_model = AutoModel(
            model=model_path,
            vad_model=vad_path or None,
            vad_kwargs={"max_single_segment_time": asr_config.asr_max_segment_ms},
            device=device,
            disable_update=True,  # 禁止联网检查模型更新，走纯本地
        )
        logger.success("SenseVoice 模型初始化成功")
        return _asr_model
    except Exception as e:
        logger.error(f"SenseVoice 模型初始化失败：{str(e)}", exc_info=True)
        raise  # 向上抛出异常，由调用方处理


def transcribe_audio(audio_path: str) -> str:
    """
    将音频文件转写为带标点的纯文本（SenseVoice + FSMN-VAD 长音频切分）
    :param audio_path: 音频文件路径（支持 mp3/wav/m4a/flac 等，长音频会自动按 VAD 切段）
    :return: 转写后的纯文本（已做标点、数字逆文本规整）
    :raise: 音频不存在或转写过程中的异常，由调用方捕获处理
    """
    # 入参合法性校验
    if not audio_path or not os.path.exists(audio_path):
        logger.warning(f"音频文件不存在或路径为空：{audio_path}")
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")

    logger.info(f"开始转写音频：{audio_path}")
    try:
        # 加载 SenseVoice 模型单例
        model = get_asr_model()
        # 执行转写：batch_size_s 控制动态批处理时长，merge_vad 合并被 VAD 切碎的同句短段
        result = model.generate(
            input=audio_path,
            cache={},
            language=asr_config.asr_language or "zh",
            use_itn=True,
            batch_size_s=asr_config.asr_batch_size_s,
            merge_vad=True,
            merge_length_s=asr_config.asr_merge_length_s,
        )
        # 结果为空（如纯静音）时返回空串，由调用方决定如何处理
        if not result or not result[0].get("text"):
            logger.warning(f"音频转写结果为空：{audio_path}")
            return ""

        raw_text = result[0]["text"]
        # 清洗 SenseVoice 输出的特殊 token（<|zh|>、<|NEUTRAL|>、<|Speech|> 等），保留纯文本 + 标点
        text = rich_transcription_postprocess(raw_text)
        logger.success(f"音频转写完成，共 {len(text)} 字")
        return text
    except Exception as e:
        logger.error(f"音频转写失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做降级处理


"""
核心设计亮点&适配说明：
1. 单例模式：SenseVoice 模型加载耗时较长，全局仅初始化一次，避免每个音频都重复加载；
2. 长音频切分：内置 FSMN-VAD，把几小时音频按语音活动自动切段再逐段识别，避免一次性送入
   超长音频导致内存溢出或漏识别（听书音频动辄数小时，此为必需能力）；
3. 逆文本规整：use_itn=True 自动把"二零二四年"转成"2024年"并补齐标点，识别结果可直接入库检索；
4. rich_transcription_postprocess 清洗：去掉语种/情感/事件特殊 token，避免污染后续切分与检索；
5. 单一职责：本模块只负责"音频→文本"，结果落盘/入库交由调用方（导入侧写 md、检索侧直接作为 query）；
6. 分级日志 + 异常不吞：初始化/转写/失败全程可追踪，异常向上抛出由节点层做降级。
"""

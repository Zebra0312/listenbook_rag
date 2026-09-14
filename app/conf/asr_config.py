# 导入核心依赖：数据类、环境变量读取
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（保持和其它配置类一致，只需执行一次）
load_dotenv()

# 定义语音识别配置（适配 FunASR SenseVoice 的所有配置，类名 asr_config）
@dataclass
class ASRConfig:
    sensevoice_path: str   # SenseVoice 主模型本地路径
    fsmn_vad_path: str     # FSMN-VAD 长音频切分模型本地路径
    asr_device: str        # 运行设备(cpu/cuda:0)
    asr_language: str      # 识别语言(zh/en/yue/ja/ko/auto)
    asr_batch_size_s: int  # 动态批处理时长(秒)，越大吞吐越高、越吃内存
    asr_max_segment_ms: int  # 单个切段的最大时长(毫秒)，超过则强制切分
    asr_merge_length_s: int  # 相邻短段的合并阈值(秒)，避免把一句话切碎
    asr_debug: bool        # 调试开关（1=True/0=False）

# 实例化配置对象，和 embedding_config 风格保持一致
asr_config = ASRConfig(
    sensevoice_path=os.getenv("SENSEVOICE_MODEL_PATH"),
    fsmn_vad_path=os.getenv("FSMN_VAD_MODEL_PATH"),
    asr_device=os.getenv("ASR_DEVICE"),
    asr_language=os.getenv("ASR_LANGUAGE"),
    # 数值型配置统一带默认值，避免 .env 缺失时 int(None) 报错
    asr_batch_size_s=int(os.getenv("ASR_BATCH_SIZE_S", "60")),
    asr_max_segment_ms=int(os.getenv("ASR_MAX_SINGLE_SEGMENT_MS", "30000")),
    asr_merge_length_s=int(os.getenv("ASR_MERGE_LENGTH_S", "15")),
    # 特殊处理：将.env中的1/0转为布尔值，兼容常见的数字/字符串格式
    asr_debug=os.getenv("ASR_DEBUG") in ("1", "True", "true", 1)
)

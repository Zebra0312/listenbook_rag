# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（确保os.getenv能获取到MinIO相关配置）
load_dotenv()


# 定义MinIO对象存储服务配置（与LLMConfig风格一致，字段对应.env配置项）
@dataclass
class MinIOConfig:
    endpoint: str    # MinIO服务地址（含http/https和端口）
    public_endpoint: str  # 浏览器访问图片/语音时用的地址（见下方说明）
    access_key: str  # MinIO访问密钥（对应MINIO_ACCESS_KEY）
    secret_key: str  # MinIO秘钥（对应MINIO_SECRET_KEY）
    bucket_name: str # MinIO默认存储桶名（知识库文件专用）
    minio_img_dir: str #Minio存储图片的文件夹
    audio_bucket_name: str # 语音提问音频专用桶名（录音存档）
    minio_secure: bool # 是否使用ssl加密 http 还是 https


# 实例化MinIO配置对象，自动从.env读取配置并绑定
minio_config = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    # 对外（浏览器）访问地址：默认与 endpoint 相同。
    # 当 MinIO 在虚拟机/内网、宿主机做了端口映射时才需要单独指定，
    # 例如后端直连 192.168.10.124:9000，而局域网同事只能走 192.168.12.196:9000。
    # 不配 MINIO_PUBLIC_ENDPOINT 时自动回落到 MINIO_ENDPOINT，行为不变。
    public_endpoint=os.getenv("MINIO_PUBLIC_ENDPOINT") or os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),
    audio_bucket_name=os.getenv("MINIO_AUDIO_BUCKET"),
    minio_secure=os.getenv("MINIO_SECURE") == "True"
)
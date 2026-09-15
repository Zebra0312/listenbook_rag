# 导入Python内置模块
import os
import json

# 导入MinIO官方Python SDK核心类（用于MinIO对象存储的客户端操作）
from minio import Minio

# 导入项目内部配置与日志工具
from app.conf.minio_config import minio_config  # MinIO相关配置（端点、密钥、桶名等）
from app.core.logger import logger            # 项目统一日志工具

# 全局MinIO客户端实例（单例模式，避免重复创建连接，提升性能）
_minio_client = None


# 1. 定义MinIO客户端连接创建函数（私有函数，仅内部调用）
def _create_minio_client() -> Minio:
    """
    创建并返回MinIO客户端连接
    核心作用：读取配置文件中的MinIO参数，初始化客户端连接
    :return: 初始化完成的MinIO客户端对象
    """
    return Minio(
        endpoint=minio_config.endpoint,        # MinIO服务端点（IP:端口）
        access_key=minio_config.access_key,    # MinIO访问密钥
        secret_key=minio_config.secret_key,    # MinIO秘密密钥
        secure=minio_config.minio_secure       # 是否启用HTTPS（True/False）
    )


# 2. 定义桶访问策略生成函数（私有函数，仅内部调用）
def _set_bucket_policy(bucket_name: str) -> str:
    """
    生成MinIO桶的访问策略字符串（JSON格式）
    核心策略：允许所有用户（Principal: "*"）对桶内所有对象执行读取操作（s3:GetObject）
    适配场景：图片上传后需公开访问（如MD中图片在线URL）
    :param bucket_name: 目标桶名
    :return: 序列化后的JSON格式访问策略字符串
    """
    # 策略模板（遵循AWS S3策略规范，MinIO兼容该规范）
    policy = {
        "Version": "2012-10-17",  # 策略版本（固定值，兼容S3标准）
        "Statement": [
            {
                "Effect": "Allow",  # 策略效果：允许访问
                "Principal": {"AWS": ["*"]},  # 授权对象：所有用户
                "Action": ["s3:GetObject"],  # 授权操作：读取桶内对象
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],  # 授权范围：桶内所有对象
            }
        ],
    }
    # 将字典策略序列化为JSON字符串，供MinIO设置使用
    return json.dumps(policy)


# 3. 定义桶初始化函数（私有函数，仅内部调用）
def _create_bucket_ready(client: Minio, bucket_name: str = None):
    """
    检查MinIO桶是否存在，不存在则创建，并设置访问策略
    核心作用：确保上传所需的桶已就绪，避免上传失败
    :param client: 已初始化的MinIO客户端对象
    :param bucket_name: 目标桶名；不传则用配置里的默认桶（知识库图片桶）
    """
    bucket_name = bucket_name or minio_config.bucket_name  # 未指定则用默认桶
    # 检查桶是否存在
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)  # 不存在则创建桶
        logger.info(f"MinIO桶 {bucket_name} 已创建")
    else:
        # 桶已存在，无需重复创建
        logger.info(f"MinIO桶 {bucket_name} 已存在，无需重复创建")
    # 关键：无论桶是本次新建还是早已存在，都要确保公开读策略已设置。
    # 历史 bug：策略原本只在“新建桶”分支里设置，导致已存在的桶永远没有策略，
    # 浏览器匿名访问图片会返回 403。set_bucket_policy 是幂等的，重复设置无副作用。
    try:
        client.set_bucket_policy(bucket_name, _set_bucket_policy(bucket_name))
        logger.info(f"MinIO桶 {bucket_name} 公开读策略已设置（允许匿名 s3:GetObject）")
    except Exception as e:
        logger.error(f"设置MinIO桶 {bucket_name} 访问策略失败：{e}")


def get_minio_client() -> Minio:
    """
    获取全局MinIO客户端（懒加载模式，无锁版本，适配单线程场景）
    核心逻辑：
    1. 首次调用：初始化客户端 + 检查/创建桶 + 设置策略，将客户端实例赋值给全局变量
    2. 后续调用：直接复用全局客户端实例，避免重复创建连接（提升性能、节省资源）
    :return: 全局唯一的MinIO客户端对象
    """
    # 声明使用全局变量（修改全局变量需显式声明）
    global _minio_client

    # 懒加载：仅在客户端未初始化时执行创建逻辑
    if _minio_client is None:
        logger.info("开始初始化MinIO客户端（首次调用，执行懒加载）")
        client = _create_minio_client()          # 创建客户端连接
        _create_bucket_ready(client)             # 检查并初始化默认桶（知识库图片）
        # 语音提问音频桶（录音存档），与图片桶分开管理，便于单独清理与扩容
        if minio_config.audio_bucket_name:
            _create_bucket_ready(client, minio_config.audio_bucket_name)
        _minio_client = client                   # 赋值给全局变量，供后续复用
        logger.info("MinIO客户端初始化完成，已就绪可使用")

    # 复用全局客户端实例，直接返回
    return _minio_client


# 4. 语音音频：拼装公开访问地址
def audio_public_url(object_name: str) -> str:
    """
    拼出语音音频在MinIO上的公开访问地址
    核心作用：桶已设置公开读策略，浏览器 <audio> 可直接播放该 URL
    :param object_name: 桶内对象名（可带目录前缀）
    :return: 完整 http(s) URL
    """
    bucket = minio_config.audio_bucket_name
    rel = object_name.lstrip("/")
    scheme = "https" if minio_config.minio_secure else "http"
    return f"{scheme}://{minio_config.public_endpoint}/{bucket}/{rel}"


# 5. 语音音频：上传本地文件并返回公开URL
def upload_audio_file(local_path: str, object_name: str, content_type: str = "audio/mpeg") -> str:
    """
    把本地音频文件上传到语音专用桶，返回可公开访问的URL
    核心作用：让用户发过的语音长期可回放（刷新页面/换设备都不丢），
             而不是只存在于浏览器内存（blob URL）里、刷新即失效。
    :param local_path: 本地音频文件路径
    :param object_name: 桶内对象名（建议带日期目录，如 speak_content/2026-09-15/xxx.mp3）
    :param content_type: MIME 类型，默认 audio/mpeg
    :return: 公开访问URL
    :raise: 未配置音频桶或上传失败时抛出异常，由调用方决定如何降级
    """
    bucket = minio_config.audio_bucket_name
    if not bucket:
        raise ValueError("未配置 MINIO_AUDIO_BUCKET，无法上传语音音频")
    client = get_minio_client()
    rel = object_name.lstrip("/")
    client.fput_object(
        bucket_name=bucket,
        object_name=rel,
        file_path=local_path,
        content_type=content_type,
    )
    url = audio_public_url(rel)
    logger.info(f"语音音频已上传 MinIO：{bucket}/{rel} -> {url}")
    return url

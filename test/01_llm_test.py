"""
01 · 大模型连通性测试

作用：导入链路依赖两个大模型，任何一个不可用都会让导入失败，这里先做前置体检。
    - 文本模型 LLM_DEFAULT_MODEL：node_item_name_recognition 识别书籍主体
    - 视觉模型 VL_MODEL          ：node_md_img 生成图片摘要

用法：
    cd listenbook_rag
    uv run python test/01_llm_test.py
"""
import sys
from pathlib import Path

# 保证可以直接 `python test/01_llm_test.py` 运行（无需手动设置 PYTHONPATH）
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from app.conf.lm_config import lm_config
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client


def check_config() -> bool:
    """校验 .env 中的关键配置是否齐全"""
    missing = []
    if not lm_config.api_key:
        missing.append("OPENAI_API_KEY")
    if not lm_config.base_url:
        missing.append("OPENAI_BASE_URL")
    if not lm_config.llm_model:
        missing.append("LLM_DEFAULT_MODEL")
    if not lm_config.lv_model:
        missing.append("VL_MODEL")
    if missing:
        logger.error(f"配置缺失：{', '.join(missing)}，请检查 listenbook_rag/.env")
        return False
    logger.info(f"配置检查通过：文本模型={lm_config.llm_model}，视觉模型={lm_config.lv_model}")
    return True


def test_text_llm() -> bool:
    """文本模型连通性（书籍主体识别用）"""
    logger.info(f"----- 测试文本模型：{lm_config.llm_model} -----")
    try:
        llm = get_llm_client()
        result = llm.invoke("请只回复两个字：成功")
        content = (result.content or "").strip()
        logger.success(f"文本模型响应：{content}")
        return bool(content)
    except Exception as e:
        logger.error(f"文本模型调用失败：{e}")
        return False


def test_vision_llm() -> bool:
    """视觉模型连通性（图片摘要用）。注意：这里只发纯文本，验证的是模型与鉴权通不通"""
    logger.info(f"----- 测试视觉模型：{lm_config.lv_model} -----")
    try:
        vl_model = get_llm_client(model=lm_config.lv_model)
        result = vl_model.invoke("请只回复两个字：成功")
        content = (result.content or "").strip()
        logger.success(f"视觉模型响应：{content}")
        return bool(content)
    except Exception as e:
        logger.error(f"视觉模型调用失败：{e}")
        return False


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("01 · 大模型连通性测试 开始")
    logger.info("=" * 70)

    if not check_config():
        logger.warning("排查提示：打开 listenbook_rag/.env，补齐百炼的 API Key 与模型名后重试")
        sys.exit(1)

    text_ok = test_text_llm()
    vision_ok = test_vision_llm()

    logger.info("-" * 70)
    logger.info(f"文本模型：{'✅ 通过' if text_ok else '❌ 失败'}")
    logger.info(f"视觉模型：{'✅ 通过' if vision_ok else '❌ 失败'}")
    if not (text_ok and vision_ok):
        logger.warning("排查提示：确认 OPENAI_BASE_URL 是否指向百炼兼容模式地址、模型名是否已开通")
    logger.info("=" * 70)
    sys.exit(0 if (text_ok and vision_ok) else 1)

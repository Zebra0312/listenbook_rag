"""
05 · 检索问答全链路测试（后端流程）

作用：不经过前端页面，直接调用 LangGraph 检索图跑完整条问答链路，并打印中间结果。
      链路：node_item_name_confirm → 三路并行（向量检索 / HyDE 检索 / 网络搜索）
            → node_rrf → node_rerank → node_answer_output

前置条件：
    - 已执行过 test/04_import_test.py（切片库里有数据，且主体库里能对齐到书名）；
    - Milvus 与 MongoDB 可用；LLM 可用；本地 BGE-Reranker-large 模型存在。

用法：
    cd listenbook_rag
    uv run python test/05_query_test.py
    uv run python test/05_query_test.py "《三体》适合谁听？"
    uv run python test/05_query_test.py --session my_sess "有没有适合通勤听的悬疑小说"
    uv run python test/05_query_test.py --dry-run        # 只检查图与依赖，不发起真实问答
"""
import argparse
import re
import sys
import time
from pathlib import Path

# 保证可以直接 `python test/05_query_test.py` 运行（无需手动设置 PYTHONPATH）
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.query_process.agent.main_graph import kb_query_app
from app.query_process.agent.state import create_query_default_state

LINE = "-" * 70
DEFAULT_QUESTION = "《三体》适合谁听？核心看点是什么？"
# 用于统计答案里出现的图片地址
IMG_RE = re.compile(r"https?://[^\s)\]<>\"'】]*?\.(?:png|jpe?g|gif|webp|bmp|svg)", re.IGNORECASE)


def check_env() -> bool:
    """检查检索链路依赖的服务与配置"""
    ok = True
    # Milvus
    client = get_milvus_client()
    if not client:
        logger.error("Milvus 连接失败，请先启动 Milvus")
        ok = False
    else:
        for name in (milvus_config.chunks_collection, milvus_config.item_name_collection):
            if not client.has_collection(collection_name=name):
                logger.error(f"集合 {name} 不存在，请先执行 test/04_import_test.py 导入数据")
                ok = False
            else:
                logger.info(f"集合 {name} 存在 ✅")
    return ok


def run_query(session_id: str, question: str) -> dict:
    """执行检索图，返回最终状态"""
    init_state = create_query_default_state(
        session_id=session_id,
        original_query=question,
        is_stream=False,
    )

    logger.info(f"会话ID：{session_id}")
    logger.info(f"用户问题：{question}")
    logger.info(LINE)

    start = time.time()
    final_state = kb_query_app.invoke(init_state)
    logger.info(f"链路耗时：{time.time() - start:.2f}s")
    return final_state


def report(state: dict) -> bool:
    """打印检索过程与最终答案"""
    logger.info("===== 主体确认 =====")
    item_names = state.get("item_names") or []
    logger.info(f"识别并确认的书籍主体：{item_names or '（未确认）'}")
    logger.info(f"改写后的问题：{state.get('rewritten_query', '')}")

    if state.get("answer") and not state.get("rrf_chunks"):
        # 走到"澄清 / 未找到书籍"的分支
        logger.info(LINE)
        logger.info("===== 直接返回的兜底/澄清答案 =====")
        logger.info(state["answer"])
        return True

    logger.info(LINE)
    logger.info("===== 三路召回 =====")
    logger.info(f"向量检索命中：{len(state.get('embedding_chunks') or [])} 条")
    logger.info(f"HyDE 检索命中：{len(state.get('hyde_embedding_chunks') or [])} 条")
    logger.info(f"网络搜索命中：{len(state.get('web_search_docs') or [])} 条")
    logger.info(f"RRF 融合后：{len(state.get('rrf_chunks') or [])} 条")
    reranked = state.get("reranked_docs") or []
    logger.info(f"重排截断后：{len(reranked)} 条")

    if reranked:
        logger.info(LINE)
        logger.info("===== 重排 Top-3（验证书籍元数据是否贯通）=====")
        for i, doc in enumerate(reranked[:3], start=1):
            logger.info(
                f"  {i}. score={float(doc.get('score') or 0):.4f}，"
                f"书名={doc.get('book_name', '')}，作者={doc.get('author', '')}，"
                f"类型={doc.get('content_type', '')}，来源文件={doc.get('source_file', '')}"
            )
            logger.info(f"     片段：{(doc.get('text') or '')[:60]}...")

    answer = state.get("answer") or ""
    logger.info(LINE)
    logger.info("===== 最终答案 =====")
    logger.info(answer if answer else "（空）")

    images = []
    for doc in reranked:
        images.extend(IMG_RE.findall(doc.get("text") or ""))
    if images:
        logger.info(f"（答案可引用的图片地址：{len(set(images))} 个）")

    # 校验项
    ok = bool(item_names) and bool(answer) and len(reranked) > 0
    logger.info(LINE)
    logger.info(f"已确认书籍主体：{'✅' if item_names else '❌'}")
    logger.info(f"检索到参考切片：{'✅' if reranked else '❌'}")
    logger.info(f"生成最终答案  ：{'✅' if answer else '❌'}")
    return ok


def show_history(session_id: str) -> None:
    """回查 MongoDB 里的会话记录，验证多轮记忆是否写入"""
    try:
        from app.clients.mongo_history_utils import get_recent_messages
        history = get_recent_messages(session_id, limit=4)
    except Exception as e:
        logger.warning(f"读取历史记录失败：{e}")
        return
    logger.info(LINE)
    logger.info(f"===== 会话历史（最近 {len(history)} 条）=====")
    for item in history:
        role = item.get("role")
        text = (item.get("text") or "").replace("\n", " ")
        logger.info(f"  [{role}] {text[:70]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="听书智库 · 检索问答全链路测试")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="提问内容")
    parser.add_argument("--session", default=None, help="会话ID（默认按时间戳生成）")
    parser.add_argument("--dry-run", action="store_true", help="只检查图与依赖，不发起真实问答")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("05 · 检索问答全链路测试 开始")
    logger.info("=" * 70)

    if not check_env():
        logger.warning("排查提示：先执行 test/03_milvus_test.py 确认服务，再执行 test/04_import_test.py 导入数据")
        sys.exit(1)

    if args.dry_run:
        logger.info("----- 干跑模式（--dry-run）-----")
        logger.info(f"检索图节点：{list(kb_query_app.get_graph().nodes.keys())}")
        logger.info(f"切片集合：{milvus_config.chunks_collection}；主体集合：{milvus_config.item_name_collection}")
        logger.info(f"默认问题：{DEFAULT_QUESTION}")
        logger.success("图与依赖检查通过（未发起真实问答）")
        logger.info("=" * 70)
        sys.exit(0)

    session_id = args.session or f"test_query_{int(time.time())}"
    try:
        state = run_query(session_id, args.question)
    except Exception:
        logger.exception("检索链路执行失败")
        logger.warning(
            "排查提示：1) test/01_llm_test.py 确认大模型可用；2) test/03_milvus_test.py 确认向量库；"
            "3) 本地 BGE-Reranker-large 模型路径（.env: BGE_RERANKER_LARGE）是否正确"
        )
        sys.exit(1)

    ok = report(state)
    show_history(session_id)

    logger.info("=" * 70)
    if ok:
        logger.success("05 测试通过：检索问答链路端到端正常")
        sys.exit(0)
    logger.error("05 测试未通过，请按上方 ❌ 项排查")
    sys.exit(1)

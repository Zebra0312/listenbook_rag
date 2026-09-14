"""
04 · 导入全链路测试（后端流程）

作用：不经过前端页面，直接调用 LangGraph 导入图跑完整条导入链路，并逐节点打印执行进度。
      链路：node_entry → (pdf_to_md) → md_img → document_split
            → item_name_recognition → bge_embedding → import_milvus

用到的测试文件按以下优先级选择：
    1) 命令行传入的路径      python test/04_import_test.py doc/三体_书籍简介.md
    2) doc/ 目录下第一个 .md / .pdf
    3) test/samples/ 下的示例文件（仅用于开箱即跑）

用法：
    cd listenbook_rag
    uv run python test/04_import_test.py
    uv run python test/04_import_test.py doc/某某书籍_听书笔记.md

注意：
    - 本测试会真实写入 Milvus（集合名取自 .env：CHUNKS_COLLECTION / ITEM_NAME_COLLECTION），
      并按 item_name 幂等覆盖，重复运行不会产生重复数据；
    - 测试会把源文件复制到 output/test_import/<文件名>/ 再导入（模拟前端上传流程），
      因此 doc/ 与 test/samples/ 不会被 *_new.md、backup.json 等中间产物污染；
    - PDF 入口依赖 MinerU 在线服务，MD 入口不需要，只想快速验证链路建议先用 MD。
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

# 保证可以直接 `python test/04_import_test.py` 运行（无需手动设置 PYTHONPATH）
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.path_util import PROJECT_ROOT

# 支持的输入格式
SUPPORTED_SUFFIXES = (".pdf", ".md")
# 每行输出的分隔线
LINE = "-" * 70
# doc / samples 目录
DOC_DIR = PROJECT_ROOT / "doc"
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


def pick_test_file(cli_path: str | None) -> Path | None:
    """按 命令行参数 → doc 目录 → 示例目录 的优先级挑选测试文件"""
    # 1) 命令行指定
    if cli_path:
        path = Path(cli_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            logger.error(f"指定的测试文件不存在：{path}")
            return None
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            logger.error(f"不支持的格式：{path.suffix}（仅支持 {SUPPORTED_SUFFIXES}）")
            return None
        return path

    # 2) doc 目录下第一个受支持的文件
    if DOC_DIR.exists():
        for item in sorted(DOC_DIR.iterdir()):
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES:
                logger.info(f"未指定文件，自动选用 doc/ 下的：{item.name}")
                return item

    # 3) 示例文件兜底
    if SAMPLES_DIR.exists():
        for item in sorted(SAMPLES_DIR.iterdir()):
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES:
                logger.warning(f"doc/ 目录为空，改用示例文件：{item.name}（正式验证请放入自己的听书资料）")
                return item

    logger.error("没有找到可用的测试文件")
    logger.info("请把待导入的 PDF/MD 放进 listenbook_rag/doc/，或直接传路径给本脚本")
    return None


def run_import(file_path: Path) -> dict:
    """执行导入图，逐节点打印进度，返回最终状态"""
    # 每次测试单独建一个输出目录，避免和正式数据混在一起
    task_id = f"test_import_{int(time.time())}"
    output_dir = PROJECT_ROOT / "output" / "test_import" / file_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    # 先把源文件复制到输出目录再导入，模拟 /upload 的真实流程：
    # node_md_img 会写 *_new.md、node_document_split 会写 backup.json，
    # 从 output/ 下的副本导入可以保证 doc/ 与 test/samples/ 不被产物污染。
    working_file = output_dir / file_path.name
    shutil.copy2(file_path, working_file)

    init_state = create_default_state(
        task_id=task_id,
        local_file_path=str(working_file),
        local_dir=str(output_dir),
    )

    logger.info(f"任务ID：{task_id}")
    logger.info(f"源文件：{file_path}")
    logger.info(f"工作副本：{working_file}")
    logger.info(f"输出目录：{output_dir}")
    logger.info(LINE)

    # stream_mode="updates"：每完成一个节点产出一条 {节点名: 该节点的状态更新}
    final_state: dict = {}
    total_start = time.time()
    for step in kb_import_app.stream(init_state, stream_mode="updates"):
        for node_name, node_delta in (step or {}).items():
            if isinstance(node_delta, dict):
                final_state.update(node_delta)
            logger.info(f"✅ 节点执行完成：{node_name}")
    logger.info(LINE)
    logger.info(f"链路总耗时：{time.time() - total_start:.2f}s")
    return final_state


def report(final_state: dict) -> bool:
    """打印导入结果与关键校验项"""
    chunks = final_state.get("chunks") or []
    logger.info("===== 导入结果 =====")
    logger.info(f"识别主体 item_name ：{final_state.get('item_name', '')}")
    logger.info(f"文档切分切片数      ：{len(chunks)}")

    if not chunks:
        logger.error("切片数为 0，导入未产生有效数据")
        return False

    first = chunks[0]
    logger.info(
        "首片元数据         ："
        f"book_name={first.get('book_name', '')}，author={first.get('author', '')}，"
        f"content_type={first.get('content_type', '')}，duration={first.get('duration', '') or '（无）'}"
    )
    logger.info(f"来源文件 source_file：{first.get('source_file', '')}")

    # 内容类型分布（验证文件名关键词推断是否生效）
    type_dist: dict = {}
    for chunk in chunks:
        key = chunk.get("content_type", "")
        type_dist[key] = type_dist.get(key, 0) + 1
    logger.info(f"内容类型分布        ：{type_dist}")

    # 核心校验项
    has_embedding = all("dense_vector" in c and "sparse_vector" in c for c in chunks)
    has_chunk_id = all("chunk_id" in c for c in chunks)
    has_meta = all(
        all(k in c for k in ("content_type", "book_name", "author", "source_file", "source_path"))
        for c in chunks
    )
    logger.info(LINE)
    logger.info(f"全部切片完成向量化  ：{'✅ 是' if has_embedding else '❌ 否'}")
    logger.info(f"全部切片回填 chunk_id：{'✅ 是' if has_chunk_id else '❌ 否'}")
    logger.info(f"书籍域元数据齐全    ：{'✅ 是' if has_meta else '❌ 否'}")
    return has_embedding and has_chunk_id and has_meta


def verify_in_milvus(item_name: str, expected_count: int) -> bool:
    """回到 Milvus 里确认数据真的落库了（数量 + 抽样）"""
    logger.info(LINE)
    logger.info("===== Milvus 落库校验 =====")
    if not item_name:
        logger.error("item_name 为空，无法校验")
        return False

    collection = milvus_config.chunks_collection
    client = get_milvus_client()
    if not client:
        logger.error("Milvus 连接失败，跳过落库校验")
        return False

    logger.info(f"切片集合：{collection}")
    if not client.has_collection(collection_name=collection):
        logger.error("集合不存在，说明入库未成功")
        return False

    expr = f'item_name == "{escape_milvus_string(item_name)}"'
    count = 0
    sample = []
    try:
        # 刚写入的数据需要 flush + 重新加载才会对查询可见，
        # 否则 count(*) 可能读到插入前的旧值（出现"命中数少于本次切片数"的假象）
        try:
            client.flush(collection_name=collection)
        except Exception as e:
            logger.warning(f"flush 失败（不影响主流程）：{e}")
        client.load_collection(collection_name=collection)

        # 查询可见性存在秒级延迟，最多重试 3 次
        for attempt in range(1, 4):
            count_res = client.query(
                collection_name=collection,
                filter=expr,
                # 注意：使用 count(*) 聚合时不能带 limit（Milvus 会报
                # "count entities with pagination is not allowed"）
                output_fields=["count(*)"],
            )
            count = int(list(count_res[0].values())[0]) if count_res else 0
            if count >= expected_count:
                break
            logger.info(f"第 {attempt} 次查询命中 {count} 条，少于本次切片数，1 秒后重试...")
            time.sleep(1)

        sample = client.query(
            collection_name=collection,
            filter=expr,
            output_fields=["chunk_id", "title", "content_type", "book_name", "author"],
            limit=3,
        )
    except Exception as e:
        logger.error(f"查询失败：{e}")
        return False

    logger.info(f"按 item_name [{item_name}] 命中：{count} 条（本次导入 {expected_count} 条）")
    for row in sample:
        logger.info(
            f"  抽样：chunk_id={row.get('chunk_id')}，title={row.get('title')}，"
            f"content_type={row.get('content_type')}，book_name={row.get('book_name')}"
        )

    if count < expected_count:
        logger.warning("命中数仍少于本次切片数，可能是查询可见性延迟；可稍后重跑 test/03_milvus_test.py 复核")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="听书智库 · 导入全链路测试")
    parser.add_argument("file", nargs="?", default=None, help="待导入的 PDF/MD 路径（相对项目根目录）")
    parser.add_argument("--dry-run", action="store_true", help="只做文件选择与路由检查，不调用大模型 / 不写 Milvus")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("04 · 导入全链路测试 开始")
    logger.info("=" * 70)

    test_file = pick_test_file(args.file)
    if not test_file:
        sys.exit(1)

    if args.dry_run:
        # 干跑：只检查文件与链路入口，不触发任何外部服务
        suffix = test_file.suffix.lower()
        route = "node_entry → node_pdf_to_md → node_md_img（依赖 MinerU 在线服务）" \
            if suffix == ".pdf" else "node_entry → node_md_img（不依赖 MinerU）"
        logger.info("----- 干跑模式（--dry-run）-----")
        logger.info(f"命中文件：{test_file}")
        logger.info(f"文件大小：{test_file.stat().st_size} 字节")
        logger.info(f"入口路由：{route}")
        logger.info(f"后续节点：node_document_split → node_item_name_recognition → node_bge_embedding → node_import_milvus")
        logger.info(f"写入集合：{milvus_config.chunks_collection} / {milvus_config.item_name_collection}")
        logger.success("文件与路由检查通过（未执行真实导入）")
        logger.info("=" * 70)
        sys.exit(0)

    try:
        state = run_import(test_file)
    except Exception:
        logger.exception("导入链路执行失败")
        logger.warning(
            "排查提示：MD 入口请先跑 test/01_llm_test.py 与 test/02_bgem3_test.py、"
            "test/03_milvus_test.py；PDF 入口还需检查 .env 中的 MINERU_API_TOKEN"
        )
        sys.exit(1)

    report_ok = report(state)
    verify_ok = verify_in_milvus(state.get("item_name", ""), len(state.get("chunks") or []))

    logger.info("=" * 70)
    if report_ok and verify_ok:
        logger.success("04 测试通过：导入链路端到端正常，数据已落库")
        sys.exit(0)
    logger.error("04 测试未通过，请按上方 ❌ 项排查")
    sys.exit(1)

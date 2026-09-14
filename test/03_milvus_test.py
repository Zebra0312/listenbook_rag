"""
03 · Milvus 连接与集合状态检查

作用：导入前确认向量库可达，并查看两个集合（切片 / 书籍主体）是否存在、字段是否齐全、数据量多少。
      集合不存在属于正常情况（首次导入会自动创建），这里只做"告知"。

用法：
    cd listenbook_rag
    uv run python test/03_milvus_test.py
"""
import sys
from pathlib import Path

# 保证可以直接 `python test/03_milvus_test.py` 运行（无需手动设置 PYTHONPATH）
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

# 切片集合应当具备的字段（与 node_import_milvus 的 schema 对应）
EXPECTED_CHUNK_FIELDS = {
    "chunk_id", "content", "title", "parent_title", "part", "file_title",
    "content_type", "book_name", "author", "item_name", "category",
    "duration", "source_file", "source_path",
    "dense_vector", "sparse_vector",
}

# 书籍主体集合应当具备的字段（与 node_item_name_recognition 的 schema 对应）
EXPECTED_ITEM_FIELDS = {"pk", "file_title", "item_name", "dense_vector", "sparse_vector"}


def get_pk_field(client, collection_name: str) -> str:
    """从集合结构中取出主键字段名（切片集合是 chunk_id，主体集合是 pk）"""
    try:
        desc = client.describe_collection(collection_name=collection_name)
        for field in desc.get("fields", []):
            if field.get("is_primary"):
                return field.get("name")
    except Exception as e:
        logger.warning(f"读取主键字段失败：{e}")
    return ""


def get_row_count(client, collection_name: str, pk_field: str) -> int:
    """
    获取集合数据量。
    说明：get_collection_stats 的 row_count 依赖后台统计刷新，刚导入完可能仍返回 0，
    因此优先用 count(*) 聚合查询，失败时再退回 stats。
    """
    if pk_field:
        try:
            res = client.query(
                collection_name=collection_name,
                # 注意：count(*) 聚合不能带 limit，否则 Milvus 报
                # "count entities with pagination is not allowed"
                filter=f"{pk_field} >= 0",
                output_fields=["count(*)"],
            )
            if res:
                return int(list(res[0].values())[0])
        except Exception as e:
            logger.warning(f"count(*) 查询失败，回退 stats：{e}")
    try:
        stats = client.get_collection_stats(collection_name=collection_name)
        return int(stats.get("row_count", 0))
    except Exception as e:
        logger.warning(f"获取集合 {collection_name} 数据量失败：{e}")
    return -1


def inspect_collection(client, collection_name: str) -> None:
    """检查单个集合：是否存在 / 字段 / 数据量"""
    logger.info(f"----- 集合：{collection_name} -----")
    if not collection_name:
        logger.error("集合名为空，请检查 .env 中 CHUNKS_COLLECTION / ITEM_NAME_COLLECTION")
        return

    if not client.has_collection(collection_name=collection_name):
        logger.warning("集合不存在（首次导入时会自动创建，属正常情况）")
        return

    try:
        client.load_collection(collection_name=collection_name)
    except Exception as e:
        logger.warning(f"load_collection 失败（可能已在内存中）：{e}")

    pk_field = get_pk_field(client, collection_name)
    count = get_row_count(client, collection_name, pk_field)
    logger.info(f"数据量：{count if count >= 0 else '未知'} 条")

    try:
        desc = client.describe_collection(collection_name=collection_name)
        fields = [f.get("name") for f in desc.get("fields", [])]
        logger.info(f"字段列表（{len(fields)} 个）：{', '.join(fields)}")
        # 按集合类型分别校验字段完整性
        if collection_name == milvus_config.chunks_collection:
            expected = EXPECTED_CHUNK_FIELDS
            label = "切片集合"
        else:
            expected = EXPECTED_ITEM_FIELDS
            label = "主体集合"
        missing = expected - set(fields)
        if missing:
            logger.warning(
                f"{label}缺少字段：{', '.join(sorted(missing))}。"
                "若该集合是旧版 schema 创建的，需删掉集合并重新导入"
            )
        else:
            logger.success(f"{label}字段完整")
    except Exception as e:
        logger.warning(f"读取集合结构失败：{e}")


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("03 · Milvus 连接与集合状态检查 开始")
    logger.info("=" * 70)

    if not milvus_config.milvus_url:
        logger.error("配置缺失：.env 中未设置 MILVUS_URL")
        sys.exit(1)

    logger.info(f"Milvus 地址：{milvus_config.milvus_url}")
    client = get_milvus_client()
    if not client:
        logger.error("Milvus 连接失败")
        logger.warning("排查提示：1) Milvus 服务是否已启动；2) 端口是否为 19530；3) 地址是否可访问")
        sys.exit(1)
    logger.success("Milvus 连接成功")

    try:
        all_collections = client.list_collections()
        logger.info(f"当前库中的集合：{all_collections or '（空）'}")
    except Exception as e:
        logger.warning(f"列出集合失败：{e}")

    inspect_collection(client, milvus_config.chunks_collection)
    inspect_collection(client, milvus_config.item_name_collection)

    logger.info("=" * 70)
    logger.info("03 检查结束")

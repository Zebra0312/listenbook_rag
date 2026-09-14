"""
02 · BGE-M3 向量模型测试

作用：验证稠密（1024 维）+ 稀疏向量的生成链路，这是导入侧 node_bge_embedding
      与检索侧的公共依赖。首次运行可能需要加载本地模型（约几百 MB），耗时较长。

用法：
    cd listenbook_rag
    uv run python test/02_bgem3_test.py
"""
import sys
import time
from pathlib import Path

# 保证可以直接 `python test/02_bgem3_test.py` 运行（无需手动设置 PYTHONPATH）
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from app.conf.embedding_config import embedding_config
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings

# 期望的稠密向量维度（BGE-M3 固定 1024，与 Milvus schema 必须一致）
EXPECTED_DIM = 1024

# 用书籍域语料做测试，比 helloworld 更能反映真实分词语义
TEXTS = [
    "三体是刘慈欣创作的科幻小说，曾获雨果奖最佳长篇小说奖。",
    "推荐一本适合通勤听的悬疑小说",
]


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("02 · BGE-M3 向量模型测试 开始")
    logger.info("=" * 70)
    logger.info(
        f"模型配置：路径={embedding_config.bge_m3_path or '未配置（将走 BAAI/bge-m3 在线下载）'}，"
        f"设备={embedding_config.bge_device}，fp16={embedding_config.bge_fp16}"
    )

    try:
        start = time.time()
        result = generate_embeddings(TEXTS)
        cost = time.time() - start
    except Exception as e:
        logger.error(f"向量生成失败：{e}")
        logger.warning(
            "排查提示：1) BGE_M3_PATH 指向的本地模型是否存在；"
            "2) BGE_DEVICE=cpu 时 BGE_FP16 必须为 0；3) 首次运行需联网下载模型"
        )
        sys.exit(1)

    dense_list = result.get("dense") or []
    sparse_list = result.get("sparse") or []

    # 逐项校验
    ok = True
    if len(dense_list) != len(TEXTS):
        logger.error(f"稠密向量条数异常：期望 {len(TEXTS)}，实际 {len(dense_list)}")
        ok = False
    if len(sparse_list) != len(TEXTS):
        logger.error(f"稀疏向量条数异常：期望 {len(TEXTS)}，实际 {len(sparse_list)}")
        ok = False

    if dense_list:
        dim = len(dense_list[0])
        logger.info(f"稠密向量维度：{dim}（期望 {EXPECTED_DIM}）")
        if dim != EXPECTED_DIM:
            logger.error("稠密向量维度与 EMBEDDING_DIM / Milvus schema 不一致，入库会失败")
            ok = False

    for idx, sparse in enumerate(sparse_list, start=1):
        if not sparse:
            logger.error(f"第 {idx} 条稀疏向量为空（词袋特征数为 0）")
            ok = False
        else:
            logger.info(f"第 {idx} 条稀疏向量特征数：{len(sparse)}")

    logger.info("-" * 70)
    logger.info(f"耗时：{cost:.2f}s")
    if ok:
        logger.success("02 测试通过：稠密 / 稀疏向量生成正常")
    else:
        logger.error("02 测试失败，请按上方提示排查")
    logger.info("=" * 70)
    sys.exit(0 if ok else 1)

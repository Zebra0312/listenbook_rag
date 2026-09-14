"""
test 包：听书智库后端流程测试

约定：
- 文件名以序号开头，按依赖顺序排列，可单独运行；
- 所有脚本都自带 sys.path 引导，支持直接 `uv run python test/xx.py`，
  不需要手动设置 PYTHONPATH；
- 只做"验证"不做"修复"，失败时打印可执行的排查建议。

脚本清单：
    01_llm_test.py     大模型（文本 + 视觉）连通性
    02_bgem3_test.py   BGE-M3 稠密 / 稀疏向量生成
    03_milvus_test.py  Milvus 连接与集合状态
    04_import_test.py  导入全链路（PDF/MD → 切分 → 主体识别 → 向量 → Milvus）
"""

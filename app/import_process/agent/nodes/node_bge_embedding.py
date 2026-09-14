import os
import sys

from dotenv import load_dotenv

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task

@step_log("step_1_validate_input")
def step_1_validate_input(state):
    # 获取chunks
    chunks = state.get("chunks")
    # 判断chunks是否为空
    if not chunks:
        raise ValueError("切片数据异常，请重试")
    return chunks

@step_log("step_2_generate_embeddings")
def step_2_generate_embeddings(chunks):
    # 创建最终存储切片的变量
    final_chunks = []
    # 获取切片的数量
    chunk_size = len(chunks)
    # 设置每个批次处理的切片的数量
    batch_size = 5
    # 通过切片的数量对chunks进行遍历
    for i in range(0, chunk_size, batch_size):
        # 获取此批次需要处理的切片
        batch_chunks = chunks[i:i + batch_size]
        # 创建存储拼接主体名称和切片内容的变量
        texts = []
        try:
            # 对此批次需要处理的切片进行遍历
            for chunk in batch_chunks:
                # 分别获取每个切片的item_name和content
                item_name = chunk["item_name"]
                content = chunk["content"]
                # 对item_name和content进行拼接
                # 听书项目：主体是书籍条目（书名-作者），因此拼接前缀用"书名/内容"
                text = f"书名：{item_name}，内容：{content}" if item_name else content
                # 存储每个切片拼接之后的结果
                texts.append(text)
            # 将texts转换为向量
            embeddings = generate_embeddings(texts)
            # 判断embeddings是否为空
            if not embeddings:
                logger.warning("获取切片所对应的向量失败")
                # 进行兜底处理
                final_chunks.extend(batch_chunks)
            else:
                # 遍历batch_chunks
                for index, doc in enumerate(batch_chunks):
                    # 对chunk进行浅拷贝
                    chunk = doc.copy()
                    # 回填稠密向量和稀疏向量（字段名必须与Milvus集合schema中的字段名一致）
                    chunk["dense_vector"] = embeddings["dense"][index]
                    chunk["sparse_vector"] = embeddings["sparse"][index]
                    # 存储到final_chunks中
                    final_chunks.append(chunk)
        except Exception as e:
            logger.error("当前批次的切片转换为向量失败")
            # 进行兜底处理
            final_chunks.extend(batch_chunks)
    return final_chunks


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    实现内容:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    """
    # 记录任务的状态为运行中
    add_running_task(state["task_id"], "node_bge_embedding")
    # 步骤1：输入数据校验，核心chunks无效则抛出异常
    chunks = step_1_validate_input(state)
    # 步骤2：批量生成双向量，为切片绑定向量字段
    final_chunks = step_2_generate_embeddings(chunks)
    # 更新状态
    state["chunks"] = final_chunks
    # 记录任务的状态为已完成
    add_done_task(state["task_id"], "node_bge_embedding")
    return state

# ==========================================
# 本地单元测试入口
# 功能：独立验证向量化节点全链路逻辑，无需启动整个LangGraph流程
# 适用场景：本地开发、调试、模型有效性验证
# ==========================================
if __name__ == '__main__':
    # 加载环境变量：定位项目根目录下的.env，读取模型路径/设备等配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造模拟测试状态：模拟上游节点输出的chunks数据，贴合真实业务场景
    test_state = ImportGraphState({
        "task_id": "test_task_embedding_001",  # 测试任务ID
        "chunks": [  # 模拟带item_name的文本切片（上游书籍主体识别节点产出）
            {
                "content": "《三体》讲述了地球人类文明与三体文明的信息交流与生死搏杀。",
                "title": "内容简介",
                "item_name": "三体-刘慈欣",
                "file_title": "三体_书籍简介.md"
            },
            {
                "content": "刘慈欣，中国科幻小说代表作家，代表作《三体》三部曲。",
                "title": "作者介绍",
                "item_name": "三体-刘慈欣",
                "file_title": "三体_书籍简介.md"
            }
        ]
    })

    # 执行本地测试
    logger.info("=== BGE-M3向量化节点本地单元测试启动 ===")
    try:
        # 调用核心节点函数
        result_state = node_bge_embedding(test_state)
        # 提取测试结果
        result_chunks = result_state.get("chunks", [])

        # 验证向量生成结果（打印向量字段是否存在）
        for idx, chunk in enumerate(result_chunks):
            has_dense = "dense_vector" in chunk
            has_sparse = "sparse_vector" in chunk
            logger.info(
                f"第{idx + 1}条切片：稠密向量生成{'' if has_dense else '未'}成功 | 稀疏向量生成{'' if has_sparse else '未'}成功")

    except Exception as e:
        logger.error(f"=== 向量化节点本地测试失败 ===" f"错误原因：{str(e)}", exc_info=True)
        # 新手友好提示：给出核心排查方向
        logger.warning("排查提示：请检查BGE-M3模型路径、显存是否充足、环境变量配置是否正确")

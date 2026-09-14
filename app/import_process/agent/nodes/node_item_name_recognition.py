import os
import re
import sys
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task

# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500

# ---------------------------------------------------------------------------
# 听书域元数据规则（内容类型 / 书名作者拆分 / 有声书时长）
# 说明：当前以「文件名关键词」推断 content_type，属于零成本、结果稳定的起步方案；
#      若后续需要更准，可改成让大模型直接输出 JSON。
# ---------------------------------------------------------------------------
# 内容类型枚举（与 README / 需求文档保持一致）
CONTENT_TYPE_AUDIOBOOK_INFO = "audiobook_info"    # 有声书信息
CONTENT_TYPE_BOOK_INTRO = "book_intro"            # 书籍简介
CONTENT_TYPE_AUTHOR_INTRO = "author_intro"        # 作者介绍
CONTENT_TYPE_LISTENING_NOTE = "listening_note"    # 听书笔记
CONTENT_TYPE_RECOMMENDATION = "recommendation"    # 推荐运营资料
CONTENT_TYPE_COMMENT_SUMMARY = "comment_summary"  # 用户评论摘要
CONTENT_TYPE_FAQ = "faq"                          # 常见问答

# 关键词 → 内容类型 的匹配顺序（命中即返回，顺序即优先级）
CONTENT_TYPE_KEYWORDS = [
    (("有声书", "音频", "演播", "主播", "时长", "音频信息"), CONTENT_TYPE_AUDIOBOOK_INFO),
    (("作者介绍", "作者简介", "作者"), CONTENT_TYPE_AUTHOR_INTRO),
    (("听书笔记", "笔记", "书评", "读书笔记"), CONTENT_TYPE_LISTENING_NOTE),
    (("推荐", "运营", "推荐语", "卖点", "亮点"), CONTENT_TYPE_RECOMMENDATION),
    (("评论", "口碑", "读者评价", "读后感", "评论摘要"), CONTENT_TYPE_COMMENT_SUMMARY),
    (("问答", "常见问题", "faq", "q&a"), CONTENT_TYPE_FAQ),
    (("简介", "内容简介", "书籍简介", "概要"), CONTENT_TYPE_BOOK_INTRO),
]

# 书名与作者的分隔符（大模型通常返回"三体-刘慈欣"这类复合主体名）
BOOK_AUTHOR_SEPARATORS = ("-", "－", "—", "–", "_", "：", ":", "/", "|")

# 有声书时长提取：仅在 content_type 为 audiobook_info 时启用
DURATION_PATTERN = re.compile(r"(?:总时长|时长|播放时长|音频长度|音频时长)\s*[:：]?\s*([^\n，。;；|）)]{1,30})")


def _infer_content_type(file_title: str) -> str:
    """根据文件名关键词推断内容类型，未命中时兜底为书籍简介。"""
    text = (file_title or "").lower()
    for keywords, content_type in CONTENT_TYPE_KEYWORDS:
        for keyword in keywords:
            if keyword.lower() in text:
                return content_type
    return CONTENT_TYPE_BOOK_INTRO


def _split_book_name_author(item_name: str):
    """把"三体-刘慈欣"这类复合主体名拆成 (书名, 作者)，拆不出作者时作者为空。"""
    text = (item_name or "").strip()
    for separator in BOOK_AUTHOR_SEPARATORS:
        if separator in text:
            book_name, _, author = text.partition(separator)
            book_name = book_name.strip()
            author = author.strip()
            if book_name and author:
                return book_name, author
    return text, ""


def _extract_duration(md_content: str, content_type: str) -> str:
    """仅有声书类内容尝试提取时长（如"时长：12小时30分钟"），取不到返回空串。"""
    if content_type != CONTENT_TYPE_AUDIOBOOK_INFO or not md_content:
        return ""
    match = DURATION_PATTERN.search(md_content[:2000])
    if not match:
        return ""
    return match.group(1).strip()

@step_log("step_1_get_chunks_and_file_title")
def step_1_get_chunks_and_file_title(state):
    # 分别获取chunks和file_tile
    file_title = state.get("file_title")
    chunks = state.get("chunks")
    # 判断chunks是否为空
    if not chunks:
        raise RuntimeError("chunks为空，即没有任何的切片")
    # 判断file_title是否为空
    if not file_title:
        file_title = Path(state.get("md_path")).stem
        state["file_title"] = file_title
    return chunks, file_title

@step_log("step_2_build_context")
def step_2_build_context(chunks):
    # 创建存储切片处理之后的数据的变量
    parts = []
    # 记录当前切片的字符数
    total_chars = 0
    # 对前若干个chunks进行遍历（累计字符数达到阈值即停止）
    for idx, chunk in enumerate(chunks, start=1):
        # 获取切片的标题和内容
        title = chunk["title"]
        content = chunk["content"]
        # 将切片组装为：切片:idx，标题:title，内容:content
        data = f"切片：{idx}，标题：{title}，内容：{content}"
        # 存储data
        parts.append(data)
        # 记录已存储的切片的总字符数
        total_chars += len(data)
        # 判断total_chars是否超过了指定的阈值CONTEXT_TOTAL_MAX_CHARS
        if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
            break
    # 将处理之后的切片拼接为字符串
    context = "\n\n".join(parts)
    # 进行兜底处理，防止上下文超过指定阈值
    context = context[:CONTEXT_TOTAL_MAX_CHARS]
    return context

@step_log("step_3_call_llm")
def step_3_call_llm(context, file_title):
    # 分别获取用户提示词和系统提示词
    # 注意：听书项目使用 book_recognition_system（模板对应 product_recognition_system）
    human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
    system_prompt = load_prompt("book_recognition_system")
    # 将human_prompt和system_prompt组成提示词
    messages = [
        SystemMessage(system_prompt),
        HumanMessage(human_prompt),
    ]
    # 获取大模型对象
    llm = get_llm_client()
    # 创建链对象
    chain = llm | StrOutputParser()
    # 调用链对象
    item_name = chain.invoke(messages)
    # 判断item_name是否为空
    if not item_name:
        item_name = file_title
    # 去掉可能存在的引号与首尾空白
    return item_name.strip().strip("“”\"'").strip()

@step_log("step_4_backfill_book_metadata")
def step_4_backfill_book_metadata(state, item_name, chunks):
    """
    步骤4：书籍主体回填 + 条目级元数据注入

    为什么在这一步做：item_name（书名-作者）在本步才产出，而需求文档要求的
    content_type / book_name / author / category / duration / source_file /
    source_path 都属于"每个切片都要带上的条目元数据"，因此统一在这里回填，
    保证 chunks 的字段与 Milvus 集合 schema 一一对应。
    """
    # 更新状态中的item_name
    state["item_name"] = item_name

    # 基础文件信息（source_file 取原始上传文件名，取不到则用 file_title 兜底）
    local_file_path = state.get("local_file_path") or ""
    source_file = Path(local_file_path).name if local_file_path else ""
    if not source_file:
        source_file = f"{state.get('file_title', '')}.md"
    source_path = str(state.get("md_path") or "")

    # 内容类型推断 + 书名/作者拆分 + 时长提取
    file_title = state.get("file_title", "")
    content_type = _infer_content_type(file_title)
    book_name, author = _split_book_name_author(item_name)
    duration = _extract_duration(state.get("md_content", ""), content_type)
    # category（类别/标签）当前无可靠来源，先留空，后续可由大模型或后台配置补齐
    category = ""

    logger.info(
        f"书籍元数据：item_name={item_name}，book_name={book_name}，author={author}，"
        f"content_type={content_type}，source_file={source_file}"
    )

    # 更新每个切片，添加书籍域元数据
    for chunk in chunks:
        chunk["item_name"] = item_name
        chunk["content_type"] = content_type
        chunk["book_name"] = book_name
        chunk["author"] = author
        chunk["category"] = category
        chunk["duration"] = duration
        chunk["source_file"] = source_file
        chunk["source_path"] = source_path
    # 更新状态中的chunks
    state["chunks"] = chunks

@step_log("step_5_generate_embeddings")
def step_5_generate_embeddings(item_name):
    # 将item_name生成稠密向量和稀疏向量
    embeddings = generate_embeddings([item_name])
    # 返回稠密向量和稀疏向量
    return embeddings["dense"][0], embeddings["sparse"][0]

@step_log("step_6_save_to_vector_db")
def step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector):
    # 获取milvus的客户端对象
    milvus_client = get_milvus_client()
    # 若milvus中没有item_name集合，则创建
    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
        # 设置集合的结构
        schema = milvus_client.create_schema(
            auto_id=True,  # 集合中的主键自增
            enable_dynamic_field=True,  # 开启动态字段，允许向向量数据库不存在的字段进行赋值
        )
        # 设置集合的字段
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 设置集合的索引
        index_params = milvus_client.prepare_index_params()
        # 设置稠密向量的索引
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            index_name="dense_vector_index",
            metric_type="COSINE",
        )
        # 设置稀疏向量的索引
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_vector_index",
            metric_type="IP",
        )
        # 创建集合
        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,
            index_params=index_params
        )
    # 将item_name相关的数据删除
    milvus_client.delete(
        collection_name=milvus_config.item_name_collection,
        filter=f"item_name == '{item_name}'"
    )
    # 准备数据
    data = {
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector,
    }
    # 保存数据
    milvus_client.insert(
        collection_name=milvus_config.item_name_collection,
        data=[data]
    )


@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的书籍条目名称 (Item Name)。
    实现内容:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是哪本书 (如: "三体-刘慈欣")。
    3. 存入 state["item_name"]，并为每个切片回填书籍域元数据。
    4. 生成主体向量，幂等写入 kb_item_names 集合（集合名来自 .env）。
    """
    # 记录任务的状态为运行中
    add_running_task(state["task_id"], "node_item_name_recognition")
    # 步骤1：校验和取值 （file_title,chunks）
    chunks, file_title = step_1_get_chunks_and_file_title(state)
    # 步骤2：构建上下文环境  chunks -> 拼接成context文本
    context = step_2_build_context(chunks)
    # 步骤3：调用模型，拼接提示词，识别chunks对应item_name
    item_name = step_3_call_llm(context, file_title)
    # 步骤4：书籍主体回填 + 条目级元数据注入（content_type / book_name / author ...）
    step_4_backfill_book_metadata(state, item_name, chunks)
    # 步骤5：item_name生成向量（稠密/稀疏）
    dense_vector, sparse_vector = step_5_generate_embeddings(item_name)
    # 步骤6：存储向量到向量数据库 item_name集合 (pk / file_title / item_name / 稠密 和 稀疏)
    step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector)
    # 记录任务的状态为已完成
    add_done_task(state["task_id"], "node_item_name_recognition")
    return state

if __name__ == "__main__":
    logger.info("=== 开始执行书籍主体识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "三体_书籍简介",  # 模拟文件标题
            "local_file_path": "doc/三体_书籍简介.md",  # 模拟原始文件名（source_file 用）
            "md_path": "output/三体_书籍简介/三体_书籍简介_new.md",
            "md_content": "《三体》是刘慈欣创作的长篇科幻小说，时长：21小时30分钟。",
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "内容简介",
                    "content": "《三体》讲述了地球人类文明与三体文明的信息交流、生死搏杀及两个文明在宇宙中的兴衰历程。"
                },
                {
                    "title": "作者介绍",
                    "content": "刘慈欣，中国科幻小说代表作家，代表作《三体》三部曲、《球状闪电》等。"
                },
                {
                    "title": "有声书信息",
                    "content": "《三体》有声书由中央广播电视总台录制，演播：李野墨，总时长：21小时30分钟。"
                }
            ]
        })

        # 2. 调用书籍主体识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 书籍主体识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别主体名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        first_chunk = result_state.get('chunks', [{}])[0]
        logger.info(f"第一个切片元数据：{first_chunk}")

    except Exception as e:
        logger.error(f"书籍主体识别节点本地测试失败，原因：{str(e)}", exc_info=True)

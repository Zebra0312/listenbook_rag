from typing import TypedDict
import copy
from app.core.logger import logger

class ImportGraphState(TypedDict):
    """
    图的状态定义，包含所有节点产生和消费的数据字段。
    TypedDict 让我们在代码中能有自动补全和类型检查。
    使用字典式访问（如state["task_id"]、state.get("chunks")）

    听书智库说明：
    - 本项目中"主体"指书籍条目（识别结果形如"三体-刘慈欣"），字段名沿用 item_name，
      既作为检索过滤键，也作为按主体幂等清理旧数据的依据。
    - 需求文档要求的条目级元数据（content_type / book_name / author / category /
      duration / source_file / source_path）不需要单独的 state 字段，随 chunks
      一起流转，由 node_item_name_recognition 的元数据回填步骤统一注入。
    """
    task_id: str          # 任务唯一ID，用于追踪日志

    # --- 流程控制标记 ---
    is_md_read_enabled: bool   # 是否启用 Markdown 读取路径
    is_pdf_read_enabled: bool  # 是否启用 PDF 读取路径
    is_mp3_read_enabled: bool  # 是否启用 MP3 音频转写路径

    # --- 路径相关 ---
    local_dir: str        # 当前工作目录或输出目录
    local_file_path: str  # 原始输入文件路径
    file_title: str       # 文件标题（文件名去后缀）
    pdf_path: str         # PDF 文件路径 (如果输入是PDF)
    md_path: str          # Markdown 文件路径 (转换后或直接输入的)
    mp3_path: str         # MP3 音频文件路径 (如果输入是音频)

    # --- 内容数据 ---
    md_content: str       # Markdown 的全文内容
    chunks: list          # 切片后的文本列表，包含 metadata（含书籍域元数据）
    item_name: str        # 识别出的书籍主体名称 (如: "三体-刘慈欣")，用于增强检索与幂等清理

    # --- 数据库相关 ---
    embeddings_content: list # 包含向量数据的列表，准备写入 Milvus


# 建议定一个初始化对象，方便后续使用
# 定义图状态的默认初始值
graph_default_state: ImportGraphState = {
    "task_id":"",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_mp3_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "mp3_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}

def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖
    Args:
        **overrides: 要覆盖的字段（关键字参数解包）
    Returns:
        新的状态实例
    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """

    # 默认状态
    state = copy.deepcopy(graph_default_state)
    # 用 overrides 覆盖默认值
    state.update(overrides)
    # 返回创建好的状态字典实例
    return state

def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    """
    return copy.deepcopy(graph_default_state)

if __name__ == "__main__":
    """
    测试
    """
    # 创建默认状态
    state = create_default_state(local_file_path="三体_书籍简介.md")
    logger.info(state)

from langgraph.constants import END
from langgraph.graph import StateGraph

from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_mp3_to_text import node_mp3_to_text
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.state import ImportGraphState

# 创建工作流对象
workflow = StateGraph(ImportGraphState)

# 添加节点
workflow.add_node(node_entry)
workflow.add_node(node_pdf_to_md)
workflow.add_node(node_md_img)
workflow.add_node(node_mp3_to_text)
workflow.add_node(node_document_split)
workflow.add_node(node_item_name_recognition)
workflow.add_node(node_bge_embedding)
workflow.add_node(node_import_milvus)

# 创建条件边的路径函数
# 判断state中的is_md_read_enabled、is_pdf_read_enabled，决定下一个节点
# is_md_read_enabled=True -- > node_md_img
# is_pdf_read_enabled=True -- > node_pdf_to_md
def condition_fun(state: ImportGraphState):
    if state["is_md_read_enabled"]:
        # 判断is_md_read_enabled是否为True
        # 若为True，说明当前上传的是md文件，下一个节点是node_md_img
        return "node_md_img"
    elif state["is_pdf_read_enabled"]:
        # 判断is_pdf_read_enabled是否为True
        # 若为True，说明当前上传的是pdf文件，下一个节点是node_pdf_to_md
        return "node_pdf_to_md"
    elif state["is_mp3_read_enabled"]:
        # 判断is_mp3_read_enabled是否为True
        # 若为True，说明当前上传的是音频文件，下一个节点是node_mp3_to_text（转写为纯文本）
        return "node_mp3_to_text"
    else:
        # 当is_md_read_enabled、is_pdf_read_enabled、is_mp3_read_enabled都为False
        # 说明上传的文件不是pdf/md/mp3文件，当前项目不支持，下一个节点是END
        return END


# 添加边
# 设置初始节点
# 等价于workflow.add_edge(START, "node_entry")
workflow.set_entry_point("node_entry")
# 设置条件边
workflow.add_conditional_edges(
    "node_entry",
    condition_fun,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        "node_mp3_to_text": "node_mp3_to_text",
        END: END
    }
)
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
# mp3 转写后直接进入文档切分（音频无图片，跳过 node_md_img），后续流程完全复用
workflow.add_edge("node_mp3_to_text", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

# 创建图对象
kb_import_app = workflow.compile()

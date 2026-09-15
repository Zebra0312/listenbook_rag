# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


# 定义mcp的服务配置
@dataclass
class McpConfig:
    mcp_base_url: str
    api_key: str
    search_tool: str  # 主工具名（默认版权书查询，优先使用）
    fallback_tool: str  # 兜底工具名（主工具无结果时使用；留空表示不兜底）

mcp_config = McpConfig(
    mcp_base_url=os.getenv("MCP_DASHSCOPE_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    search_tool=os.getenv("MCP_SEARCH_TOOL", "copyrightBookSearch"),
    fallback_tool=os.getenv("MCP_SEARCH_TOOL_FALLBACK", "internetBookSearch")
)

import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from app.conf.bailian_mcp_config import mcp_config
from app.core.logger import logger, node_log
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task


# 调用MCP工具
async def mcp_call_streamable(query, tool_name=None):
    """
    连接百炼书籍查询 MCP 并调用工具。

    :param query: 查询词（书名或改写后的问题）
    :param tool_name: 工具名；为空时用配置里的主工具（MCP_SEARCH_TOOL）
    """
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": mcp_config.api_key},
            "timeout": 300,
            "sse_read_timeout": 300,
            "terminate_on_close": True,
        },
        max_retry_attempts=2,
    )
    try:
        await search_mcp.connect()
        result = await search_mcp.call_tool(
            tool_name=tool_name or mcp_config.search_tool,
            arguments={"query": query},
        )
        return result
    finally:
        await search_mcp.cleanup()


def _s(value):
    """
    安全转字符串。

    MCP 返回的字段类型不统一（book_id 是 int、book_score 可能是 float/str），
    直接 .strip() 会抛 "'int' object has no attribute 'strip'"。
    """
    if value is None:
        return ""
    # 数字 0 在 Milvus/MCP 里通常表示"无值"（如空标题），当作空串处理，
    # 否则来源卡片上会显示一个没有意义的 "0"
    if isinstance(value, (int, float)) and value == 0:
        return ""
    return str(value).strip()


def _book_name_of(item_name):
    """
    从条目名里取出书名部分。

    库里条目名形如"三体-刘慈欣"（书名-作者），而查书工具对"书名"最敏感，
    直接把"三体-刘慈欣"整串丢进去会带偏召回。分隔口径与
    node_item_name_confirm 的书名对齐逻辑保持一致（按 - / — / ： / : 切第一段）。
    """
    s = _s(item_name)
    for sep in ("-", "—", "：", ":"):
        if sep in s:
            head = s.split(sep, 1)[0].strip()
            if head:
                return head
    return s


def build_search_query(state):
    """
    构造查询词：优先书名（可多本），没有书名时退化成改写后的问题。

    为什么不用 rewritten_query 打头：实测把「《三体》有声书的演播者是谁」这类
    完整问句丢给查书工具，会返回一堆不相干的小说（谁与争锋 / 良宵谁与共…）；
    换成书名「三体」则两个工具都精准命中《三体》/刘慈欣。
    """
    item_names = state.get("item_names") or []
    if isinstance(item_names, str):
        item_names = [item_names]
    names = [_book_name_of(n) for n in item_names]
    # 去掉空值并去重保序
    names = [n for n in names if n]
    if names:
        return "、".join(dict.fromkeys(names))
    return (state.get("rewritten_query") or "").strip()


# 多本书查询时，最多分别查询几本（每本一次 MCP 调用，避免请求过多）
MAX_QUERY_BOOKS = 3


def build_search_queries(state):
    """
    构造查询词列表（多本书时拆开分别查）。

    为什么要拆：实测把「斗破苍穹、斗罗大陆」作为一个查询词丢给工具，
    只会返回第一本（斗破苍穹）的结果，第二本完全拿不到、自然也带不出它的封面；
    拆成两次独立查询后两本书都能召回。书名超过 MAX_QUERY_BOOKS 本时，
    退化回「、」连接的组合查询（1 次调用），避免请求过多。
    """
    item_names = state.get("item_names") or []
    if isinstance(item_names, str):
        item_names = [item_names]
    names = [_book_name_of(n) for n in item_names]
    names = list(dict.fromkeys(n for n in names if n))
    if names:
        return names if len(names) <= MAX_QUERY_BOOKS else ["、".join(names)]
    fallback = (state.get("rewritten_query") or "").strip()
    return [fallback] if fallback else []


async def _query_one(query):
    """
    查询单个书名：主工具优先，无结果时用兜底工具补充。

    :param query: 查询词（单个书名，或退化后的组合查询）
    :return: [{title, url, snippet, image_urls}, ...]
    """
    result = await mcp_call_streamable(query)
    docs = parse_search_docs(result.content[0].text, query)
    if not docs and mcp_config.fallback_tool:
        logger.info(
            f"书籍查询主工具（{mcp_config.search_tool}）无结果，"
            f"改用兜底工具（{mcp_config.fallback_tool}）"
        )
        result = await mcp_call_streamable(query, mcp_config.fallback_tool)
        docs = parse_search_docs(result.content[0].text, query)
    return docs


async def _search_many(queries):
    """
    并发查询多个书名，按「书名 + 链接」去重合并。

    都是只读的网络请求，并发不会把单次耗时叠加（总耗时≈最慢的那一次）。
    单个查询失败不影响其它查询（return_exceptions）。
    """
    if len(queries) == 1:
        return await _query_one(queries[0])
    batches = await asyncio.gather(
        *[_query_one(q) for q in queries], return_exceptions=True
    )
    merged, seen = [], set()
    for i, batch in enumerate(batches):
        if isinstance(batch, Exception):
            logger.warning(f"书籍查询「{queries[i]}」失败，跳过：{batch}")
            continue
        for d in batch:
            key = (_s(d.get("title")), _s(d.get("url")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)
    return merged


def _norm_name(value):
    """书名归一化：去书名号/空格/常见标点并转小写，用于一致性比对。"""
    s = _s(value)
    for ch in "《》「」『』\"'‘’“” 　·・-—_:：,，.。、()（）[]【】!":
        s = s.replace(ch, "")
    return s.lower()


def _split_query_names(query):
    """把查询词拆成若干归一化书名（兼容「、」「，」分隔的多本书查询）。"""
    raw = _s(query).replace("，", "、")
    return [x for x in (_norm_name(part) for part in raw.split("、")) if x]


def _name_matches(query, book_name):
    """
    判断 MCP 返回的书名与本次查询是否为**同一本书**。

    为什么要这个闸门：查**版权库里没有的书**时（例如《蛊真人》），
    copyrightBookSearch 会返回"相近的其它版权书"（师叔你的法宝太不正经了 / 苗疆蛊事…），
    这些书**带有封面**但与用户问的书毫不相干，直接展示就是"图文不符"。
    宁可不配图，也不配错图。

    判定口径刻意用**归一化后完全相等**，而不是"互相包含"：
    查询「三体」若用包含判定，《三体文明》《三体：死神永生》都会被算命中，
    于是给《三体》配上了别的书的封面 —— 依旧是图文不符。
    """
    b = _norm_name(book_name)
    if not b:
        return False
    return b in _split_query_names(query)


def _book_to_doc(book, query=""):
    """
    将书籍查询 MCP 返回的单本书，映射为下游统一的文档结构。

    输入（书籍对象）形如:
        {"book_name": "三体", "author_name": "刘慈欣",
         "introduction": "……", "tags": "科幻,少儿",
         "book_score": 9.2, "landingpage_url": "https://..."}

    输出: {"title", "url", "snippet"}，与网络搜索路保持一致，
          这样 node_rerank 的合并逻辑无需任何改动。
    """
    # 书名 + 作者 作为标题，便于答案里标注来源
    book_name = _s(book.get("book_name"))
    author = _s(book.get("author_name"))
    title = f"{book_name} · {author}" if book_name and author else (book_name or author)

    # 详情页链接：internetBookSearch 通常不返回，用 book_id 按官方格式补出，
    # 保证答案里的来源标注带得上链接（拼不出就留空，不编造）。
    url = _s(book.get("landingpage_url")) or _s(book.get("url"))
    if not url and _s(book.get("book_id")):
        url = f"https://t.shuqi.com/book/{_s(book['book_id'])}.html?from=znyx_mcp"

    # 正文：简介打底，追加作者/标签/评分等元信息，供重排与答案引用
    intro = _s(book.get("introduction"))
    meta_parts = []
    if author:
        meta_parts.append(f"作者：{author}")
    tags = _s(book.get("tags"))
    if tags:
        meta_parts.append(f"标签：{tags}")
    score = book.get("book_score")
    if score not in (None, "", 0):
        meta_parts.append(f"评分：{score}")
    category = _s(book.get("first_category_name"))
    if category:
        meta_parts.append(f"分类：{category}")

    snippet = intro
    if meta_parts:
        meta = "；".join(meta_parts)
        snippet = f"{intro}\n{meta}" if intro else meta

    # 封面 / 书卡：书籍查询 MCP 返回的是**真实可用的图片地址**（本地切片里往往没有），
    # 一并带给下游，供"问封面/看图"时展示；internetBookSearch 不返回这两个字段，允许为空。
    # 图片闸门：只有书名与本次查询对得上时才采用它的封面/书卡。
    # 查库外书时版权书工具会返回不相干的版权书，那些封面必须丢掉（否则图文不符）。
    images = []
    if query and _name_matches(query, book_name):
        for key in ("cover_image_url", "book_card"):
            u = _s(book.get(key))
            if u and u not in images:
                images.append(u)

    return {"title": title, "url": url, "snippet": snippet.strip(), "image_urls": images}


def parse_search_docs(text, query=""):
    """
    把 MCP 返回的文本解析成 [{title, url, snippet}]。

    兼容两种返回体：书籍查询 MCP 返回「书籍对象数组」；
    老的 WebSearch 返回 {"pages": [...]}。两者都映射到同一结构，
    因此只改 .env 的 MCP_DASHSCOPE_BASE_URL 即可在两种服务间切换。
    """
    payload = json.loads(text)
    docs = []
    if isinstance(payload, dict):
        for page in payload.get("pages", []):
            docs.append(
                {
                    "title": _s(page.get("title")),
                    "url": _s(page.get("url")),
                    "snippet": _s(page.get("snippet")),
                }
            )
    elif isinstance(payload, list):
        for book in payload:
            if isinstance(book, dict):
                docs.append(_book_to_doc(book, query))
    return docs


@node_log("node_web_search_mcp")
def node_web_search_mcp(state: QueryGraphState):
    """
    节点: 书籍查询 (node_web_search_mcp)
    实现内容:
    1. 通过 MCP 协议连接百炼「书旗小说」书籍查询服务；
    2. 查询词优先取书名（item_names），多本书时拆开分别查询（组合查询只能召回第一本），退化到改写后的问题；
    3. 工具按官方策略：主工具（默认 copyrightBookSearch，版权书优先）查不到时，
       用 fallback 工具（默认 internetBookSearch）全网兜底；
    4. 整理为统一的 {title, url, snippet} 结构，不参与 RRF 融合，留给重排阶段合并。
    """
    # 记录当前任务的状态为进行中
    add_running_task(state["session_id"], "node_web_search_mcp", state["is_stream"])
    # 构造查询词：多本书时拆成多个查询（组合查询只能召回第一本）
    queries = build_search_queries(state)
    # 创建存储最终结果的列表
    results = []
    # 判断查询词是否为空
    if queries:
        # 说明：书籍查询依赖外部 MCP 服务，属于"锦上添花"的一路召回；
        # 这里做降级保护，MCP 不可用/超时/返回异常时只记日志并返回空列表，
        # 不影响向量检索与 HyDE 检索两路本地召回。
        try:
            results = asyncio.run(_search_many(queries))
        except Exception as e:
            logger.warning(f"书籍查询（MCP）调用失败，本次降级为仅本地召回：{e}")
    # 记录当前任务的状态为已完成
    add_done_task(state["session_id"], "node_web_search_mcp", state["is_stream"])
    return {"web_search_docs": results}


if __name__ == "__main__":
    init_state = {
        "session_id": "abc",
        "is_stream": False,
        "item_names": ["三体-刘慈欣"],
        "rewritten_query": "《三体》有声书的演播者是谁",
    }
    print("查询词:", build_search_query(init_state))
    result = node_web_search_mcp(init_state)
    print(result)

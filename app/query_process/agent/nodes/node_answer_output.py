import re

from app.clients.mongo_history_utils import save_chat_message
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    set_task_result,
    get_done_task_list,
    TASK_STATUS_COMPLETED,
)

# 上下文最大字符数
MAX_CONTEXT_CHARS = 12000

# 本地知识库与书籍查询 MCP 都没有召回内容时的兜底话术。
# 原来这句话写在 node_item_name_confirm 的兜底分支里；现在那条路要放行 MCP 兜底，
# 所以"最终仍没查到"的提示挪到这里，依据重排结果是否为空来决定。
NO_CONTEXT_ANSWER = "抱歉，未找到相关书籍，请提供书名或作者，以便我为您查询。"

# 内容类型枚举 → 中文标签（用于在参考内容里给大模型更好读的来源标注）
CONTENT_TYPE_CN = {
    "audiobook_info": "有声书信息",
    "book_intro": "书籍简介",
    "author_intro": "作者介绍",
    "listening_note": "听书笔记",
    "recommendation": "推荐运营资料",
    "comment_summary": "用户评论摘要",
    "faq": "常见问答",
}

# 参考来源最多回传/展示的条数（够用即可，避免前端列表过长）
MAX_SOURCES = 8

# 外部书籍库（MCP）图片最多展示的张数。
# 需求：用本地知识库以外的数据回答时，顺手把相关性最高的封面带上，
# 而不是非要用户明确要图才给；限量避免答案下方堆一大片图。
MAX_MCP_IMAGES = 2


def _s(value):
    """
    安全转字符串。

    MCP / Milvus 返回的字段类型不统一（例如 chunk_id、book_id 是 int，
    book_score 可能是 float 或 str），直接 .strip() 会抛
    "'int' object has no attribute 'strip'"。这里统一收口。
    """
    if value is None:
        return ""
    # 数字 0 在 Milvus/MCP 里通常表示"无值"（如空标题），当作空串处理，
    # 否则来源卡片上会显示一个没有意义的 "0"
    if isinstance(value, (int, float)) and value == 0:
        return ""
    return str(value).strip()


def _collect_sources(state: QueryGraphState):
    """
    整理本次答案用到的参考来源，供前端展示"来自智库 / 来自 MCP 服务"。

    来源判定沿用重排阶段写入的 source 字段：local → 智库（本地知识库切片），
    web → MCP 书籍查询服务。按类型输出不同字段：
      智库：书名 / 作者 / 内容类型（中文）/ 来源文件 / chunk_id
      MCP ：标题 / 详情页链接
    顺序与重排得分一致（已按分数降序），并做去重。
    """
    docs = state.get("reranked_docs") or []
    sources = []
    seen = set()
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        raw_source = _s(doc.get("source"))
        kind = "mcp" if raw_source == "web" else "kb"
        if kind == "mcp":
            item = {
                "kind": "mcp",
                "title": _s(doc.get("title")),
                "url": _s(doc.get("url")),
            }
            # 标题和链接都为空 → 没有可展示的信息，跳过
            if not item["title"] and not item["url"]:
                continue
            key = ("mcp", item["title"], item["url"])
        else:
            item = {
                "kind": "kb",
                "title": _s(doc.get("title")),
                "book_name": _s(doc.get("book_name")),
                "author": _s(doc.get("author")),
                "content_type": CONTENT_TYPE_CN.get(_s(doc.get("content_type")), ""),
                "source_file": _s(doc.get("source_file")),
                "chunk_id": _s(doc.get("chunk_id")),
            }
            key = ("kb", item["chunk_id"] or item["title"] or item["source_file"])
        if key in seen:
            continue
        seen.add(key)
        score = doc.get("score")
        if isinstance(score, (int, float)):
            item["score"] = round(float(score), 4)
        sources.append(item)
    return sources[:MAX_SOURCES]


@step_log("step_1_check_answer")
def step_1_check_answer(state: QueryGraphState):
    # 分别获取状态中answer和is_stream
    answer = state.get("answer")
    is_stream = state.get("is_stream")
    # 判断answer是否为空
    if answer:
        # 判断is_stream是否为空
        if is_stream:
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
        else:
            set_task_result(state["session_id"], "answer", answer)
        return True
    else:
        return False

def _construct_chitchat_prompt(state):
    """
    闲聊回答的提示词组装：与书籍无关的寒暄/闲聊走这里，不引用任何检索内容。

    历史沿用 answer_out 的口径（用户: / 助手:），只做上下文、不喂检索结果。
    """
    history = state.get("history") or []
    history_text = ""
    for h in history:
        role = h.get("role")
        text = h.get("text")
        if role == "user" and text:
            history_text += f"用户: {text}\n"
        elif role == "assistant" and text:
            history_text += f"助手: {text}\n"
    if not history_text:
        history_text = "无历史记录"
    return load_prompt(
        "chitchat",
        history_text=history_text,
        query=state.get("original_query") or state.get("rewritten_query") or "",
    )


@step_log("step_2_construct_prompt")
def step_2_construct_prompt(state: QueryGraphState):
    # 从状态中获取所需要的数据
    original_query = state.get("original_query")
    rewritten_query = state.get("rewritten_query")
    question = rewritten_query if rewritten_query else original_query
    reranked_docs = state.get("reranked_docs")
    item_names = state.get("item_names")
    history_list = state.get("history")
    """
    将reranked_docs中的数据转换为以下格式（带听书域来源标注，便于答案引用书名/作者/类型/文件）：
    "[1] [local] [书名=三体] [作者=刘慈欣] [内容类型=书籍简介] [来源文件=三体简介.md]
     [chunk_id=123] [score=0.9500] [title=内容简介]
     这里是文档的正文内容..."
    """
    # 处理上下文，创建存储处理之后的结果的列表
    docs = []
    # 创建记录最大字符数的变量
    used = 0
    # 对reranked_docs进行遍历
    for num, chunk in enumerate(reranked_docs, start=1):
        # 从chunk中获取所需要的数据，并存储到列表中
        text = chunk.get("text")
        if not text:
            continue
        data_list = [f"[{num}]"]
        source = chunk.get("source")
        if source:
            data_list.append(f"[{source}]")
        # 听书域元数据：书名 / 作者 / 内容类型 / 来源文件
        book_name = chunk.get("book_name")
        if book_name:
            data_list.append(f"[书名={book_name}]")
        author = chunk.get("author")
        if author:
            data_list.append(f"[作者={author}]")
        content_type = chunk.get("content_type")
        if content_type:
            data_list.append(f"[内容类型={CONTENT_TYPE_CN.get(content_type, content_type)}]")
        source_file = chunk.get("source_file")
        if source_file:
            data_list.append(f"[来源文件={source_file}]")
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            data_list.append(f"[chunk_id={chunk_id}]")
        score = chunk.get("score")
        if score is not None:
            data_list.append(f"[score={float(score):.4f}]")
        title = chunk.get("title")
        if title:
            data_list.append(f"[title={title}]")
        # 外部书籍库返回的封面/书卡图片地址：一并交给模型，
        # 用户问"封面长什么样"时可直接引用（这些地址已在图片白名单内，不会被当成编造）
        for img in (chunk.get("image_urls") or []):
            if img:
                data_list.append(f"[图片={img}]")
        # 拼接各个数据为指定格式
        doc = " ".join(data_list) + "\n" + text
        # 判断当前字符数是否超过最大字符数的阈值
        if used + len(doc) > MAX_CONTEXT_CHARS:
            break
        # 存储每条数据转换的结果
        docs.append(doc)
        # 记录本次循环的字节数
        used += len(doc) + 2
    # 将每条数据转换的结果拼接为字符串
    context = "\n\n".join(docs) if docs else "无参考内容"
    """
    将历史对话转换为以下格式：
    用户: xxx
    助手: xxx
    """
    # 处理历史记录
    history_str = ""
    # 判断历史记录是否为空
    if history_list:
        # 对history_list进行遍历
        for history in history_list:
            # 分别获取历史记录中role和text
            role = history.get("role")
            text = history.get("text")
            # 判断role是否为user，若为user，则拼接"用户: xxx"
            history_text = ""
            if role == "user" and text:
                history_text += f"用户: {text}\n"
            elif role == "assistant" and text:
                history_text += f"助手: {text}\n"
            if used + len(history_text) > MAX_CONTEXT_CHARS:
                break
            history_str += history_text
            # 记录拼接之后的字符数
            used += len(history_text)
    else:
        history_str = "无历史记录"
    """
    将书籍主体转换为以下格式：
    书籍主体1, 书籍主体2, ...
    """
    item_names_str = ", ".join(item_names)
    # 读取提示词
    prompt = load_prompt(
        "answer_out",
        context=context,
        history=history_str,
        item_names=item_names_str,
        question=question,
    )
    return prompt

@step_log("step_3_generate_response")
def step_3_generate_response(state: QueryGraphState, prompt: str):
    # 从状态中获取session_id和is_stream
    session_id = state.get("session_id")
    is_stream = state.get("is_stream")
    # 获取大模型对象
    llm = get_llm_client()
    # 判断是否为流式调用
    if is_stream:
        # 创建存储最终answer的变量
        answer = ""
        try:
            # 以流式方式调用大模型
            for chunk in llm.stream(prompt):
                # 当使用流式调用大模型时，每次返回封装了token的对象
                # 每次返回的具体的token
                content = chunk.content
                # 将每次返回的token添加到队列中
                push_to_session(session_id, SSEEvent.DELTA, {"delta": content})
                # 将每次的token拼接，获取最终的answer
                answer += content
        except Exception as e:
            push_to_session(session_id, SSEEvent.ERROR, {"error": e})
        # 更新状态
        state["answer"] = answer
    else:
        try:
            # 表示非流式，则直接调用大模型
            response = llm.invoke(prompt)
            # 获取最终的answer
            answer = response.content
            # 设置当前任务的状态
            set_task_result(session_id, "answer", answer)
            # 更新状态
            state["answer"] = answer
        except Exception as e:
            state["answer"] = "抱歉，生成回答时出现错误。"
    return state

# 从最终的文档中提取图片的url
def _extract_images_from_docs(reranked_docs):
    # 创建存储提取的图片url的列表
    image_urls = []
    # 创建正则表达式
    pattern = r"!\[.*?\]\((.*?)\)"
    # 已加入的外部书籍库图片计数（用于限量）
    mcp_added = 0
    # 对文档进行遍历
    for doc in reranked_docs:
        # 获取文档中的url
        url = doc.get("url")
        # 判断url是否为空
        if url:
            if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                if url not in image_urls:
                    image_urls.append(url)
        # 获取文档中的text
        text = doc.get("text")
        if text:
            # 提供正则表达式提取文档中图片的url
            matches = re.findall(pattern, text)
            # 判断matches是否为空
            if matches:
                for match in matches:
                    # match 可能是空串（切片里存在 ![alt]() 这类空地址写法），空值不能入白名单，
                    # 否则上下文会拼出 [图片=] 这种空标注，模型照抄后就会显示成乱码标记
                    if match and match not in image_urls:
                        image_urls.append(match)
        # 外部书籍库（MCP）返回的封面/书卡图片地址。
        # docs 已按重排得分降序，所以先遍历到的就是「相关性稍高」的那几本；
        # 最多取 MAX_MCP_IMAGES 张，避免答案下方堆图
        for img in (doc.get("image_urls") or []):
            if mcp_added >= MAX_MCP_IMAGES:
                break
            if img and img not in image_urls:
                image_urls.append(img)
                mcp_added += 1
    return image_urls


# 图片 URL：惰性匹配到"图片扩展名"为止，避免把后面的中文说明文字一起吞掉
_IMG_URL_RE = re.compile(r"https?://[^\s)\]<>\"'】]*?\.(?:png|jpe?g|gif|webp|bmp|svg)", re.IGNORECASE)
# 扩展名后面紧跟的 ?query / #hash（用于原样保留，例如预签名地址）
_URL_TAIL_RE = re.compile(r"[?#][^\s)\]<>\"'】]*")
# 正文里的 URL 候选（贪婪，交给 _extract_image_url 再裁剪）
_URL_BARE_RE = re.compile(r"https?://[^\s)\]<>\"'】]+")
# Markdown 图片语法 ![alt](url)
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
# 【图片】 标记（兼容 [图片] 写法）
_IMG_MARKER_RE = re.compile(r"【\s*图片\s*】|\[\s*图片\s*\]")

# 上下文里的图片标注 [图片=<url>]：模型有时会把它原样抄进答案（甚至抄成空的 [图片=]）。
# 这种标记不是给用户看的正文，一律删除 —— 图片由系统通过 final.image_urls 自动附在答案下方。
_CTX_IMG_TAG_RE = re.compile(r"[\[【]\s*图片\s*=\s*[^\]】]*[\]】]")


def _ensure_text_answer(answer: str, docs, image_urls) -> str:
    """
    保证答案里有正文。

    背景：模型偶尔只输出【图片】区块、一个字正文都不写（尤其问「我想看 XX」这类视觉化措辞时），
    前端就会呈现"空白答案 + 一张图"。这里检测到这种情况时，补一句基于参考内容的说明。
    """
    text = str(answer or "")
    # 去掉【图片】区块后是否还有正文
    head = text.split("【图片】", 1)[0].strip()
    if head or not image_urls:
        return answer
    # 取一个可读的书名：优先 book_name；MCP 的 title 形如"书名 · 作者"，取书名部分
    name = ""
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        name = (doc.get("book_name") or "").strip()
        if not name:
            title = (doc.get("title") or "").strip()
            name = title.split("·", 1)[0].strip() if "·" in title else title
        if name:
            break
    lead = f"以下是《{name}》的相关图片：" if name else "以下是相关图片："
    tail = text.split("【图片】", 1)[1] if "【图片】" in text else ""
    return f"{lead}\n\n【图片】{tail}".strip() if tail else lead


def _extract_image_url(text: str):
    """从一段文本里抽取第一个图片 URL；不是图片地址则返回 None。保留 ?query/#hash。"""
    m = _IMG_URL_RE.search(text or "")
    if not m:
        return None
    url = m.group(0)
    rest = (text or "")[m.end():]
    if rest[:1] in ("?", "#"):
        tail = _URL_TAIL_RE.match(rest)
        if tail:
            url += tail.group(0)
    return url


def _norm_img_url(u: str) -> str:
    """归一化图片 URL 用于比对：去包裹符号、去 query/fragment、去结尾标点。"""
    s = str(u or "").strip().strip("<>").rstrip(".,;，。、")
    return s.split("?")[0].split("#")[0]


def _sanitize_answer_images(answer: str, allowed_urls) -> str:
    """
    清洗答案里的图片地址，防止大模型编造/幻觉出参考内容中不存在的图片链接
    （典型表现：参考资料无图时编出 https://example.com/xxx.jpg 这类占位地址）。

    规则：
    0) 先删除模型抄自上下文的 [图片=<url>] 标注（含被抄成空的 [图片=]）——它属于参考内容的字段标注，不是正文；
    1) 白名单 = 参考切片里真实存在的图片 URL，只有白名单内的地址才允许保留；
    2) Markdown 图片 ![](url) 中 url 不在白名单 → 整段删除；
    3) 正文中游离的、非白名单图片 URL → 删除该 URL（普通网页链接不受影响）；
    4) 【图片】区块只保留白名单内的 URL；若一个都不剩 → 连【图片】标题一起删除。
    """
    if not answer:
        return answer
    allowed = {_norm_img_url(u) for u in (allowed_urls or []) if u}

    def _keep(url: str) -> bool:
        return _norm_img_url(url) in allowed

    # 0) 先清掉模型从上下文抄来的 [图片=<url>] 标注（含被抄成空的 [图片=]）
    answer = _CTX_IMG_TAG_RE.sub("", answer)

    # 1) 去掉参考内容里没有的 Markdown 图片
    answer = _MD_IMG_RE.sub(lambda m: m.group(0) if _keep(m.group(1)) else "", answer)

    def _clean_prose(text: str) -> str:
        """正文里的图片 URL 白名单过滤（普通网页链接原样保留）。"""
        def _bare(m):
            raw = m.group(0)
            url = _extract_image_url(raw)
            if url is None or _keep(url):
                return raw
            return raw.replace(url, "", 1)
        return _URL_BARE_RE.sub(_bare, text)

    # 2) 以最后一个【图片】标记为界，拆分"正文"与"图片区块"
    last_marker = None
    for m in _IMG_MARKER_RE.finditer(answer):
        last_marker = m

    if last_marker is None:
        return _clean_prose(answer).strip()

    head = _clean_prose(answer[:last_marker.start()]).rstrip()
    tail = answer[last_marker.end():]
    kept = []
    for line in tail.splitlines():
        url = _extract_image_url(line)
        if url and _keep(url) and url not in kept:
            kept.append(url)
    # 一张真实图片都没有 → 整个区块丢掉，避免前端把编造地址当图片展示
    return (f"{head}\n\n【图片】\n" + "\n".join(kept) if kept else head).strip()


@step_log("step_4_write_history")
def step_4_write_history(state: QueryGraphState, image_urls, sources=None):
    # 保存历史记录
    session_id = state.get("session_id")
    item_names = state.get("item_names")
    answer = state.get("answer")
    # 判断answer是否有值
    if answer:
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            rewritten_query="",
            item_names=item_names,
            image_urls=image_urls,
            sources=sources or [],
            message_id=None
        )



@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState):
    """
    节点: 答案生成 (node_answer_output)
    实现内容:
    1. 若 state 中已有 answer（书籍主体需澄清 / 未找到书籍的兜底话术）直接透传；
       若本地库与 MCP 都无召回内容，则直接落兜底话术，不让模型凭空编造；
    2. 组装 Prompt（参考切片带书名/作者/内容类型/来源文件 + 历史对话 + 用户问题）；
    3. 流式或同步调用大模型，流式时逐 token 推送 delta 事件；
    4. 图片白名单校验，防止编造图片地址；
    5. 写入历史并推送 final 事件（携带清洗后的答案、图片地址与参考来源）。
    """
    # 记录当前任务的状态为进行中
    add_running_task(state["session_id"], "node_answer_output", state["is_stream"])
    # 闲聊分流（由 node_item_name_confirm 判定）：与书籍无关的寒暄/闲聊，
    # 直接用闲聊提示词回答，不走检索，也不会被下面「无召回 → 兜底话术」覆盖成未找到书籍
    if state.get("is_chitchat"):
        prompt = _construct_chitchat_prompt(state)
        state["prompt"] = prompt
        state = step_3_generate_response(state, prompt)
    else:
        # 阶段零：防幻觉兜底 —— 本地知识库与书籍查询 MCP 都没召回内容时，
        # 直接把兜底话术写进 answer（先写再判存在，后续 delta/final 推送逻辑无需特殊处理），
        # 否则会把"无参考内容"交给模型，诱导它凭空编造。
        if not state.get("answer") and not (state.get("reranked_docs") or []):
            logger.info(
                f"{state.get('session_id')} 本地知识库与书籍查询 MCP 均无召回内容，返回兜底话术"
            )
            state["answer"] = NO_CONTEXT_ANSWER
        # 阶段一：检查answer是否存在,如果存在直接输出answer中的答案
        answer_exists = step_1_check_answer(state)
        # 判断answer是否存在
        if not answer_exists:
            prompt = step_2_construct_prompt(state)
            state["prompt"] = prompt
            state = step_3_generate_response(state, prompt)
    # 提取图片URL（白名单：仅参考切片里真实存在的图片地址，用于历史记录和前端展示）
    image_urls = _extract_images_from_docs(state.get("reranked_docs") or [])
    # 参考来源（智库切片 / MCP 书籍）：提前算好，供历史留存与 final 事件使用
    sources = _collect_sources(state)
    if state.get("answer"):
        # 清洗答案中的图片地址：丢弃模型编造的 URL（如参考资料无图时幻觉出的 example.com 占位地址）
        cleaned = _sanitize_answer_images(state["answer"], image_urls)
        if cleaned != state["answer"]:
            logger.warning(
                f"{state.get('session_id')} 答案中的图片地址已被清洗（疑似模型编造，已按参考内容白名单过滤）"
            )
        state["answer"] = cleaned
        # 兜底：模型可能只输出图片、完全不给正文（问"我想看 XX"这类措辞时容易触发），
        # 那样前端就只有一张图、没有文字。这里补一句基于参考内容的说明。
        state["answer"] = _ensure_text_answer(
            state["answer"], state.get("reranked_docs") or [], image_urls
        )
        step_4_write_history(state, image_urls=image_urls, sources=sources)
    # 记录当前任务的状态为已完成
    # 注意：必须先于 final 事件推送。前端收到 final 后会立即关闭 SSE 连接，
    # 之后再推的 progress 无人接收，进度条会残留“⏳ 生成答案 / 状态：处理中”
    add_done_task(state["session_id"], "node_answer_output", state["is_stream"])
    # 将图片和最终answer推送到浏览器端
    if state.get("is_stream"):
        push_to_session(
            state['session_id'],
            SSEEvent.FINAL,
            {
                "answer": state["answer"],
                "image_urls": image_urls,  # 发送图片URL给前端
                "sources": sources,  # 参考来源：智库切片 / MCP 书籍（前端做来源展示）
                # 随最终答案一并下发进度快照：前端 close 连接后收不到后续 progress，
                # 靠它渲染出“全部已完成”的收尾状态
                "status": TASK_STATUS_COMPLETED,
                "done_list": get_done_task_list(state["session_id"]),
                "running_list": [],
            }
        )
    return state

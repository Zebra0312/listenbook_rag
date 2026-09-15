import json
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from app.clients.milvus_utils import get_milvus_client, hybrid_search, create_hybrid_search_requests
from app.clients.mongo_history_utils import save_chat_message, get_recent_messages, update_message_item_names
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 听书域补充：切片库里存的主体名是"书名-作者"复合形式（如"三体-刘慈欣"），
# 而用户提问通常只说书名（如"三体"），单纯靠向量相似度会落在 0.6~0.85 的候选区间，
# 导致每次都要求用户澄清。这里按分隔符取"书名部分"做包含式对齐来兜住这种情况。
NAME_SEPARATORS = ("-", "－", "—", "–", "_", "：", ":", "/", "|")


def _book_name_part(name: str) -> str:
    """从"三体-刘慈欣"取出"三体"；没有分隔符时原样返回。"""
    text = (name or "").strip()
    for separator in NAME_SEPARATORS:
        if separator in text:
            return text.partition(separator)[0].strip()
    return text

# 本轮问题里出现这些词，才允许沿用历史会话里的书名（指代消解的正常场景）。
# 注意：这里刻意不收单字「本」——「推荐几本小说」「买一本」里的「本」是量词，
# 会把它误判成指代词，反而把历史书名保留下来；因此只保留「本书 / 这本 / 那本」这类组合。
REFER_WORDS = (
    "它", "他", "她", "这", "那", "该", "此",
    "刚才", "上面", "前面", "之前", "上述",
    "本书", "本作", "这本", "那本", "该本",
    "另一本", "前一本", "前面那本",
)


def _strip_history_leaked_names(query, names):
    """
    剔掉从历史里继承来、本轮问题根本没提的书名。

    背景：本轮用户只发了「你好」，模型却返回了上一轮的 ['斗破苍穹', '斗罗大陆']，
    于是检索去查这两本书，答案变成两本书的比较，与用户问题完全无关。
    提示词已明确禁止该行为，这里再加一道代码闸门兜底（模型不总听话）。

    判定：只有当本轮问题里出现了书名，或出现了明确的指代词
    （它 / 这 / 那 / 本 / 刚才 / 上面 …）时，才允许沿用；否则一律丢弃。

    :param query: 本轮用户原始问题
    :param names: 模型返回的 item_names
    :return: 过滤后的 item_names
    """
    if not names:
        return []
    q = str(query or "")
    has_refer = any(w in q for w in REFER_WORDS)
    kept = []
    for n in names:
        if has_refer:
            kept.append(n)
            continue
        book = _book_name_part(str(n))
        # 书名或其条目名在本轮问题里字面出现 → 说明是用户自己提的，保留
        if book and (book in q or str(n) in q):
            kept.append(n)
    return kept

@step_log("step_3_extract_info")
def step_3_extract_info(original_query, history_list):
    """
    # 步骤3: 从用户的问题中提取item_names（书籍主体）并重写用户问题
    :param original_query: 用户原始提出的问题
    :param history_list: 步骤1 通过get_recent_messages方法 传入会话session_id获得的历史记录
    :return: 重写用户问题
    """
    # 遍历history_list，将每个历史对话拼接，role(user/ai):text
    history_text = ""
    for history in history_list:
        history_text += f"{history['role']}: {history['text']}\n"
    # 读取rewritten_query_and_itemnames.prompt文件获取提示词
    prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=original_query)
    # 组织用户提示词和系统提示词
    messages = [
        SystemMessage("你是一个专业的听书平台知识助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(prompt),
    ]
    try:
        # 获取大模型对象
        llm = get_llm_client()
        # 调用大模型
        response = llm.invoke(messages)
        # 获取大模型输出的内容
        result = response.content
        # 判断result是否是json代码块，```json{key:value}```
        if result.startswith("```json"):
            result = result.replace("```json", "").replace("```", "")
        # 将字符串格式的json转换为Python对象
        extract_result = json.loads(result)
        # 判断结果中是否包含item_names
        if "item_names" not in extract_result:
            extract_result["item_names"] = []
        # 判断结果中是否包含rewritten_query
        if "rewritten_query" not in extract_result:
            extract_result["rewritten_query"] = original_query
        # 闸门：剔掉从历史继承来、本轮问题根本没提的书名（模型不总听提示词的话）
        raw_names = extract_result.get("item_names") or []
        kept = _strip_history_leaked_names(original_query, raw_names)
        if len(kept) != len(raw_names):
            logger.info(
                f"本轮问题未提及书籍（{original_query!r}），"
                f"已丢弃从历史继承的书名：{raw_names} -> {kept}"
            )
        extract_result["item_names"] = kept
        # is_chitchat 兜底：模型漏返回时默认 False；但只要有书名，就绝不可能是纯闲聊
        if "is_chitchat" not in extract_result:
            extract_result["is_chitchat"] = False
        if kept:
            extract_result["is_chitchat"] = False
        return extract_result
    except Exception as e:
        logger.error(f"提取书籍主体并且重写用户问题时出现了异常：{e}")
        return {"item_names": [], "rewritten_query": original_query, "is_chitchat": False}


@step_log("step_4_vectorize_and_query")
def step_4_vectorize_and_query(item_names):
    """
        results=[
            {
                "extracted_name":从用户的问题中提取的书籍主体,
                "matches":[
                    {
                        "item_name":从向量数据库中检索到的书籍主体,
                        "score":分数
                    },
                    ...
                ]
            },
            ...
        ]
    """
    # 创建存储最终结果的列表
    results = []
    try:
        # 获取Milvus的客户端
        milvus_client = get_milvus_client()
        # 判断milvus_client是否为空
        if not milvus_client:
            logger.error("获取Milvus客户端失败")
            return results
        # 获取要检索的集合（书籍主体集合 listenbook_item_names）
        collection_name = milvus_config.item_name_collection
        # 判断collection_name是否为空
        if not collection_name:
            logger.error("获取书籍主体的集合名称失败")
            return results
        # 获取item_names所对应的向量
        embeddings = generate_embeddings(item_names)
        # 对item_names进行遍历，在向量数据库中进行检索
        for i in range(len(item_names)):
            # 分别获取每个item_name所对应的稠密向量和稀疏向量
            dense_vector = embeddings["dense"][i]
            sparse_vector = embeddings["sparse"][i]
            # 设置稠密向量和稀疏向量的检索方式
            reqs = create_hybrid_search_requests(dense_vector=dense_vector, sparse_vector=sparse_vector, limit=5)
            # 进行混合检索
            """
                混合检索的结果的结构：
                [
                    [
                        {
                            'pk': 468868229533272140,
                            'distance': 0.9151462912559509,
                            'entity': {'item_name': '三体-刘慈欣'}
                        }
                    ]
                ]
            """
            hybrid_search_results = hybrid_search(
                client=milvus_client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                norm_score=True,
                limit=5,
                output_fields=["item_name"]
            )
            # 创建存储检索的结果的列表
            matches = []
            # 判断检索的结果是否为空
            if hybrid_search_results and len(hybrid_search_results) > 0:
                # 对检索的结果进行遍历
                for result in hybrid_search_results[0]:
                    matches.append(
                        {
                            "item_name": result["entity"]["item_name"],
                            "score": result["distance"]
                        }
                    )
            # 存储最终的结果
            results.append(
                {
                    "extracted_name": item_names[i],
                    "matches": matches
                }
            )
        return results
    except Exception as e:
        logger.error(f"混合检索item_name失败，{e}")


@step_log("step_5_align_item_names")
def step_5_align_item_names(query_results):
    # 创建存储已确认的item_name的列表
    confirmed_item_names: List[str] = []
    # 创建存储待确认的item_name的列表
    options: List[str] = []
    # 遍历query_results
    for result in query_results:
        # 获取从用户的问题中提取出的extracted_name
        extracted_name = result["extracted_name"]
        # 获取extracted_name所检索的数据
        matches = result["matches"]
        # 将所检索到的数据根据分数进行倒序排序
        matches.sort(key=lambda match: match["score"], reverse=True)
        # 判断matches是否为空
        if not matches:
            logger.warning(f"{extracted_name}没有检索到任何数据")
            continue
        # 补充判定（听书域）：按"书名部分"做包含式对齐
        # 例：抽取到"三体"，库中存在"三体-刘慈欣" → 直接确认为同名书籍的全部主体
        target_book = _book_name_part(extracted_name)
        exact_names = [
            match["item_name"] for match in matches
            if target_book and _book_name_part(match["item_name"]) == target_book
        ]
        if exact_names:
            logger.info(f"{extracted_name} 按书名精确对齐到：{exact_names}")
            confirmed_item_names.extend(exact_names)
            continue
        # 分别获取高分数（>=0.85）和中间分数（>=0.6 and < 0.85）的数据
        high = [match for match in matches if match["score"] >= 0.85]
        middle = [match for match in matches if match["score"] >= 0.6]
        # 判断high中数据的数量，若只有一条，则直接作为已确认的item_name
        if len(high) == 1:
            confirmed_item_names.append(high[0]["item_name"])
            continue
        # 判断high中数据的数量，若有多条，优先找检索的item_name和extracted_name一致的数据
        # 创建存储已确认的item_name的变量
        picked = None
        if len(high) > 1:
            # 遍历检索的数据
            for item in high:
                if item["item_name"] == extracted_name:
                    picked = item
                    continue
            # 判断picked是否为None，若为None，表示0.85以上没有数据的item_name和extracted_name一致
            # 直接将0.85以上的数据中分数最高的作为已确认的item_name
            if not picked:
                picked = high[0]
            # 保存已确认的item_name
            confirmed_item_names.append(picked["item_name"])
            continue
        # 表示没有已确认的item_name，即所检索的数据的score在0.6-0.85之间
        # 将0.6-0.85之间的数据中的前三个作为待确认的item_name
        if len(middle) > 0:
            for item in middle[:3]:
                # 保存待确认的item_name
                options.append(item["item_name"])
    return {
        "confirmed_item_names": list(set(confirmed_item_names)),
        "options": list(set(options))
    }


@step_log("step_6_check_confirmation")
def step_6_check_confirmation(state, align_result, session_id, history_list, rewritten_query, extracted_names=None):
    # 分别获取已确认和待确认的item_name的列表
    confirmed = align_result.get("confirmed_item_names", [])
    options = align_result.get("options", [])
    # 分支1：有已确认的item_name
    if confirmed:
        # 更新历史记录中item_names
        # 先获取要修改的数据的_id
        ids = []
        # 遍历历史记录
        for history in history_list:
            # 若历史记录的item_names为空，进行更新，就需要记录历史记录的_id
            if not history.get("item_names"):
                ids.append(history.get("_id"))
        # 判断ids是否为空，若不为空则修改历史记录的item_names
        if ids:
            update_message_item_names(ids, confirmed)
        # 更新状态
        state["item_names"] = confirmed
        state["rewritten_query"] = rewritten_query
        # 判断state中是否有answer，若有则删除
        if state.get("answer"):
            del state["answer"]
        return state
    # 分支2/3：本地知识库没有精确命中（有 0.6~0.85 的近似候选 options，或连候选都没有）
    #
    # 【重要】这里**不写 answer**，也不做反问澄清。原因：
    #   1. 反问会让"库外书籍"永远卡在澄清循环里——例如问《斗罗大陆》，库里只有不相干的
    #      近似项，就会一直重复"您想查询以下哪本书：XXX？"；
    #   2. 「本地库没有」不等于「这本书不存在」，此时正该让书籍查询 MCP 去查库外书籍，
    #      最终由 node_rerank 统一打分判定，而不是在入口就把流程掐断。
    # 只要往 state 写 answer，main_graph.condition_fun 就会直接收尾，三路检索全都跑不到。
    if options:
        logger.info(
            f"库内无精确匹配，近似候选={options}，不做反问，改由三路检索 + 重排判定"
        )
    else:
        logger.info("库内无任何候选，不做反问，改由三路检索 + 重排判定")
    # 保留抽取到的书名作为本地过滤条件：
    #   ① 库里没有这本书时，Milvus 过滤结果自然为空（不报错），不会把噪声带进重排；
    #   ② 书籍查询 MCP 能从 item_names 拿到"干净的书名"，而不是整句问话——
    #      实测整句「我要查询斗罗大陆」会让工具命中《斗破苍穹》，而「斗罗大陆」精准命中。
    state["item_names"] = [n for n in (extracted_names or []) if n]
    # 改写后的问题：MCP 优先用书名，取不到书名时才退化到它
    state["rewritten_query"] = rewritten_query or state.get("original_query") or ""
    # 清掉可能残留的 answer，避免被短路
    if state.get("answer"):
        del state["answer"]
    return state


@step_log("step_7_write_history")
def step_7_write_history(state, session_id, history_list, rewritten_query, message_id):
    # 判断状态中answer，若有值则保存历史记录，若没有值，更新历史记录
    if state.get("answer"):
        save_chat_message(session_id, "assistant", state["answer"], "", [])
    # 更新历史记录
    save_chat_message(
        session_id=session_id,  # 会话ID，关联所属会话
        role="user",  # 消息角色：用户
        text=state["original_query"],  # 消息内容：用户原始查询
        rewritten_query=rewritten_query,  # 补充step3改写后的完整问题
        item_names=state.get("item_names", []),  # 补充关联的书籍主体列表
        message_id=message_id,  # 消息ID，指定更新已存在的用户消息（而非新增）
        audio_url=state.get("audio_url", "")  # 语音音频URL（update 为 $set 覆盖，必须一并带上）
    )
    return state


@node_log("node_item_name_confirm")
def node_item_name_confirm(state : QueryGraphState):
    """
    节点: 书籍主体确认 (node_item_name_confirm)
    实现内容:
    1. 读取会话历史 → 2. 保存当轮用户消息 → 3. LLM 提取书籍主体并改写问题
    → 4. 在书籍主体集合中向量对齐 → 5. 置信度分级（确认 / 候选 / 兜底）
    → 6. 回填历史 → 7. 写入最终历史
    """
    # 记录当前任务的状态为进行中
    add_running_task(state["session_id"], "node_item_name_confirm", state["is_stream"])

    # 分别获取session_id,original_query,is_stream
    session_id = state["session_id"]
    original_query = state["original_query"]
    is_stream = state["is_stream"]

    # 步骤1: 获取历史记录
    history_list = get_recent_messages(session_id)
    # 保存历史记录到状态中
    state["history"] = history_list
    # 步骤2：将当前用户的问题保存到MongoDB中，返回的message_id是添加的数据的唯一标识
    # audio_url：语音提问时带音频地址（供刷新后回放），文本提问为空串
    message_id = save_chat_message(session_id, "user", original_query, "", [], audio_url=state.get("audio_url", ""))
    # 步骤3: 从用户的问题中提取item_names并重写用户问题
    extract_result = step_3_extract_info(original_query,history_list)
    # 分别获取提取的item_names和重写之后的问题rewritten_query
    item_names = extract_result.get("item_names")
    rewritten_query = extract_result.get("rewritten_query")
    is_chitchat = extract_result.get("is_chitchat", False)

    # 闲聊分流（2026-09-15，方案 C）：
    # 本轮被判定为「与书籍无关的寒暄/闲聊」且没抽到任何书名时，不检索、不调 MCP，
    # 直接交由 node_answer_output 用闲聊提示词生成回答。
    # 这里只打标记、**不写 answer**——若在这里生成 answer，会被 step_7 写一次 assistant，
    # 到 answer_output 又写一次，造成历史重复。
    if is_chitchat and not item_names:
        logger.info(f"识别为闲聊/寒暄，跳过检索，交由答案节点闲聊回答：{original_query!r}")
        state["is_chitchat"] = True
        state["item_names"] = []
        state["rewritten_query"] = original_query
        state = step_7_write_history(state, session_id, history_list, rewritten_query, message_id)
        add_done_task(state["session_id"], "node_item_name_confirm", state["is_stream"])
        return state

    # 更新状态中的rewritten_query
    state["rewritten_query"] = rewritten_query

    # 创建存储对齐之后结果的字典
    align_result = {}
    if len(item_names) > 0:
        # 步骤4：通过item_names在向量数据库中进行检索
        query_results = step_4_vectorize_and_query(item_names)
        # 步骤5：获取对齐的结果
        align_result = step_5_align_item_names(query_results)
    else:
        logger.info("Node: 未提取到书籍名，跳过向量检索")

    # 步骤6：检查确认状态
    state = step_6_check_confirmation(state, align_result, session_id, history_list, rewritten_query, item_names)
    # 步骤7：写入最终历史
    final_state = step_7_write_history(state, session_id, history_list, rewritten_query, message_id)

    # 记录当前任务的状态为已完成
    add_done_task(state["session_id"], "node_item_name_confirm", state["is_stream"])
    return final_state



if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "《三体》适合谁听？",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(result_state)

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认书籍主体: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到书籍主体 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")

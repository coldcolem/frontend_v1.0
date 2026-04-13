"""
问答对话页面 - 支持带引文来源的 RAG 智能问答
"""

import streamlit as st
from typing import Optional
from html import escape

st.set_page_config(page_title="智能问答 | RAG Search", page_icon="·", layout="wide")

# ══════════════════════════════════════════════════════════════════════
# 全局样式
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* 顶部标题栏 */
.top-bar {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.top-bar h2 {
    font-size: 1.4rem;
    font-weight: 600;
    margin: 0;
    color: inherit;
}
.top-bar span {
    font-size: 0.85rem;
    color: inherit;
    opacity: 0.6;
}

/* 用户消息 */
.msg-user-wrap {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1rem;
}
.msg-user {
    border-radius: 14px 14px 3px 14px;
    padding: 0.7rem 1rem;
    max-width: 68%;
    font-size: 0.93rem;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    background: var(--streamlit-dark-primary-color, #2d3748);
    color: var(--streamlit-dark-font-color, #f7fafc);
}

/* AI 消息 */
.msg-ai-wrap { margin-bottom: 0.5rem; }
.msg-ai {
    border-radius: 3px 14px 14px 14px;
    padding: 0.85rem 1.1rem;
    max-width: 80%;
    font-size: 0.93rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
    background: var(--streamlit-background-color, #ffffff);
    border: 1px solid var(--streamlit-border-color, #e2e8f0);
    color: inherit;
}

/* 角色标签 */
.role-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    color: inherit;
    opacity: 0.6;
}
.role-label-user { text-align: right; }

/* 引文区域 - 紧凑折叠 */
.citations-wrap {
    max-width: 80%;
    margin-bottom: 0.8rem;
}
.citation-summary {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex-wrap: wrap;
    font-size: 0.75rem;
}
.citation-tag {
    padding: 0.12rem 0.5rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 500;
    background: var(--streamlit-secondary-background-color, #edf2f7);
    color: inherit;
}

/* 单条引文卡片 */
.citation-card {
    border: 1px solid var(--streamlit-border-color, #e5e7eb);
    border-left: 2px solid #d97706;
    border-radius: 4px 6px 6px 4px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.35rem;
    font-size: 0.78rem;
    background: var(--streamlit-secondary-background-color, #fafaf9);
    color: inherit;
}
.citation-details {
    margin-top: 0.15rem;
}
.citation-details summary {
    list-style: none;
    cursor: pointer;
    font-size: 0.72rem;
    font-weight: 500;
    color: #975a16;
}
.citation-details summary::-webkit-details-marker { display: none; }
.citation-preview {
    line-height: 1.5;
    font-size: 0.78rem;
    color: inherit;
    opacity: 0.8;
}

/* 空状态 */
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
    color: inherit;
    opacity: 0.5;
}
.empty-state h3 { font-size: 1.05rem; font-weight: 500; margin-bottom: 0.4rem; color: inherit; }
.empty-state p  { font-size: 0.85rem; color: inherit; }

/* 侧边栏 */
.sb-status { font-size: 0.83rem; line-height: 1.9; color: inherit; }
.sb-dot-on  { color: #48bb78; }
.sb-dot-off { color: #fc8181; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════
def _score_info(score: float):
    """根据 FAISS L2 距离返回颜色和标签。距离越小越相关。"""
    if score < 0.3:
        return "#48bb78", "高度相关"
    elif score < 0.7:
        return "#ed8936", "中等相关"
    else:
        return "#fc8181", "弱相关"


def _clean_text(text: str, max_len: int = 500) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) > max_len:
        return cleaned[:max_len].rstrip() + "..."
    return cleaned


def render_citations(citations: list):
    if not citations:
        return
    sorted_citations = sorted(
        citations, key=lambda x: x.score if x.score is not None else float("inf")
    )

    # 紧凑的来源摘要行
    source_tags = " ".join(
        f'<span class="citation-tag">{c.source}</span>' for c in sorted_citations
    )
    st.markdown(
        f"""
    <div class="citation-summary">
        <span style="color:#9ca3af;font-size:0.72rem">参考来源 {len(sorted_citations)} 条</span>
        {source_tags}
    </div>""",
        unsafe_allow_html=True,
    )

    # 折叠详情面板
    with st.expander("查看引文详情", expanded=False):
        for index, c in enumerate(sorted_citations, start=1):
            color, label = (
                _score_info(c.score) if c.score is not None else ("#a0aec0", "未知")
            )
            page_text = f"第 {c.page} 页" if c.page else "页码未知"

            with st.container(border=True):
                col_left, col_right = st.columns([8, 2])
                with col_left:
                    st.markdown(f"**{index}. {c.source}**")
                with col_right:
                    st.markdown(
                        f'<span style="font-size:0.72rem;color:{color}">{label}</span>',
                        unsafe_allow_html=True,
                    )
                st.caption(page_text)

            with st.expander(f"内容预览"):
                st.markdown(_clean_text(c.content, 500), unsafe_allow_html=False)


def render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]
    citations = msg.get("citations", [])

    content_safe = escape(content)

    if role == "user":
        st.markdown(
            f"""
        <div class="role-label role-label-user">You</div>
        <div class="msg-user-wrap"><div class="msg-user">{content_safe}</div></div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="role-label">Assistant</div>
        <div class="msg-ai-wrap"><div class="msg-ai">{content_safe}</div></div>
        """,
            unsafe_allow_html=True,
        )
        if citations:
            render_citations(citations)


# ══════════════════════════════════════════════════════════════════════
# 会话状态 - 多对话管理
# ══════════════════════════════════════════════════════════════════════
import time

def _generate_conversation_id():
    return str(int(time.time() * 1000))

def _get_or_create_conversation():
    """获取或创建当前对话"""
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    
    if "current_conversation_id" not in st.session_state:
        # 创建第一个对话
        conv_id = _generate_conversation_id()
        st.session_state.conversations[conv_id] = {
            "id": conv_id,
            "title": "新对话",
            "messages": [],
            "created_at": time.time()
        }
        st.session_state.current_conversation_id = conv_id
    
    return st.session_state.current_conversation_id

# 初始化当前对话
current_conv_id = _get_or_create_conversation()
current_conversation = st.session_state.conversations.get(current_conv_id)
chat_messages = current_conversation["messages"] if current_conversation else []

# 控制输入框是否可用
if "input_disabled" not in st.session_state:
    st.session_state.input_disabled = False

# 控制停止标志
if "cancel_requested" not in st.session_state:
    st.session_state.cancel_requested = False

# ══════════════════════════════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("**RAG Search**")
    st.caption("检索增强生成问答系统")
    st.divider()

    backend = st.session_state.get("backend")
    is_ready = backend is not None and getattr(backend, "llm", None) is not None
    has_index = is_ready and getattr(backend, "vector_store", None) is not None
    chunk_cnt = getattr(backend, "_chunk_count", 0) if is_ready else 0

    on = '<span class="sb-dot-on">●</span>'
    off = '<span class="sb-dot-off">●</span>'
    st.markdown(
        f"""
    <div class="sb-status">
        {on if is_ready else off}&nbsp; 后端服务<br>
        {on if has_index else off}&nbsp; 知识库&nbsp;
        {"<span style='color:#a0aec0;font-size:0.75rem'>(" + str(chunk_cnt) + " chunks)</span>" if has_index else ""}
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    # 对话管理区域
    st.markdown("**对话列表**")
    
    # 新建对话按钮
    if st.button("+ 新建对话", use_container_width=True):
        conv_id = _generate_conversation_id()
        st.session_state.conversations[conv_id] = {
            "id": conv_id,
            "title": "新对话",
            "messages": [],
            "created_at": time.time()
        }
        st.session_state.current_conversation_id = conv_id
        st.session_state.input_disabled = False
        st.session_state.cancel_requested = False
        st.rerun()
    
    # 对话列表
    conversations_list = list(st.session_state.conversations.values())
    conversations_list.sort(key=lambda x: x["created_at"], reverse=True)
    
    # 显示对话列表
    for conv in conversations_list:
        conv_id = conv["id"]
        is_current = conv_id == current_conv_id
        msg_count = len(conv["messages"])
        preview = conv["messages"][0]["content"][:15] + "..." if msg_count > 0 else "新对话"
        
        # 使用3:1比例，适配手机端
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(
                f"{'● ' if is_current else ''}{preview}",
                key=f"conv_{conv_id}",
                use_container_width=True,
            ):
                st.session_state.current_conversation_id = conv_id
                st.session_state.input_disabled = False
                st.session_state.cancel_requested = False
                st.rerun()
        with col2:
            if st.button("删除", key=f"del_{conv_id}", help="删除对话", use_container_width=True):
                del st.session_state.conversations[conv_id]
                if st.session_state.current_conversation_id == conv_id:
                    remaining = list(st.session_state.conversations.keys())
                    if remaining:
                        # 切换到第一个剩余对话
                        st.session_state.current_conversation_id = remaining[0]
                    else:
                        # 没有剩余对话，创建一个新的
                        new_conv_id = _generate_conversation_id()
                        st.session_state.conversations[new_conv_id] = {
                            "id": new_conv_id,
                            "title": "新对话",
                            "messages": [],
                            "created_at": time.time()
                        }
                        st.session_state.current_conversation_id = new_conv_id
                st.rerun()
    
    # 当前对话信息
    rounds = len(chat_messages) // 2
    if rounds:
        st.caption(f"当前对话：{rounds} 轮对话")

    st.divider()

# ══════════════════════════════════════════════════════════════════════
# 主区域
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="top-bar">
    <h2>智能问答</h2>
    <span>基于文档的 RAG 检索问答</span>
</div>
""",
    unsafe_allow_html=True,
)

# 消息列表
if not chat_messages:
    st.markdown(
        """
    <div class="empty-state">
        <h3>开始提问</h3>
        <p>在下方输入框输入问题，系统将从知识库中检索相关内容并给出回答，<br>每条回答附有可溯源的引文来源。</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
else:
    for msg in chat_messages:
        render_message(msg)

# 提示信息
if not is_ready:
    st.warning("服务尚未初始化")
    st.caption(
        "请先前往 **Settings** 页面配置 LLM 与 Embedding 服务，并点击「初始化后端」。"
    )
    st.page_link(
        "pages/0_Settings.py",
        label="前往 Settings",
        icon=":material/settings:",
        use_container_width=True,
    )
elif not has_index:
    st.warning("知识库为空")
    st.caption("请先前往 **Knowledge Base** 上传文档，构建向量索引后再进行问答。")
    st.page_link(
        "pages/1_Knowledge_Base.py",
        label="前往 Knowledge Base",
        icon=":material/folder:",
        use_container_width=True,
    )
else:
    # 停止按钮（仅在处理中显示）
    if st.session_state.input_disabled:
        col_btn, col_txt = st.columns([1, 4])
        with col_btn:
            if st.button("停止生成", type="secondary"):
                st.session_state.cancel_requested = True
                st.rerun()
        with col_txt:
            st.caption("正在生成回复中...")

    # 使用 disabled 参数控制输入框
    user_input = st.chat_input(
        "输入问题..." if not st.session_state.input_disabled else "等待回复中...",
        key="chat_input",
        disabled=st.session_state.input_disabled
    )

    if user_input and user_input.strip():
        query = user_input.strip()
        # 保存查询并禁用输入框
        st.session_state.pending_query = query
        st.session_state.input_disabled = True
        # 切换到当前对话（防止切换对话后消息发到错误的对话）
        st.session_state.current_conversation_id = current_conv_id
        st.rerun()

# 处理待处理的查询（回复生成）
if st.session_state.input_disabled and "pending_query" in st.session_state:
    # 检查是否取消了请求
    if st.session_state.cancel_requested:
        st.session_state.input_disabled = False
        st.session_state.cancel_requested = False
        st.session_state.pop("pending_query", None)
        st.rerun()
    
    query = st.session_state.pending_query
    current_conv_id = st.session_state.current_conversation_id
    conversation = st.session_state.conversations[current_conv_id]

    # 添加用户消息到当前对话
    conversation["messages"].append(
        {"role": "user", "content": query, "citations": []}
    )

    # 更新对话标题（如果是对话的第一条消息）
    if len(conversation["messages"]) == 1:
        title = query[:30] + "..." if len(query) > 30 else query
        conversation["title"] = title

    history_for_backend = [
        {"role": m["role"], "content": m["content"]}
        for m in conversation["messages"][:-1]
    ]

    # 使用placeholder显示动态状态
    status_placeholder = st.empty()
    
    with st.spinner("检索中..."):
        try:
            # 模拟检查取消状态（在实际AI调用前）
            if st.session_state.cancel_requested:
                status_placeholder.warning("请求已取消")
                # 移除刚添加的用户消息
                conversation["messages"].pop()
                st.session_state.input_disabled = False
                st.session_state.cancel_requested = False
                st.session_state.pop("pending_query", None)
                st.rerun()
            
            result = backend.chat(query, history_for_backend)
            
            # 检查AI调用后是否取消了
            if st.session_state.cancel_requested:
                status_placeholder.warning("请求已取消（回复已被忽略）")
                conversation["messages"].pop()  # 移除用户消息
                conversation["messages"].pop()  # 移除之前的助手回复（如果有）
                st.session_state.input_disabled = False
                st.session_state.cancel_requested = False
                st.session_state.pop("pending_query", None)
                st.rerun()
            
            answer = result["answer"]
            citations = result["citations"]
        except Exception as e:
            answer = f"发生错误：{e}"
            citations = []

    # 添加助手回复
    conversation["messages"].append(
        {"role": "assistant", "content": answer, "citations": citations}
    )

    # 重新启用输入框
    st.session_state.input_disabled = False
    st.session_state.cancel_requested = False
    st.session_state.pop("pending_query", None)
    st.rerun()

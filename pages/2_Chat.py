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
    color: #1a202c;
    margin: 0;
}
.top-bar span {
    font-size: 0.85rem;
    color: #a0aec0;
}

/* 用户消息 */
.msg-user-wrap {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1rem;
}
.msg-user {
    background: #2d3748;
    color: #f7fafc;
    border-radius: 14px 14px 3px 14px;
    padding: 0.7rem 1rem;
    max-width: 68%;
    font-size: 0.93rem;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
}

/* AI 消息 */
.msg-ai-wrap { margin-bottom: 0.5rem; }
.msg-ai {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 3px 14px 14px 14px;
    padding: 0.85rem 1.1rem;
    max-width: 80%;
    font-size: 0.93rem;
    line-height: 1.7;
    color: #2d3748;
    white-space: pre-wrap;
    word-break: break-word;
}

/* 角色标签 */
.role-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #a0aec0;
    margin-bottom: 0.3rem;
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
    background: #edf2f7;
    color: #4a5568;
    padding: 0.12rem 0.5rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 500;
}

/* 单条引文卡片 */
.citation-card {
    background: #fafaf9;
    border: 1px solid #e5e7eb;
    border-left: 2px solid #d97706;
    border-radius: 4px 6px 6px 4px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.35rem;
    font-size: 0.78rem;
}
.citation-details {
    margin-top: 0.15rem;
}
.citation-details summary {
    list-style: none;
    cursor: pointer;
    color: #975a16;
    font-size: 0.72rem;
    font-weight: 500;
}
.citation-details summary::-webkit-details-marker { display: none; }
.citation-preview {
    color: #4a5568;
    line-height: 1.5;
    font-size: 0.78rem;
}

/* 空状态 */
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
    color: #cbd5e0;
}
.empty-state h3 { color: #a0aec0; font-size: 1.05rem; font-weight: 500; margin-bottom: 0.4rem; }
.empty-state p  { font-size: 0.85rem; }

/* 侧边栏 */
.sb-status { font-size: 0.83rem; color: #4a5568; line-height: 1.9; }
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
# 会话状态
# ══════════════════════════════════════════════════════════════════════
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

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

    if st.button("清空对话", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

    rounds = len(st.session_state.chat_messages) // 2
    if rounds:
        st.caption(f"{rounds} 轮对话")

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
if not st.session_state.chat_messages:
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
    for msg in st.session_state.chat_messages:
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
    user_input = st.chat_input("输入问题...", key="chat_input")

    if user_input and user_input.strip():
        query = user_input.strip()

        st.session_state.chat_messages.append(
            {"role": "user", "content": query, "citations": []}
        )

        history_for_backend = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages[:-1]
        ]

        with st.spinner("检索中..."):
            try:
                result = backend.chat(query, history_for_backend)
                answer = result["answer"]
                citations = result["citations"]
            except Exception as e:
                answer = f"发生错误：{e}"
                citations = []

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer, "citations": citations}
        )

        st.rerun()

import streamlit as st
from core.rag_backend import LangChainRAGBackend

st.set_page_config(page_title="RAG Search | Home", page_icon="·", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.page-title { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.2rem; color: inherit; }
.page-subtitle { font-size: 0.88rem; margin-bottom: 1.5rem; color: inherit; opacity: 0.7; }
.status-card { padding: 1rem 1.2rem; border-radius: 0.5rem; margin-bottom: 0.8rem; }
.status-ok { border-left: 3px solid #22c55e; background: rgba(34, 197, 94, 0.1); }
.status-warn { border-left: 3px solid #eab308; background: rgba(234, 179, 8, 0.1); }
.status-error { border-left: 3px solid #ef4444; background: rgba(239, 68, 68, 0.1); }
.stat-value { font-size: 1.6rem; font-weight: 600; color: inherit; }
.stat-label { font-size: 0.8rem; color: inherit; opacity: 0.6; }
.status-text-ok { color: #22c55e; font-weight: 500; }
.status-text-warn { color: #eab308; font-weight: 500; }
.status-text-error { color: #ef4444; font-weight: 500; }
.status-hint { font-size: 0.85rem; margin-left: 0.5rem; opacity: 0.7; }
</style>
""", unsafe_allow_html=True)

# ── 持久化 Backend 实例 ─────────────────────────────────────────────
if "backend" not in st.session_state:
    st.session_state.backend = LangChainRAGBackend()

backend = st.session_state.backend

# ── 页面标题 ────────────────────────────────────────────────────────
st.markdown('<div class="page-title">RAG Search</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">基于检索增强生成（RAG）的智能文档问答系统</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 状态面板
# ══════════════════════════════════════════════════════════════════
is_initialized = backend.llm is not None and backend.embeddings is not None
has_docs = backend.vector_store is not None

if is_initialized and has_docs:
    st.markdown("""<div class="status-card status-ok">
    <span class="status-text-ok">● 系统就绪</span>
    <span class="status-hint">可前往 Chat 开始问答</span></div>""", unsafe_allow_html=True)
elif is_initialized:
    st.markdown("""<div class="status-card status-warn">
    <span class="status-text-warn">● 已初始化</span>
    <span class="status-hint">请前往 Knowledge Base 上传文档</span></div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="status-card status-error">
    <span class="status-text-error">● 未初始化</span>
    <span class="status-hint">请前往 Settings 配置服务并初始化后端</span></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 统计信息
# ══════════════════════════════════════════════════════════════════
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Chunk 数量**")
    st.markdown(f'<div class="stat-value">{backend._chunk_count}</div><div class="stat-label">已入库文档块</div>',
                unsafe_allow_html=True)

with col2:
    files = backend.get_knowledge_files()
    st.markdown("**知识库文件**")
    st.markdown(f'<div class="stat-value">{len(files)}</div><div class="stat-label">已上传文档</div>',
                unsafe_allow_html=True)

with col3:
    status_text = "就绪" if has_docs else "空库"
    st.markdown("**索引状态**")
    st.markdown(f'<div class="stat-value" style="font-size:1.2rem">{status_text}</div><div class="stat-label">向量索引</div>',
                unsafe_allow_html=True)

st.divider()

# 快捷导航
st.markdown("**快速导航**")
ncol1, ncol2, ncol3 = st.columns(3)

with ncol1:
    st.page_link("pages/0_Settings.py", label="Settings", icon=":material/settings:", use_container_width=True)

with ncol2:
    st.page_link("pages/1_Knowledge_Base.py", label="Knowledge Base", icon=":material/folder:", use_container_width=True)

with ncol3:
    st.page_link("pages/2_Chat.py", label="Chat", icon=":material/chat:", use_container_width=True)

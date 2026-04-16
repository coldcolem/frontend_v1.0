import streamlit as st
import threading
from typing import Any, Optional
from core.interfaces import LLMConfig, EmbedConfig
from core.rag_backend import LangChainRAGBackend

# 超时时间（秒）
PING_TIMEOUT = 10


def _ping_with_timeout(backend, config, ping_type):
    """带超时的 ping 函数，返回 None 表示超时"""
    result: dict[str, Optional[bool | str]] = {"ok": None, "error": None}
    finished = threading.Event()

    def _target():
        try:
            if ping_type == "llm":
                result["ok"] = backend.ping_llm(config)
            else:
                result["ok"] = backend.ping_embedding(config)
        except Exception as e:
            result["error"] = str(e)
        finally:
            finished.set()

    t = threading.Thread(target=_target)
    t.daemon = True
    t.start()
    if not finished.wait(timeout=PING_TIMEOUT):
        # 超时，终止线程
        return None
    if result["error"]:
        raise Exception(result["error"])
    return result["ok"]


st.set_page_config(page_title="RAG Search | 服务配置", page_icon="·", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.page-title { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.2rem; color: inherit; }
.page-subtitle { font-size: 0.88rem; margin-bottom: 1.2rem; color: inherit; opacity: 0.7; }
.field-label { font-size: 0.8rem; font-weight: 500; margin-bottom: 0.2rem; color: inherit; }
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── 预设配置 ──────────────────────────────────────────────────────────
LLM_PRESETS = {
    "自定义": {"base_url": "", "model": ""},
    "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
    "DeepSeek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "硅基流动": {"base_url": "https://api.siliconflow.cn/v1", "model": "Qwen/Qwen2.5-7B-Instruct"},
    "阿里云百炼": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
    "Groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.1-70b-versatile"},
    "Together AI": {"base_url": "https://api.together.ai/v1", "model": "meta-llama/Llama-3-70b-chat-hf"},
    "Anthropic": {"base_url": "https://api.anthropic.com/v1", "model": "claude-3-5-sonnet-20241022"},
}

EMBEDDING_PRESETS = {
    "自定义": {"base_url": "", "model": ""},
    "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "text-embedding-3-small"},
    "智谱 AI": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "embedding-3"},
    "阿里云百炼": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "text-embedding-v3"},
    "MiniMax": {"base_url": "https://api.minimax.chat/v1", "model": "embo-01"},
    "Cohere": {"base_url": "https://api.cohere.ai/v1", "model": "embed-english-v3.0"},
}

# ── 持久化 Backend 实例 ─────────────────────────────────────────────
if "backend" not in st.session_state:
    st.session_state.backend = LangChainRAGBackend()

# ── 持久化配置字段（切换页面后保留） ──────────────────────────────────
_cfg_defaults = {
    "cfg_llm_base_url":  "",
    "cfg_llm_api_key":   "",
    "cfg_llm_model":     "",
    "cfg_emb_base_url":  "",
    "cfg_emb_api_key":   "",
    "cfg_emb_model":     "",
}
for _k, _v in _cfg_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── 页面标题 ────────────────────────────────────────────────────────
st.markdown('<div class="page-title">服务配置</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">选择预设配置或手动填写，完成后初始化后端。</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

# ══════════════════════════════════════════════════════════════════
# LLM 配置列
# ══════════════════════════════════════════════════════════════════
with col1:
    st.markdown("**LLM 配置**")
    st.caption("支持 OpenAI 兼容协议（OpenAI、DeepSeek、硅基流动、阿里云百炼等）")

    # 预设选择器
    llm_preset = st.selectbox(
        "选择预设",
        options=list(LLM_PRESETS.keys()),
        index=list(LLM_PRESETS.keys()).index(st.session_state.get("_cfg_llm_preset", "自定义")),
        key="_w_llm_preset"
    )

    # 获取预设值
    preset_llm = LLM_PRESETS[llm_preset]

    # 当选择预设时，更新 session_state 以便 text_input 能显示
    if llm_preset != "自定义":
        st.session_state["_w_llm_base_url"] = preset_llm["base_url"]
        st.session_state["_w_llm_model"] = preset_llm["model"]

    st.session_state["_cfg_llm_preset"] = llm_preset

    # Base URL - 下拉预填但可编辑
    llm_base_url = st.text_input(
        "Base URL（留空使用 OpenAI 官方地址）",
        placeholder="https://api.example.com/v1",
        key="_w_llm_base_url"
    )

    # API Key - 始终需要手动填写
    llm_api_key = st.text_input(
        "API Key",
        value=st.session_state.cfg_llm_api_key,
        type="password",
        placeholder="sk-...",
        key="_w_llm_api_key"
    )

    # Model - 下拉预填但可编辑
    llm_model = st.text_input(
        "Model Name",
        placeholder="gpt-3.5-turbo",
        key="_w_llm_model"
    )

    # 同步到持久化 key（仅"自定义"模式或用户手动编辑时保存）
    st.session_state.cfg_llm_base_url = llm_base_url
    st.session_state.cfg_llm_api_key  = llm_api_key
    st.session_state.cfg_llm_model    = llm_model

    if st.button("测试 LLM 连通性", use_container_width=True):
        with st.spinner(f"连接中（超时{PING_TIMEOUT}秒）..."):
            try:
                ok = _ping_with_timeout(
                    st.session_state.backend,
                    LLMConfig(base_url=llm_base_url or None, api_key=llm_api_key, model_name=llm_model),
                    "llm"
                )
                if ok is None:
                    st.error(f"LLM 连接超时（{PING_TIMEOUT}秒），请检查网络或地址")
                elif ok:
                    st.success("LLM 连通正常")
                else:
                    st.error("LLM 连接失败，请检查配置")
            except Exception as e:
                st.error(f"LLM 连接失败：{e}")

# ══════════════════════════════════════════════════════════════════
# Embedding 配置列
# ══════════════════════════════════════════════════════════════════
with col2:
    st.markdown("**Embedding 配置**")
    st.caption("支持 OpenAI 兼容协议（OpenAI、智谱、阿里云百炼等）")

    # 预设选择器
    emb_preset = st.selectbox(
        "选择预设",
        options=list(EMBEDDING_PRESETS.keys()),
        index=list(EMBEDDING_PRESETS.keys()).index(st.session_state.get("_cfg_emb_preset", "自定义")),
        key="_w_emb_preset"
    )

    # 获取预设值
    preset_emb = EMBEDDING_PRESETS[emb_preset]

    # 当选择预设时，更新 session_state 以便 text_input 能显示
    if emb_preset != "自定义":
        st.session_state["_w_emb_base_url"] = preset_emb["base_url"]
        st.session_state["_w_emb_model"] = preset_emb["model"]

    st.session_state["_cfg_emb_preset"] = emb_preset

    # Base URL - 下拉预填但可编辑
    emb_base_url = st.text_input(
        "Base URL（留空使用 OpenAI 官方地址）",
        placeholder="https://api.example.com/v1",
        key="_w_emb_base_url"
    )

    # API Key - 始终需要手动填写
    emb_api_key = st.text_input(
        "API Key",
        value=st.session_state.cfg_emb_api_key,
        type="password",
        placeholder="sk-...",
        key="_w_emb_api_key"
    )

    # Model - 下拉预填但可编辑
    emb_model = st.text_input(
        "Model Name",
        placeholder="text-embedding-ada-002",
        key="_w_emb_model"
    )

    # 同步到持久化 key
    st.session_state.cfg_emb_base_url = emb_base_url
    st.session_state.cfg_emb_api_key  = emb_api_key
    st.session_state.cfg_emb_model    = emb_model

    if st.button("测试 Embedding 连通性", use_container_width=True):
        with st.spinner(f"连接中（超时{PING_TIMEOUT}秒）..."):
            try:
                ok = _ping_with_timeout(
                    st.session_state.backend,
                    EmbedConfig(base_url=emb_base_url or None, api_key=emb_api_key, model_name=emb_model),
                    "embedding"
                )
                if ok is None:
                    st.error(f"Embedding 连接超时（{PING_TIMEOUT}秒），请检查网络或地址")
                elif ok:
                    st.success("Embedding 连通正常")
                else:
                    st.error("Embedding 连接失败，请检查配置")
            except Exception as e:
                st.error(f"Embedding 连接失败：{e}")

st.divider()

# ══════════════════════════════════════════════════════════════════
# 初始化 Backend
# ══════════════════════════════════════════════════════════════════
backend = st.session_state.backend

# 初始化按钮
if st.button("初始化后端", type="primary", use_container_width=True):
    llm_config = LLMConfig(
        base_url=llm_base_url or None,
        api_key=llm_api_key,
        model_name=llm_model
    )
    emb_config = EmbedConfig(
        base_url=emb_base_url or None,
        api_key=emb_api_key,
        model_name=emb_model
    )

    # 分别测试连通性（带超时）
    with st.spinner(f"正在验证 LLM 连接（超时{PING_TIMEOUT}秒）..."):
        try:
            llm_ok = _ping_with_timeout(backend, llm_config, "llm")
        except Exception as e:
            llm_ok = False

    if llm_ok is None:
        st.error(f"LLM 连接超时（{PING_TIMEOUT}秒），请检查网络或地址")
    elif not llm_ok:
        st.error("LLM 连接失败，请检查 API Key、Base URL 和模型名称")
    else:
        with st.spinner(f"正在验证 Embedding 连接（超时{PING_TIMEOUT}秒）..."):
            try:
                emb_ok = _ping_with_timeout(backend, emb_config, "embedding")
            except Exception as e:
                emb_ok = False

        if emb_ok is None:
            st.error(f"Embedding 连接超时（{PING_TIMEOUT}秒），请检查网络或地址")
        elif not emb_ok:
            st.error("Embedding 连接失败，请检查 API Key、Base URL 和模型名称")
        else:
            # 两个都通过才执行初始化
            with st.spinner("正在初始化后端..."):
                ok = backend.initialize(llm_config, emb_config)
            if ok:
                st.success("初始化成功")
                st.rerun()
            else:
                st.error("初始化失败，请检查配置后重试。")

# 显示当前状态
is_initialized = backend.llm is not None and backend.embeddings is not None
if is_initialized:
    st.info("当前状态：已初始化")
else:
    st.warning("当前状态：未初始化")

st.divider()

st.markdown("**知识库状态**")
st.caption(f"当前已入库 {backend._chunk_count} 个 Chunk，索引状态：{'就绪' if backend.vector_store else '空库'}")

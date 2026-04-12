"""
RAG 智能文档问答系统 - Web 全栈前端
角色 B：负责 UI 开发、状态管理、前后打通、部署运维

依赖安装：pip install streamlit requests
运行方式：streamlit run app.py
"""

import streamlit as st
import time
import os
import sys
import tempfile
from datetime import datetime
import requests
from typing import List, Dict, Any, Optional

# ========================================================================
# 🔌 A 同学接口导入
# ========================================================================
try:
    from core.interfaces import IRAGBackend, LLMConfig, EmbedConfig, Citation
    from core.rag_backend import LangChainRAGBackend
    INTERFACE_AVAILABLE = True
except ImportError as e:
    INTERFACE_AVAILABLE = False
    st.warning(f"⚠️ 未找到后端模块，将使用 Mock 模式: {e}")

# ========================================================================
# 🧩 预设模型配置
# ========================================================================
LLM_PRESETS = {
    "阿里云百炼 - Qwen Plus": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "阿里云百炼 - Qwen Turbo": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-turbo",
    },
    "阿里云百炼 - Qwen Max": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max",
    },
    "OpenAI - GPT-4o": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    "OpenAI - GPT-4o Mini": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "OpenAI - GPT-4": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4",
    },
    "OpenAI - GPT-3.5 Turbo": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-3.5-turbo",
    },
    "DeepSeek - DeepSeek Chat": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "智谱 AI - GLM-4": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4",
    },
    "智谱 AI - GLM-4-Flash": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
    },
    "自定义": {
        "base_url": "",
        "model": "",
    },
}

EMBED_PRESETS = {
    "阿里云 - text-embedding-v3": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "text-embedding-v3",
    },
    "阿里云 - text-embedding-ada-002": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "text-embedding-ada-002",
    },
    "OpenAI - text-embedding-3-small": {
        "base_url": "https://api.openai.com/v1",
        "model": "text-embedding-3-small",
    },
    "OpenAI - text-embedding-3-large": {
        "base_url": "https://api.openai.com/v1",
        "model": "text-embedding-3-large",
    },
    "DeepSeek - text-embedding": {
        "base_url": "https://api.deepseek.com",
        "model": "text-embedding-2",
    },
    "智谱 AI - embedding-3": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "embedding-3",
    },
    "自定义": {
        "base_url": "",
        "model": "",
    },
}


# ========================================================================
# 🌐 真实 API 调用类（对接 A 同学后使用）
# ========================================================================
class RealRAGEngine:
    """
    真实 RAG 引擎，直接调用 A 同学的 IRAGBackend 接口
    """
    
    def __init__(
        self,
        rag_backend: IRAGBackend,
        llm_config: Optional[LLMConfig] = None,
        embed_config: Optional[EmbedConfig] = None
    ) -> None:
        """
        初始化真实 RAG 引擎
        
        Args:
            rag_backend: A 同学实现的 IRAGBackend 实例
            llm_config: LLM 配置（可选，有默认值）
            embed_config: Embedding 配置（可选，有默认值）
        """
        self.backend: IRAGBackend = rag_backend
        self.llm_config: Optional[LLMConfig] = llm_config
        self.embed_config: Optional[EmbedConfig] = embed_config
        self._initialized: bool = False
    
    def initialize(self) -> bool:
        """初始化后端（连接 LLM 和 Embedding）"""
        if not INTERFACE_AVAILABLE:
            return False
        
        if self.llm_config and self.embed_config:
            try:
                self._initialized = self.backend.initialize(
                    self.llm_config,
                    self.embed_config
                )
                return self._initialized
            except Exception as e:
                st.error(f"初始化失败: {e}")
                return False
        return False
    
    def upload_file(self, file_bytes: bytes, file_name: str) -> bool:
        """
        上传文件并解析入库
        
        Args:
            file_bytes: 文件二进制内容
            file_name: 文件名
        
        Returns:
            bool: 处理是否成功
        """
        if not INTERFACE_AVAILABLE:
            # Mock 模式回退
            time.sleep(1.5)
            return True
        
        try:
            # 1. 保存到临时文件
            with tempfile.NamedTemporaryFile(
                mode='wb',
                delete=False,
                suffix=os.path.splitext(file_name)[1]
            ) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path: str = tmp_file.name
            
            try:
                # 2. 调用 A 的 ingest_documents
                success: bool = self.backend.ingest_documents(
                    file_paths=[tmp_path],
                    source_names={tmp_path: file_name}
                )
                return success
            finally:
                # 3. 清理临时文件
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            st.error(f"文件处理失败: {e}")
            return False
    
    def ask_question(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        发送问答请求
        
        Args:
            query: 用户问题
            chat_history: 对话历史（可选）
        
        Returns:
            str: 格式化后的回答文本
        """
        if not INTERFACE_AVAILABLE:
            # Mock 模式回退
            time.sleep(0.8)
            return "Mock 模式回答"
        
        try:
            # 调用 A 的 chat 方法
            result: Dict[str, Any] = self.backend.chat(
                query=query,
                chat_history=chat_history
            )
            
            answer: str = result.get('answer', '抱歉，未能生成回答')
            citations: List[Citation] = result.get('citations', [])
            
            # 格式化带引文的回答
            return self._format_answer_with_citations(answer, citations)
            
        except Exception as e:
            return f"⚠️ 问答处理失败：{str(e)}"
    
    def _format_answer_with_citations(self, answer: str, citations: List[Citation]) -> str:
        """格式化带引文的回答"""
        formatted: str = answer
        
        if citations:
            formatted += "\n\n---\n**📚 参考来源：**\n"
            for i, cite in enumerate(citations[:3], 1):  # 最多显示3条
                source_name: str = cite.source or "未知来源"
                page_info: str = f" (第{cite.page}页)" if cite.page else ""
                formatted += f"\n{i}. **{source_name}**{page_info}\n   > {cite.content[:150]}..."
        
        return formatted
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """获取已上传文档列表"""
        if not INTERFACE_AVAILABLE:
            return []
        
        try:
            return self.backend.get_knowledge_files()
        except Exception:
            return []
    
    def delete_document(self, source_name: str) -> bool:
        """删除文档"""
        if not INTERFACE_AVAILABLE:
            return True
        
        try:
            return self.backend.delete_knowledge_files(sources=[source_name])
        except Exception:
            return False
    
    def get_ingest_stats(self) -> Dict[str, Any]:
        """获取最近入库统计"""
        if not INTERFACE_AVAILABLE:
            return {}
        
        try:
            return self.backend.get_last_ingest_stats()
        except Exception:
            return {}


# ========================================================================
# 🎨 UI 样式配置
# ========================================================================
def load_custom_css():
    """加载自定义 CSS 样式"""
    st.markdown("""
    <style>
    /* 全局背景 */
    .stApp {
        background-color: #212121 !important;
    }
    

    
    /* 主内容区 - 紧凑布局 */
    .main .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* 增大边栏切换按钮 */
    [data-testid="stSidebarCollapsedControl"] {
        padding: 12px 16px !important;
        font-size: 16px !important;
    }
    
    /* 减少元素间距 */
    .element-container {
        margin-bottom: 0.3rem !important;
    }
    
    /* 移除 stChatMessageContent 的默认样式由自定义 HTML 替代 */
    
    /* 聊天容器 */
    .stChatContainer {
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* 输入框 */
    .stTextInput > div > div > input {
        background-color: #40414F !important;
        color: #ECECEC !important;
        border: 1px solid #565869 !important;
        border-radius: 6px !important;
        font-size: 14px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #10A37F !important;
        box-shadow: 0 0 0 1px rgba(16, 163, 127, 0.3) !important;
    }
    
    /* 按钮 */
    .stButton > button {
        background-color: #10A37F !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-size: 13px !important;
        padding: 4px 10px !important;
        pointer-events: auto !important;
    }
    
    .stButton > button:hover {
        background-color: #0d8a6c !important;
    }
    
    /* 确保按钮可点击 */
    .stButton {
        pointer-events: auto !important;
    }
    
    /* 侧边栏样式 - 不添加任何可能影响开关的样式 */
    
    /* 文件上传 */
    .stFileUploader > div {
        background-color: #343541 !important;
        border: 1px dashed #565869 !important;
        border-radius: 6px !important;
        padding: 8px !important;
    }
    
    /* 隐藏默认 footer */
    footer {
        display: none !important;
    }
    
    /* 滚动条 */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #212121;
    }
    ::-webkit-scrollbar-thumb {
        background: #40414F;
        border-radius: 3px;
    }
    
    /* 分割线 */
    hr {
        border-color: #343541 !important;
        margin: 8px 0 !important;
    }
    
    /* 指标 */
    [data-testid="stMetricValue"] {
        color: #10A37F !important;
        font-size: 18px !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
    }
    
    /* 可展开区域 */
    .streamlit-expanderHeader {
        background-color: #2D2D2D !important;
        border-radius: 4px !important;
        color: #ECECEC !important;
        font-size: 12px !important;
    }
    
    /* 欢迎页文字 */
    .welcome-title {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        text-align: center;
        margin-bottom: 1rem !important;
        color: #ECECEC !important;
    }
    
    .welcome-subtitle {
        text-align: center;
        color: #9CA3AF !important;
        font-size: 14px !important;
        margin-bottom: 1.5rem !important;
    }
    
    .welcome-box {
        background-color: #2D2D2D !important;
        border-radius: 8px !important;
        padding: 16px !important;
        color: #ECECEC !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .welcome-box h4 {
        font-size: 14px !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
        color: #ECECEC !important;
    }
    
    .welcome-box ol {
        margin: 8px 0 !important;
        padding-left: 20px !important;
    }
    
    .welcome-box li {
        margin-bottom: 4px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ========================================================================
# 💾 持久化存储
# ========================================================================
import json

def get_upload_history_path():
    """获取上传历史文件路径"""
    return os.path.join(os.path.dirname(__file__), "upload_history.json")

def load_upload_history():
    """从文件加载上传历史"""
    path = get_upload_history_path()
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_upload_history(history):
    """保存上传历史到文件"""
    path = get_upload_history_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"保存历史失败: {e}")


# ========================================================================
# 🔧 状态管理初始化
# ========================================================================
def init_rag_engine() -> Any:
    """初始化 RAG 引擎（支持 Mock 或真实模式）"""
    
    # 优先使用真实引擎
    if INTERFACE_AVAILABLE:
        try:
            # 使用 Session State 中的配置
            llm_config = LLMConfig(
                base_url=st.session_state.llm_base_url or None,
                api_key=st.session_state.llm_api_key,
                model_name=st.session_state.llm_model,
                temperature=0.7
            )
            
            # Embedding 配置：为空时复用 LLM 配置
            embed_api_key = st.session_state.embed_api_key.strip() or st.session_state.llm_api_key
            embed_base_url = st.session_state.embed_base_url.strip() or st.session_state.llm_base_url
            
            embed_config = EmbedConfig(
                base_url=embed_base_url or None,
                api_key=embed_api_key,
                model_name=st.session_state.embed_model
            )
            
            # 创建并初始化 A 同学的后端
            backend = LangChainRAGBackend()
            engine = RealRAGEngine(
                rag_backend=backend,
                llm_config=llm_config,
                embed_config=embed_config
            )
            
            # 初始化连接
            if engine.initialize():
                st.session_state.api_configured = True
                return engine
            else:
                st.warning("⚠️ RAG 后端初始化失败，请检查配置")
                return None
                
        except ImportError as e:
            st.info(f"ℹ️ 未找到 RAG 后端实现: {e}")
            return None
        except Exception as e:
            st.error(f"❌ 初始化 RAG 后端出错: {e}")
            return None
    else:
        return None


def test_api_connection():
    """测试 API 连通性"""
    if not INTERFACE_AVAILABLE:
        return False, False
    
    llm_ok = False
    embed_ok = False
    
    try:
        backend = LangChainRAGBackend()
        
        llm_config = LLMConfig(
            base_url=st.session_state.llm_base_url or None,
            api_key=st.session_state.llm_api_key,
            model_name=st.session_state.llm_model,
            temperature=0.7
        )
        llm_ok = backend.ping_llm(llm_config)
        st.session_state.llm_connected = llm_ok
    except Exception:
        st.session_state.llm_connected = False
    
    try:
        backend = LangChainRAGBackend()
        
        # Embedding 配置：为空时复用 LLM 配置
        embed_api_key = st.session_state.embed_api_key.strip() or st.session_state.llm_api_key
        embed_base_url = st.session_state.embed_base_url.strip() or st.session_state.llm_base_url
        
        embed_config = EmbedConfig(
            base_url=embed_base_url or None,
            api_key=embed_api_key,
            model_name=st.session_state.embed_model
        )
        embed_ok = backend.ping_embedding(embed_config)
        st.session_state.embed_connected = embed_ok
    except Exception:
        st.session_state.embed_connected = False
    
    st.session_state.has_tested = True
    return llm_ok, embed_ok


def init_session_state():
    """统一初始化所有 Session State"""
    
    # 多对话管理
    if "conversations" not in st.session_state:
        # 创建默认对话
        import uuid
        default_id = str(uuid.uuid4())[:8]
        st.session_state.conversations = {
            default_id: {
                "title": "新对话",
                "messages": [],
                "qa_count": 0
            }
        }
        st.session_state.current_conversation_id = default_id
    
    # 当前对话的消息（便捷访问）
    if "messages" not in st.session_state:
        st.session_state.messages = st.session_state.conversations.get(
            st.session_state.current_conversation_id, {}
        ).get("messages", [])
    
    # API 配置状态
    if "api_configured" not in st.session_state:
        st.session_state.api_configured = False
    
    # LLM 配置（必须）
    if "llm_base_url" not in st.session_state:
        st.session_state.llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "qwen-plus"
    
    # Embedding 配置（可选，为空时复用 LLM 配置）
    if "embed_base_url" not in st.session_state:
        st.session_state.embed_base_url = ""
    if "embed_api_key" not in st.session_state:
        st.session_state.embed_api_key = ""
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = "text-embedding-v3"
    
    # 连接测试结果
    if "llm_connected" not in st.session_state:
        st.session_state.llm_connected = None  # None 表示未测试
    if "embed_connected" not in st.session_state:
        st.session_state.embed_connected = None
    if "has_tested" not in st.session_state:
        st.session_state.has_tested = False
    
    # RAG 引擎实例
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = init_rag_engine()
        st.session_state.config_version = 0  # 配置版本号
    
    # 配置版本号（用于检测配置变化）
    if "config_version" not in st.session_state:
        st.session_state.config_version = 0
    
    # 当前文档
    if "current_doc" not in st.session_state:
        st.session_state.current_doc = None
    
    # 文档就绪状态
    if "doc_ready" not in st.session_state:
        st.session_state.doc_ready = False
    
    # 处理状态
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # 错误消息
    if "error_msg" not in st.session_state:
        st.session_state.error_msg = None
    
    # 对话统计
    if "qa_count" not in st.session_state:
        st.session_state.qa_count = 0
    
    # 文档上传历史（从文件持久化加载）
    if "upload_history" not in st.session_state:
        st.session_state.upload_history = load_upload_history()
    
    # 知识库（已加载的文档列表）
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = []
    
    # 禁用的文档列表（文档名 -> True）
    if "disabled_docs" not in st.session_state:
        st.session_state.disabled_docs = {}
    
    # 侧边栏状态 - 强制展开
    st.session_state.sidebar_collapsed = False


def reset_chat_history():
    """重置当前对话历史"""
    conv_id = st.session_state.current_conversation_id
    if conv_id in st.session_state.conversations:
        st.session_state.conversations[conv_id]["messages"] = []
        st.session_state.conversations[conv_id]["qa_count"] = 0
    st.session_state.messages = []


def clear_error():
    """清除错误消息"""
    st.session_state.error_msg = None


def create_new_conversation():
    """创建新对话"""
    import uuid
    new_id = str(uuid.uuid4())[:8]
    st.session_state.conversations[new_id] = {
        "title": "新对话",
        "messages": [],
        "qa_count": 0
    }
    st.session_state.current_conversation_id = new_id
    st.session_state.messages = []
    st.rerun()


def switch_conversation(conv_id):
    """切换到指定对话"""
    if conv_id in st.session_state.conversations:
        st.session_state.current_conversation_id = conv_id
        st.session_state.messages = st.session_state.conversations[conv_id]["messages"]


def delete_conversation(conv_id):
    """删除指定对话"""
    if len(st.session_state.conversations) > 1:
        del st.session_state.conversations[conv_id]
        if st.session_state.current_conversation_id == conv_id:
            # 切换到第一个对话
            st.session_state.current_conversation_id = list(st.session_state.conversations.keys())[0]
            st.session_state.messages = st.session_state.conversations[st.session_state.current_conversation_id]["messages"]


def update_conversation_title(conv_id, title):
    """更新对话标题"""
    if conv_id in st.session_state.conversations:
        st.session_state.conversations[conv_id]["title"] = title


def disable_document(doc_name):
    """禁用文档（暂时，不参与检索但保留）"""
    st.session_state.disabled_docs[doc_name] = True


def enable_document(doc_name):
    """启用文档（恢复）"""
    if doc_name in st.session_state.disabled_docs:
        del st.session_state.disabled_docs[doc_name]


def delete_document(doc_name):
    """删除文档（永久，从知识库和向量库中移除）"""
    # 从知识库列表移除
    st.session_state.knowledge_base = [
        doc for doc in st.session_state.knowledge_base 
        if doc.get('name') != doc_name
    ]
    # 从禁用列表移除
    if doc_name in st.session_state.disabled_docs:
        del st.session_state.disabled_docs[doc_name]
    # 从上传历史移除
    st.session_state.upload_history = [
        doc for doc in st.session_state.upload_history 
        if doc.get('name') != doc_name
    ]
    # 保存上传历史
    save_upload_history(st.session_state.upload_history)
    # 从向量库删除
    try:
        if st.session_state.rag_engine:
            st.session_state.rag_engine.delete_documents([doc_name])
    except Exception:
        pass


def get_active_documents():
    """获取活跃文档列表（排除禁用的）"""
    return [
        doc for doc in st.session_state.knowledge_base 
        if doc.get('name') not in st.session_state.disabled_docs
    ]


def get_current_messages():
    """获取当前对话的消息"""
    conv_id = st.session_state.current_conversation_id
    return st.session_state.conversations.get(conv_id, {}).get("messages", [])


def save_message_to_current(role, content):
    """保存消息到当前对话"""
    conv_id = st.session_state.current_conversation_id
    if conv_id in st.session_state.conversations:
        st.session_state.conversations[conv_id]["messages"].append({
            "role": role,
            "content": content
        })
        # 更新标题为第一条用户消息
        if role == "user" and st.session_state.conversations[conv_id]["title"] == "新对话":
            title = content[:20] + "..." if len(content) > 20 else content
            st.session_state.conversations[conv_id]["title"] = title
        # 更新统计
        if role == "assistant":
            st.session_state.conversations[conv_id]["qa_count"] += 1


# ========================================================================
# 📤 文件上传处理
# ========================================================================
def handle_file_upload(uploaded_file: Any) -> bool:
    """处理文件上传"""
    if uploaded_file is None:
        return False
    
    # 检查 API Key 和模型是否已配置
    if not st.session_state.llm_api_key:
        st.error("⚠️ 请先配置服务：展开「服务配置」→ 选择模型 → 填写 API Key → 保存配置")
        return False
    
    if not st.session_state.llm_model:
        st.error("⚠️ 请先选择对话模型：展开「服务配置」→ 选择模型 → 保存配置")
        return False
    
    # 只在首次上传或引擎为空时初始化，或配置变化时重新初始化
    current_version = st.session_state.config_version
    if (st.session_state.rag_engine is None or 
        getattr(st.session_state, '_engine_version', -1) != current_version):
        st.session_state.rag_engine = init_rag_engine()
        st.session_state._engine_version = current_version
        if st.session_state.rag_engine is None:
            st.error("⚠️ RAG 引擎初始化失败")
            return False
        # 如果是配置变化导致的重新初始化，清空知识库列表
        if current_version > 0 and not st.session_state.knowledge_base:
            pass  # 正常情况
    
    file_name: str = uploaded_file.name
    
    # 检测是否已存在该文档
    existing_names = [doc['name'] for doc in st.session_state.knowledge_base]
    if file_name in existing_names:
        st.toast(f"📄 {file_name} 已加载", icon="ℹ️")
        return True
    
    st.session_state.processing = True
    st.session_state.error_msg = None
    
    try:
        # 调用 RAG 引擎上传
        success = st.session_state.rag_engine.upload_file(
            uploaded_file.read(),
            file_name
        )
        
        if success:
            # 添加到知识库列表
            st.session_state.knowledge_base.append({
                'name': file_name,
                'time': datetime.now().strftime("%H:%M:%S"),
                'size': len(uploaded_file.getvalue())
            })
            st.session_state.doc_ready = True
            st.session_state.upload_history.append({
                'name': file_name,
                'time': datetime.now().strftime("%H:%M:%S"),
                'size': len(uploaded_file.getvalue())
            })
            save_upload_history(st.session_state.upload_history)
            
            st.toast(f"✅ {file_name} 已添加至知识库", icon="🎉")
            return True
        else:
            st.session_state.error_msg = "后端返回处理失败，请检查文件是否损坏"
            return False
            
    except Exception as e:
        st.session_state.error_msg = f"处理异常：{str(e)}"
        return False
    finally:
        st.session_state.processing = False


# ========================================================================
# 💬 问答处理
# ========================================================================
def handle_question(prompt: str) -> str:
    """处理用户问题，支持多轮对话"""
    if not st.session_state.doc_ready:
        return "⚠️ 请先上传文档后再提问"
    
    try:
        # 构建 chat_history（传递给 A 的接口用于多轮对话）
        chat_history: Optional[List[Dict[str, str]]] = None
        if INTERFACE_AVAILABLE:
            # 获取当前对话的消息
            current_messages = get_current_messages()
            # 只保留最近10轮对话作为历史
            recent_messages = [
                msg for msg in current_messages
                if msg.get("role") in ["user", "assistant"]
            ][-20:]  # 10轮 = 20条消息
            
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in recent_messages
            ]
        
        # 调用问答接口
        answer: str = st.session_state.rag_engine.ask_question(
            query=prompt,
            chat_history=chat_history
        )
        return answer
        
    except Exception as e:
        return f"⚠️ 回答生成失败：{str(e)}"





# ========================================================================
# 🎯 侧边栏组件（包含知识库 + 对话管理）
# ========================================================================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("### 知识库")
        
        # 文件上传（支持多文件）
        uploaded_files = st.file_uploader(
            "选择文档（支持多选）",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        # 显示已加载文档数量
        if st.session_state.knowledge_base:
            st.success(f"已加载 {len(st.session_state.knowledge_base)} 个文档")
        
        # 按钮行
        col1, col2 = st.columns(2)
        with col1:
            upload_btn = st.button("批量上传", use_container_width=True)
        with col2:
            clear_btn = st.button("清空", use_container_width=True)
        
        if upload_btn and uploaded_files:
            # 检查配置
            if not st.session_state.llm_api_key:
                st.error("⚠️ 请先配置服务：展开「服务配置」→ 选择模型 → 填写 API Key → 保存配置")
            elif not st.session_state.llm_model:
                st.error("⚠️ 请先选择对话模型：展开「服务配置」→ 选择模型 → 保存配置")
            else:
                with st.spinner("处理中..."):
                    for uploaded_file in uploaded_files:
                        handle_file_upload(uploaded_file)
                    st.rerun()
        
        if clear_btn:
            st.session_state.current_doc = None
            st.session_state.doc_ready = False
            st.session_state.knowledge_base = []
            st.session_state.upload_history = []
            st.session_state.rag_engine = None  # 重置引擎以清空向量库
            st.session_state._engine_version = -1  # 重置引擎版本
            save_upload_history([])
            reset_chat_history()
            st.rerun()
        
        st.divider()
        
        # API 配置面板
        with st.expander("服务配置", expanded=not st.session_state.api_configured):
            st.markdown("**LLM 配置**")
            llm_preset_options = list(LLM_PRESETS.keys())
            llm_preset_default = 0
            if st.session_state.llm_model and st.session_state.llm_base_url:
                for i, (name, config) in enumerate(LLM_PRESETS.items()):
                    if config["model"] == st.session_state.llm_model and config["base_url"] == st.session_state.llm_base_url:
                        llm_preset_default = i
                        break
            
            selected_llm_preset = st.selectbox("选择模型", options=llm_preset_options, index=llm_preset_default)
            llm_preset = LLM_PRESETS[selected_llm_preset]
            llm_base_url = llm_preset["base_url"]
            llm_model = llm_preset["model"]
            
            if selected_llm_preset == "自定义":
                llm_base_url = st.text_input("Base URL", value=st.session_state.llm_base_url, placeholder="https://api.openai.com/v1")
                llm_model = st.text_input("模型名称", value=st.session_state.llm_model, placeholder="gpt-4o")
            else:
                st.caption(f"Endpoint: {llm_base_url} | Model: {llm_model}")
            
            llm_api_key = st.text_input("API Key", value=st.session_state.llm_api_key, type="password", placeholder="输入 API 密钥")
            
            st.markdown("**Embedding 配置** (可选，留空则复用 LLM 配置)")
            embed_base_url = st.text_input("Embedding Base URL", value=st.session_state.embed_base_url, placeholder="留空则复用 LLM Base URL")
            embed_api_key = st.text_input("Embedding API Key", value=st.session_state.embed_api_key, type="password", placeholder="留空则复用 LLM API Key")
            embed_model = st.text_input("Embedding 模型", value=st.session_state.embed_model, placeholder="text-embedding-v3")
            
            col_save, col_test = st.columns(2)
            with col_save:
                save_btn = st.button("保存配置", use_container_width=True)
            with col_test:
                test_btn = st.button("测试连接", use_container_width=True)
            
            if save_btn:
                if not llm_api_key:
                    st.error("LLM API Key 不能为空")
                else:
                    st.session_state.llm_base_url = llm_base_url
                    st.session_state.llm_api_key = llm_api_key
                    st.session_state.llm_model = llm_model
                    st.session_state.embed_base_url = embed_base_url
                    st.session_state.embed_api_key = embed_api_key
                    st.session_state.embed_model = embed_model
                    st.session_state.rag_engine = init_rag_engine()
                    st.session_state.config_version += 1  # 配置已更新
                    st.session_state.knowledge_base = []  # 清空知识库（需重新入库）
                    save_upload_history([])
                    st.rerun()
            
            if test_btn:
                st.session_state.llm_base_url = llm_base_url
                st.session_state.llm_api_key = llm_api_key
                st.session_state.llm_model = llm_model
                st.session_state.embed_base_url = embed_base_url
                st.session_state.embed_api_key = embed_api_key
                st.session_state.embed_model = embed_model
                with st.spinner("测试中..."):
                    llm_ok, embed_ok = test_api_connection()
                if llm_ok:
                    st.success("LLM 连接成功")
                else:
                    st.error("LLM 连接失败")
                if embed_ok:
                    st.success("Embedding 连接成功")
                else:
                    st.error("Embedding 连接失败")
            
            if st.session_state.has_tested:
                st.markdown("---")
                status_col1, status_col2 = st.columns(2)
                with status_col1:
                    llm_status = "✅" if st.session_state.llm_connected else "❌"
                    st.markdown(f"LLM: {llm_status}")
                with status_col2:
                    embed_status = "✅" if st.session_state.embed_connected else "❌"
                    st.markdown(f"Embedding: {embed_status}")
        
        # 对话管理（可展开面板）
        with st.expander("💬 对话管理", expanded=False):
            # 新建对话按钮
            if st.button("➕ 新建对话", use_container_width=True):
                create_new_conversation()
            
            # 对话列表
            conversations = st.session_state.conversations
            current_id = st.session_state.current_conversation_id
            conv_count = len(conversations)
            
            st.caption(f"共 {conv_count} 个对话")
            
            for conv_id, conv in reversed(list(conversations.items())):
                is_active = conv_id == current_id
                title = conv.get("title", "新对话")[:18]
                msg_count = len(conv.get("messages", []))
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    btn_label = f"💬 {title}"
                    if is_active:
                        st.markdown(f"**▶ {title}**")
                        st.caption(f"   {msg_count} 条消息")
                    else:
                        if st.button(btn_label, key=f"conv_{conv_id}", use_container_width=True):
                            switch_conversation(conv_id)
                            st.rerun()
                with col2:
                    if conv_count > 1:
                        if st.button("×", key=f"del_{conv_id}", help="删除"):
                            delete_conversation(conv_id)
                            st.rerun()
        
        st.divider()
        
        # 已加载文档列表（下拉栏形式）
        if st.session_state.knowledge_base:
            doc_count = len(st.session_state.knowledge_base)
            disabled_count = len(st.session_state.disabled_docs)
            status_text = f"📚 已加载文档 ({doc_count})"
            if disabled_count > 0:
                status_text += f" · {disabled_count} 已禁用"
            with st.expander(status_text):
                for i, doc in enumerate(st.session_state.knowledge_base):
                    name = doc.get('name', '未知')
                    size = doc.get('size', 0)
                    time = doc.get('time', '')
                    is_disabled = name in st.session_state.disabled_docs
                    
                    if size > 1024 * 1024:
                        size_str = f"{size / (1024*1024):.1f} MB"
                    elif size > 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size} B"
                    
                    # 文档信息行
                    status_icon = "⏸️" if is_disabled else "✅"
                    st.caption(f"{status_icon} {name} ({size_str}) {time}")
                    
                    # 操作按钮行
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        if is_disabled:
                            if st.button("启用", key=f"enable_{name}", use_container_width=True):
                                enable_document(name)
                                st.rerun()
                        else:
                            if st.button("禁用", key=f"disable_{name}", use_container_width=True):
                                disable_document(name)
                                st.rerun()
                    with col2:
                        if st.button("删除", key=f"delete_{name}", use_container_width=True):
                            delete_document(name)
                            st.rerun()
                    st.divider()


# ========================================================================
# 💬 主对话界面
# ========================================================================
def render_chat_interface():
    """渲染主对话界面"""
    
    # 欢迎消息（当前对话无消息时）
    current_messages = get_current_messages()
    if not current_messages:
        st.markdown('<p class="welcome-title">RAG 智能问答</p>', unsafe_allow_html=True)
        st.markdown('<p class="welcome-subtitle">基于检索增强生成技术的知识交互系统</p>', unsafe_allow_html=True)
        
        # 如果已有加载的文档，显示知识库状态（只显示活跃的）
        active_docs = get_active_documents()
        if active_docs:
            doc_list = "、".join([doc['name'] for doc in active_docs])
            st.info(f"📚 知识库已加载 {len(active_docs)} 个文档：{doc_list}")
        elif st.session_state.knowledge_base:
            st.warning(f"⚠️ 当前所有 {len(st.session_state.knowledge_base)} 个文档均已禁用，请到侧边栏启用")
        
        st.markdown("""
        <div class="welcome-box">
            <h4>欢迎使用</h4>
            <ol>
                <li>在左侧上传文档（PDF/TXT）</li>
                <li>等待系统完成解析入库</li>
                <li>在下方输入问题开始问答</li>
                <li>侧边栏「对话管理」可切换多个对话</li>
            </ol>
            <hr style="margin: 12px 0;">
            <h4>支持功能</h4>
            <ul style="margin: 0; padding-left: 18px;">
                <li>多格式文档解析</li>
                <li>智能向量检索</li>
                <li>多轮对话上下文</li>
                <li>多对话独立管理</li>
                <li>参考来源标注</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 渲染历史消息 - 用户消息右侧（头像在右），AI 消息左侧
    current_messages = get_current_messages()
    for msg in current_messages:
        if msg["role"] == "user":
            # 用户消息：气泡在左，头像在右
            col1, col2 = st.columns([0.05, 0.9])
            with col1:
                st.markdown('<div style="width:36px;height:36px;background:#10A37F;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:14px;font-weight:bold;">U</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="background-color:#10A37F;color:white;padding:12px 16px;border-radius:0 12px 12px 12px;text-align:left;max-width:100%;margin-bottom:8px;">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            # AI 消息：头像在左，气泡在右
            col1, col2 = st.columns([0.03, 0.9])
            with col1:
                st.markdown('<div style="width:36px;height:36px;background:#343541;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:14px;font-weight:bold;">A</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="background-color:#343541;color:#ECECEC;padding:12px 16px;border-radius:0 12px 12px 12px;text-align:left;max-width:100%;margin-bottom:8px;">{msg["content"]}</div>', unsafe_allow_html=True)
    
    # 问答输入框
    if st.session_state.knowledge_base:
        doc_count = len(st.session_state.knowledge_base)
        placeholder = f"向知识库（{doc_count}个文档）提问..."
    else:
        placeholder = "请先上传文档后提问"
    disabled = not st.session_state.doc_ready or st.session_state.processing
    
    if prompt := st.chat_input(placeholder, disabled=disabled):
        # 保存用户消息到当前对话
        save_message_to_current("user", prompt)
        st.session_state.messages = get_current_messages()
        
        # 渲染用户消息 - 头像在右，气泡在左
        col1, col2 = st.columns([0.05, 0.9])
        with col1:
            st.markdown('<div style="width:36px;height:36px;background:#10A37F;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:14px;font-weight:bold;">U</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div style="background-color:#10A37F;color:white;padding:12px 16px;border-radius:0 12px 12px 12px;text-align:left;max-width:100%;margin-bottom:8px;">{prompt}</div>', unsafe_allow_html=True)
        
        # 生成 AI 回复 - 左侧
        col1, col2 = st.columns([0.03, 0.9])
        with col1:
            st.markdown('<div style="width:36px;height:36px;background:#343541;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:14px;font-weight:bold;">A</div>', unsafe_allow_html=True)
        with col2:
            with st.spinner("思考中..."):
                answer = handle_question(prompt)
                st.markdown(f'<div style="background-color:#343541;color:#ECECEC;padding:12px 16px;border-radius:0 12px 12px 12px;text-align:left;max-width:100%;margin-bottom:8px;">{answer}</div>', unsafe_allow_html=True)
                
                # 保存 AI 回复到当前对话
                save_message_to_current("assistant", answer)
                st.session_state.messages = get_current_messages()


# ========================================================================
# 🚀 主函数
# ========================================================================
def main():
    """应用主入口"""
    
    # 页面配置
    st.set_page_config(
        page_title="RAG 智能问答助手",
        page_icon="R",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 加载自定义样式
    load_custom_css()
    
    # 初始化状态
    init_session_state()
    
    # 渲染左侧边栏（包含知识库 + 对话管理）
    render_sidebar()
    
    # 渲染主对话界面
    render_chat_interface()


if __name__ == "__main__":
    main()


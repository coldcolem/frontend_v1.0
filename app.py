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
# 🧪 临时 Mock 类（前端独立调试用，对接 A 后请删除或设为 fallback）
# ========================================================================
class MockRAGEngine:
    """模拟 RAG 引擎，供前端独立开发和演示使用"""
    
    def __init__(self) -> None:
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.qa_history: List[Dict[str, str]] = []
    
    def upload_file(self, file_bytes: bytes, file_name: str) -> bool:
        """模拟文档解析与向量入库"""
        time.sleep(1.5)  # 模拟处理延迟
        
        # 模拟不同文件类型
        ext = file_name.split('.')[-1].lower()
        if ext not in ['pdf', 'txt', 'md', 'docx', 'doc']:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        # 模拟成功入库
        self.documents[file_name] = {
            'size': len(file_bytes),
            'upload_time': datetime.now().isoformat(),
            'chunks': len(file_bytes) // 100  # 模拟分块数
        }
        return True
    
    def ask_question(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """模拟 RAG 检索与生成回答"""
        time.sleep(0.8)  # 模拟 RAG 检索延迟
        
        current_doc = st.session_state.get('current_doc', '未知文档')
        
        responses = [
            f"✅ 根据《{current_doc}》中的内容，我找到了以下相关信息：\n\n"
            f"**检索片段 1**：相关章节介绍了核心概念和关键要点...\n\n"
            f"**检索片段 2**：文档中还提到了实施步骤和注意事项...\n\n"
            f"基于以上检索结果，针对您的问题「{query}」，我的回答是：\n\n"
            f"根据文档描述，这个问题涉及多个方面。首先...其次...最后...",
            
            f"📚 我在《{current_doc}》中检索到以下相关内容：\n\n"
            f"相关段落表明，该问题的解决方案需要考虑以下几个因素：\n\n"
            f"1. **技术层面**：涉及系统架构和算法设计\n"
            f"2. **实践层面**：需要参考文档中的案例分析\n"
            f"3. **优化层面**：可参考文档建议的最佳实践\n\n"
            f"针对「{query}」的具体建议是...",
            
            f"🔍 已从《{current_doc}》检索到匹配内容：\n\n"
            f"文档中的关键信息显示，这个问题应当这样理解：\n\n"
            f"**核心观点**：文档指出了问题的本质和解决方向\n\n"
            f"**具体说明**：详细描述了操作步骤和预期结果\n\n"
            f"**参考建议**：可以结合实际情况进行调整\n\n"
            f"针对「{query}」的回答：**具体答案内容...**"
        ]
        
        import random
        response = random.choice(responses)
        self.qa_history.append({'query': query, 'response': response[:50]})
        return response
    
    def get_document_list(self) -> List[str]:
        """获取已上传文档列表"""
        return list(self.documents.keys())
    
    def delete_document(self, file_name: str) -> bool:
        """删除文档"""
        if file_name in self.documents:
            del self.documents[file_name]
            return True
        return False


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
                st.warning("⚠️ RAG 后端初始化失败，将使用 Mock 模式")
                return MockRAGEngine()
                
        except ImportError as e:
            st.info(f"ℹ️ 未找到 RAG 后端实现，使用 Mock 模式: {e}")
            return MockRAGEngine()
        except Exception as e:
            st.error(f"❌ 初始化 RAG 后端出错: {e}")
            return MockRAGEngine()
    else:
        return MockRAGEngine()


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
    
    # 对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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
    
    # 文档上传历史
    if "upload_history" not in st.session_state:
        st.session_state.upload_history = []
    
    # 侧边栏状态 - 强制展开
    st.session_state.sidebar_collapsed = False


def reset_chat_history():
    """重置对话历史"""
    st.session_state.messages = []
    st.session_state.qa_count = 0


def clear_error():
    """清除错误消息"""
    st.session_state.error_msg = None


# ========================================================================
# 📤 文件上传处理
# ========================================================================
def handle_file_upload(uploaded_file: Any) -> bool:
    """处理文件上传"""
    if uploaded_file is None:
        return False
    
    # 检查 API Key 是否已配置
    if not st.session_state.llm_api_key:
        st.session_state.error_msg = "请先在侧边栏填写并保存 LLM API Key"
        st.error("⚠️ 请先配置 API Key：填写 LLM API Key 后点击「保存配置」")
        return False
    
    # 使用最新配置重新初始化 RAG 引擎
    st.session_state.rag_engine = init_rag_engine()
    
    file_name: str = uploaded_file.name
    
    # 检测是否是新文件
    if file_name == st.session_state.current_doc and st.session_state.doc_ready:
        st.toast("📄 该文档已加载，无需重复上传", icon="ℹ️")
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
            st.session_state.current_doc = file_name
            st.session_state.doc_ready = True
            st.session_state.upload_history.append({
                'name': file_name,
                'time': datetime.now().strftime("%H:%M:%S"),
                'size': len(uploaded_file.getvalue())
            })
            reset_chat_history()
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"""📄 **《{file_name}》已成功加载！**

✅ 文档已解析并入库至向量知识库。

💡 **使用建议**：
• 请在下方输入您的问题
• 可以询问文档内容、概念解释、操作步骤等
• 上传新文档将自动切换知识库""",
                "timestamp": datetime.now().isoformat()
            })
            st.toast("🎉 文档处理完成，可以开始对话了！", icon="✅")
            return True
        else:
            st.session_state.error_msg = "后端返回处理失败，请检查文件是否损坏"
            return False
            
    except Exception as e:
        st.session_state.error_msg = f"处理异常：{str(e)}"
        st.session_state.doc_ready = False
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
            # 只保留最近10轮对话作为历史
            recent_messages = [
                msg for msg in st.session_state.messages
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
        st.session_state.qa_count += 1
        return answer
        
    except Exception as e:
        return f"⚠️ 回答生成失败：{str(e)}"


# ========================================================================
# 🎯 侧边栏组件
# ========================================================================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("### 知识库")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "上传文档 (PDF/TXT/MD)",
            type=["pdf", "txt", "md", "docx", "doc"]
        )
        
        # 按钮行
        col1, col2 = st.columns(2)
        with col1:
            upload_btn = st.button("上传", use_container_width=True)
        with col2:
            clear_btn = st.button("清空", use_container_width=True)
        
        if upload_btn and uploaded_file:
            with st.spinner("处理中..."):
                handle_file_upload(uploaded_file)
                st.rerun()
        
        if clear_btn:
            st.session_state.current_doc = None
            st.session_state.doc_ready = False
            st.session_state.upload_history = []
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
            
            selected_llm_preset = st.radio("选择模型", options=llm_preset_options, index=llm_preset_default, horizontal=True, label_visibility="collapsed")
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
        
        st.divider()
        
        if st.session_state.doc_ready:
            st.text(f"已加载: {st.session_state.current_doc}")
            col3, col4 = st.columns(2)
            with col3:
                st.metric("问答", st.session_state.qa_count)
            with col4:
                st.metric("文档", len(st.session_state.upload_history))
        else:
            st.text("等待上传文档")
        
        st.divider()
        
        if st.button("新对话", use_container_width=True):
            reset_chat_history()
            st.rerun()
        
        if st.session_state.upload_history:
            with st.expander("上传历史"):
                for i, doc in enumerate(reversed(st.session_state.upload_history[-5:])):
                    st.caption(f"{i+1}. {doc['name']}")


# ========================================================================
# 💬 主对话界面
# ========================================================================
def render_chat_interface():
    """渲染主对话界面"""
    
    # 欢迎消息（首次打开）
    if not st.session_state.messages:
        st.markdown('<p class="welcome-title">RAG 智能问答</p>', unsafe_allow_html=True)
        st.markdown('<p class="welcome-subtitle">基于检索增强生成技术的知识交互系统</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="welcome-box">
            <h4>欢迎使用</h4>
            <ol>
                <li>在左侧上传文档（PDF/TXT/MD）</li>
                <li>等待系统完成解析入库</li>
                <li>在下方输入问题开始问答</li>
            </ol>
            <hr style="margin: 12px 0;">
            <h4>支持功能</h4>
            <ul style="margin: 0; padding-left: 18px;">
                <li>多格式文档解析</li>
                <li>智能向量检索</li>
                <li>多轮对话上下文</li>
                <li>参考来源标注</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 渲染历史消息 - 用户消息右侧，AI 消息左侧
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            # 用户消息：右侧，带头像
            col1, col2 = st.columns([0.9, 0.05])
            with col1:
                st.markdown(f'<div style="background-color:#10A37F;color:white;padding:12px 16px;border-radius:12px 0 12px 12px;text-align:left;max-width:100%;">{msg["content"]}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div style="width:36px;height:36px;background:#10A37F;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:14px;font-weight:bold;">U</div>', unsafe_allow_html=True)
        else:
            # AI 消息：左侧
            col1, col2 = st.columns([0.03, 0.9])
            with col1:
                st.markdown('<div style="width:36px;height:36px;background:#343541;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:14px;font-weight:bold;">A</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="background-color:#343541;color:#ECECEC;padding:12px 16px;border-radius:0 12px 12px 12px;text-align:left;max-width:100%;">{msg["content"]}</div>', unsafe_allow_html=True)
    
    # 问答输入框
    placeholder = "输入您的问题..." if st.session_state.doc_ready else "请先上传文档后提问"
    disabled = not st.session_state.doc_ready or st.session_state.processing
    
    if prompt := st.chat_input(placeholder, disabled=disabled):
        # 添加用户消息到历史
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # 渲染用户消息 - 右侧
        col1, col2 = st.columns([0.9, 0.05])
        with col1:
            st.markdown(f'<div style="background-color:#10A37F;color:white;padding:12px 16px;border-radius:12px 0 12px 12px;text-align:left;max-width:100%;">{prompt}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div style="width:36px;height:36px;background:#10A37F;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:14px;font-weight:bold;">U</div>', unsafe_allow_html=True)
        
        # 生成 AI 回复 - 左侧
        col1, col2 = st.columns([0.03, 0.9])
        with col1:
            st.markdown('<div style="width:36px;height:36px;background:#343541;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:14px;font-weight:bold;">A</div>', unsafe_allow_html=True)
        with col2:
            with st.spinner("思考中..."):
                answer = handle_question(prompt)
                st.markdown(f'<div style="background-color:#343541;color:#ECECEC;padding:12px 16px;border-radius:0 12px 12px 12px;text-align:left;max-width:100%;">{answer}</div>', unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })


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
    
    # 渲染界面
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()

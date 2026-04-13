from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM 连接配置。仅支持 OpenAI 兼容协议（含国产代理服务）。"""

    base_url: Optional[str]  # OpenAI 兼容接口的 Base URL，留空则使用官方地址
    api_key: str
    model_name: str
    temperature: float = 0.7


@dataclass
class EmbedConfig:
    """Embedding 连接配置。仅支持 OpenAI 兼容协议（含国产代理服务）。"""

    base_url: Optional[str]  # OpenAI 兼容接口的 Base URL，留空则使用官方地址
    api_key: str
    model_name: str


@dataclass
class Citation:
    """单条引文来源的结构化描述。"""

    content: str  # 被引用的原文片段
    source: str = "未知来源"  # 文件名 / URL 等来源标识
    source_path: Optional[str] = None  # 完整来源路径（展示层可用于 tooltip）
    page: Optional[int] = None  # 页码（PDF 适用）
    score: Optional[float] = None  # 相似度距离（越小越相关）
    chunk_index: int = 0  # 在检索结果中的排序位置


class IRAGBackend(ABC):
    """
    RAG核心服务的抽象基类接口约束。
    """

    @abstractmethod
    def ping_llm(self, config: LLMConfig) -> bool:
        """测试大语言模型配置连通性"""
        pass

    @abstractmethod
    def ping_embedding(self, config: EmbedConfig) -> bool:
        """测试向量模型连通性"""
        pass

    @abstractmethod
    def initialize(self, llm_config: LLMConfig, embed_config: EmbedConfig) -> bool:
        """初始化整个 RAG 后端实例"""
        pass

    @abstractmethod
    def ingest_documents(
        self, file_paths: List[str], source_names: Optional[Dict[str, str]] = None
    ) -> bool:
        """解析切分并入库文档。source_names 用于将临时路径映射为用户可读文件名。"""
        pass

    @abstractmethod
    def get_knowledge_files(self) -> List[Dict[str, Any]]:
        """返回当前会话知识库中的文件摘要列表。"""
        pass

    @abstractmethod
    def delete_knowledge_files(self, sources: List[str]) -> bool:
        """按来源标识删除文件并重建索引。"""
        pass

    @abstractmethod
    def get_last_ingest_stats(self) -> Dict[str, Any]:
        """返回最近一次入库的耗时与规模统计。"""
        pass

    @abstractmethod
    def chat(
        self, query: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        RAG 对话请求。

        Returns:
            dict with keys:
                - answer (str): LLM 生成的回答
                - citations (List[Citation]): 检索到的引文来源列表
        """
        pass

    @abstractmethod
    def _generate_hypothetical_answer(self, query: str) -> str:
        """HyDE: 生成假设性答案用于检索增强"""
        pass

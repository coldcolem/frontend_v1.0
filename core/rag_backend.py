import logging
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Tuple, cast
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from core.interfaces import IRAGBackend, LLMConfig, EmbedConfig, Citation

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Streamlit Cloud 内存保护：单 Session 最多允许缓存的向量 Chunk 数量
_MAX_CHUNKS = 2000


class LangChainRAGBackend(IRAGBackend):
    def __init__(self):
        self.llm: Optional["BaseChatModel"] = None
        self.embeddings: Optional["Embeddings"] = None
        self.vector_store = None
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus: List[Document] = []
        self._chunk_count = 0  # 当前已入库的 Chunk 总数
        self._embed_batch_candidates = (128, 64, 32)
        self._embedding_cache: Dict[str, List[float]] = {}
        self._embedding_cache_limit = 4000
        self._last_ingest_stats: Dict[str, Any] = {}

    @staticmethod
    def _sanitize_retrieved_text(text: str) -> str:
        """清理检索文本中的 UI HTML 残留，避免模型把前端片段当知识复述。"""
        if not text:
            return ""
        sanitized = text
        if "citation-" in sanitized or "<div" in sanitized or "<span" in sanitized:
            sanitized = re.sub(r"<[^>]+>", " ", sanitized)
            sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized

    @staticmethod
    def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
        """将用户输入的 endpoint 规范为 OpenAI 兼容的根地址。"""
        if not base_url:
            return None

        normalized = base_url.strip().rstrip("/")
        lower = normalized.lower()
        suffixes = (
            "/chat/completions",
            "/completions",
            "/embeddings",
        )
        for suffix in suffixes:
            if lower.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break

        return normalized or None

    def _build_llm(self, config: LLMConfig):
        """内部工厂：构建 OpenAI 兼容协议的 LLM 客户端。"""
        return ChatOpenAI(
            api_key=cast(Any, config.api_key),
            base_url=self._normalize_base_url(config.base_url),
            model=config.model_name,
            temperature=config.temperature,
            max_retries=2,
        )

    def _build_embeddings(self, config: EmbedConfig):
        """内部工厂：构建 OpenAI 兼容协议的 Embedding 客户端。

        使用 SafeOpenAIEmbeddings 子类，同时覆盖 embed_query 和 embed_documents：
        - 仅发送 input / model 两个参数，剥离 encoding_format / dimensions 等
          OpenAI 特有字段，避免第三方兼容 API 返回 'No schema matches' 错误。
        - 将 input 始终包装为列表，兼容部分严格 schema 的国产服务。
        """
        _model_name = config.model_name
        cache = self._embedding_cache
        cache_limit = self._embedding_cache_limit
        batch_candidates = self._embed_batch_candidates

        class SafeOpenAIEmbeddings(OpenAIEmbeddings):
            @staticmethod
            def _cache_put(text: str, embedding: List[float]):
                if text in cache:
                    return
                if len(cache) >= cache_limit:
                    # 轻量缓存淘汰：移除最早插入项
                    oldest_key = next(iter(cache))
                    cache.pop(oldest_key, None)
                cache[text] = embedding

            def embed_documents(
                self, texts: List[str], chunk_size: Optional[int] = None, **kwargs: Any
            ) -> List[List[float]]:
                """直接调用底层 client，只传 input 与 model，不附加额外字段。"""
                if not texts:
                    return []

                # 先查缓存并收集缺失文本（按文本去重）
                text_to_embedding: Dict[str, List[float]] = {}
                unique_missing: List[str] = []
                seen_missing = set()
                for text in texts:
                    if text in cache:
                        text_to_embedding[text] = cache[text]
                    elif text not in seen_missing:
                        unique_missing.append(text)
                        seen_missing.add(text)

                if unique_missing:
                    last_error: Optional[Exception] = None

                    def _embed_batch(batch: List[str]) -> List[List[float]]:
                        try:
                            response = self.client.create(
                                input=batch, model=_model_name
                            )
                            sorted_data = sorted(response.data, key=lambda d: d.index)
                            return [d.embedding for d in sorted_data]
                        except Exception as e:
                            error_text = str(e).lower()
                            if "schema" in error_text and "input" in error_text:
                                fallback_result = []
                                for single_text in batch:
                                    single = self.client.create(
                                        input=single_text, model=_model_name
                                    )
                                    fallback_result.append(single.data[0].embedding)
                                return fallback_result
                            raise

                    for batch_size in batch_candidates:
                        try:
                            missing_embeddings: List[List[float]] = []
                            batches = [
                                unique_missing[i : i + batch_size]
                                for i in range(0, len(unique_missing), batch_size)
                            ]

                            max_workers = max(1, min(4, len(batches)))
                            if max_workers == 1:
                                for batch in batches:
                                    missing_embeddings.extend(_embed_batch(batch))
                            else:
                                indexed_results: Dict[int, List[List[float]]] = {}
                                with ThreadPoolExecutor(
                                    max_workers=max_workers
                                ) as executor:
                                    future_map = {
                                        executor.submit(_embed_batch, batch): idx
                                        for idx, batch in enumerate(batches)
                                    }
                                    for future in as_completed(future_map):
                                        idx = future_map[future]
                                        indexed_results[idx] = future.result()

                                for idx in range(len(batches)):
                                    missing_embeddings.extend(indexed_results[idx])

                            for text, emb in zip(unique_missing, missing_embeddings):
                                text_to_embedding[text] = emb
                                self._cache_put(text, emb)
                            last_error = None
                            break
                        except Exception as e:
                            last_error = e

                    if last_error is not None:
                        raise last_error

                return [text_to_embedding[text] for text in texts]

            def embed_query(self, text: str, **kwargs: Any) -> List[float]:
                return self.embed_documents([text])[0]

        return SafeOpenAIEmbeddings(
            api_key=cast(Any, config.api_key),
            base_url=self._normalize_base_url(config.base_url),
            model=config.model_name,
            check_embedding_ctx_length=False,
        )

    def ping_llm(self, config: LLMConfig) -> bool:
        try:
            temp_llm = self._build_llm(config)
            response = temp_llm.invoke("Hi")
            return response is not None
        except Exception as e:
            logger.error(f"LLM Ping 失败: {e}")
            return False

    def ping_embedding(self, config: EmbedConfig) -> bool:
        try:
            temp_embed = self._build_embeddings(config)
            result = temp_embed.embed_query("test")
            return len(result) > 0
        except Exception as e:
            logger.error(f"Embedding Ping 失败: {e}")
            return False

    def initialize(self, llm_config: LLMConfig, embed_config: EmbedConfig) -> bool:
        try:
            # 先验证 LLM 连接
            if not self.ping_llm(llm_config):
                raise RuntimeError("LLM 连接验证失败")
            self.llm = self._build_llm(llm_config)

            # 验证 Embedding 连接
            if not self.ping_embedding(embed_config):
                raise RuntimeError("Embedding 连接验证失败")
            self.embeddings = self._build_embeddings(embed_config)

            # FAISS 将在摄入文档时按需初始化
            return True
        except Exception as e:
            logger.error(f"后端初始化失败: {e}")
            # 初始化失败时清除可能已部分设置的实例
            self.llm = None
            self.embeddings = None
            return False

    def clear_index(self):
        """清空内存中的知识库索引以释放资源"""
        self.vector_store = None
        self._bm25_index = None
        self._bm25_corpus = []
        self._chunk_count = 0
        logger.info("知识库索引已清空，内存释放。")

    @staticmethod
    def _load_single_file(path: str) -> Tuple[List[Document], Optional[str]]:
        try:
            ext = os.path.splitext(path)[1].lower()

            if ext == ".pdf":
                try:
                    loader = PyMuPDFLoader(path)
                    docs = loader.load()
                except Exception:
                    loader = PyPDFLoader(path)
                    docs = loader.load()
            elif ext == ".md":
                from langchain_community.document_loaders import (
                    UnstructuredMarkdownLoader,
                )

                loader = UnstructuredMarkdownLoader(path)
                docs = loader.load()
            elif ext == ".docx":
                from langchain_community.document_loaders import (
                    UnstructuredWordDocumentLoader,
                )

                loader = UnstructuredWordDocumentLoader(path)
                docs = loader.load()
            else:
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
            return docs, None
        except Exception as e:
            logger.error(f"文件加载异常 {path}: {e}")
            return [], str(e)

    @staticmethod
    def _iter_index_docs(vector_store) -> List[Document]:
        if vector_store is None:
            return []
        docstore = getattr(vector_store, "docstore", None)
        docs_dict = getattr(docstore, "_dict", {}) if docstore is not None else {}
        return [doc for doc in docs_dict.values() if isinstance(doc, Document)]

    def _apply_source_names(
        self, docs: List[Document], source_names: Optional[Dict[str, str]] = None
    ):
        if not source_names:
            return
        for doc in docs:
            metadata = doc.metadata or {}
            original_source = metadata.get("source")
            if original_source and original_source in source_names:
                metadata["source"] = source_names[original_source]
            doc.metadata = metadata

    def _clean_documents(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs

        cleaned = []
        for doc in docs:
            text = doc.page_content
            if not text:
                continue
            text = self._clean_text_content(text)
            if text:
                doc.page_content = text
                cleaned.append(doc)

        pdf_sources = set()
        for doc in cleaned:
            source = doc.metadata.get("source", "") if doc.metadata else ""
            if source.lower().endswith(".pdf"):
                pdf_sources.add(source)

        if pdf_sources:
            cleaned = self._clean_pdf_header_footer(cleaned, pdf_sources)

        return cleaned

    @staticmethod
    def _clean_text_content(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text

    def _clean_pdf_header_footer(
        self, docs: List[Document], pdf_sources: set
    ) -> List[Document]:
        source_docs: Dict[str, List[Document]] = defaultdict(list)
        for doc in docs:
            source = doc.metadata.get("source", "") if doc.metadata else ""
            if source in pdf_sources:
                source_docs[source].append(doc)

        for source, source_docs_list in source_docs.items():
            header_footer_patterns = self._detect_header_footer_patterns(
                source_docs_list
            )
            for doc in source_docs_list:
                original = doc.page_content
                cleaned = original
                for pattern in header_footer_patterns:
                    cleaned = re.sub(pattern, "", cleaned)
                cleaned = self._clean_text_content(cleaned)
                doc.page_content = cleaned

        return docs

    def _detect_header_footer_patterns(
        self, docs: List[Document], min_repeat: int = 3
    ) -> List[str]:
        page_content_by_pos: Dict[str, List[str]] = defaultdict(list)

        for doc in docs:
            content = doc.page_content
            if not content:
                continue
            lines = content.split("\n")
            if not lines:
                continue

            first_line = lines[0].strip()
            last_line = lines[-1].strip()

            if first_line and len(first_line) < 100:
                page_content_by_pos["first"].append(first_line)
            if last_line and len(last_line) < 100:
                page_content_by_pos["last"].append(last_line)

        patterns = []

        for pos_lines in page_content_by_pos.values():
            counter = Counter(pos_lines)
            for text, count in counter.items():
                if count >= min_repeat and len(text) >= 3:
                    escaped = re.escape(text)
                    patterns.append(escaped)

        return patterns

    def _update_bm25_index(self, docs: List[Document]):
        self._bm25_corpus.extend(docs)
        if len(self._bm25_corpus) > 0:
            corpus_texts = [doc.page_content for doc in self._bm25_corpus]
            self._bm25_index = BM25Okapi(corpus_texts)

    def _hybrid_search(
        self, query: str, top_k: int = 10, alpha: float = 0.5
    ) -> List[Tuple[Document, float]]:
        if not self.vector_store or not self.embeddings:
            raise RuntimeError("服务未初始化完全。")

        hyde_query = self._generate_hypothetical_answer(query)

        vector_scores: Dict[int, float] = {}
        hyde_embedding = self.embeddings.embed_query(hyde_query)
        raw_vector_docs = self.vector_store.similarity_search_with_score_by_vector(
            hyde_embedding, k=top_k * 2
        )
        for doc, score in raw_vector_docs:
            doc_id = id(doc)
            vector_scores[doc_id] = score

        bm25_scores: Dict[int, float] = {}
        bm25_results: List[float] = []
        if self._bm25_index:
            bm25_results = cast(
                List[float], list(self._bm25_index.get_scores(query.split()))
            )
            for idx, score in enumerate(bm25_results):
                if idx < len(self._bm25_corpus):
                    doc_id = id(self._bm25_corpus[idx])
                    bm25_scores[doc_id] = score

        all_doc_ids = set(vector_scores.keys()) | set(bm25_scores.keys())

        if not vector_scores:
            filtered = [
                (self._bm25_corpus[idx], score)
                for idx, score in enumerate(bm25_results[:top_k])
                if score > 0
            ]
            return filtered[:top_k]

        if not bm25_scores:
            filtered = [(doc, score) for doc, score in raw_vector_docs if score <= 0.7]
            return filtered[:top_k]

        vector_min = min(vector_scores.values()) if vector_scores else 1
        vector_max = max(vector_scores.values()) if vector_scores else 1

        bm25_max = max(bm25_scores.values()) if bm25_scores else 1

        results: List[Tuple[Document, float, int]] = []

        for doc_id in all_doc_ids:
            doc = None
            if doc_id in vector_scores:
                for d, _ in raw_vector_docs:
                    if id(d) == doc_id:
                        doc = d
                        break
            if doc is None:
                for d in self._bm25_corpus:
                    if id(d) == doc_id:
                        doc = d
                        break
            if doc is None:
                continue

            vector_score = vector_scores.get(doc_id, vector_max)
            norm_vector = 1 - (vector_score - vector_min) / (
                vector_max - vector_min + 1e-8
            )

            bm25_score = bm25_scores.get(doc_id, 0)
            norm_bm25 = bm25_score / (bm25_max + 1e-8)

            final_score = alpha * norm_vector + (1 - alpha) * norm_bm25

            results.append((doc, final_score, doc_id))

        results.sort(key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score, _ in results[:top_k]]

    def _generate_hypothetical_answer(self, query: str) -> str:
        """HyDE: 使用 LLM 生成假设性答案，用于增强检索"""
        prompt = f"""请针对以下问题生成一个假设性的答案。
要求：
1. 答案要具体、完整，包含关键细节
2. 即使信息不完整，也要基于常识给出合理推断
3. 只输出答案内容，不要添加解释、备注或其他文字

问题：{query}
回答："""
        try:
            llm = self.llm
            if llm is None:
                raise RuntimeError("服务未初始化完全。")

            response = llm.invoke(prompt)
            content = response.content
            if isinstance(content, str):
                return content.strip()
            return query
        except Exception as e:
            logger.warning(f"HyDE 假设答案生成失败，回退到原始查询: {e}")
            return query

    def _build_file_summaries(self) -> List[Dict[str, Any]]:
        docs = self._iter_index_docs(self.vector_store)
        grouped: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "source": "未知来源",
                "file_name": "未知来源",
                "chunk_count": 0,
                "page_count": 0,
                "pages": set(),
            }
        )

        for doc in docs:
            metadata = doc.metadata or {}
            source = metadata.get("source") or "未知来源"
            item = grouped[source]
            item["source"] = source
            item["file_name"] = (
                os.path.basename(source) if source != "未知来源" else source
            )
            item["chunk_count"] += 1
            page = metadata.get("page")
            if page is not None:
                page_num = int(page) + 1
                item["pages"].add(page_num)

        result = []
        for item in grouped.values():
            pages = sorted(list(item["pages"]))
            item["page_count"] = len(pages)
            item["pages"] = pages
            result.append(item)

        result.sort(key=lambda x: x["file_name"].lower())
        return result

    def ingest_documents(
        self, file_paths: List[str], source_names: Optional[Dict[str, str]] = None
    ) -> bool:
        """解析、切分并入库文档。

        Returns:
            True 表示入库成功。

        Raises:
            RuntimeError: 后端未初始化、文件加载失败、容量超限或向量化失败时抛出，
                          携带可读的错误信息供前端展示。
        """
        if not self.embeddings:
            raise RuntimeError("后端服务尚未初始化。请先调用 initialize()。")

        ingest_started_at = time.perf_counter()
        all_docs = []
        load_errors = []

        load_started_at = time.perf_counter()
        max_workers = max(1, min(4, len(file_paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._load_single_file, path): path
                for path in file_paths
            }
            for future in as_completed(future_map):
                path = future_map[future]
                docs, error = future.result()
                if error:
                    load_errors.append(f"{os.path.basename(path)}: {error}")
                else:
                    all_docs.extend(docs)
        self._apply_source_names(all_docs, source_names)
        load_cost_ms = int((time.perf_counter() - load_started_at) * 1000)

        if not all_docs:
            error_detail = "；".join(load_errors) if load_errors else "文档内容为空"
            raise RuntimeError(f"文件加载失败 — {error_detail}")

        all_docs = self._clean_documents(all_docs)

        # 进行文本切分
        split_started_at = time.perf_counter()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1400, chunk_overlap=100
        )
        splits = text_splitter.split_documents(all_docs)
        split_cost_ms = int((time.perf_counter() - split_started_at) * 1000)

        # OOM 保护：检查是否超过单 Session 的最大 Chunk 数量限制
        if self._chunk_count + len(splits) > _MAX_CHUNKS:
            raise RuntimeError(
                f"容量超限：当前已有 {self._chunk_count} 个 Chunk，"
                f"新增 {len(splits)} 个将超过上限 {_MAX_CHUNKS}。"
                "请先清空知识库后再上传。"
            )

        try:
            ingest_stage_started_at = time.perf_counter()
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(splits, self.embeddings)
            else:
                self.vector_store.add_documents(documents=splits)
            self._update_bm25_index(splits)
            ingest_cost_ms = int((time.perf_counter() - ingest_stage_started_at) * 1000)
            self._chunk_count += len(splits)
            total_cost_ms = int((time.perf_counter() - ingest_started_at) * 1000)
            self._last_ingest_stats = {
                "total_ms": total_cost_ms,
                "load_ms": load_cost_ms,
                "split_ms": split_cost_ms,
                "index_ms": ingest_cost_ms,
                "files": len(file_paths),
                "docs": len(all_docs),
                "chunks": len(splits),
                "cache_size": len(self._embedding_cache),
                "load_errors": load_errors,
            }
            logger.info(f"入库成功，当前 Chunk 总数：{self._chunk_count}/{_MAX_CHUNKS}")
            return True
        except Exception as e:
            logger.error(f"向量入库失败: {e}")
            error_text = str(e)
            lower_text = error_text.lower()
            hint = ""
            if "schema" in lower_text and "input" in lower_text:
                hint = (
                    "；请检查 Embedding Base URL 是否为服务根地址（如 .../v1），"
                    "不要填写到具体 endpoint（如 .../chat/completions 或 .../embeddings）"
                )
            raise RuntimeError(f"向量化入库失败: {error_text}{hint}")

    def get_knowledge_files(self) -> List[Dict[str, Any]]:
        return self._build_file_summaries()

    def delete_knowledge_files(self, sources: List[str]) -> bool:
        if not sources:
            return False
        if self.embeddings is None:
            raise RuntimeError("Embedding 服务尚未初始化。")

        source_set = set(sources)
        docs = self._iter_index_docs(self.vector_store)
        if not docs:
            return False

        keep_docs: List[Document] = []
        for doc in docs:
            metadata = doc.metadata or {}
            source = metadata.get("source") or "未知来源"
            if source not in source_set:
                keep_docs.append(doc)

        if not keep_docs:
            self.clear_index()
            return True

        self.vector_store = FAISS.from_documents(keep_docs, self.embeddings)
        self._bm25_corpus = keep_docs
        if len(self._bm25_corpus) > 0:
            corpus_texts = [doc.page_content for doc in self._bm25_corpus]
            self._bm25_index = BM25Okapi(corpus_texts)
        self._chunk_count = len(keep_docs)
        logger.info(
            f"已删除 {len(docs) - len(keep_docs)} 个 Chunk，当前 Chunk 总数：{self._chunk_count}/{_MAX_CHUNKS}"
        )
        return True

    def get_last_ingest_stats(self) -> Dict[str, Any]:
        return self._last_ingest_stats

    def chat(
        self, query: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        if self.llm is None or self.embeddings is None or self.vector_store is None:
            raise RuntimeError("服务未初始化完全。")

        docs_with_scores = self._hybrid_search(query, top_k=10, alpha=0.5)

        if not docs_with_scores:
            return {"answer": "根据现有文档无法回答该问题", "citations": []}

        # 2. 构建上下文和引文列表
        citations: List[Citation] = []
        context_parts = []
        for i, (doc, score) in enumerate(docs_with_scores):
            # 提取来源信息
            metadata = doc.metadata or {}
            raw_source = metadata.get("source", "未知来源")
            # 仅保留文件名部分，去除完整路径
            source_name = os.path.basename(raw_source) if raw_source else "未知来源"
            page = metadata.get("page")

            citations.append(
                Citation(
                    content=self._sanitize_retrieved_text(doc.page_content),
                    source=source_name,
                    source_path=raw_source,
                    page=int(page) + 1 if page is not None else None,  # 转为 1-indexed
                    score=float(score),
                    chunk_index=i,
                )
            )
            context_parts.append(self._sanitize_retrieved_text(doc.page_content))

        context_text = "\n\n".join(context_parts)

        # 3. 组装历史消息
        system_prompt = f"""你是一个专业的文档问答助手，基于提供的参考文档回答用户问题。

## 回答规范
1. **严格依据文档**：只使用参考文档中的信息，禁止编造
2. **不确认时声明**：如果文档没有相关信息，必须明确说"根据提供的文档无法回答该问题"，不要猜测
3. **引用格式**：每个关键论点必须标注来源，格式：[来源: 文件名, 第X页]（如页码未知则省略）
4. **回答结构**：优先使用 bullet points 或编号列表，保持简洁
5. **专业语气**：准确、专业，避免口语化表达

## 参考文档
{context_text}

请根据以上参考文档回答用户问题。回答时必须严格遵守上述规范。"""

        messages: List[Any] = [SystemMessage(content=system_prompt)]

        if chat_history:
            for message in chat_history:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        # 4. 添加当前提问
        messages.append(HumanMessage(content=query))

        # 5. 生成回答
        llm = self.llm
        if llm is None:
            raise RuntimeError("服务未初始化完全。")

        response = llm.invoke(messages)

        return {"answer": response.content, "citations": citations}

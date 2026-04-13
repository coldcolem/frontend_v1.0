"""
知识库管理页面 - 上传文档并构建向量知识库
"""

import streamlit as st
import tempfile
import os

st.set_page_config(page_title="知识库管理 | RAG Search", page_icon="·", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.page-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
    color: inherit;
}
.page-subtitle {
    font-size: 0.88rem;
    margin-bottom: 1.5rem;
    color: inherit;
    opacity: 0.7;
}
.stat-card {
    border: 1px solid var(--streamlit-border-color, #e2e8f0);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    text-align: center;
}
.stat-label { font-size: 0.75rem; font-weight: 500; margin-bottom: 0.2rem; opacity: 0.7; }
.stat-value { font-size: 1.4rem; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="page-title">知识库管理</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">上传文档、构建向量索引</div>', unsafe_allow_html=True
)

backend = st.session_state.get("backend")
if (
    backend is None
    or not getattr(backend, "llm", None)
    or not getattr(backend, "embeddings", None)
):
    st.warning("服务尚未初始化")
    st.caption("请先前往 **Settings** 页面配置 LLM 与 Embedding 服务，并点击「初始化后端」。")
    st.page_link("pages/0_Settings.py", label="前往 Settings", icon=":material/settings:", use_container_width=True)
    st.stop()

# 状态指标
chunk_count = getattr(backend, "_chunk_count", 0)
has_index = backend.vector_store is not None
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">已入库 Chunks</div><div class="stat-value">{chunk_count}</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">索引状态</div><div class="stat-value">{"就绪" if has_index else "空库"}</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">容量上限</div><div class="stat-value">2000</div></div>',
        unsafe_allow_html=True,
    )

st.divider()

# 上传与管理
tab_upload, tab_manage = st.tabs(["上传文档", "文件管理"])

with tab_upload:
    st.markdown("**上传文档**")
    # 使用动态 key 来重置上传组件
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    uploaded_files = st.file_uploader(
        "支持 PDF、TXT、Markdown、DOCX 格式，可多选",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

    if uploaded_files:
        st.caption(
            f"已选择 {len(uploaded_files)} 个文件：{', '.join(f.name for f in uploaded_files)}"
        )

    if st.button(
        "开始入库",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        tmp_paths = []
        source_names = {}
        try:
            with st.spinner("正在处理文档..."):
                for uf in uploaded_files:
                    suffix = os.path.splitext(uf.name)[1]
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(uf.read())
                        tmp_paths.append(tmp.name)
                        source_names[tmp.name] = uf.name

                ok = backend.ingest_documents(tmp_paths, source_names=source_names)

            if ok:
                stats = backend.get_last_ingest_stats()
                total_s = stats.get("total_ms", 0) / 1000
                load_s = stats.get("load_ms", 0) / 1000
                split_s = stats.get("split_ms", 0) / 1000
                index_s = stats.get("index_ms", 0) / 1000
                chunks = stats.get("chunks", 0)
                st.success(f"入库成功。当前 Chunks：{backend._chunk_count}")
                st.caption(
                    f"本次处理：{chunks} chunks，"
                    f"总耗时 {total_s:.2f}s（加载 {load_s:.2f}s / 切分 {split_s:.2f}s / 向量索引 {index_s:.2f}s）"
                )
                # 重置上传组件，防止重复提交
                st.session_state.uploader_key += 1
                st.rerun()
        except RuntimeError as e:
            st.error(f"入库失败：{e}")
        except Exception as e:
            st.error(f"入库过程中发生未知错误：{e}")
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except Exception:
                    pass

    st.divider()
    st.markdown("**清空知识库**")
    st.caption("将释放当前会话的所有向量内存，操作不可撤销。")
    if st.button("清空知识库", type="secondary"):
        backend.clear_index()
        st.success("知识库已清空。")
        st.rerun()

with tab_manage:
    st.markdown("**已入库文件**")
    files = backend.get_knowledge_files()
    if not files:
        st.info("当前暂无已入库文件。")
    else:
        display_rows = []
        for item in files:
            page_preview = "-"
            if item["pages"]:
                if len(item["pages"]) <= 6:
                    page_preview = ", ".join(str(p) for p in item["pages"])
                else:
                    page_preview = f"{item['pages'][0]} ... {item['pages'][-1]}"
            display_rows.append(
                {
                    "文件名": item["file_name"],
                    "Chunks": item["chunk_count"],
                    "页数": item["page_count"],
                    "页码预览": page_preview,
                }
            )
        st.dataframe(display_rows, use_container_width=True, hide_index=True)

        source_options = []
        for idx, item in enumerate(files, start=1):
            label = f"{idx}. {item['file_name']} ({item['chunk_count']} chunks)"
            source_options.append((label, item["source"]))
        selected_label = st.selectbox(
            "选择要删除的文件", [x[0] for x in source_options]
        )

        if st.button("删除选中文件", type="secondary", use_container_width=True):
            target_source = next(
                (x[1] for x in source_options if x[0] == selected_label), None
            )
            if target_source is None:
                st.warning("未找到目标文件，请刷新后重试。")
                st.stop()
            ok = backend.delete_knowledge_files([target_source])
            if ok:
                st.success("文件已删除并重建索引。")
                st.rerun()
            else:
                st.warning("未执行删除：索引为空或目标不存在。")

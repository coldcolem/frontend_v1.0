# Streamlit RAG 搜索引擎项目 (RagSearch)

RagSearch 是一个基于 **Streamlit** + **LangChain** 构建的云端 RAG (Retrieval-Augmented Generation) 文档问答系统，界面风格仿照 [RAGFlow](https://ragflow.io/)。支持上传 PDF 文档，自动完成切片、向量化存储到 Pinecone 云端数据库，然后基于文档内容进行智能多轮对话。

> **设计目标**: 全部 AI 能力通过在线 API 提供（LLM / Embedding / 向量数据库），无任何本地模型依赖，可无缝托管至 Streamlit Cloud 运行。

### 1. 准备开发环境

```powershell
# 创建虚拟环境（Windows PowerShell）
python -m venv venv
.\venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

```bash
streamlit run app.py
```

```bash
.\venv\Scripts\activate
streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`。

---
- 所有新功能需先在 `core/interfaces.py` 中定义抽象方法，再实现


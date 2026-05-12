# my-rag-demo

**一个用于学习 RAG 技术的实践项目** 🚀

RAG（Retrieval-Augmented Generation，检索增强生成）是当前最流行的 AI 应用开发模式之一。本项目通过清晰的代码和详细的注释，帮助你从零理解并动手实现一个完整的 RAG 系统。

---

## 📖 什么是 RAG？

RAG = **检索（Retrieval）** + **增强（Augmented）** + **生成（Generation）**

它解决了大语言模型的三大痛点：
- ❌ **知识截止**：无法获取最新信息 → RAG 可接入实时知识库
- ❌ **幻觉问题**：生成错误内容 → RAG 基于真实文档回答
- ❌ **领域不足**：对特定领域知识了解有限 → RAG 可注入私有知识

## 🏗️ RAG 工作流程

```
文档 → 加载 → 分割(Chunk) → 嵌入(Embedding) → 向量库
                                                    ↓
用户问题 → 查询向量 → 相似度检索 → 相关文档块 → LLM → 答案
```

---

## 🗂️ 项目结构

```
my-rag-demo/
├── rag/                    # 核心模块
│   ├── __init__.py
│   ├── loader.py           # 文档加载器（TXT、PDF、Markdown）
│   ├── splitter.py         # 文本分割器
│   ├── embedder.py         # 嵌入模型（OpenAI / Ollama / HuggingFace）
│   ├── retriever.py        # 向量库与检索器（ChromaDB）
│   ├── generator.py        # 语言模型（OpenAI / Ollama）
│   └── pipeline.py         # 完整 RAG 流水线
├── examples/
│   ├── simple_rag.py       # 入门示例
│   └── advanced_rag.py     # 进阶示例（持久化、调试、自定义提示词）
├── notebooks/
│   └── rag_tutorial.ipynb  # Jupyter 交互教程
├── data/
│   ├── rag_introduction.txt    # RAG 技术介绍
│   └── langchain_intro.txt     # LangChain 框架介绍
├── tests/
│   └── test_rag.py         # 单元测试
├── requirements.txt
└── .env.example            # 环境变量配置示例
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 填写你的 API Key
# 使用 OpenAI：填写 OPENAI_API_KEY
# 使用本地模型：安装 Ollama 后无需 API Key
```

### 3. 运行示例

**入门示例**（推荐新手从这里开始）：
```bash
python examples/simple_rag.py
```

**进阶示例**（持久化、调试、自定义提示词）：
```bash
python examples/advanced_rag.py
```

**Jupyter 交互教程**（边运行边学习）：
```bash
jupyter notebook notebooks/rag_tutorial.ipynb
```

### 4. 运行测试

```bash
pytest tests/ -v
```

---

## 💡 模型选择

### 方式一：OpenAI（推荐入门，需要 API Key）

```python
from rag.embedder import get_embeddings
from rag.generator import get_llm

embeddings = get_embeddings("openai")                    # 嵌入模型
llm = get_llm("openai", model="gpt-4o-mini")            # 语言模型
```

### 方式二：Ollama 本地模型（完全免费，无需 API Key）

1. 安装 Ollama：https://ollama.com/download
2. 拉取模型：
   ```bash
   ollama pull qwen2.5              # 语言模型（中文推荐）
   ollama pull nomic-embed-text     # 嵌入模型
   ```
3. 在代码中切换：
   ```python
   embeddings = get_embeddings("ollama", model="nomic-embed-text")
   llm = get_llm("ollama", model="qwen2.5")
   ```

---

## 📝 代码示例

### 最简示例（5 行核心代码）

```python
from rag import RAGPipeline
from rag.embedder import get_embeddings
from rag.generator import get_llm

pipeline = RAGPipeline(
    embeddings=get_embeddings("openai"),
    llm=get_llm("openai"),
)
pipeline.build(["data/rag_introduction.txt"])
answer = pipeline.ask("什么是 RAG？")
print(answer)
```

### 持久化知识库（避免重复嵌入）

```python
pipeline = RAGPipeline(
    embeddings=get_embeddings("openai"),
    llm=get_llm("openai"),
    persist_dir="./my_knowledge_base",  # 指定持久化目录
)

# 首次运行：构建并保存
pipeline.build(["my_document.txt"])

# 之后直接加载，无需重新嵌入
pipeline.load()
answer = pipeline.ask("你的问题")
```

### 调试检索结果

```python
# 查看检索到的原始文档块
docs = pipeline.retrieve("你的查询")
for doc in docs:
    print(f"来源: {doc.metadata['source']}")
    print(f"内容: {doc.page_content}")
```

---

## 🔧 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 500 | 每个文本块的最大字符数。越小检索越精准，但上下文可能不完整 |
| `chunk_overlap` | 50 | 相邻块的重叠字符数。保证语义连续性 |
| `retrieval_k` | 4 | 检索返回的相关块数量。越多上下文越丰富，但 Token 消耗更多 |
| `persist_dir` | None | 向量库持久化目录。None 则使用内存模式 |

---

## 📚 学习路径

1. **阅读 `data/rag_introduction.txt`** - 理解 RAG 基本概念
2. **运行 `examples/simple_rag.py`** - 跑通第一个 RAG 程序
3. **逐模块阅读 `rag/` 目录** - 理解每个组件的作用
4. **尝试 `examples/advanced_rag.py`** - 学习进阶技巧
5. **修改 `system_prompt`** - 尝试定制 LLM 的回答风格
6. **加载自己的文档** - 将本项目应用到你的实际场景

---

## 🛠️ 技术栈

- **[LangChain](https://python.langchain.com/)** - RAG 框架
- **[ChromaDB](https://www.trychroma.com/)** - 向量数据库
- **[OpenAI](https://platform.openai.com/)** - LLM 和嵌入模型（可替换）
- **[Ollama](https://ollama.com/)** - 本地模型运行时（可选）

---

## 📄 License

MIT

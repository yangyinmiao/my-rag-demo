"""
RAG 完整流水线模块

将所有组件串联起来，实现完整的 RAG（检索增强生成）流程：

  文档 → 加载 → 分割 → 嵌入 → 向量库 → 检索 → 生成 → 答案

RAGPipeline 类提供了两种使用方式：
1. 一次性构建：适合首次使用或小型知识库
2. 持久化：构建后保存到磁盘，下次直接加载，无需重新处理文档
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag.loader import load_documents
from rag.splitter import split_documents
from rag.retriever import build_vector_store, load_vector_store, get_retriever
from rag.generator import DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _format_docs(docs: List[Document]) -> str:
    """将检索到的文档块格式化为字符串，作为 LLM 的上下文"""
    return "\n\n---\n\n".join(
        f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
        for doc in docs
    )


class RAGPipeline:
    """
    RAG 流水线

    使用示例：
    ```python
    from rag import RAGPipeline
    from rag.embedder import get_embeddings
    from rag.generator import get_llm

    # 初始化
    pipeline = RAGPipeline(
        embeddings=get_embeddings("openai"),
        llm=get_llm("openai"),
    )

    # 构建知识库
    pipeline.build(["data/my_document.txt"])

    # 提问
    answer = pipeline.ask("这份文档讲了什么？")
    print(answer)
    ```
    """

    def __init__(
        self,
        embeddings: Embeddings,
        llm: BaseChatModel,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 4,
        persist_dir: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        """
        初始化 RAG 流水线

        Args:
            embeddings: 嵌入模型（用于将文本转为向量）
            llm: 语言模型（用于生成答案）
            chunk_size: 文本块大小（字符数）
            chunk_overlap: 相邻块重叠大小（字符数）
            retrieval_k: 每次检索返回的相关块数量
            persist_dir: 向量库持久化目录，None 则使用内存模式
            system_prompt: 系统提示词，用于指导 LLM 的回答方式
        """
        self.embeddings = embeddings
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.persist_dir = persist_dir
        self.system_prompt = system_prompt

        self._vector_store = None
        self._retriever = None
        self._chain = None

    def build(self, file_paths: List[str]) -> "RAGPipeline":
        """
        构建知识库（加载文档 → 分割 → 嵌入 → 存入向量库）

        Args:
            file_paths: 要加载的文档路径列表

        Returns:
            self（支持链式调用）
        """
        print("=" * 50)
        print("开始构建 RAG 知识库...")

        # 第一步：加载文档
        print("\n[第1步] 加载文档...")
        documents = load_documents(file_paths)
        print(f"  ✓ 成功加载 {len(documents)} 篇文档")

        # 第二步：分割文档
        print("\n[第2步] 分割文档...")
        chunks = split_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        print(f"  ✓ 分割为 {len(chunks)} 个块")

        # 第三步：构建向量库
        print("\n[第3步] 构建向量库（生成嵌入向量）...")
        self._vector_store = build_vector_store(
            documents=chunks,
            embeddings=self.embeddings,
            persist_dir=self.persist_dir,
        )
        print("  ✓ 向量库构建完成")

        # 第四步：创建检索器和链
        self._setup_chain()

        print("\n" + "=" * 50)
        print("✅ 知识库构建完成！可以开始提问了。")
        print("=" * 50)
        return self

    def load(self) -> "RAGPipeline":
        """
        从磁盘加载已持久化的向量库（无需重新处理文档）

        Returns:
            self（支持链式调用）

        Raises:
            ValueError: 未设置 persist_dir 时抛出
        """
        if not self.persist_dir:
            raise ValueError("请在初始化时设置 persist_dir 才能加载持久化向量库")

        print(f"[加载] 从 {self.persist_dir} 加载向量库...")
        self._vector_store = load_vector_store(
            persist_dir=self.persist_dir,
            embeddings=self.embeddings,
        )
        self._setup_chain()
        print("✅ 向量库加载完成！可以开始提问了。")
        return self

    def _setup_chain(self) -> None:
        """构建 LangChain 检索链（内部方法）"""
        # 创建检索器
        self._retriever = get_retriever(
            self._vector_store,
            k=self.retrieval_k,
        )

        # 构建提示词模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{question}"),
        ])

        # 构建 RAG 链：检索 → 格式化 → 提示词 → LLM → 解析输出
        self._chain = (
            {
                "context": self._retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """
        向知识库提问并获取答案

        Args:
            question: 用户问题

        Returns:
            基于知识库内容生成的答案

        Raises:
            RuntimeError: 知识库尚未构建或加载时抛出
        """
        if self._chain is None:
            raise RuntimeError("请先调用 build() 或 load() 构建知识库")

        print(f"\n❓ 问题: {question}")
        answer = self._chain.invoke(question)
        print(f"💡 答案: {answer}")
        return answer

    def retrieve(self, query: str) -> List[Document]:
        """
        仅执行检索步骤，返回相关文档块（不生成答案）

        用于调试和理解检索效果。

        Args:
            query: 查询文本

        Returns:
            最相关的文档块列表
        """
        if self._retriever is None:
            raise RuntimeError("请先调用 build() 或 load() 构建知识库")

        docs = self._retriever.invoke(query)
        print(f"\n[检索结果] 查询: '{query}'")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知")
            print(f"  [{i}] 来源: {source}")
            print(f"      内容: {doc.page_content[:100]}...")
        return docs

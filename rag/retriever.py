"""
向量存储与检索器模块

RAG 的第四步：将文本块的向量存入数据库，并根据查询检索最相关的块。

向量数据库会存储每个文本块的：
- 向量表示（用于相似度计算）
- 原始文本内容
- 元数据（来源文件、页码等）

检索时，将查询文本转换为向量，然后找出向量空间中最近的 k 个块。
"""

import logging
import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


def build_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    persist_dir: Optional[str] = None,
    collection_name: str = "rag_demo",
) -> VectorStore:
    """
    将文档块构建为 ChromaDB 向量数据库

    Args:
        documents: 已分割的 Document 块列表
        embeddings: 嵌入模型实例
        persist_dir: 向量库持久化目录，None 则使用内存模式（重启后数据丢失）
        collection_name: 集合名称，用于区分不同的知识库

    Returns:
        ChromaDB 向量存储实例
    """
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    logger.info(
        "向量库：成功构建，共 %d 个块%s",
        len(documents),
        f"，已持久化到 {persist_dir}" if persist_dir else "（内存模式）",
    )
    return vector_store


def load_vector_store(
    persist_dir: str,
    embeddings: Embeddings,
    collection_name: str = "rag_demo",
) -> VectorStore:
    """
    从磁盘加载已持久化的向量数据库

    Args:
        persist_dir: 向量库持久化目录
        embeddings: 嵌入模型实例（必须与构建时使用的模型一致）
        collection_name: 集合名称

    Returns:
        ChromaDB 向量存储实例
    """
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    logger.info("向量库：从 %s 加载成功", persist_dir)
    return vector_store


def get_retriever(
    vector_store: VectorStore,
    k: int = 4,
    search_type: str = "similarity",
) -> object:
    """
    从向量库创建检索器

    Args:
        vector_store: 向量存储实例
        k: 每次检索返回的最相关文档块数量
        search_type: 检索类型
            - "similarity": 余弦相似度（最常用）
            - "mmr": 最大边际相关性（减少重复，增加多样性）

    Returns:
        LangChain 检索器实例
    """
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )
    logger.info("检索器：search_type=%s, k=%d", search_type, k)
    return retriever


def retrieve(
    retriever,
    query: str,
) -> List[Document]:
    """
    使用检索器查找与查询最相关的文档块

    Args:
        retriever: 检索器实例
        query: 用户查询文本

    Returns:
        最相关的 Document 块列表
    """
    docs = retriever.invoke(query)
    print(f"[检索] 查询: '{query}' → 找到 {len(docs)} 个相关块")
    return docs

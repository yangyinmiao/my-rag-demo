"""
文本分割器模块

RAG 的第二步：将长文档分割成适合嵌入和检索的小块（chunk）。

关键参数说明：
- chunk_size: 每个块的最大字符数，影响上下文完整性
- chunk_overlap: 相邻块之间的重叠字符数，保证语义连续性
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

logger = logging.getLogger(__name__)


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    使用递归字符分割器分割文档（推荐）

    RecursiveCharacterTextSplitter 会按照 ["\n\n", "\n", " ", ""] 的顺序
    尝试分割，尽量保持语义完整性。

    Args:
        documents: 待分割的 Document 列表
        chunk_size: 每个块的最大字符数（默认 500）
        chunk_overlap: 相邻块的重叠字符数（默认 50）

    Returns:
        分割后的 Document 列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,  # 在 metadata 中记录块的起始位置
    )
    chunks = splitter.split_documents(documents)
    logger.info("分割：原始文档 %d 篇 → 分割为 %d 个块", len(documents), len(chunks))
    return chunks


def split_by_character(
    documents: List[Document],
    separator: str = "\n\n",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    使用固定字符分割器分割文档

    适合有明确段落分隔符的文档，例如用空行分隔的文档。

    Args:
        documents: 待分割的 Document 列表
        separator: 分隔符（默认双换行，即段落分隔）
        chunk_size: 每个块的最大字符数
        chunk_overlap: 相邻块的重叠字符数

    Returns:
        分割后的 Document 列表
    """
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)

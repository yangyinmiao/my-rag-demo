"""
文档加载器模块

RAG 的第一步：将各种格式的文档加载为 LangChain Document 对象。
支持 TXT、PDF、Markdown、目录等多种来源。
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)


def load_text(file_path: str, encoding: str = "utf-8") -> List[Document]:
    """
    加载纯文本文件 (.txt)

    Args:
        file_path: 文件路径
        encoding: 文件编码，默认 utf-8

    Returns:
        Document 列表
    """
    loader = TextLoader(file_path, encoding=encoding)
    return loader.load()


def load_pdf(file_path: str) -> List[Document]:
    """
    加载 PDF 文件

    Args:
        file_path: PDF 文件路径

    Returns:
        Document 列表（每页一个 Document）
    """
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_markdown(file_path: str) -> List[Document]:
    """
    加载 Markdown 文件

    Args:
        file_path: Markdown 文件路径

    Returns:
        Document 列表
    """
    loader = UnstructuredMarkdownLoader(file_path)
    return loader.load()


def load_directory(
    dir_path: str,
    glob_pattern: str = "**/*.txt",
    encoding: str = "utf-8",
) -> List[Document]:
    """
    批量加载目录中的文件

    Args:
        dir_path: 目录路径
        glob_pattern: 文件匹配模式，例如 "**/*.txt", "**/*.pdf"
        encoding: 文件编码

    Returns:
        Document 列表
    """
    loader = DirectoryLoader(
        dir_path,
        glob=glob_pattern,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": encoding},
        show_progress=True,
    )
    return loader.load()


def load_documents(paths: List[str]) -> List[Document]:
    """
    根据文件扩展名自动选择加载器，批量加载多个文件

    Args:
        paths: 文件路径列表

    Returns:
        Document 列表
    """
    documents = []
    for path in paths:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            docs = load_pdf(path)
        elif ext in (".md", ".markdown"):
            docs = load_markdown(path)
        elif ext == ".txt":
            docs = load_text(path)
        else:
            # 默认尝试按文本加载
            docs = load_text(path)
        documents.extend(docs)
    return documents

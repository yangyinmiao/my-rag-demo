"""
RAG (检索增强生成) 学习 Demo

本包实现了一个完整的 RAG 流水线，包含以下核心组件：
- loader: 文档加载器
- splitter: 文本分割器
- embedder: 文本嵌入器
- retriever: 文档检索器
- generator: 答案生成器
- pipeline: 完整的 RAG 流水线
"""

from rag.pipeline import RAGPipeline

__all__ = ["RAGPipeline"]

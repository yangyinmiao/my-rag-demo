"""
文本嵌入器模块

RAG 的第三步：将文本块转换为向量表示（embeddings）。

嵌入（Embedding）是将文本映射到高维向量空间的过程，
语义相似的文本在向量空间中距离更近，这是 RAG 能够检索相关内容的基础。

支持：
- OpenAI Embeddings（在线服务）
- Ollama Embeddings（本地模型，免费）
- HuggingFace Embeddings（本地模型，需下载）
"""

import os
from typing import Optional

from langchain_core.embeddings import Embeddings


def get_openai_embeddings(
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Embeddings:
    """
    获取 OpenAI 嵌入模型

    Args:
        model: 模型名称，可选 text-embedding-3-small / text-embedding-3-large
        api_key: OpenAI API Key，默认从环境变量 OPENAI_API_KEY 读取
        base_url: API 基础 URL，使用代理时设置

    Returns:
        OpenAI 嵌入模型实例
    """
    from langchain_openai import OpenAIEmbeddings

    kwargs = {
        "model": model,
        "api_key": api_key or os.getenv("OPENAI_API_KEY"),
    }
    if base_url:
        kwargs["base_url"] = base_url
    elif os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")

    return OpenAIEmbeddings(**kwargs)


def get_ollama_embeddings(
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> Embeddings:
    """
    获取 Ollama 本地嵌入模型（免费，需要本地运行 Ollama）

    使用前请先安装并启动 Ollama：
    1. 安装：https://ollama.com/download
    2. 拉取模型：ollama pull nomic-embed-text
    3. 启动服务（通常自动启动）

    Args:
        model: 模型名称，推荐 nomic-embed-text
        base_url: Ollama 服务地址

    Returns:
        Ollama 嵌入模型实例
    """
    from langchain_community.embeddings import OllamaEmbeddings

    return OllamaEmbeddings(model=model, base_url=base_url)


def get_huggingface_embeddings(
    model_name: str = "BAAI/bge-small-zh-v1.5",
) -> Embeddings:
    """
    获取 HuggingFace 本地嵌入模型（免费，需下载模型）

    推荐中文模型：
    - BAAI/bge-small-zh-v1.5（小模型，速度快）
    - BAAI/bge-large-zh-v1.5（大模型，效果更好）

    Args:
        model_name: HuggingFace 模型名称或本地路径

    Returns:
        HuggingFace 嵌入模型实例
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)


def get_embeddings(provider: str = "openai", **kwargs) -> Embeddings:
    """
    根据提供商名称获取嵌入模型（工厂函数）

    Args:
        provider: 提供商，可选 "openai" / "ollama" / "huggingface"
        **kwargs: 传递给对应嵌入函数的参数

    Returns:
        嵌入模型实例

    Examples:
        >>> embeddings = get_embeddings("openai")
        >>> embeddings = get_embeddings("ollama", model="nomic-embed-text")
        >>> embeddings = get_embeddings("huggingface", model_name="BAAI/bge-small-zh-v1.5")
    """
    providers = {
        "openai": get_openai_embeddings,
        "ollama": get_ollama_embeddings,
        "huggingface": get_huggingface_embeddings,
    }
    if provider not in providers:
        raise ValueError(
            f"不支持的嵌入提供商: {provider}，"
            f"请选择: {list(providers.keys())}"
        )
    return providers[provider](**kwargs)

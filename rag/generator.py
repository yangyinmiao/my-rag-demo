"""
答案生成器模块

RAG 的第五步：将检索到的相关文档块与用户问题组合，让大语言模型生成回答。

这是 RAG（检索增强生成）的"生成"部分：
- 检索（Retrieval）：找到相关文档块
- 增强（Augmented）：将文档块作为上下文
- 生成（Generation）：LLM 基于上下文生成答案

支持 OpenAI 和 Ollama（本地）两种 LLM。
"""

import os
from typing import Optional

from langchain_core.language_models import BaseChatModel


# 默认的系统提示词（中文）
DEFAULT_SYSTEM_PROMPT = """你是一个专业的问答助手。请根据以下提供的参考资料来回答用户的问题。

规则：
1. 只使用参考资料中的信息来回答问题
2. 如果参考资料中没有相关信息，请直接说"根据提供的资料，我无法回答这个问题"
3. 回答要准确、简洁、有条理
4. 可以适当引用原文，但要用自己的话组织答案

参考资料：
{context}
"""


def get_openai_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseChatModel:
    """
    获取 OpenAI 语言模型

    Args:
        model: 模型名称，推荐 gpt-4o-mini（性价比高）或 gpt-4o（能力更强）
        temperature: 生成温度，0.0 表示确定性输出，1.0 表示更有创意
        api_key: OpenAI API Key，默认从环境变量 OPENAI_API_KEY 读取
        base_url: API 基础 URL，使用代理时设置

    Returns:
        OpenAI ChatModel 实例
    """
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key or os.getenv("OPENAI_API_KEY"),
    }
    if base_url:
        kwargs["base_url"] = base_url
    elif os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")

    return ChatOpenAI(**kwargs)


def get_ollama_llm(
    model: str = "qwen2.5",
    temperature: float = 0.0,
    base_url: str = "http://localhost:11434",
) -> BaseChatModel:
    """
    获取 Ollama 本地语言模型（免费，无需 API Key）

    使用前请先安装并启动 Ollama，然后拉取模型：
    - ollama pull qwen2.5      （通义千问，中文能力强）
    - ollama pull llama3.2     （Meta Llama，英文能力强）

    Args:
        model: 模型名称
        temperature: 生成温度
        base_url: Ollama 服务地址

    Returns:
        Ollama ChatModel 实例
    """
    from langchain_community.llms import Ollama

    return Ollama(model=model, temperature=temperature, base_url=base_url)


def get_llm(provider: str = "openai", **kwargs) -> BaseChatModel:
    """
    根据提供商名称获取语言模型（工厂函数）

    Args:
        provider: 提供商，可选 "openai" / "ollama"
        **kwargs: 传递给对应模型函数的参数

    Returns:
        语言模型实例

    Examples:
        >>> llm = get_llm("openai", model="gpt-4o-mini")
        >>> llm = get_llm("ollama", model="qwen2.5")
    """
    providers = {
        "openai": get_openai_llm,
        "ollama": get_ollama_llm,
    }
    if provider not in providers:
        raise ValueError(
            f"不支持的 LLM 提供商: {provider}，"
            f"请选择: {list(providers.keys())}"
        )
    return providers[provider](**kwargs)

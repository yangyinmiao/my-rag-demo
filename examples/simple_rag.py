"""
简单 RAG 示例

演示最基础的 RAG 流程：加载文档 → 构建知识库 → 提问
运行前请确保已设置 OPENAI_API_KEY 环境变量

使用方法：
  1. 复制 .env.example 为 .env 并填写 API Key
  2. 运行：python examples/simple_rag.py
"""

import os
import sys

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

from rag import RAGPipeline
from rag.embedder import get_embeddings
from rag.generator import get_llm


def main():
    print("=" * 60)
    print("RAG 学习 Demo - 简单示例")
    print("=" * 60)

    # ========== 配置 ==========
    # 方式一：使用 OpenAI（需要 API Key）
    embeddings = get_embeddings("openai")
    llm = get_llm("openai", model="gpt-4o-mini")

    # 方式二：使用 Ollama 本地模型（免费，需要本地运行 Ollama）
    # embeddings = get_embeddings("ollama", model="nomic-embed-text")
    # llm = get_llm("ollama", model="qwen2.5")

    # ========== 构建知识库 ==========
    # 指定要加载的文档
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    files = [
        os.path.join(data_dir, "rag_introduction.txt"),
        os.path.join(data_dir, "langchain_intro.txt"),
    ]

    # 初始化 RAG 流水线
    pipeline = RAGPipeline(
        embeddings=embeddings,
        llm=llm,
        chunk_size=400,    # 每个文本块最大 400 字符
        chunk_overlap=50,  # 相邻块重叠 50 字符
        retrieval_k=3,     # 检索最相关的 3 个块
    )

    # 构建知识库
    pipeline.build(files)

    # ========== 提问 ==========
    questions = [
        "什么是 RAG？它解决了什么问题？",
        "RAG 的工作原理是什么？请分步骤说明",
        "LangChain 是什么？有哪些核心概念？",
        "RAG 和微调（Fine-tuning）有什么区别？",
    ]

    print("\n" + "=" * 60)
    print("开始问答...")
    print("=" * 60)

    for question in questions:
        print(f"\n{'─' * 50}")
        answer = pipeline.ask(question)
        print(f"{'─' * 50}")

    print("\n✅ 示例运行完成！")


if __name__ == "__main__":
    main()

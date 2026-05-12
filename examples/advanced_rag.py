"""
进阶 RAG 示例

演示以下进阶功能：
1. 持久化向量库（避免重复嵌入）
2. 调试模式（查看检索到的原始文档块）
3. 自定义提示词
4. 流式输出

使用方法：
  python examples/advanced_rag.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from rag import RAGPipeline
from rag.embedder import get_embeddings
from rag.generator import get_llm


# ========== 自定义提示词 ==========
CUSTOM_PROMPT = """你是一位 AI 技术专家，擅长用通俗易懂的语言解释复杂概念。
请基于以下参考资料回答问题，回答要：
- 结构清晰，使用数字列表或小标题
- 语言简洁，避免不必要的重复
- 如果资料中没有相关信息，直接说明

参考资料：
{context}
"""


def demo_basic(pipeline: RAGPipeline) -> None:
    """演示基础问答"""
    print("\n" + "━" * 60)
    print("📌 演示1：基础问答")
    print("━" * 60)

    pipeline.ask("RAG 系统由哪些核心组件构成？")


def demo_retrieval_debug(pipeline: RAGPipeline) -> None:
    """演示检索调试：查看检索到的原始文档块"""
    print("\n" + "━" * 60)
    print("📌 演示2：检索调试（查看原始文档块）")
    print("━" * 60)

    query = "向量数据库有哪些选择？"
    print(f"查询: {query}")
    docs = pipeline.retrieve(query)

    print(f"\n检索到 {len(docs)} 个相关文档块：")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- 块 {i} ---")
        print(f"来源: {doc.metadata.get('source', '未知')}")
        print(f"内容:\n{doc.page_content}")


def demo_persistent_store() -> None:
    """演示持久化向量库：构建一次，多次使用"""
    print("\n" + "━" * 60)
    print("📌 演示3：持久化向量库")
    print("━" * 60)

    persist_dir = "/tmp/rag_demo_chroma"
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    files = [os.path.join(data_dir, "rag_introduction.txt")]

    embeddings = get_embeddings("openai")
    llm = get_llm("openai", model="gpt-4o-mini")

    if os.path.exists(persist_dir):
        # 向量库已存在，直接加载
        print(f"发现已有向量库，直接加载（跳过嵌入步骤）...")
        pipeline = RAGPipeline(
            embeddings=embeddings,
            llm=llm,
            persist_dir=persist_dir,
        )
        pipeline.load()
    else:
        # 首次构建并持久化
        print(f"首次构建，将持久化到 {persist_dir} ...")
        pipeline = RAGPipeline(
            embeddings=embeddings,
            llm=llm,
            persist_dir=persist_dir,
        )
        pipeline.build(files)

    pipeline.ask("RAG 的知识库构建分哪几步？")


def main():
    print("=" * 60)
    print("RAG 学习 Demo - 进阶示例")
    print("=" * 60)

    # 初始化基础 pipeline
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    files = [
        os.path.join(data_dir, "rag_introduction.txt"),
        os.path.join(data_dir, "langchain_intro.txt"),
    ]

    embeddings = get_embeddings("openai")
    llm = get_llm("openai", model="gpt-4o-mini")

    pipeline = RAGPipeline(
        embeddings=embeddings,
        llm=llm,
        chunk_size=400,
        chunk_overlap=50,
        retrieval_k=3,
        system_prompt=CUSTOM_PROMPT,  # 使用自定义提示词
    )
    pipeline.build(files)

    # 运行各演示
    demo_basic(pipeline)
    demo_retrieval_debug(pipeline)
    demo_persistent_store()

    print("\n✅ 进阶示例运行完成！")


if __name__ == "__main__":
    main()

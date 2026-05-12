"""
RAG 单元测试

使用 Mock 对象测试各模块功能，无需真实的 API Key 或模型。
运行方式：
  pytest tests/ -v
"""

import os
import sys
from unittest.mock import MagicMock, patch, mock_open

import pytest

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document


# ========== 测试数据 ==========

SAMPLE_DOCS = [
    Document(
        page_content="RAG 是检索增强生成技术，结合了信息检索和语言生成。",
        metadata={"source": "test.txt"},
    ),
    Document(
        page_content="LangChain 是一个用于构建 LLM 应用的开源框架。",
        metadata={"source": "test.txt"},
    ),
    Document(
        page_content="向量数据库用于存储和检索高维向量，是 RAG 的核心组件之一。",
        metadata={"source": "test.txt"},
    ),
]


# ========== 测试文本分割器 ==========

class TestSplitter:
    """测试文本分割模块"""

    def test_split_documents_returns_chunks(self):
        """分割后应返回文档块列表"""
        from rag.splitter import split_documents

        # 使用较长文档测试分割
        long_doc = Document(
            page_content="A" * 1200,
            metadata={"source": "test.txt"},
        )
        chunks = split_documents([long_doc], chunk_size=500, chunk_overlap=50)

        assert len(chunks) > 1  # 应被分割为多块

    def test_split_preserves_metadata(self):
        """分割后的块应保留原始元数据"""
        from rag.splitter import split_documents

        doc = Document(
            page_content="这是一段测试文本。" * 50,
            metadata={"source": "my_file.txt", "page": 1},
        )
        chunks = split_documents([doc], chunk_size=100, chunk_overlap=10)

        assert all("source" in chunk.metadata for chunk in chunks)
        assert all(chunk.metadata["source"] == "my_file.txt" for chunk in chunks)

    def test_split_short_document_returns_single_chunk(self):
        """短文档不应被分割"""
        from rag.splitter import split_documents

        short_doc = Document(
            page_content="这是一段很短的文本。",
            metadata={"source": "test.txt"},
        )
        chunks = split_documents([short_doc], chunk_size=500, chunk_overlap=50)

        assert len(chunks) == 1
        assert chunks[0].page_content == short_doc.page_content

    def test_split_empty_document_list(self):
        """空文档列表应返回空列表"""
        from rag.splitter import split_documents

        chunks = split_documents([], chunk_size=500, chunk_overlap=50)
        assert chunks == []

    def test_split_multiple_documents(self):
        """多篇文档应全部被处理"""
        from rag.splitter import split_documents

        chunks = split_documents(SAMPLE_DOCS, chunk_size=50, chunk_overlap=5)
        # 每篇文档至少生成一个块
        assert len(chunks) >= len(SAMPLE_DOCS)


# ========== 测试加载器 ==========

class TestLoader:
    """测试文档加载模块"""

    def test_load_text_file(self, tmp_path):
        """应能正确加载文本文件"""
        from rag.loader import load_text

        # 创建临时文本文件
        test_file = tmp_path / "test.txt"
        test_file.write_text("这是测试内容\n第二行内容", encoding="utf-8")

        docs = load_text(str(test_file))

        assert len(docs) == 1
        assert "这是测试内容" in docs[0].page_content

    def test_load_documents_auto_detect_txt(self, tmp_path):
        """load_documents 应能自动识别 txt 文件"""
        from rag.loader import load_documents

        test_file = tmp_path / "sample.txt"
        test_file.write_text("测试内容", encoding="utf-8")

        docs = load_documents([str(test_file)])
        assert len(docs) >= 1

    def test_load_documents_multiple_files(self, tmp_path):
        """应能批量加载多个文件"""
        from rag.loader import load_documents

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("文件1内容", encoding="utf-8")
        file2.write_text("文件2内容", encoding="utf-8")

        docs = load_documents([str(file1), str(file2)])
        assert len(docs) == 2


# ========== 测试嵌入器 ==========

class TestEmbedder:
    """测试嵌入模型模块"""

    def test_get_embeddings_invalid_provider(self):
        """不支持的提供商应抛出 ValueError"""
        from rag.embedder import get_embeddings

        with pytest.raises(ValueError, match="不支持的嵌入提供商"):
            get_embeddings("invalid_provider")

    def test_get_embeddings_openai_returns_instance(self):
        """get_embeddings('openai') 应返回 OpenAIEmbeddings 实例"""
        from rag.embedder import get_embeddings
        from langchain_openai import OpenAIEmbeddings

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            embeddings = get_embeddings("openai")
        assert isinstance(embeddings, OpenAIEmbeddings)

    def test_get_embeddings_openai_custom_model(self):
        """应能传入自定义模型名"""
        from rag.embedder import get_embeddings
        from langchain_openai import OpenAIEmbeddings

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            embeddings = get_embeddings("openai", model="text-embedding-3-large")
        assert isinstance(embeddings, OpenAIEmbeddings)
        assert embeddings.model == "text-embedding-3-large"


# ========== 测试生成器 ==========

class TestGenerator:
    """测试 LLM 生成器模块"""

    def test_get_llm_invalid_provider(self):
        """不支持的提供商应抛出 ValueError"""
        from rag.generator import get_llm

        with pytest.raises(ValueError, match="不支持的 LLM 提供商"):
            get_llm("invalid_provider")

    def test_get_openai_llm_returns_instance(self):
        """get_llm('openai') 应返回 ChatOpenAI 实例"""
        from rag.generator import get_llm
        from langchain_openai import ChatOpenAI

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            llm = get_llm("openai")
        assert isinstance(llm, ChatOpenAI)

    def test_get_openai_llm_custom_model(self):
        """应能传入自定义模型名"""
        from rag.generator import get_llm
        from langchain_openai import ChatOpenAI

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            llm = get_llm("openai", model="gpt-4o")
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-4o"


# ========== 测试 RAG Pipeline ==========

class TestRAGPipeline:
    """测试 RAG 完整流水线"""

    def _make_pipeline(self):
        """创建带 Mock 依赖的 RAGPipeline"""
        from rag.pipeline import RAGPipeline

        mock_embeddings = MagicMock()
        mock_llm = MagicMock()
        return RAGPipeline(
            embeddings=mock_embeddings,
            llm=mock_llm,
            chunk_size=200,
            chunk_overlap=20,
            retrieval_k=2,
        )

    def test_ask_before_build_raises_error(self):
        """未构建知识库时调用 ask 应抛出 RuntimeError"""
        pipeline = self._make_pipeline()
        with pytest.raises(RuntimeError, match="请先调用 build"):
            pipeline.ask("测试问题")

    def test_retrieve_before_build_raises_error(self):
        """未构建知识库时调用 retrieve 应抛出 RuntimeError"""
        pipeline = self._make_pipeline()
        with pytest.raises(RuntimeError, match="请先调用 build"):
            pipeline.retrieve("测试查询")

    def test_load_without_persist_dir_raises_error(self):
        """未设置 persist_dir 时调用 load 应抛出 ValueError"""
        pipeline = self._make_pipeline()
        with pytest.raises(ValueError, match="请在初始化时设置 persist_dir"):
            pipeline.load()

    def test_build_and_ask(self, tmp_path):
        """构建知识库后应能成功提问"""
        from rag.pipeline import RAGPipeline

        # 创建临时文档
        doc_file = tmp_path / "test.txt"
        doc_file.write_text(
            "RAG 是检索增强生成技术，" * 20,  # 足够长以产生多个块
            encoding="utf-8",
        )

        # Mock 嵌入模型：返回固定维度向量（数量与输入文本数匹配）
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
        mock_embeddings.embed_query.return_value = [0.1] * 1536

        # Mock LLM：返回固定答案
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "RAG 是检索增强生成技术。"
        mock_llm.invoke.return_value = mock_response

        # Mock 向量库和检索器
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = SAMPLE_DOCS[:2]
        mock_vector_store.as_retriever.return_value = mock_retriever

        with patch("rag.retriever.build_vector_store", return_value=mock_vector_store):
            pipeline = RAGPipeline(
                embeddings=mock_embeddings,
                llm=mock_llm,
                chunk_size=100,
                chunk_overlap=10,
                retrieval_k=2,
            )
            pipeline.build([str(doc_file)])

        # 验证链已建立
        assert pipeline._chain is not None

    def test_pipeline_init_defaults(self):
        """验证初始化时的默认参数"""
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(
            embeddings=MagicMock(),
            llm=MagicMock(),
        )
        assert pipeline.chunk_size == 500
        assert pipeline.chunk_overlap == 50
        assert pipeline.retrieval_k == 4
        assert pipeline.persist_dir is None
        assert pipeline._chain is None

    def test_format_docs(self):
        """验证文档格式化函数"""
        from rag.pipeline import _format_docs

        docs = [
            Document(page_content="内容1", metadata={"source": "file1.txt"}),
            Document(page_content="内容2", metadata={"source": "file2.txt"}),
        ]
        result = _format_docs(docs)

        assert "内容1" in result
        assert "内容2" in result
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "---" in result  # 分隔符

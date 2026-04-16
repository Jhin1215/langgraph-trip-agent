
import re
from pathlib import Path


from langchain_core.tools import tool

from deep_agent.config import PROJECT_ROOT
from deep_agent.embeddings import QWenEmbeddings
from deep_agent.env_util import QWEN_API_KEY, QWEN_BASE_URL
from deep_agent.retrievers import VectorStoreRetriever

FAQ_PATH = PROJECT_ROOT / "assets" / "faq" / "order_faq.md"


def _load_faq_docs(faq_path: Path) -> list[dict]:
    """
    读取 FAQ 文档中的内容，并按二级标题切分成多个文档快
    Args:
        faq_path: FAQ文档的路径

    Returns:
        多个文档快构成的字典列表
    """
    with faq_path.open('r', encoding='utf-8') as f:
        faq_text = f.read()
    chunks = re.split(r"(?=\n##)", faq_text)
    return [{'page_content': chunk.strip()} for chunk in chunks if chunk.strip()]


faq_docs = _load_faq_docs(FAQ_PATH)
qwen_embedding_model = QWenEmbeddings(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    model_name="text-embedding-v4",
    dim=1024,
)
retriever = VectorStoreRetriever.from_docs(faq_docs, qwen_embedding_model)


@tool
def lookup_policy(query: str) -> str:
    """
    查询航班政策 FAQ。
    在进行改签、取消、退票等操作前，可先调用该工具检索相关政策。
    """
    docs = retriever.query(query, qwen_embedding_model, k=2)
    return "\n\n".join(doc["page_content"] for doc in docs)

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI


class MyEembeddings(Protocol):
    def embed_documents(self, docs: list[str]) -> NDArray[np.float32]:
        ...

    def embed_text(self, text: str) -> NDArray[np.float32]:
        ...


class QWenEmbeddings:
    def __init__(
            self,
            api_key: str,
            base_url: str,
            model_name: str = "text-embedding-v4",
            dim: int = 1024
    ):
        self.model_name = model_name
        self.dim = dim
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def _embed(self, inputs: list[str]) -> NDArray[np.float32]:
        # 通过 client 调用模型
        resp = self.client.embeddings.create(
            model=self.model_name,
            input=inputs,
            dimensions=self.dim,
            encoding_format="float"
        )
        # OpenAI Python SDK 中的 Json 格式使用 . 访问
        vectors = np.asarray([item.embedding for item in resp.data], dtype=np.float32)
        # 显示归一化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # 放值除 0 错误
        vectors = vectors / np.clip(norms, 1e-12, None)
        return vectors

    def embed_documents(self, docs: list[str]) -> NDArray[np.float32]:
        return self._embed(docs)

    def embed_text(self, text: str) -> NDArray[np.float32]:
        return self._embed([text])[0]


# 本地部署了 embedding 分词模型的写法
# embedding_model = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-zh-v1.5",
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": True},
# )

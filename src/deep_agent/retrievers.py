import os
import numpy as np


class VectorStoreRetriever(object):
    """
    一个轻量本地向量检索器：
    - 初始化时将文档转成向量
    - query 时计算余弦相似度（这里依赖 embedding 已归一化，可直接点积）
    """

    def __init__(self, docs: list[dict], vectors: list[list[float]]):
        self.docs = docs
        self.arr = np.array(vectors, dtype=np.float32)

    @classmethod
    def from_docs(cls, docs: list[dict], embedding_model) -> 'VectorStoreRetriever':
        texts = [doc["page_content"] for doc in docs]
        vectors = embedding_model.embed_documents(texts)
        # cls() 就是调用 __init__ 方法
        # cls(docs, vetcotrs) == VectorStoreRetriever(docs, vectors)
        return cls(docs, vectors)

    def query(self, query: str, embedding_model, k: int = 3) -> list[dict]:
        if not self.docs:
            return []
        embed = np.array(embedding_model.embed_text(query), dtype=np.float32)
        # 计算余弦相似度，得到查询文本和每一个文本块的相似度，整个列表是一个归一化的结果
        scores = embed @ self.arr.T
        # argpartition 返回的升序排序的索引数组(和scores等长)，-k: 是取置信度最大的 k 个索引
        topk_idx = np.argpartition(scores, -k)[-k:]
        # scores[topk_idx] 根据最大的 k 个索引取元素
        #  np.argsort() 返回升序排序的索引，传入的是加入负号的，
        #  所以经过 np.argsort()得到的最小的索引就是 k 个元素里面数值最大的元素对应的索引
        top_k_idx_sorted = topk_idx[np.argsort(-scores[topk_idx])]

        return [
            # ** 是对字典解包。**{'a': 1} = 'a': 1
            {**self.docs[idx], "similarity": float(scores[idx])}
            for idx in top_k_idx_sorted
        ]

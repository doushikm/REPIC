import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

def tokenize(text):
    """Split code into searchable words. Keeps whole identifiers AND their parts."""
    tokens = []
    for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        tokens.append(ident.lower())                       # raise_for_status
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", ident)  # camelCase -> camel Case
        parts = [p for p in re.split(r"[_\s]+", spaced.lower()) if p]
        if len(parts) > 1:
            tokens.extend(parts)                           # raise, for, status
    return tokens

class Retriever:
    def __init__(self, vector_store):
        self.vs = vector_store
        data = vector_store.get(include=["documents", "metadatas"])
        self.texts = data["documents"]
        self.metas = data["metadatas"]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.reranker = None

    def _vector(self, q, n):
        return self.vs.similarity_search(q, k=n)

    def _bm25(self, q, n):
        scores = self.bm25.get_scores(tokenize(q))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [Document(page_content=self.texts[i], metadata=self.metas[i]) for i in top]

    def search(self, q, k=4, mode="vector", boost=0.0, rerank=False, candidates=20):
        if mode == "vector":
            docs = self._vector(q, max(k, candidates if rerank else k))
        elif mode == "bm25":
            docs = self._bm25(q, max(k, candidates if rerank else k))
        else:  # hybrid: Reciprocal Rank Fusion
            scores, by_key = {}, {}
            for results in (self._vector(q, candidates), self._bm25(q, candidates)):
                for rank, d in enumerate(results):
                    key = d.page_content
                    by_key[key] = d
                    scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            if boost:
                for key, d in by_key.items():
                    if d.metadata["source"].endswith(".py"):
                        scores[key] += boost
            ordered = sorted(scores, key=scores.get, reverse=True)
            docs = [by_key[key] for key in ordered]

        if rerank:
            if self.reranker is None:
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pool = docs[:10]
            pair_scores = self.reranker.predict([(q, d.page_content) for d in pool])
            pool = [d for _, d in sorted(zip(pair_scores, pool), key=lambda x: x[0], reverse=True)]
            return pool[:k]

        return docs[:k]
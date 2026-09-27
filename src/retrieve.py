"""
Retrieval Engine with Hybrid Search (Dense Cosine Similarity + BM25 Keyword Search)
and Reciprocal Rank Fusion / Weighted Scoring.
"""

import re
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi
from src.embed_store import VectorStore


def tokenize_hindi(text: str) -> List[str]:
    """
    Tokenizes Hindi/multilingual text for BM25 keyword matching.
    Removes punctuation while keeping words and alphanumeric tokens.
    """
    tokens = re.findall(r"[\w\u0900-\u097F]+", text.lower())
    return [t for t in tokens if len(t) > 1]


class HybridRetriever:
    """
    Hybrid Retriever integrating Dense Vector Search (ChromaDB) and Sparse BM25 Search.
    Supports reciprocal rank fusion (RRF) and metadata filtering.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        chunks: Optional[List[Dict[str, Any]]] = None,
        alpha: float = 0.7,
    ):
        self.vector_store = vector_store
        self.alpha = alpha
        self.chunks = chunks or []
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunk_lookup: Dict[int, Dict[str, Any]] = {}

        if self.chunks:
            self._build_bm25_index(self.chunks)

    def _build_bm25_index(self, chunks: List[Dict[str, Any]]):
        """Builds in-memory BM25 index over all document chunks."""
        self.chunks = chunks
        self.chunk_lookup = {c["chunk_id"]: c for c in chunks}
        corpus_tokens = [tokenize_hindi(c["text"]) for c in chunks]
        cleaned_corpus = [tokens if tokens else ["unk"] for tokens in corpus_tokens]
        self.bm25_index = BM25Okapi(cleaned_corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        chunk_type_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes retrieval for a given query (Hindi or English).
        Combines Dense embeddings with BM25 keyword matching if enabled.
        """
        dense_results = self.vector_store.query(query_text=query, top_k=max(top_k * 2, 10))

        if not use_hybrid or not self.bm25_index:
            filtered = self._apply_filters(dense_results, chunk_type_filter, page_filter)
            return filtered[:top_k]

        query_tokens = tokenize_hindi(query)
        bm25_scores = {}
        if query_tokens:
            raw_bm25_scores = self.bm25_index.get_scores(query_tokens)
            max_bm25 = max(raw_bm25_scores) if max(raw_bm25_scores) > 0 else 1.0
            for idx, raw_score in enumerate(raw_bm25_scores):
                cid = self.chunks[idx]["chunk_id"]
                bm25_scores[cid] = raw_score / max_bm25

        combined_scores: Dict[int, Dict[str, Any]] = {}

        for r in dense_results:
            cid = r["chunk_id"]
            dense_s = r["score"]
            sparse_s = bm25_scores.get(cid, 0.0)
            hybrid_score = (self.alpha * dense_s) + ((1.0 - self.alpha) * sparse_s)

            combined_scores[cid] = {
                **r,
                "score": round(hybrid_score, 4),
                "dense_score": dense_s,
                "sparse_score": round(sparse_s, 4),
            }

        sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for cid, sparse_s in sorted_bm25:
            if cid not in combined_scores and sparse_s > 0.1:
                chunk_data = self.chunk_lookup.get(cid)
                if chunk_data:
                    hybrid_score = (1.0 - self.alpha) * sparse_s
                    combined_scores[cid] = {
                        "chunk_id": cid,
                        "page_number": chunk_data["page_number"],
                        "section_heading": chunk_data["section_heading"],
                        "chunk_type": chunk_data["chunk_type"],
                        "text": chunk_data["text"],
                        "score": round(hybrid_score, 4),
                        "dense_score": 0.0,
                        "sparse_score": round(sparse_s, 4),
                        "metadata": chunk_data,
                    }

        ranked_results = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)
        filtered = self._apply_filters(ranked_results, chunk_type_filter, page_filter)
        return filtered[:top_k]

    def _apply_filters(
        self,
        results: List[Dict[str, Any]],
        chunk_type_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        filtered = []
        for r in results:
            if chunk_type_filter and r.get("chunk_type") != chunk_type_filter:
                continue
            if page_filter is not None and r.get("page_number") != page_filter:
                continue
            filtered.append(r)
        return filtered

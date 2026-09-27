"""
Multilingual Embedding and ChromaDB Vector Store Module.
Handles vector generation using multilingual embeddings (intfloat/multilingual-e5-base),
persists vectors with comprehensive metadata, and executes cosine similarity search.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


import threading
import torch

try:
    torch.set_num_threads(1)
except Exception:
    pass

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"
DEFAULT_COLLECTION_NAME = "hindi_rag_kalam"
DEFAULT_PERSIST_DIR = "data/chroma_db"

_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_MODEL_LOCK = threading.Lock()


class EmbeddingManager:
    """
    Manages multilingual sentence embedding generation.
    Formats query and passage prefixes specifically optimized for multilingual-e5 models.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.is_e5 = "e5" in model_name.lower()

    @property
    def model(self) -> SentenceTransformer:
        with _MODEL_LOCK:
            if self.model_name not in _MODEL_CACHE:
                _MODEL_CACHE[self.model_name] = SentenceTransformer(
                    self.model_name,
                    model_kwargs={"use_safetensors": False},
                )
            return _MODEL_CACHE[self.model_name]

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """Embeds document chunks (prepending 'passage: ' if using e5 models)."""
        formatted = [f"passage: {t}" if self.is_e5 else t for t in texts]
        embeddings = self.model.encode(formatted, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embeds search query (prepending 'query: ' if using e5 models)."""
        formatted = f"query: {query}" if self.is_e5 else query
        embedding = self.model.encode([formatted], normalize_embeddings=True, show_progress_bar=False)
        return embedding[0].tolist()


class VectorStore:
    """
    Encapsulates ChromaDB persistent client and operations.
    Maintains collections with cosine similarity and rich chunk metadata.
    """

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embed_manager: Optional[EmbeddingManager] = None,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embed_manager = embed_manager or EmbeddingManager()

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64) -> int:
        """
        Computes embeddings and saves chunks with full metadata to ChromaDB.
        """
        if not chunks:
            return 0

        # Existing count
        existing = self.collection.get()
        existing_ids = set(existing.get("ids", []))

        # Filter new or replace
        ids_to_add = []
        texts_to_add = []
        metadatas_to_add = []

        for c in chunks:
            cid = str(c["chunk_id"])
            ids_to_add.append(cid)
            texts_to_add.append(c["text"])
            metadatas_to_add.append({
                "chunk_id": int(c["chunk_id"]),
                "page_number": int(c["page_number"]),
                "section_heading": str(c["section_heading"]),
                "chunk_type": str(c.get("chunk_type", "prose")),
                "char_start": int(c.get("char_start", 0)),
                "char_end": int(c.get("char_end", len(c["text"]))),
                "token_count": int(c.get("token_count", 0)),
                "source_doc": str(c.get("source_doc", "")),
            })

        # Process in batches
        for i in range(0, len(ids_to_add), batch_size):
            b_ids = ids_to_add[i : i + batch_size]
            b_texts = texts_to_add[i : i + batch_size]
            b_meta = metadatas_to_add[i : i + batch_size]

            b_embeddings = self.embed_manager.embed_passages(b_texts)

            # Upsert into ChromaDB
            self.collection.upsert(
                ids=b_ids,
                documents=b_texts,
                embeddings=b_embeddings,
                metadatas=b_meta,
            )

        return len(ids_to_add)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search against the vector store.
        Returns top_k items with documents, metadata, and similarity scores.
        """
        query_vec = self.embed_manager.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for doc, meta, dist, cid in zip(docs, metas, distances, ids):
                # Cosine distance to similarity score: 1 - cosine_distance
                score = round(max(0.0, min(1.0, 1.0 - float(dist))), 4)
                formatted_results.append({
                    "chunk_id": meta.get("chunk_id", int(cid) if cid.isdigit() else cid),
                    "page_number": meta.get("page_number"),
                    "section_heading": meta.get("section_heading"),
                    "chunk_type": meta.get("chunk_type"),
                    "text": doc,
                    "score": score,
                    "distance": float(dist),
                    "metadata": meta,
                })

        return formatted_results

    def reset(self):
        """Clears the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

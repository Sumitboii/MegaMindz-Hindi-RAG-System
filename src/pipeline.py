"""
End-to-End Hindi RAG Pipeline.
Coordinates Ingestion, Chunking, Multilingual Embedding, ChromaDB Storage,
Hybrid Retrieval, Grounded Generation, and Programmatic Citations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from src.chunk import chunk_document
from src.embed_store import EmbeddingManager, VectorStore, DEFAULT_MODEL_NAME, DEFAULT_PERSIST_DIR
from src.generate import LLMClient, generate_grounded_answer
from src.ingest import extract_pdf_document
from src.retrieve import HybridRetriever


class HindiRAGPipeline:
    """
    Complete Hindi RAG Pipeline instance.
    """

    def __init__(
        self,
        pdf_path: str = "data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf",
        persist_dir: str = DEFAULT_PERSIST_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        llm_provider: str = "auto",
        llm_model: Optional[str] = None,
        force_reindex: bool = False,
    ):
        self.pdf_path = Path(pdf_path)
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.embed_manager = EmbeddingManager(model_name=model_name)
        self.vector_store = VectorStore(
            persist_directory=self.persist_dir,
            embed_manager=self.embed_manager,
        )
        self.llm_client = LLMClient(provider=llm_provider, model_name=llm_model)
        self.chunks: List[Dict[str, Any]] = []
        self.retriever: Optional[HybridRetriever] = None

        self._initialize(force_reindex=force_reindex)

    def _initialize(self, force_reindex: bool = False):
        """Loads and indexes the document if not already indexed."""
        needs_indexing = force_reindex or (self.vector_store.count() == 0)

        # Always extract chunks to support BM25 and exact text lookup
        if self.pdf_path.exists():
            pages_data = extract_pdf_document(self.pdf_path)
            self.chunks = chunk_document(
                pages_data=pages_data,
                source_doc_name=self.pdf_path.name,
            )
        else:
            print(f"Warning: PDF file not found at {self.pdf_path}")
            self.chunks = []

        if needs_indexing and self.chunks:
            if force_reindex:
                self.vector_store.reset()
            self.vector_store.add_chunks(self.chunks)

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            chunks=self.chunks,
            alpha=0.75,
        )

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        chunk_type_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end question answering pipeline with citations.
        """
        if not self.retriever:
            raise RuntimeError("Pipeline retriever is not initialized.")

        # 1. Retrieve most relevant chunks
        retrieved_chunks = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            use_hybrid=use_hybrid,
            chunk_type_filter=chunk_type_filter,
            page_filter=page_filter,
        )

        # 2. Generate grounded response with citations
        result = generate_grounded_answer(
            query=query,
            retrieved_chunks=retrieved_chunks,
            llm_client=self.llm_client,
        )

        return result

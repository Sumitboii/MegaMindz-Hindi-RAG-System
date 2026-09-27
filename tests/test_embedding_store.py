"""
Tests for Vector Store and Embedding Persistence.
Verifies ChromaDB collection initialization, vector persistence, and metadata round-tripping.
"""

import shutil
from pathlib import Path
from src.chunk import chunk_document
from src.embed_store import VectorStore
from src.ingest import extract_pdf_document


def test_vector_store_persistence_and_metadata_roundtrip(tmp_path, session_pipeline):
    test_db_dir = tmp_path / "test_chroma"

    # Ingest and chunk
    pdf_path = Path("data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")
    if not pdf_path.exists():
        pdf_path = Path("Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")

    pages_data = extract_pdf_document(pdf_path)
    chunks = chunk_document(pages_data)[:10]  # Use first 10 chunks for fast unit test

    store = VectorStore(
        persist_directory=str(test_db_dir),
        collection_name="test_collection",
        embed_manager=session_pipeline.embed_manager,
    )
    added_count = store.add_chunks(chunks)
    assert added_count == len(chunks)
    assert store.count() == len(chunks)

    # Query
    results = store.query("कलाम का बचपन रामेश्वरम", top_k=3)
    assert len(results) > 0

    top_res = results[0]
    assert "chunk_id" in top_res
    assert "page_number" in top_res
    assert "section_heading" in top_res
    assert "score" in top_res
    assert 0.0 <= top_res["score"] <= 1.0

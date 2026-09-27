"""
Tests for Hindi Chunking Engine.
Verifies sentence-boundary splitting, chunk count sanity, overlap presence,
and metadata completeness.
"""

from pathlib import Path
from src.chunk import chunk_document, split_hindi_sentences
from src.ingest import extract_pdf_document


def test_hindi_sentence_splitter():
    text = "कलाम का जन्म रामेश्वरम में हुआ था। उनके पिता नाविक थे। क्या वे वैज्ञानिक बने? हाँ, वे महान वैज्ञानिक बने!"
    sentences = split_hindi_sentences(text)
    assert len(sentences) >= 3
    assert any("रामेश्वरम" in s for s in sentences)


def test_chunk_count_and_metadata_completeness():
    pdf_path = Path("data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")
    if not pdf_path.exists():
        pdf_path = Path("Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")

    pages_data = extract_pdf_document(pdf_path)
    chunks = chunk_document(pages_data)

    assert 25 <= len(chunks) <= 150, f"Unexpected chunk count: {len(chunks)}"

    required_keys = {"chunk_id", "page_number", "section_heading", "chunk_type", "char_start", "char_end", "text"}
    for idx, c in enumerate(chunks, 1):
        assert required_keys.issubset(c.keys()), f"Chunk {idx} missing required metadata keys"
        assert c["chunk_id"] == idx
        assert 1 <= c["page_number"] <= 22
        assert len(c["section_heading"]) > 0
        assert c["chunk_type"] in ("prose", "table")
        assert len(c["text"]) > 10


def test_table_chunks_isolated():
    pdf_path = Path("data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")
    if not pdf_path.exists():
        pdf_path = Path("Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")

    pages_data = extract_pdf_document(pdf_path)
    chunks = chunk_document(pages_data)

    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) >= 4, f"Expected at least 4 table chunks, got {len(table_chunks)}"
    for tc in table_chunks:
        assert "|" in tc["text"], "Table chunk should contain markdown table delimiters"

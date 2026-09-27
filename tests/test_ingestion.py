"""
Tests for PDF Ingestion and Table Extraction.
Verifies PDF loading, page count integrity, and structured table extraction.
"""

from pathlib import Path
import pytest
from src.ingest import extract_pdf_document, extract_tables_from_page


def test_pdf_loads_and_no_pages_dropped():
    pdf_path = Path("data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")
    if not pdf_path.exists():
        pdf_path = Path("Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")

    assert pdf_path.exists(), f"PDF document not found at {pdf_path}"

    pages_data = extract_pdf_document(pdf_path)
    # The document has 22 pages
    assert len(pages_data) == 22, f"Expected 22 pages, but found {len(pages_data)}"

    # Ensure each page has non-empty text and correct 1-indexed page_number
    for idx, p in enumerate(pages_data, 1):
        assert p["page_number"] == idx
        assert len(p["text"]) > 0
        assert p["section_heading"] is not None


def test_both_tables_extracted():
    pdf_path = Path("data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")
    if not pdf_path.exists():
        pdf_path = Path("Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")

    pages_data = extract_pdf_document(pdf_path)

    # Awards table is on page 16
    page_16 = next(p for p in pages_data if p["page_number"] == 16)
    assert len(page_16["tables"]) >= 1, "Page 16 awards table not extracted"
    awards_table = page_16["tables"][0]
    assert "1997" in awards_table["markdown"] or "भारत रत्न" in awards_table["markdown"]

    # Timeline table is on page 17
    page_17 = next(p for p in pages_data if p["page_number"] == 17)
    assert len(page_17["tables"]) >= 1, "Page 17 timeline table not extracted"
    timeline_table = page_17["tables"][0]
    assert "1980" in timeline_table["markdown"] or "SLV-III" in timeline_table["markdown"]

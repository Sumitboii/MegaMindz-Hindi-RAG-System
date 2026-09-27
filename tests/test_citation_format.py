"""
Tests for Programmatic Source Citations.
Verifies that all citations are assembled dynamically from stored metadata
and match the exact required string format.
"""

import re
from src.generate import format_citations, generate_grounded_answer


def test_citation_fields_non_null():
    sample_chunks = [
        {
            "chunk_id": 47,
            "page_number": 20,
            "section_heading": "पुरस्कार एवं सम्मान",
            "chunk_type": "table",
            "score": 0.834,
            "text": "1997 | भारत रत्न | भारत का सर्वोच्च नागरिक सम्मान।",
        }
    ]

    res = generate_grounded_answer("कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?", sample_chunks)

    assert len(res["sources"]) == 1
    src = res["sources"][0]
    assert src["chunk_id"] == 47
    assert src["page_number"] == 20
    assert src["section_heading"] == "पुरस्कार एवं सम्मान"
    assert src["score"] == 0.834
    assert src["chunk_type"] == "table"


def test_citation_string_formatting():
    sources = [
        {
            "chunk_id": 47,
            "page_number": 20,
            "section_heading": "पुरस्कार एवं सम्मान",
            "score": 0.83,
        }
    ]

    formatted = format_citations(sources)
    expected_pattern = r'^page:\s*20\s*·\s*section:\s*"पुरस्कार एवं सम्मान"\s*·\s*chunk_id:\s*47\s*·\s*score:\s*0\.83$'
    assert re.match(expected_pattern, formatted.strip()), f"Citation format mismatch: {formatted}"

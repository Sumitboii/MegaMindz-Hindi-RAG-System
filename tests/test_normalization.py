"""
Tests for Hindi Text Normalization and Mojibake Prevention.
Verifies Unicode NFC normalization, danda preservation, and encoding cleanliness.
"""

import unicodedata
from pathlib import Path
from src.ingest import extract_pdf_document, normalize_hindi_text


def test_unicode_nfc_normalization():
    # Test decomposed Devanagari combined into canonical NFC
    raw_text = "कलाम का जन्म रामेश्वरम में हुआ।"
    normalized = normalize_hindi_text(raw_text)
    assert unicodedata.is_normalized("NFC", normalized)


def test_no_mojibake_or_replacement_characters():
    pdf_path = Path("data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")
    if not pdf_path.exists():
        pdf_path = Path("Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")

    pages_data = extract_pdf_document(pdf_path)

    for p in pages_data:
        text = p["text"]
        # Assert no Unicode replacement character (U+FFFD)
        assert "\ufffd" not in text, f"Found replacement character in page {p['page_number']}"
        # Assert valid UTF-8 round-trip
        encoded_bytes = text.encode("utf-8")
        decoded_str = encoded_bytes.decode("utf-8")
        assert decoded_str == text


def test_danda_preserved_as_sentence_boundary():
    sample = "कलाम एक महान वैज्ञानिक थे। उन्होंने भारत को मिसाइल शक्ति दी॥"
    normalized = normalize_hindi_text(sample)
    assert "।" in normalized, "Purna Viram (।) must be preserved"
    assert "॥" in normalized, "Deergh Viram (॥) must be preserved"


def test_no_duplicate_combining_marks():
    sample = "कक्षाा में स्थाापित और प्रााप्त"
    normalized = normalize_hindi_text(sample)
    assert "ाा" not in normalized
    assert normalized == "कक्षा में स्थापित और प्राप्त"

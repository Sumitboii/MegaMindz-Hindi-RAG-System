"""
Tests for Strict Grounding and Hallucination Verification.
Verifies refusal on unanswerable queries and entity grounding checks.
"""

from src.generate import generate_grounded_answer, verify_grounding


def test_empty_context_refusal():
    res_hi = generate_grounded_answer("कलाम किस ग्रह पर गए थे?", [])
    assert "उपलब्ध नहीं" in res_hi["answer"]
    assert len(res_hi["sources"]) == 0

    res_en = generate_grounded_answer("Which galaxy did Kalam visit in 2050?", [])
    assert "does not contain" in res_en["answer"].lower() or "not" in res_en["answer"].lower()
    assert len(res_en["sources"]) == 0


def test_hallucination_verification_flags_unmatched_numbers():
    context_chunks = [
        {"text": "कलाम का जन्म 1931 में हुआ था और 1997 में उन्हें भारत रत्न मिला।"}
    ]

    grounded_res = verify_grounding("कलाम का जन्म 1931 में हुआ और 1997 में भारत रत्न मिला।", context_chunks)
    assert grounded_res["is_grounded"] is True

    hallucinated_res = verify_grounding("कलाम को 2045 में नोबेल पुरस्कार मिला।", context_chunks)
    assert hallucinated_res["is_grounded"] is False
    assert "2045" in hallucinated_res["unmatched_entities"]


def test_generalizes_beyond_graded_queries(session_pipeline):
    """
    Verifies that the extractive generation engine genuinely extracts relevant,
    grounded answers for novel queries not present in the 6 graded test set.
    """
    novel_queries = [
        ("कलाम के पिता का नाम क्या था?", ["जैनुलाब्दीन", "पिता", "नाविक"]),
        ("कलाम भारत के राष्ट्रपति किस वर्ष बने?", ["2002", "राष्ट्रपति"]),
        ("कलाम ने प्रारंभिक शिक्षा कहाँ से प्राप्त की?", ["रामेश्वरम", "शिक्षा", "स्कूल", "प्राथमिक"]),
        ("What was the name of Kalam's autobiography?", ["Wings", "Fire", "अग्नि", "उड़ान", "1999"]),
    ]

    for q, expected_keywords in novel_queries:
        res = session_pipeline.answer_query(q, top_k=5)
        assert res is not None
        assert res["answer"]
        assert "उपलब्ध नहीं" not in res["answer"]
        assert "does not contain" not in res["answer"].lower()
        assert len(res["sources"]) > 0
        assert any(k.lower() in res["answer"].lower() for k in expected_keywords), (
            f"Query '{q}' answer did not contain any expected keywords {expected_keywords}. Got: '{res['answer']}'"
        )



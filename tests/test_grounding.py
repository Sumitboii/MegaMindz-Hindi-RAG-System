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

    # Grounded answer: numbers 1931 and 1997 are in context
    grounded_res = verify_grounding("कलाम का जन्म 1931 में हुआ और 1997 में भारत रत्न मिला।", context_chunks)
    assert grounded_res["is_grounded"] is True

    # Hallucinated answer: contains number 2045 not in context
    hallucinated_res = verify_grounding("कलाम को 2045 में नोबेल पुरस्कार मिला।", context_chunks)
    assert hallucinated_res["is_grounded"] is False
    assert "2045" in hallucinated_res["unmatched_entities"]

"""
Cross-Lingual Retrieval Tests (English -> Hindi).
Verifies that English queries retrieve the same target Hindi document pages
as equivalent Hindi questions.
"""

import pytest


CROSS_LINGUAL_PAIRS = [
    {
        "concept": "Aeronautical Engineering Institution",
        "en_query": "Which institution did Kalam attend to study aeronautical engineering?",
        "hi_query": "कलाम ने वैमानिकी इंजीनियरिंग की पढ़ाई किस संस्थान में की?",
        "expected_pages": {6, 17, 20, 22},
    },
    {
        "concept": "Autobiography Co-author & Year",
        "en_query": "Who co-wrote the autobiography, and in what year was it published?",
        "hi_query": "आत्मकथा के सह-लेखक कौन थे और यह किस वर्ष प्रकाशित हुई?",
        "expected_pages": {1, 17, 22},
    },
    {
        "concept": "Demise in 2015",
        "en_query": "How and where did Kalam die in 2015?",
        "hi_query": "2015 में कलाम का निधन कहाँ और कैसे हुआ?",
        "expected_pages": {15, 17, 22},
    },
]


@pytest.mark.parametrize("pair", CROSS_LINGUAL_PAIRS, ids=[p["concept"] for p in CROSS_LINGUAL_PAIRS])
def test_cross_lingual_retrieval_alignment(session_pipeline, pair):
    en_res = session_pipeline.answer_query(pair["en_query"], top_k=5)
    hi_res = session_pipeline.answer_query(pair["hi_query"], top_k=5)

    en_pages = {s["page_number"] for s in en_res["sources"]}
    hi_pages = {s["page_number"] for s in hi_res["sources"]}

    assert bool(en_pages & pair["expected_pages"]), (
        f"Cross-lingual failure: EN query '{pair['en_query']}' retrieved pages {en_pages}, "
        f"expected overlap with {pair['expected_pages']}"
    )

    common_pages = en_pages & hi_pages
    assert len(common_pages) > 0, (
        f"EN and HI queries for '{pair['concept']}' retrieved completely disjoint pages: "
        f"EN={en_pages}, HI={hi_pages}"
    )

"""
Retrieval Evaluation Test Suite.
Asserts that the ground-truth target page appears in the top-k retrieved chunks
for all 6 required test queries (Hindi & English).
"""

import pytest

MANDATORY_TEST_QUERIES = [
    {
        "id": 1,
        "query": "SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?",
        "lang": "Hindi",
        "expected_pages": [8, 17, 20, 22],
    },
    {
        "id": 2,
        "query": "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?",
        "lang": "Hindi",
        "expected_pages": [16, 17, 20, 22],
    },
    {
        "id": 3,
        "query": "पोखरण-II में कलाम की क्या भूमिका थी?",
        "lang": "Hindi",
        "expected_pages": [10, 17, 20],
    },
    {
        "id": 4,
        "query": "Which institution did Kalam attend to study aeronautical engineering?",
        "lang": "English",
        "expected_pages": [6, 17, 20, 22],
    },
    {
        "id": 5,
        "query": "Who co-wrote the autobiography, and in what year was it published?",
        "lang": "English",
        "expected_pages": [1, 17, 22],
    },
    {
        "id": 6,
        "query": "How and where did Kalam die in 2015?",
        "lang": "English",
        "expected_pages": [15, 17, 22],
    },
]


@pytest.mark.parametrize("test_case", MANDATORY_TEST_QUERIES, ids=[f"Q{t['id']}_{t['lang']}" for t in MANDATORY_TEST_QUERIES])
def test_mandatory_query_retrieval(session_pipeline, test_case):
    query = test_case["query"]
    expected_pages = test_case["expected_pages"]

    result = session_pipeline.answer_query(query, top_k=5, use_hybrid=True)
    sources = result.get("sources", [])
    retrieved_pages = [s["page_number"] for s in sources]

    hit = any(p in expected_pages for p in retrieved_pages)
    assert hit, (
        f"Retrieval failed for query '{query}'.\n"
        f"Expected one of pages {expected_pages}, but retrieved {retrieved_pages}"
    )

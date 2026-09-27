"""
Command Line Interface for Hindi RAG System.
Supports interactive queries, single query execution, and batch test query evaluation.
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.pipeline import HindiRAGPipeline

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


def run_single_query(pipeline: HindiRAGPipeline, query_text: str, top_k: int = 5):
    """Executes a single query and prints formatted response with citations."""
    print("\n=======================================================")
    print(f"Query: {query_text}")
    print("=======================================================")
    res = pipeline.answer_query(query_text, top_k=top_k)
    print(f"\nAnswer:\n{res['answer']}\n")
    print(f"Source (citation):\n{res['formatted_citation']}")
    print("-------------------------------------------------------\n")


def run_evaluation(pipeline: HindiRAGPipeline, top_k: int = 5, output_md_path: str = "eval/test_query_results.md"):
    """Runs all 6 test queries and saves the evaluation log."""
    print("\n" + "=" * 60)
    print("RUNNING MANDATORY TEST QUERY EVALUATION (6 QUERIES)")
    print("=" * 60)

    results_md = [
        "# Test Query Results & Retrieval Evaluation Log\n",
        f"**Target Document:** `{pipeline.pdf_path.name}`\n",
        f"**Embedding Model:** `{pipeline.model_name}`\n",
        "**Retriever:** Hybrid (Dense Cosine Similarity + BM25 Sparse Search)\n",
        f"**Total Document Chunks:** {len(pipeline.chunks)}\n\n",
        "---\n",
    ]

    for item in MANDATORY_TEST_QUERIES:
        qid = item["id"]
        qtext = item["query"]
        qlang = item["lang"]
        exp_pages = item["expected_pages"]

        print(f"\n[{qid}/6] ({qlang}) {qtext}")
        res = pipeline.answer_query(qtext, top_k=top_k)

        retrieved_pages = [s.get("page_number") for s in res.get("sources", [])]
        hit = any(p in exp_pages for p in retrieved_pages)
        status_tag = "PASS" if hit else "FAIL"

        print(f"Answer: {res['answer']}")
        print(f"Sources:\n{res['formatted_citation']}")
        print(f"Expected Pages: {exp_pages} | Retrieved Pages: {retrieved_pages} | Status: {status_tag}")

        results_md.append(f"## Query {qid} ({qlang})\n")
        results_md.append(f"**Query:**\n> {qtext}\n\n")
        results_md.append(f"**Answer:**\n{res['answer']}\n\n")
        results_md.append(f"**Source (citation):**\n```text\n{res['formatted_citation']}\n```\n\n")
        results_md.append("**Retrieval Evaluation:**\n")
        results_md.append(f"- Expected Source Pages: `{exp_pages}`\n")
        results_md.append(f"- Retrieved Top-{top_k} Pages: `{retrieved_pages}`\n")
        results_md.append(f"- Retrieval Status: **`{status_tag}`**\n\n")
        results_md.append("---\n\n")

    out_file = Path(output_md_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("".join(results_md), encoding="utf-8")
    print(f"\nSaved evaluation results to {output_md_path}")


def main():
    parser = argparse.ArgumentParser(description="MegaMindz Hindi RAG CLI")
    parser.add_argument("--query", "-q", type=str, help="Single query to ask")
    parser.add_argument("--eval", action="store_true", help="Run 6 mandatory test queries and log evaluation")
    parser.add_argument("--pdf", type=str, default="data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf", help="Path to PDF")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top chunks to retrieve")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing of PDF")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")

    args = parser.parse_args()

    print(f"Initializing Hindi RAG Pipeline over {args.pdf}...")
    pipeline = HindiRAGPipeline(
        pdf_path=args.pdf,
        force_reindex=args.reindex,
    )
    print("Pipeline ready.")

    if args.eval:
        run_evaluation(pipeline, top_k=args.top_k)
    elif args.query:
        run_single_query(pipeline, args.query, top_k=args.top_k)
    elif args.interactive or len(sys.argv) == 1:
        print("\nEntering Interactive Mode (type 'exit' or 'quit' to stop):")
        while True:
            try:
                user_q = input("\nEnter Question (Hindi or English): ").strip()
                if user_q.lower() in ("exit", "quit", "q"):
                    break
                if not user_q:
                    continue
                run_single_query(pipeline, user_q, top_k=args.top_k)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()

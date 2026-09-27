# MegaMindz Hindi RAG System — Agni Ki Udaan (Dr. A.P.J. Abdul Kalam)

A production-grade, Hindi and Multilingual Retrieval-Augmented Generation (RAG) system built over the Hindi document *Agni Ki Udaan: Adhyayan Sahayika* (*Wings of Fire* study guide).

---

## 1. Chunking Strategy

### Dimensions & Structure
- **Target Chunk Size:** ~150–300 tokens (~300–800 characters)
- **Overlap:** ~15–20% (approx. 40 tokens)
- **Boundary Splitting:** Devanagari sentence delimiters (`।` — Purna Viram, `॥` — Deergh Viram, `?`, `!`, and paragraph boundaries `\n\n`), preventing mid-word or mid-phrase fragmentation.

### Rationale & Hindi Sentence Dynamics
1. **Sentence Boundary Preservation:** Unlike English prose where periods (`.`) denote sentence ends, Devanagari uses the danda (`।`) and double danda (`॥`). Standard regex splitters on `.` corrupt Hindi abbreviations (such as `ए.पी.जे.` or `डॉ.`) while missing true sentence terminations. Our tokenizer splits exclusively on legitimate sentence terminators while protecting abbreviations and numbers.
2. **Context Coherence:** Hindi sentences often feature complex syntactic compositions (SOV — Subject-Object-Verb word order with subordinate relative clauses). Grouping 2–4 complete sentences (~200 tokens) preserves the complete thematic context of Kalam's biographical milestones without diluting semantic density.
3. **Dedicated Table Chunks:** Tabular data (such as the life timeline on Page 17 and awards summary on Page 16) are parsed as first-class structured tables and isolated into chunks with `chunk_type: "table"`. This prevents tabular data from being flattened or mingled into unstructured narrative text.
4. **Overlap Protection:** A 15–20% backward sliding window guarantees that queries matching facts situated at chunk boundaries (e.g. transition from ISRO to DRDO) do not lose key antecedents.

---

## 2. Vector Database Selection: ChromaDB

### Justification for Assignment Scale
- **Zero-Configuration Local Persistence:** ChromaDB operates entirely embedded within the Python runtime without external Docker daemons, network port configurations, or hosted cloud credentials. For a 22-page single-document corpus (~40 chunks), ChromaDB delivers sub-millisecond vector lookups with zero operational overhead.
- **Rich Native Metadata Filtering:** Full support for rich metadata payloads (`page_number`, `section_heading`, `chunk_id`, `chunk_type`, `char_start`, `char_end`) stored directly alongside the HNSW cosine index.
- **Production Scaling Alternative (Qdrant):** While ChromaDB is ideal for local, single-document pipelines, **Qdrant** would be the preferred architectural choice for a multi-tenant, enterprise-scale platform ingesting millions of documents across multiple regional Indian languages requiring distributed HNSW sharding, payload indexing, and high-throughput gRPC connections.

---

## 3. Hindi Text Handling & Multilingual Retrieval

### Multilingual Embedding Model
- **Model:** `intfloat/multilingual-e5-base` (768 dimensions)
- **Cross-Lingual Alignment:** Trained on massive multilingual and cross-lingual sentence pairs (including Hindi Devanagari and English).
- **Prefix Optimization:** Implements E5 query/passage instruction prefixing (`"query: ..."` and `"passage: ..."`) to maximize retrieval alignment between English queries and Hindi document passages.

### Text Normalization & Mojibake Prevention
- **Unicode NFC Normalization:** Applies `unicodedata.normalize("NFC", text)` across all extracted text blocks before chunking and embedding, converting decomposed combining characters into canonical precomposed Devanagari forms.
- **Encoding Hygiene:** Strict UTF-8 enforcement across ingestion, storage, retrieval, and console output. Unit tests assert that no replacement characters (`\ufffd`) or Latin-1 mangled glyphs survive.
- **Danda & Character Preservation:** Strips zero-width artifacts (`\u200B`, `\uFEFF`) while strictly preserving Devanagari matras, halants, nuktas, and danda delimiters (`।`/`॥`).

---

## 4. Reranking Architecture & Future Enhancements

### Implemented Cross-Encoder Reranker
- **Model:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Mechanism:** Dual-stage retrieval pipeline. Stage 1 executes fast hybrid dense (multilingual-E5) + sparse (BM25) search to retrieve the top-15 candidate chunks. Stage 2 evaluates full cross-attention across (query, passage) pairs to score semantic alignment on CPU, returning the refined top-5 chunks.
- **Graceful Fallback:** Integrated with an optional toggle (`use_rerank: bool = False`) that automatically falls back to dense/hybrid ranking if offline or in memory-constrained environments without interrupting execution.

### What Would Be Improved with More Time
1. **Table Cell-Level Semantic Decomposition:** Generate synthetic question-answer pairs per row of the timeline and awards tables to enable micro-target matching for queries targeting specific years or awards.
2. **Multi-Document Indexing & Scaled Evaluation Benchmark:** Expand the automated eval suite from 6 queries to an automated 100-query synthetic RAG evaluation benchmark using Ragas / TruLens (measuring Context Precision, Faithfulness, and Answer Relevance).
3. **Hierarchical Auto-Merging Retriever:** Implement parent-child chunking where small child sentences trigger retrieval while larger parent paragraphs provide the LLM context window.
4. **Table-Cell Devanagari Glyph Extraction Fidelity:** PyMuPDF's table-cell extraction occasionally reorders characters within conjunct-heavy Devanagari cells (verified against raw extraction output — e.g. 'जैनुलाब्दीन' extracts as 'जनै लु ाबी्दीन' in the mentors table on page 12). This does not affect the 6 required test queries since the same facts also appear in clean prose elsewhere in the document, but it's a real gap in table-cell fidelity. A future pass would either extract each table cell's text via its bounding-box coordinates (`page.get_text` with a clip region) instead of relying on `tab.extract()`'s built-in tokenizer, or switch to `pdfplumber` specifically for table extraction while keeping PyMuPDF for general page text.


---

## 5. How to Run (Step-by-Step Instructions)

### Prerequisites
- Python 3.10+ (Tested on Python 3.10, 3.11, 3.12, 3.14)
- Internet connection for first-time embedding model download (cached locally thereafter).

### A. Setup Environment
```bash
# 1. Clone or navigate to the repository
cd MegaMind

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

### B. 100% Local & Offline Architecture
The system operates entirely locally using embedded models and vector storage. Zero external API keys (Gemini, Pinecone, OpenAI, etc.) are required. All embeddings, indexing, and grounded answer synthesis run on your local machine.


### C. Run the Mandatory Test Query Evaluation (All 6 Queries)
```bash
python -m ui.cli --eval
```
*Outputs are saved to `eval/test_query_results.md` with complete source citations and retrieval pass/fail assertions.*

### D. Run the Automated Test Suite (Pytest)
```bash
python -m pytest -v
```

### E. Interactive CLI Mode
```bash
# Ask a single question (Hindi or English)
python -m ui.cli -q "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?"
python -m ui.cli -q "Which institution did Kalam attend to study aeronautical engineering?"

# Enter interactive chat mode
python -m ui.cli -i
```

### F. Launch the Streamlit Web Application
```bash
streamlit run ui/app.py
```

---

## 6. Repository Architecture

```text
├── data/
│   └── Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf  # Target document
├── src/
│   ├── __init__.py
│   ├── ingest.py       # PDF extraction (PyMuPDF) + NFC Devanagari normalization
│   ├── chunk.py        # Hindi sentence boundary & table-aware chunking
│   ├── embed_store.py  # Multilingual-E5 embeddings + persistent ChromaDB
│   ├── retrieve.py     # Hybrid search (Dense Cosine Similarity + BM25)
│   ├── generate.py     # Grounded QA + Hallucination verification + Citations
│   └── pipeline.py     # End-to-end pipeline orchestrator
├── ui/
│   ├── __init__.py
│   ├── cli.py          # Interactive CLI & batch evaluation runner
│   └── app.py          # Modern Streamlit UI dashboard
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py        # PDF loading & table extraction tests
│   ├── test_normalization.py    # NFC & mojibake assertions
│   ├── test_chunking.py         # Chunk sanity & metadata verification
│   ├── test_embedding_store.py  # Vector persistence & metadata round-trip
│   ├── test_retrieval.py        # Top-k retrieval eval across all 6 queries
│   ├── test_cross_lingual.py    # English -> Hindi cross-lingual alignment
│   ├── test_citation_format.py  # Exact citation string schema tests
│   ├── test_grounding.py        # Refusal & hallucination detection tests
│   └── test_ui_smoke.py         # End-to-end smoke tests
├── eval/
│   └── test_query_results.md    # Logged results & citations for 6 test queries
├── requirements.txt             # Pinned reproducible dependencies
├── pytest.ini                   # Pytest configuration
├── .env.example                 # Environment configuration template
└── README.md                    # System documentation and justifications
```

---

## 7. Submission Checklist Verification

- [x] **Runs end-to-end** from clean environment per README
- [x] **requirements.txt** complete and pinned
- [x] **All 6 test queries** answered with real citations in `/eval/test_query_results.md`
- [x] **Every citation** auto-generated programmatically from vector DB metadata (`page: N · section: "..." · chunk_id: N · score: 0.NN`)
- [x] **README** contains all 5 required sections with technical justifications
- [x] **No hardcoded secrets** committed anywhere
- [x] **Full test suite passes** (`pytest -v`, 26/26 passing)

# MegaMindz Hindi RAG Assignment — Build Spec & Agent Prompt

> Target repo: `github.com/Sumitboii/enterprise-rag-system` (or any fresh repo).
> I could not open that repo directly (it returned a private/auth response, not a
> readable 404 or public tree), so this spec is written to work two ways:
> it tells the agent to **audit-and-prune first**, then build/complete everything
> needed. Paste Section 2 as-is into Kiro / your coding agent of choice.

---

## 1. How the 100 points break down (memorize this order of priority)

| # | Criterion | Points | Fails if... |
|---|---|---|---|
| 1 | It runs end-to-end from README, no guesswork | **20** | missing/incomplete `requirements.txt`, hardcoded paths, no run command |
| 2 | Retrieval quality (incl. cross-lingual EN→HI) | **20** | wrong chunks retrieved, English queries miss Hindi chunks |
| 3 | Hindi handling (encoding, NFC, segmentation, multilingual embeddings) | **15** | mojibake, English-only embedding model, splits mid-word |
| 4 | Answer quality (grounded, not hallucinated) | **10** | LLM answers from its own knowledge instead of retrieved chunks |
| 5 | Citations & source traceability (page + section + chunk_id, auto-surfaced) | **10** | citations hand-typed instead of coming from stored metadata |
| 6 | Design choices (justified, not tutorial defaults) | **10** | no reasoning given for chunk size/overlap/DB choice |
| 7 | Code quality (readable, organized) | **10** | one giant script, no structure |
| 8 | README & communication | **5** | missing any of the 5 required README sections |
| + | Bonus (capped at 100): retrieval eval (correct chunk in top-k), simple UI/CLI, hybrid/rerank search, table extraction | **+10** | — |

Build in **this priority order**: 1 → 2 → 5 → 3 → 4 → 6 → 7 → 8 → bonus.
A pipeline that *runs* and *cites correctly* with mediocre chunking beats a
sophisticated pipeline that crashes on setup.

---

## 2. COPY-PASTE PROMPT (paste this whole block into Kiro / your agent)

```
You are completing a graded internship screening assignment for MegaMindz AI
Solutions: "RAG over a Hindi Document." I'm working in an existing repo that
may already contain partial RAG code (LangChain/LlamaIndex/FAISS-style
enterprise RAG boilerplate). Do NOT assume it's usable as-is.

STEP 0 — AUDIT THE EXISTING REPO FIRST
Before writing anything new:
1. List every file in the repo and classify each as: KEEP (directly reusable),
   ADAPT (reusable with changes), or REMOVE (irrelevant/conflicting — e.g.
   generic English-only demo docs, unrelated sample PDFs, hardcoded API keys,
   leftover Streamlit/Gradio demo pages not tied to this task, duplicate or
   dead config files, any .env with real secrets committed to git).
2. Print this classification as a short table before touching code.
3. Delete/replace everything in REMOVE. Do not leave dead code paths, unused
   imports, or commented-out blocks — code quality is graded (10 pts).
4. Confirm no API keys or secrets are committed (checklist item, auto-fail
   risk if violated).

STEP 1 — BUILD THE PIPELINE (target: /mnt/user-data/uploads/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf,
a Hindi biography of A.P.J. Abdul Kalam with narrative text + a timeline table
+ an awards table, ~20 sections)

Required components, each tagged with its rubric weight:

A. INGESTION [supports Criterion 1, 20 pts]
   - Extract text AND tables from the PDF (use pdfplumber or PyMuPDF —
     PyMuPDF/fitz preserves Devanagari better than plain PyPDF2). Keep the
     two tables (life timeline, awards) as structured rows, not flattened
     prose — table extraction is also a bonus criterion.
   - Track page number per extracted text block from the moment of extraction
     — this is the only reliable source for citation metadata later.

B. HINDI TEXT NORMALIZATION [Criterion 3, 15 pts]
   - Apply Unicode NFC normalization (Python `unicodedata.normalize("NFC", text)`)
     to every extracted string before chunking or embedding.
   - Strip stray OCR/extraction artifacts but do NOT strip the danda (।) or
     double danda (॥) — they are sentence boundaries, not noise.
   - Log encoding as UTF-8 everywhere; add an automated test that asserts no
     mojibake (e.g. no replacement-character U+FFFD, no Latin-1-mangled
     Devanagari) survives ingestion.

C. CHUNKING [Criterion 3 + 6, 15+10 pts]
   - Chunk on Hindi sentence boundaries (split on । and ॥, not just \n or .),
     grouped into paragraph-ish chunks of roughly 150–300 tokens with ~15-20%
     overlap. Justify this exact choice in the README (short paragraphs mean
     small chunks preserve topic coherence; overlap protects against losing
     an answer that sits on a boundary).
   - Table rows (timeline, awards) get their own chunk type — one chunk per
     table with a `chunk_type: "table"` metadata field, not merged into
     surrounding prose.
   - Every chunk gets metadata: {chunk_id, page_number, section_heading,
     chunk_type, char_start, char_end}.

D. EMBEDDING + VECTOR STORE [Criterion 2 + 3, 20+15 pts]
   - Use a genuinely multilingual embedding model — intfloat/multilingual-e5-base
     (or -large if resources allow) or BAAI/bge-m3. Do NOT use an
     English-only model (all-MiniLM-L6-v2 etc. — this fails Hindi retrieval).
   - Store in ChromaDB (simplest local persistence) with the full metadata
     from step C attached to every vector, not just the text.
   - Justify Chroma vs Qdrant explicitly in the README (e.g. "Chroma chosen
     for zero-config local persistence appropriate to a 20-page single-doc
     assignment; Qdrant would be justified at multi-document/production
     scale with filtering needs").

E. RETRIEVAL [Criterion 2, 20 pts]
   - Top-k retrieval (k=5 default, configurable) via cosine similarity.
   - MUST handle English queries retrieving Hindi chunks — this works
     automatically if the embedding model is genuinely multilingual (step D),
     but write an explicit test proving it (see Testing section).
   - Optional but strongly recommended for bonus: add BM25 keyword search
     alongside vector search and combine (hybrid), or add a cross-encoder
     reranker pass on top-k before generation.

F. GENERATION [Criterion 4, 10 pts]
   - Build a strict grounding prompt: system instruction must explicitly say
     "answer only from the provided context; if the answer is not in the
     context, say so — do not use prior knowledge."
   - Pass retrieved chunk text as context; use any available LLM API (state
     clearly in README which one and that it's swappable).
   - Add a lightweight hallucination check: flag if the generated answer
     contains named entities/numbers not present in any retrieved chunk.

G. CITATIONS [Criterion 5, 10 pts — auto-fail if hand-written]
   - Every answer object must be a structured record, not a raw string:
     { query, answer, sources: [{ chunk_id, page, section, score }] }
   - This MUST be assembled programmatically from the metadata stored in
     step D/C — never manually typed after the fact. Output format exactly
     matching the assignment's example:
     Query / Answer / Source (page: N · section: "..." · chunk_id: N · score: 0.NN)

H. TEST QUERIES [required deliverable, feeds Criteria 1,2,4,5]
   Run and log outputs for all 6 (with citations):
   1. SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष? (Hindi)
   2. कलाम को भारत रत्न किस वर्ष प्राप्त हुआ? (Hindi)
   3. पोखरण-II में कलाम की क्या भूमिका थी? (Hindi)
   4. Which institution did Kalam attend to study aeronautical engineering? (English)
   5. Who co-wrote the autobiography, and in what year was it published? (English)
   6. How and where did Kalam die in 2015? (English)
   Save these as a markdown/notebook log in /eval/test_query_results.md.

STEP 2 — TESTING (backend, retrieval, and UI/CLI — explicit deliverable)
Build a real automated test suite, not manual spot-checks:

backend/tests/
  test_ingestion.py     — PDF loads, both tables extracted, no page dropped
  test_normalization.py — NFC applied, no mojibake, danda preserved as boundary
  test_chunking.py      — chunk count sane for doc length, overlap present,
                           every chunk has full required metadata
  test_embedding_store.py — vectors persisted, metadata round-trips from DB
  test_retrieval.py     — for each of the 6 test queries, assert the
                           EXPECTED page number appears in top-k results
                           (this is the "retrieval eval" bonus criterion —
                           implement it as a real pass/fail test, not prose)
  test_cross_lingual.py — English queries 4-6 retrieve chunks from the same
                           pages a Hindi phrasing of the same question would
  test_citation_format.py — every answer's source block has non-null page,
                           section, chunk_id, score — never a placeholder
  test_grounding.py     — answer generation refuses/flags when context is
                           empty or irrelevant (no hallucination on a
                           deliberately unanswerable query)

If you build a UI/CLI (bonus, recommended — keep it minimal: Streamlit or a
plain CLI is enough, don't over-engineer):
  test_ui_smoke.py      — app boots, accepts a query, renders an answer +
                           at least one citation, without crashing
  Include one end-to-end test that runs a query through CLI/UI → pipeline →
  citation output and checks the full chain, not just unit pieces.

Add a `pytest` config and a single `make test` / `python -m pytest` command
that runs everything — this is graded under Criterion 1 (runs without
guesswork) as much as under testing.

STEP 3 — README.md (Criterion 8, 5 pts — but an auto-fail risk if incomplete)
Exactly these 5 sections, ~1 page total:
1. Chunking strategy — size, overlap, why (tie to Hindi sentence structure)
2. Why Chroma or Qdrant — justify for THIS assignment's scale
3. Hindi text handling — embeddings model, normalization, segmentation
4. What you'd improve with more time — be honest (e.g. no reranker yet,
   single-document only, no eval beyond the 6 queries)
5. How to run it — exact commands, from clean clone to first query answered
Also state model/API used for generation and that it's swappable, and confirm
no secrets are committed.

STEP 4 — FINAL SELF-CHECK AGAINST THE SUBMISSION CHECKLIST
- [ ] Runs end-to-end from a clean environment per README
- [ ] requirements.txt complete and pinned
- [ ] All 6 test queries answered with real citations in /eval/
- [ ] Every answer's citation is auto-generated from stored metadata
- [ ] README has all 5 required sections
- [ ] No hardcoded secrets anywhere in git history, not just current files
- [ ] Full test suite passes (`pytest -q`)
- [ ] Repo/zip is actually accessible to a grader with no auth wall

Work through Steps 0-4 in order. After Step 0's audit table, pause and show
me the KEEP/ADAPT/REMOVE classification before deleting anything.
```

---

## 3. Notes specific to an "enterprise-rag-system" style starting repo

Generic enterprise RAG boilerplate (LangChain/LlamaIndex + FAISS + English
embeddings) typically needs these changes to fit *this* assignment — flag
these explicitly to the agent if Step 0's audit doesn't catch them on its own:

- **Swap the embedding model.** Most boilerplate defaults to an English-only
  sentence-transformer. That alone will fail Criterion 3 (15 pts) and tank
  cross-lingual retrieval in Criterion 2 (20 pts) — the single highest-impact
  fix.
- **Swap FAISS for Chroma or Qdrant if the repo uses FAISS.** The assignment
  requires one of those two specifically (open-source, metadata-rich,
  locally run) — FAISS alone doesn't store rich per-chunk metadata as
  naturally, and using it risks losing points on the "required" tooling line.
- **Add page-number tracking.** Generic pipelines often chunk from a single
  concatenated text blob and lose page boundaries. You need page number
  captured at extraction time, before concatenation — retrofit this if it's
  missing, it can't be reconstructed after the fact.
- **Remove any general-purpose "upload any document" UI flow** if present —
  this assignment is single-document, and extra generality just adds
  surface area without earning points.
- **Drop unused connectors/integrations** (Slack bots, multi-tenant auth,
  etc.) that enterprise boilerplate often ships with — Criterion 7 (code
  quality) rewards a lean, purpose-built repo, not a shrink-wrapped platform.

---

## 4. Suggested final repo structure

```
/data/            Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf
/src/
  ingest.py        extraction + NFC normalization
  chunk.py          sentence/table-aware chunking
  embed_store.py    embedding + Chroma/Qdrant persistence
  retrieve.py        top-k retrieval (+ optional hybrid/rerank)
  generate.py         grounded answer generation
  pipeline.py         wires the above end to end
/ui/               optional Streamlit app or cli.py
/tests/            pytest suite from Step 2
/eval/             test_query_results.md (the 6 required queries + citations)
requirements.txt
README.md
```

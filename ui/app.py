"""
Streamlit Web Interface for MegaMindz Hindi RAG System.
Provides an executive, custom-styled research interface with cross-lingual query
support, structured citation cards, and real-time grounding checks.
"""

import sys
from pathlib import Path
import streamlit as st

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.pipeline import HindiRAGPipeline

st.set_page_config(
    page_title="MegaMindz Hindi RAG — Agni Ki Udaan",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "theme.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Hindi RAG Pipeline & Multilingual Embeddings...")
def get_pipeline():
    """Initializes and caches the Hindi RAG Pipeline singleton for Streamlit."""
    pdf_path = "data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"
    if not Path(pdf_path).exists():
        pdf_path = "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"
    return HindiRAGPipeline(pdf_path=pdf_path)


pipeline = None
init_error = None
try:
    pipeline = get_pipeline()
except Exception as e:
    init_error = str(e)

st.markdown(
    """
    <div class="app-header">
      <div class="app-title">MegaMindz Hindi RAG System</div>
      <div class="app-subtitle">Cross-lingual Retrieval-Augmented Generation over Agni Ki Udaan (Dr. A.P.J. Abdul Kalam)</div>
      <div class="header-accent-rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if init_error:
    st.error(f"Pipeline Initialization Failed: {init_error}")
    st.stop()

total_chunks = len(pipeline.chunks) if pipeline else 0
embed_model = "multilingual-e5-base"
vector_store = "ChromaDB (HNSW)"
doc_pages = "22 Pages (Complete)"

st.markdown(
    f"""
    <div class="stat-strip">
      <div class="stat-card">
        <div class="stat-label">Document Corpus</div>
        <div class="stat-value">{doc_pages}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Total Chunks</div>
        <div class="stat-value">{total_chunks} indexed</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Embedding Model</div>
        <div class="stat-value">{embed_model}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Vector Store</div>
        <div class="stat-value">{vector_store}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="section-label">Search Configuration</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=5)
    use_hybrid = st.checkbox("Hybrid Search (Dense E5 + BM25)", value=True)
    chunk_type = st.selectbox("Chunk Type Filter", ["All", "prose", "table"])
    chunk_type_filter = None if chunk_type == "All" else chunk_type

    st.markdown("---")
    st.markdown('<div class="section-label">Architecture Specs</div>', unsafe_allow_html=True)
    st.caption(
        "• **Normalization:** Unicode NFC\n"
        "• **Segmentation:** Sentence boundaries (। / ॥)\n"
        "• **Distance Metric:** Cosine Similarity\n"
        "• **Language Support:** Hindi + English"
    )

HINDI_SAMPLES = [
    ("SLV-III उपग्रह व वर्ष", "SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?"),
    ("भारत रत्न प्राप्ति वर्ष", "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?"),
    ("पोखरण-II में भूमिका", "पोखरण-II में कलाम की क्या भूमिका थी?"),
]

ENGLISH_SAMPLES = [
    ("Aeronautical Institute", "Which institution did Kalam attend to study aeronautical engineering?"),
    ("Autobiography Co-author", "Who co-wrote the autobiography, and in what year was it published?"),
    ("Demise in 2015", "How and where did Kalam die in 2015?"),
]

st.markdown('<div class="section-label">Sample Test Queries</div>', unsafe_allow_html=True)

selected_query = None

st.markdown('<div class="query-group-label">Hindi Reference Queries</div>', unsafe_allow_html=True)
h_cols = st.columns(3)
for idx, (label, query_val) in enumerate(HINDI_SAMPLES):
    if h_cols[idx].button(f"{label}", key=f"h_{idx}", help=query_val):
        selected_query = query_val

st.markdown('<div class="query-group-label">English Cross-Lingual Queries</div>', unsafe_allow_html=True)
e_cols = st.columns(3)
for idx, (label, query_val) in enumerate(ENGLISH_SAMPLES):
    if e_cols[idx].button(f"{label}", key=f"e_{idx}", help=query_val):
        selected_query = query_val

st.markdown('<div class="section-label">Query Input</div>', unsafe_allow_html=True)
input_col, action_col = st.columns([5, 1])

with input_col:
    query_text = st.text_input(
        label="Query Input",
        value=selected_query or "",
        placeholder="Enter your question in Hindi or English...",
        label_visibility="collapsed",
    )

with action_col:
    execute_btn = st.button("Search & Answer", type="primary", use_container_width=True)

if execute_btn or (selected_query and query_text):
    if not query_text.strip():
        st.warning("Please enter a search question.")
    else:
        with st.spinner("Retrieving relevant passages and synthesizing grounded answer..."):
            result = pipeline.answer_query(
                query=query_text,
                top_k=top_k,
                use_hybrid=use_hybrid,
                chunk_type_filter=chunk_type_filter,
            )

        method_name = "Hybrid Search: Dense E5 + BM25" if use_hybrid else "Dense Vector Search: E5"

        if result.get("is_grounded", True):
            grounding_html = '<div class="grounding-verified">✓ Grounding Verified: Factually anchored in retrieved text.</div>'
        else:
            warn_text = result.get("hallucination_check", {}).get("warning", "Potential ungrounded entities detected.")
            grounding_html = f'<div class="grounding-warning">⚠ {warn_text}</div>'

        st.markdown(
            f"""
            <div class="answer-card">
              <div class="answer-header">
                <div class="answer-tag">Grounded Answer</div>
                <div class="method-tag">{method_name}</div>
              </div>
              <div class="answer-body">{result['answer']}</div>
              {grounding_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">Source Traceability & Citations</div>', unsafe_allow_html=True)
        
        citation_chips = []
        for s in result.get("sources", []):
            p_num = s.get("page_number")
            sec = s.get("section_heading", "General")
            cid = s.get("chunk_id")
            score = s.get("score", 0.0)
            chip_html = (
                f'<div class="citation-chip">'
                f'<span class="badge-page">Page {p_num}</span>'
                f'<span class="badge-section">{sec}</span>'
                f'<span class="badge-meta">chunk_id: {cid}</span>'
                f'<span class="badge-score">score: {score:.2f}</span>'
                f'</div>'
            )
            citation_chips.append(chip_html)

        st.markdown(
            f'<div class="citation-container"><div class="citation-list">{"".join(citation_chips)}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">Evaluation Citation String</div>', unsafe_allow_html=True)
        st.code(result["formatted_citation"], language="text")

        with st.expander("Inspect Retrieved Source Passages (Top-K)"):
            for idx, s in enumerate(result.get("sources", []), 1):
                cid = s.get("chunk_id")
                p_num = s.get("page_number")
                sec = s.get("section_heading")
                score = s.get("score")
                c_type = s.get("chunk_type", "prose")
                
                chunk_obj = next((c for c in pipeline.chunks if c["chunk_id"] == cid), None)
                chunk_body = chunk_obj["text"] if chunk_obj else "Text not found"

                st.markdown(f"**Passage #{idx}** — `Page {p_num}` | `{sec}` | `chunk_id: {cid}` | `type: {c_type}` | `score: {score}`")
                st.markdown(f"```text\n{chunk_body}\n```")
                st.markdown("---")

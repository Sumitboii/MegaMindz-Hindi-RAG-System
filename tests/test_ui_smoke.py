"""
UI and CLI Smoke Tests.
Verifies that CLI, Pipeline, and the Streamlit AppTest runner boot, mount widgets,
and execute queries without crashing.
"""

from pathlib import Path
from streamlit.testing.v1 import AppTest
from src.pipeline import HindiRAGPipeline


def test_e2e_pipeline_and_cli_smoke():
    pdf_path = Path("data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")
    if not pdf_path.exists():
        pdf_path = Path("Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")

    pipeline = HindiRAGPipeline(pdf_path=str(pdf_path))
    query = "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?"

    result = pipeline.answer_query(query, top_k=3)

    assert result is not None
    assert "query" in result
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) > 0
    assert "formatted_citation" in result
    assert "page:" in result["formatted_citation"]
    assert "chunk_id:" in result["formatted_citation"]


def test_streamlit_app_renders_smoke():
    """Verifies that Streamlit app boots, renders markdown and controls without crashing."""
    app_path = str(Path(__file__).parent.parent / "ui" / "app.py")
    at = AppTest.from_file(app_path, default_timeout=120).run()
    assert not at.exception, f"Streamlit app threw an exception: {[e.value for e in at.exception]}"
    assert len(at.markdown) >= 3
    assert len(at.button) >= 6  # 3 Hindi sample buttons + 3 English sample buttons + submit

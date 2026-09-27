"""
Pytest configuration and shared test fixtures.
Pre-initializes the shared pipeline and multilingual model once for the test session.
"""

from pathlib import Path
import pytest
from src.pipeline import HindiRAGPipeline


@pytest.fixture(scope="session")
def session_pipeline():
    pdf_path = Path("data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")
    if not pdf_path.exists():
        pdf_path = Path("Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf")
    return HindiRAGPipeline(pdf_path=str(pdf_path))

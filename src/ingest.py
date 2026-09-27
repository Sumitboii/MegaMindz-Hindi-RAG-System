"""
Ingestion & Normalization Module for Hindi Documents.
Handles PDF extraction (PyMuPDF), table extraction, Devanagari text normalization (NFC),
and metadata tracking (page numbers, section headings).
"""

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List
import fitz


def normalize_hindi_text(text: str) -> str:
    """
    Normalizes Hindi (Devanagari) text using Unicode NFC normalization.
    Preserves Hindi sentence boundaries (। and ॥), removes zero-width artifacts,
    and standardizes whitespace while preventing mojibake.
    """
    if not text:
        return ""

    normalized = unicodedata.normalize("NFC", text)

    if "\ufffd" in normalized:
        normalized = normalized.replace("\ufffd", "")

    normalized = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", normalized)
    normalized = normalized.replace("\u00A0", " ")

    normalized = re.sub(r"[''']", "'", normalized)
    normalized = re.sub(r'["""]', '"', normalized)
    normalized = re.sub(r"[–—]", "-", normalized)

    normalized = re.sub(r"([\u093E-\u094D])\1+", r"\1", normalized)

    lines = [line.strip() for line in normalized.splitlines()]
    clean_lines = []
    for line in lines:
        if line:
            clean_line = re.sub(r"[ \t]+", " ", line)
            clean_lines.append(clean_line)

    return "\n".join(clean_lines)


def detect_section_heading(text: str, page_number: int) -> str:
    """
    Identifies section headings from the top of the page or text block.
    Matches patterns like '1 · ए.पी.जे. अब्दुल कलाम कौन थे?', '15 · जीवन-रेखा (टाइमलाइन)', etc.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return f"पृष्ठ {page_number}"

    heading_pattern = re.compile(r"^(\d+\s*[·\-–]\s*.+|विषय-सूची|अग्नि की उड़ान|प्रस्तावना|उपसंहार)", re.IGNORECASE)
    for line in lines[:3]:
        if heading_pattern.match(line):
            return line

    return lines[0] if lines else f"पृष्ठ {page_number}"


def extract_tables_from_page(page: fitz.Page, page_number: int) -> List[Dict[str, Any]]:
    """
    Extracts structured tables from a PDF page using PyMuPDF's table finder.
    Returns structured markdown table representation and metadata.
    """
    tables_data = []
    try:
        tabs = page.find_tables()
        if tabs and tabs.tables:
            for idx, tab in enumerate(tabs.tables):
                raw_rows = tab.extract()
                if not raw_rows or len(raw_rows) < 2:
                    continue

                cleaned_rows = []
                for row in raw_rows:
                    cleaned_row = [
                        normalize_hindi_text(cell).replace("\n", " ") if cell else ""
                        for cell in row
                    ]
                    cleaned_rows.append(cleaned_row)

                headers = cleaned_rows[0]
                md_table_lines = [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(["---"] * len(headers)) + " |",
                ]
                for r in cleaned_rows[1:]:
                    md_table_lines.append("| " + " | ".join(r) + " |")

                md_table_text = "\n".join(md_table_lines)
                tables_data.append({
                    "table_index": idx + 1,
                    "page_number": page_number,
                    "headers": headers,
                    "rows": cleaned_rows[1:],
                    "markdown": md_table_text,
                })
    except Exception as e:
        print(f"Warning: table extraction on page {page_number} encountered: {e}")

    return tables_data


def extract_pdf_document(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """
    Extracts text, section headings, and structured tables from all pages of the target PDF.
    Guarantees page number tracking and NFC normalization.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"Target PDF not found at {pdf_path}")

    doc = fitz.open(str(pdf_file))
    pages_data = []

    for page_idx in range(len(doc)):
        page_number = page_idx + 1
        page = doc[page_idx]

        raw_text = page.get_text("text")
        normalized_text = normalize_hindi_text(raw_text)
        section_heading = detect_section_heading(normalized_text, page_number)
        tables = extract_tables_from_page(page, page_number)

        pages_data.append({
            "page_number": page_number,
            "section_heading": section_heading,
            "text": normalized_text,
            "tables": tables,
            "char_count": len(normalized_text),
        })

    doc.close()
    return pages_data

"""
Hindi Sentence-Boundary & Table-Aware Chunking Engine.
Splits text respecting Devanagari sentence delimiters (।, ॥, ?, \n\n), creates
overlapping token/character chunks (150-300 tokens, 15-20% overlap), and preserves
table structures as dedicated table chunks.
"""

import re
from typing import Any, Dict, List
from src.ingest import normalize_hindi_text


def split_hindi_sentences(text: str) -> List[str]:
    """
    Splits Hindi text into distinct sentences honoring Devanagari delimiters
    (। - Purna Viram, ॥ - Deergh Viram, ?, !) and paragraph breaks.
    Protects common abbreviations (e.g. ए.पी.जे., डॉ., SLV-III).
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []

    for para in paragraphs:
        raw_splits = re.split(r"([।॥?!]+|\n+)", para)
        current = ""
        for part in raw_splits:
            if not part:
                continue
            current += part
            if re.match(r"[।॥?!]+", part) or "\n" in part:
                s = current.strip()
                if s:
                    sentences.append(s)
                current = ""
        if current.strip():
            sentences.append(current.strip())

    return sentences


def estimate_token_count(text: str) -> int:
    """
    Approximates token count for Hindi/multilingual text.
    In Devanagari subword tokenizers (like E5/XLM-RoBERTa), 1 token is ~3-4 characters or ~1 word.
    """
    words = text.split()
    return max(1, int(len(words) * 1.3))


def chunk_document(
    pages_data: List[Dict[str, Any]],
    source_doc_name: str = "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf",
    target_chunk_tokens: int = 220,
    overlap_tokens: int = 40,
) -> List[Dict[str, Any]]:
    """
    Produces structured chunks across all pages.
    1. Extracts dedicated structured chunks for tables (chunk_type: 'table').
    2. Groups prose sentences into cohesive chunks of ~150-300 tokens with 15-20% overlap.
    3. Attaches full metadata to every chunk.
    """
    all_chunks: List[Dict[str, Any]] = []
    chunk_id_counter = 1

    for page_info in pages_data:
        page_num = page_info["page_number"]
        section_heading = page_info["section_heading"]
        page_text = page_info["text"]
        tables = page_info.get("tables", [])

        for tab in tables:
            table_md = tab["markdown"]
            tab_chunk_text = (
                f"### तालिका (खंड: {section_heading} - पृष्ठ {page_num})\n"
                f"{table_md}"
            )
            tab_token_count = estimate_token_count(tab_chunk_text)

            all_chunks.append({
                "chunk_id": chunk_id_counter,
                "text": tab_chunk_text,
                "page_number": page_num,
                "section_heading": section_heading,
                "chunk_type": "table",
                "char_start": 0,
                "char_end": len(tab_chunk_text),
                "token_count": tab_token_count,
                "source_doc": source_doc_name,
            })
            chunk_id_counter += 1

        sentences = split_hindi_sentences(page_text)
        if not sentences:
            continue

        sentence_info_list = []
        cur_pos = 0
        for s in sentences:
            s_clean = normalize_hindi_text(s)
            s_tokens = estimate_token_count(s_clean)
            sentence_info_list.append({
                "text": s_clean,
                "tokens": s_tokens,
                "char_len": len(s_clean),
                "start": cur_pos,
                "end": cur_pos + len(s_clean),
            })
            cur_pos += len(s_clean) + 1

        idx = 0
        while idx < len(sentence_info_list):
            current_chunk_sentences = []
            current_tokens = 0
            start_char = sentence_info_list[idx]["start"]
            end_char = sentence_info_list[idx]["end"]

            j = idx
            while j < len(sentence_info_list):
                s_item = sentence_info_list[j]
                current_chunk_sentences.append(s_item["text"])
                current_tokens += s_item["tokens"]
                end_char = s_item["end"]
                j += 1
                if current_tokens >= target_chunk_tokens:
                    break

            chunk_text = " ".join(current_chunk_sentences).strip()
            if chunk_text:
                all_chunks.append({
                    "chunk_id": chunk_id_counter,
                    "text": chunk_text,
                    "page_number": page_num,
                    "section_heading": section_heading,
                    "chunk_type": "prose",
                    "char_start": start_char,
                    "char_end": end_char,
                    "token_count": current_tokens,
                    "source_doc": source_doc_name,
                })
                chunk_id_counter += 1

            if j >= len(sentence_info_list):
                break

            accumulated_overlap = 0
            step_back = 0
            for back_i in range(j - 1, idx, -1):
                accumulated_overlap += sentence_info_list[back_i]["tokens"]
                step_back += 1
                if accumulated_overlap >= overlap_tokens:
                    break

            next_idx = max(idx + 1, j - step_back)
            idx = next_idx

    return all_chunks

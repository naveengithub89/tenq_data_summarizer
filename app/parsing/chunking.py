from __future__ import annotations

from typing import Iterable, List

from app.parsing.models import TenQSection, TenQChunk


def simple_paragraph_chunker(
    section: TenQSection,
    max_chars: int = 2_000,
    overlap_chars: int = 200,
) -> List[TenQChunk]:
    """
    Very simple chunking: split section text on double newlines,
    group into ~max_chars size with overlap.
    """
    paragraphs = [p.strip() for p in section.text.split("\n\n") if p.strip()]
    chunks: list[TenQChunk] = []

    current_text = ""
    idx = 0
    for para in paragraphs:
        if len(current_text) + len(para) + 2 > max_chars and current_text:
            chunks.append(
                TenQChunk(
                    section_name=section.name,
                    section_item=section.item_number,
                    chunk_index=idx,
                    text=current_text.strip(),
                    metadata=section.metadata,
                )
            )
            idx += 1
            # carry overlap from the end of previous chunk
            current_text = current_text[-overlap_chars:]

        current_text += ("\n\n" if current_text else "") + para

    if current_text:
        chunks.append(
            TenQChunk(
                section_name=section.name,
                section_item=section.item_number,
                chunk_index=idx,
                text=current_text.strip(),
                metadata=section.metadata,
            )
        )

    return chunks

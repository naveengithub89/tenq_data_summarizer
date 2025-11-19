from __future__ import annotations

import re
from typing import List

from bs4 import BeautifulSoup  # add beautifulsoup4 to deps

from app.edgar.models import TenQMetadata
from app.parsing.chunking import simple_paragraph_chunker
from app.parsing.models import TenQSection, TenQChunk


ITEM_HEADING_RE = re.compile(r"item\s+(\d+[A-Z]?)\.\s*(.+)", re.IGNORECASE)


class TenQParser:
    """
    Parse 10-Q HTML into logical sections and chunks.

    v1 implementation uses heading heuristics only; you can refine this over time.
    """

    def parse_html(self, html: str, metadata: TenQMetadata) -> List[TenQChunk]:
        soup = BeautifulSoup(html, "html.parser")

        # convert to plaintext but keep some structure
        text = soup.get_text(separator="\n")
        lines = [l.strip() for l in text.splitlines()]

        sections: list[TenQSection] = []
        current_lines: list[str] = []
        current_name = "Unknown"
        current_item: str | None = None
        order_index = 0

        for line in lines:
            if not line:
                continue
            m = ITEM_HEADING_RE.match(line)
            if m:
                # flush previous
                if current_lines:
                    sections.append(
                        TenQSection(
                            name=current_name,
                            item_number=current_item,
                            order_index=order_index,
                            text="\n".join(current_lines),
                            metadata=metadata,
                        )
                    )
                    order_index += 1
                    current_lines = []

                current_item = m.group(1)
                current_name = m.group(2)
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                TenQSection(
                    name=current_name,
                    item_number=current_item,
                    order_index=order_index,
                    text="\n".join(current_lines),
                    metadata=metadata,
                )
            )

        chunks: list[TenQChunk] = []
        for section in sections:
            chunks.extend(simple_paragraph_chunker(section))

        return chunks

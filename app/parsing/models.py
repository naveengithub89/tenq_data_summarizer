from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.edgar.models import TenQMetadata


@dataclass
class TenQSection:
    name: str
    item_number: Optional[str]
    order_index: int
    text: str
    metadata: TenQMetadata


@dataclass
class TenQChunk:
    section_name: str
    section_item: Optional[str]
    chunk_index: int
    text: str
    metadata: TenQMetadata

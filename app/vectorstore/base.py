from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from app.parsing.models import TenQChunk


@dataclass
class ScoredChunk:
    chunk: TenQChunk
    score: float


class VectorStore(Protocol):
    async def upsert_chunks(self, chunks: Sequence[TenQChunk], embeddings: list[list[float]]) -> None:
        ...

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        *,
        ticker: str | None = None,
        cik: str | None = None,
        section_name: str | None = None,
    ) -> list[ScoredChunk]:
        ...

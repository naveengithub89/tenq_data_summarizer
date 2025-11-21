from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from app.parsing.models import TenQChunk


@dataclass
class ScoredChunk:
    chunk: TenQChunk
    score: float


class VectorStore(Protocol):
    async def upsert_chunks(
        self,
        chunks: Sequence[TenQChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Insert or update chunks with their embeddings.
        Implementations should be idempotent by using a stable key
        (e.g., accession_number + section + chunk_index).
        """
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
        """
        Vector similarity search with optional metadata filters.
        """
        ...

    async def has_accession(self, ticker: str, accession_number: str) -> bool:
        """
        Return True if any chunk exists for this ticker + accession.
        Used to gate network calls and ingestion.
        """
        ...

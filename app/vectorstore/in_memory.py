from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from app.parsing.models import TenQChunk
from app.vectorstore.base import ScoredChunk, VectorStore


@dataclass
class _Stored:
    chunk: TenQChunk
    embedding: list[float]


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store for dev & tests.
    NOTE: contents disappear when the process restarts.
    """

    def __init__(self) -> None:
        self._data: list[_Stored] = []

    async def upsert_chunks(
        self,
        chunks: Sequence[TenQChunk],
        embeddings: list[list[float]],
    ) -> None:
        for chunk, emb in zip(chunks, embeddings, strict=True):
            self._data.append(_Stored(chunk=chunk, embedding=emb))

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        *,
        ticker: str | None = None,
        cik: str | None = None,
        section_name: str | None = None,
    ) -> list[ScoredChunk]:
        def cos(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb + 1e-9)

        scored: list[ScoredChunk] = []
        for stored in self._data:
            md = stored.chunk.metadata
            if ticker and md.ticker.upper() != ticker.upper():
                continue
            if cik and md.cik != cik:
                continue
            if section_name and stored.chunk.section_name != section_name:
                continue

            scored.append(
                ScoredChunk(
                    chunk=stored.chunk,
                    score=cos(query_embedding, stored.embedding),
                )
            )

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    async def has_accession(self, ticker: str, accession_number: str) -> bool:
        t = ticker.upper()
        for stored in self._data:
            md = stored.chunk.metadata
            if md.ticker.upper() == t and md.accession_number == accession_number:
                return True
        return False

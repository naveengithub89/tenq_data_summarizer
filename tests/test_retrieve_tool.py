from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agents.insights_agent import retrieve_tenq_chunks, MAX_CHUNK_CHARS, MAX_TOP_K
from app.parsing.models import TenQChunk
from app.vectorstore.base import ScoredChunk


def md_obj(ticker: str, cik: str, accession: str):
    return SimpleNamespace(
        ticker=ticker,
        cik=cik,
        accession_number=accession,
        filing_date="2025-10-31",
        period_of_report="2025-09-27",
    )


class FakeEmbeddings:
    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self, scored: list[ScoredChunk]) -> None:
        self._scored = scored

    async def search(self, *args, **kwargs):
        # Respect top_k so retrieve_tenq_chunks can cap results.
        top_k = kwargs.get("top_k", len(self._scored))
        return self._scored[:top_k]


class FakeDeps(SimpleNamespace):
    def __init__(self, scored: list[ScoredChunk]) -> None:
        super().__init__(
            embeddings=FakeEmbeddings(),
            vector_store=FakeVectorStore(scored),
        )


class FakeCtx(SimpleNamespace):
    def __init__(self, deps) -> None:
        super().__init__(deps=deps)


@pytest.mark.asyncio
async def test_retrieve_tool_caps_top_k_and_truncates() -> None:
    long_text = "x" * (MAX_CHUNK_CHARS + 500)

    scored = [
        ScoredChunk(
            chunk=TenQChunk(
                section_name="MD&A",
                section_item="Item 2",
                chunk_index=i,
                text=long_text,
                metadata=md_obj("AAPL", "0000320193", "ACC-1"),
            ),
            score=1.0 - i * 0.01,
        )
        for i in range(MAX_TOP_K + 3)
    ]

    ctx = FakeCtx(FakeDeps(scored))
    out = await retrieve_tenq_chunks(ctx, ticker="AAPL", query="growth", top_k=999)

    assert len(out) == MAX_TOP_K
    for c in out:
        assert len(c.text) == MAX_CHUNK_CHARS

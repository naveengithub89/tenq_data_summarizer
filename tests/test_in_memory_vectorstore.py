from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.parsing.models import TenQChunk
from app.vectorstore.in_memory import InMemoryVectorStore


def md_obj(ticker: str, cik: str, accession: str):
    # Metadata must be attribute-accessible because InMemoryVectorStore does md.ticker etc.
    return SimpleNamespace(
        ticker=ticker,
        cik=cik,
        accession_number=accession,
        filing_date="2025-10-31",
        period_of_report="2025-09-27",
    )


@pytest.mark.asyncio
async def test_in_memory_vectorstore_has_accession() -> None:
    store = InMemoryVectorStore()

    chunks = [
        TenQChunk(
            section_name="MD&A",
            section_item="Item 2",
            chunk_index=0,
            text="hello world",
            metadata=md_obj("AAPL", "0000320193", "ACC-1"),
        )
    ]
    embeddings = [[1.0, 0.0, 0.0]]

    await store.upsert_chunks(chunks, embeddings)

    assert await store.has_accession("AAPL", "ACC-1") is True
    assert await store.has_accession("AAPL", "ACC-2") is False
    assert await store.has_accession("MSFT", "ACC-1") is False


@pytest.mark.asyncio
async def test_in_memory_vectorstore_search_filters() -> None:
    store = InMemoryVectorStore()

    chunks = [
        TenQChunk(
            section_name="MD&A",
            section_item="Item 2",
            chunk_index=0,
            text="foo apple",
            metadata=md_obj("AAPL", "0000320193", "ACC-1"),
        ),
        TenQChunk(
            section_name="Risk Factors",
            section_item="Item 1A",
            chunk_index=0,
            text="bar microsoft",
            metadata=md_obj("MSFT", "0000789019", "ACC-9"),
        ),
    ]

    embeddings = [
        [1.0, 0.0],  # close to query
        [0.0, 1.0],  # far from query
    ]

    await store.upsert_chunks(chunks, embeddings)

    results_all = await store.search([1.0, 0.0], top_k=10)
    assert len(results_all) == 2
    assert results_all[0].chunk.metadata.ticker == "AAPL"

    results_aapl = await store.search([1.0, 0.0], top_k=10, ticker="AAPL")
    assert len(results_aapl) == 1
    assert results_aapl[0].chunk.metadata.ticker == "AAPL"

    results_risk = await store.search([1.0, 0.0], top_k=10, section_name="Risk Factors")
    assert len(results_risk) == 1
    assert results_risk[0].chunk.section_name == "Risk Factors"

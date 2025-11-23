from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agents import orchestrator as orch
from app.edgar.models import TenQMetadata
from app.parsing.models import TenQChunk
from app.vectorstore.in_memory import InMemoryVectorStore


def md_obj(ticker: str, cik: str, accession: str):
    return SimpleNamespace(
        ticker=ticker,
        cik=cik,
        accession_number=accession,
        filing_date="2025-10-31",
        period_of_report="2025-09-27",
    )


class FakeCikResolver:
    async def resolve(self, ticker: str):
        return SimpleNamespace(cik_str="0000320193")


class FakeSubmissions:
    def __init__(self, meta: TenQMetadata):
        self.meta = meta
        self.fetch_called = 0

    async def fetch_submissions(self, cik: str):
        self.fetch_called += 1
        return {"dummy": True}

    def select_latest_10q(self, subs, target_period=None):
        return self.meta


class FakeDownloader:
    def __init__(self):
        self.download_called = 0

    async def download_primary_html(self, tenq_meta: TenQMetadata) -> str:
        self.download_called += 1
        return "filings/AAPL/test.htm"


class FakeParser:
    def parse_html(self, html: str, tenq_meta: TenQMetadata):
        return [
            TenQChunk(
                section_name="MD&A",
                section_item="Item 2",
                chunk_index=0,
                text="short text",
                metadata=md_obj(
                    tenq_meta.ticker,
                    tenq_meta.cik,
                    tenq_meta.accession_number,
                ),
            )
        ]


class FakeEmbeddings:
    async def embed_many(self, texts: list[str]):
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_orchestrator_skips_sec_when_cached_and_ingested(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    meta = TenQMetadata(
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        form_type="10-Q",
        filing_date=date(2025, 10, 31),
        period_of_report=date(2025, 9, 27),
        accession_number="ACC-1",
        primary_document="doc.htm",
    )

    store = InMemoryVectorStore()
    await store.upsert_chunks(
        [
            TenQChunk(
                section_name="MD&A",
                section_item="Item 2",
                chunk_index=0,
                text="x",
                metadata=md_obj("AAPL", "0000320193", "ACC-1"),
            )
        ],
        [[1.0, 0.0]],
    )

    fake_subs = FakeSubmissions(meta)
    fake_dl = FakeDownloader()

    deps = orch.AgentDependencies(
        edgar_client=None,
        cik_resolver=FakeCikResolver(),
        submissions=fake_subs,
        filing_downloader=fake_dl,
        tenq_parser=FakeParser(),
        embeddings=FakeEmbeddings(),
        vector_store=store,
    )

    # Patch cache file to tmp_path
    from app.edgar.metadata_cache import TenQMetadataCache
    cache_file = tmp_path / "cache.json"
    cache = TenQMetadataCache(cache_file)
    cache.set_latest("AAPL", meta)
    monkeypatch.setattr(orch, "TenQMetadataCache", lambda: TenQMetadataCache(cache_file))

    # Patch agents to avoid LLM
    async def fake_run(*args, **kwargs):
        return SimpleNamespace(output=SimpleNamespace(model_dump_json=lambda: "{}"))

    monkeypatch.setattr(orch.insights_agent, "run", fake_run)
    monkeypatch.setattr(orch.decision_agent, "run", fake_run)

    await orch.summarize_10q_for_ticker("AAPL", deps=deps)

    assert fake_subs.fetch_called == 0
    assert fake_dl.download_called == 0


@pytest.mark.asyncio
async def test_orchestrator_reingests_when_dates_change(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    old_meta = TenQMetadata(
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        form_type="10-Q",
        filing_date=date(2025, 7, 31),
        period_of_report=date(2025, 6, 28),
        accession_number="ACC-OLD",
        primary_document="old.htm",
    )

    new_meta = TenQMetadata(
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        form_type="10-Q",
        filing_date=date(2025, 10, 31),
        period_of_report=date(2025, 9, 27),
        accession_number="ACC-NEW",
        primary_document="new.htm",
    )

    store = InMemoryVectorStore()
    fake_subs = FakeSubmissions(new_meta)
    fake_dl = FakeDownloader()

    # Create stub file for downloader return
    data_file = tmp_path / "data" / "filings" / "AAPL" / "test.htm"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text("<html>stub</html>", encoding="utf-8")

    deps = orch.AgentDependencies(
        edgar_client=None,
        cik_resolver=FakeCikResolver(),
        submissions=fake_subs,
        filing_downloader=fake_dl,
        tenq_parser=FakeParser(),
        embeddings=FakeEmbeddings(),
        vector_store=store,
    )

    from app.edgar.metadata_cache import TenQMetadataCache
    cache_file = tmp_path / "cache.json"
    cache = TenQMetadataCache(cache_file)
    cache.set_latest("AAPL", old_meta)
    monkeypatch.setattr(orch, "TenQMetadataCache", lambda: TenQMetadataCache(cache_file))

    async def fake_run(*args, **kwargs):
        return SimpleNamespace(output=SimpleNamespace(model_dump_json=lambda: "{}"))

    monkeypatch.setattr(orch.insights_agent, "run", fake_run)
    monkeypatch.setattr(orch.decision_agent, "run", fake_run)

    await orch.summarize_10q_for_ticker("AAPL", deps=deps)

    assert fake_subs.fetch_called == 1
    assert fake_dl.download_called == 1
    assert await store.has_accession("AAPL", "ACC-NEW") is True

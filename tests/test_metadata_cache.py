from __future__ import annotations

from datetime import date
from pathlib import Path

from app.edgar.metadata_cache import TenQMetadataCache
from app.edgar.models import TenQMetadata


def test_metadata_cache_roundtrip(tmp_path: Path) -> None:
    cache_path = tmp_path / "latest_tenq.json"
    cache = TenQMetadataCache(cache_path)

    meta = TenQMetadata(
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        form_type="10-Q",
        filing_date=date(2025, 10, 31),
        period_of_report=date(2025, 9, 27),
        accession_number="ACC-1",
        primary_document="aapl-20250927x10q.htm",
    )

    # empty cache
    assert cache.get_latest("AAPL") is None

    # set + get (case-insensitive)
    cache.set_latest("AAPL", meta)
    cached = cache.get_latest("aapl")
    assert cached is not None

    # CachedTenQMetadata stores only gating fields
    assert cached.accession_number == "ACC-1"
    assert cached.filing_date == date(2025, 10, 31)
    assert cached.period_of_report == date(2025, 9, 27)


def test_metadata_cache_matches_only_on_dates(tmp_path: Path) -> None:
    cache = TenQMetadataCache(tmp_path / "latest_tenq.json")

    old = TenQMetadata(
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        form_type="10-Q",
        filing_date=date(2025, 7, 31),
        period_of_report=date(2025, 6, 28),
        accession_number="ACC-OLD",
        primary_document="old.htm",
    )

    same_dates = TenQMetadata(
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        form_type="10-Q/A",
        filing_date=date(2025, 7, 31),
        period_of_report=date(2025, 6, 28),
        accession_number="ACC-NEW",
        primary_document="new.htm",
    )

    diff_dates = TenQMetadata(
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        form_type="10-Q",
        filing_date=date(2025, 10, 31),
        period_of_report=date(2025, 9, 27),
        accession_number="ACC-LATEST",
        primary_document="latest.htm",
    )

    cache.set_latest("AAPL", old)
    cached = cache.get_latest("AAPL")
    assert cached is not None

    assert cached.matches(same_dates) is True
    assert cached.matches(diff_dates) is False

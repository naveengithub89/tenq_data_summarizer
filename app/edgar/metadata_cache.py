from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Optional

from app.edgar.models import TenQMetadata


@dataclass
class CachedTenQMetadata:
    cik: str
    accession_number: str
    filing_date: date
    period_of_report: Optional[date]
    primary_document: str

    @classmethod
    def from_tenq(cls, m: TenQMetadata) -> "CachedTenQMetadata":
        return cls(
            cik=m.cik,
            accession_number=m.accession_number,
            filing_date=m.filing_date,
            period_of_report=m.period_of_report,
            primary_document=m.primary_document,
        )

    def matches(self, m: TenQMetadata) -> bool:
        """
        Match only on the fields requested:
        - filed date
        - period end date (period_of_report)
        """
        return (
            self.filing_date == m.filing_date
            and self.period_of_report == m.period_of_report
        )


class TenQMetadataCache:
    """
    Disk-backed JSON cache of "latest 10-Q metadata per ticker".

    Used to skip SEC calls entirely when:
      - cached metadata exists AND
      - that accession is already ingested into the vector store.

    Stored at: data/cache/latest_tenq.json
    """

    def __init__(self, path: Path = Path("data/cache/latest_tenq.json")) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load_all(self) -> Dict[str, dict]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save_all(self, data: Dict[str, dict]) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_latest(self, ticker: str) -> Optional[CachedTenQMetadata]:
        data = self._load_all()
        key = ticker.upper()
        if key not in data:
            return None

        entry = data[key]
        return CachedTenQMetadata(
            cik=entry["cik"],
            accession_number=entry["accession_number"],
            filing_date=date.fromisoformat(entry["filing_date"]),
            period_of_report=(
                date.fromisoformat(entry["period_of_report"])
                if entry.get("period_of_report")
                else None
            ),
            primary_document=entry["primary_document"],
        )

    def set_latest(self, ticker: str, meta: TenQMetadata) -> None:
        data = self._load_all()
        key = ticker.upper()
        cached = CachedTenQMetadata.from_tenq(meta)

        data[key] = {
            "cik": cached.cik,
            "accession_number": cached.accession_number,
            "filing_date": cached.filing_date.isoformat(),
            "period_of_report": (
                cached.period_of_report.isoformat()
                if cached.period_of_report
                else None
            ),
            "primary_document": cached.primary_document,
        }
        self._save_all(data)

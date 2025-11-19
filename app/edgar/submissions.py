from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

from pydantic import BaseModel

from app.edgar.client import EdgarHttpClient
from app.edgar.models import CompanySubmissions, TenQMetadata, CikResolutionResult


class SubmissionsService:
    def __init__(self, client: EdgarHttpClient) -> None:
        self._client = client

    async def fetch_submissions(self, cik_str: str) -> CompanySubmissions:
        path = f"submissions/CIK{cik_str}.json"
        data = await self._client.get_json(path, data_host=True)

        recent = data["filings"]["recent"]
        forms: list[str] = list(recent["form"])
        filing_dates: list[date] = [date.fromisoformat(d) for d in recent["filingDate"]]
        accession_numbers: list[str] = list(recent["accessionNumber"])
        primary_docs: list[str] = list(recent["primaryDocument"])
        periods_raw: Iterable[Optional[str]] = recent.get("periodOfReport", [])
        periods = [
            date.fromisoformat(p) if p else None
            for p in periods_raw
        ]

        return CompanySubmissions(
            cik=data["cik"],
            entity_type=data.get("entityType"),
            sic=data.get("sic"),
            name=data["name"],
            tickers=data.get("tickers", []),
            forms=forms,
            filing_dates=filing_dates,
            accession_numbers=accession_numbers,
            primary_docs=primary_docs,
            periods=periods,
        )

    def select_latest_10q(
        self,
        submissions: CompanySubmissions,
        target_period: Optional[date] = None,
    ) -> Optional[TenQMetadata]:
        """
        Filter for 10-Q / 10-Q/A. If target_period is provided,
        prefer that period; otherwise take the most recent filing_date.
        """
        indices = [
            i
            for i, form in enumerate(submissions.forms)
            if form in {"10-Q", "10-Q/A"}
        ]
        if not indices:
            return None

        # naive strategy: sort by filing_date desc, optionally filter by target_period
        candidates = [
            (i, submissions.filing_dates[i])
            for i in indices
        ]

        if target_period:
            candidates = [c for c in candidates if submissions.periods[c[0]] == target_period]
            if not candidates:
                return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        idx = candidates[0][0]

        return TenQMetadata(
            ticker=submissions.tickers[0] if submissions.tickers else "",
            cik=submissions.cik,
            company_name=submissions.name,
            form_type=submissions.forms[idx],
            filing_date=submissions.filing_dates[idx],
            period_of_report=submissions.periods[idx] if submissions.periods else None,
            accession_number=submissions.accession_numbers[idx],
            primary_document=submissions.primary_docs[idx],
        )

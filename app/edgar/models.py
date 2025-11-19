from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class CikResolutionResult(BaseModel):
    ticker: str
    cik_str: str = Field(description="Zero-padded CIK string")
    cik_int: int
    company_name: str


class CompanySubmissions(BaseModel):
    cik: str
    entity_type: Optional[str] = None
    sic: Optional[str] = None
    name: str
    tickers: list[str] = []
    # Highly simplified view of the SEC submissions JSON:
    forms: list[str]
    filing_dates: list[date]
    accession_numbers: list[str]
    primary_docs: list[str]
    periods: list[Optional[date]] = []


class TenQMetadata(BaseModel):
    ticker: str
    cik: str
    company_name: str
    form_type: str
    filing_date: date
    period_of_report: Optional[date]
    accession_number: str
    primary_document: str
    xbrl_instance_document: Optional[str] = None

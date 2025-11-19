from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.edgar.models import TenQMetadata


class CompanyProfile(BaseModel):
    name: str
    ticker: str
    cik: str
    sector: Optional[str] = None
    industry: Optional[str] = None


class FinancialSummary(BaseModel):
    key_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="E.g. revenue, operating income, EPS, FCF etc.",
    )
    narrative: str = ""


class RiskItem(BaseModel):
    title: str
    description: str
    changed_since_prior: bool = False


class NotableEvent(BaseModel):
    category: str  # e.g. "M&A", "Litigation", "Restructuring"
    summary: str


class LiquiditySummary(BaseModel):
    narrative: str
    leverage_metrics: dict[str, float] = Field(default_factory=dict)


class GuidanceSummary(BaseModel):
    narrative: str
    time_horizon: Optional[str] = None


class TenQInsights(BaseModel):
    company_profile: CompanyProfile
    filing_metadata: TenQMetadata
    high_level_summary: str
    financial_summary: FinancialSummary
    risk_summary: List[RiskItem] = []
    notable_events: List[NotableEvent] = []
    liquidity_and_capital_structure: LiquiditySummary
    guidance_and_outlook: GuidanceSummary
    open_questions: List[str] = []


class DecisionEnum(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class DecisionOutput(BaseModel):
    decision: DecisionEnum
    confidence: float = Field(ge=0.0, le=1.0)
    time_horizon: str
    positives: list[str]
    negatives: list[str]
    uncertainties: list[str]
    risk_profile: str
    disclaimer: str = (
        "This is an automated, heuristic assessment based on the latest 10-Q filing and "
        "does not constitute investment advice. Do your own research."
    )

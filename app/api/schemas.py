from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel

from app.agents.models import TenQInsights, DecisionOutput


class TenQSummaryRequest(BaseModel):
    ticker: str
    filing_period: Optional[date] = None
    force_refresh: bool = False


class TenQSummaryResponse(BaseModel):
    insights: TenQInsights
    decision: DecisionOutput

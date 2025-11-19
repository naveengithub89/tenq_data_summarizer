from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.agents.orchestrator import summarize_10q_for_ticker
from app.api.schemas import TenQSummaryRequest, TenQSummaryResponse

app = FastAPI(title="SEC 10-Q Analyst API")


@app.post("/summaries/10q", response_model=TenQSummaryResponse)
async def summarize_10q(req: TenQSummaryRequest) -> TenQSummaryResponse:
    try:
        insights, decision = await summarize_10q_for_ticker(
            ticker=req.ticker,
            filing_period=req.filing_period,
        )
    except Exception as exc:  # tighten this over time
        raise HTTPException(status_code=400, detail=str(exc))

    return TenQSummaryResponse(insights=insights, decision=decision)

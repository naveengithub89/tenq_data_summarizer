from __future__ import annotations

from pydantic_ai import Agent, RunContext

from app.agents.dependencies import AgentDependencies
from app.agents.models import TenQInsights
from app.config.settings import get_settings
from app.parsing.models import TenQChunk
from app.vectorstore.base import ScoredChunk

settings = get_settings()

# ---------------------------------------------------------------------------
# Prompt template (formatted, dynamic fields)
# ---------------------------------------------------------------------------

INSIGHTS_PROMPT_TEMPLATE = """
ROLE:
You are an elite equity research analyst at a top-tier investment fund.
Your task is to analyze a company using both fundamental and macroeconomic perspectives.
Structure your response according to the framework below.

INPUT (filled by system):
- Stock Ticker / Company Name: {ticker}
- Investment Thesis: {thesis}
- Goal: {goal}

INSTRUCTIONS:
Use the following structure to deliver a clear, well-reasoned equity research report.

1. Fundamental Analysis
- Analyze revenue growth, gross margin trends, net margin trends, and free cash flow.
- Compare valuation metrics vs. sector peers (P/E, EV/EBITDA, P/S, etc.), if disclosed in the filing.
- Review insider ownership and recent insider trades, if disclosed in the filing.

2. Thesis Validation
- Provide **3 arguments supporting the thesis**, grounded in the 10-Q.
- Provide **2 counter-arguments / key risks**, grounded in the 10-Q.
- Give a final **verdict**: Bullish / Bearish / Neutral with justification.

3. Sector & Macro View
- Brief sector overview (based only on what the filing implies or explicitly states).
- Relevant macroeconomic trends impacting the company (only if mentioned or clearly indicated in the filing).
- Company’s competitive positioning vs. peers (from the filing).

4. Catalyst Watch
- List upcoming events (earnings, launches, regulation, litigation milestones, etc.) mentioned in the 10-Q.
- Identify **short-term** and **long-term** catalysts.

5. Investment Summary
- 5-bullet investment thesis recap.
- Final recommendation: **Buy / Hold / Sell**.
- Confidence level: High / Medium / Low.
- Expected timeframe (e.g., 6–12 months).

FORMATTING REQUIREMENTS:
- Use markdown *inside string fields* where helpful (bullets, bold, headings).
- Be concise, professional, and insight-driven.
- Do NOT explain your process.

TOOLING & GROUNDING RULES:
- Use `retrieve_tenq_chunks` sparingly (max 3 calls).
- Request only a small number of chunks per call.
- Base claims strictly on retrieved 10-Q text; do NOT invent numbers.
- If a required fact is not in the filing, say **"not disclosed"** or **"unknown"**.

OUTPUT RULE (CRITICAL):
Return ONLY valid JSON that matches the TenQInsights schema.
- Do NOT wrap JSON in ``` fences.
- Do NOT add any text outside the JSON.
- You MAY use markdown within JSON string values.
""".strip()


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

insights_agent = Agent(
    settings.llm_model,
    deps_type=AgentDependencies,
    output_type=TenQInsights,
    # We keep instructions concise and stable; the big template goes in the user prompt.
    instructions=(
        "You are an equity research analyst. "
        "Follow the user's prompt, which provides the detailed analysis framework. "
        "Use tools when needed, ground all claims in retrieved 10-Q text, "
        "and return ONLY JSON that matches the TenQInsights schema, with no extra text."
    ),
    retries=3,
    output_retries=3,
)

MAX_CHUNK_CHARS = 1500
MAX_TOP_K = 5


@insights_agent.tool
async def retrieve_tenq_chunks(
    ctx: RunContext[AgentDependencies],
    ticker: str,
    query: str,
    top_k: int = 20,
) -> list[TenQChunk]:
    """
    Retrieve a SMALL set of chunks for a given ticker relevant to the query.

    - top_k is capped to avoid exceeding context.
    - chunk text is truncated to MAX_CHUNK_CHARS.
    """
    effective_top_k = min(top_k, MAX_TOP_K)

    embeds = await ctx.deps.embeddings.embed_many([query])
    scored: list[ScoredChunk] = await ctx.deps.vector_store.search(
        embeds[0],
        top_k=effective_top_k,
        ticker=ticker,
    )

    limited_chunks: list[TenQChunk] = []
    for s in scored:
        chunk = s.chunk
        if len(chunk.text) > MAX_CHUNK_CHARS:
            limited_chunks.append(
                TenQChunk(
                    section_name=chunk.section_name,
                    section_item=chunk.section_item,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text[:MAX_CHUNK_CHARS],
                    metadata=chunk.metadata,
                )
            )
        else:
            limited_chunks.append(chunk)

    return limited_chunks


def build_insights_prompt(
    ticker: str,
    thesis: str | None = None,
    goal: str | None = None,
) -> str:
    """
    Helper to format the big analysis prompt.

    If thesis/goal aren't provided by the caller (e.g. API only supplies ticker),
    we fill them with sensible defaults.
    """
    thesis_text = thesis or "Not explicitly specified; infer a reasonable thesis from the latest 10-Q."
    goal_text = goal or "Summarize and analyze the latest 10-Q into the requested equity research structure."

    return INSIGHTS_PROMPT_TEMPLATE.format(
        ticker=ticker,
        thesis=thesis_text,
        goal=goal_text,
    )
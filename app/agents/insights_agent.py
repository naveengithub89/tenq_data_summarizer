from __future__ import annotations

from pydantic_ai import Agent, RunContext

from app.agents.dependencies import AgentDependencies
from app.agents.models import TenQInsights
from app.config.settings import get_settings
from app.parsing.models import TenQChunk
from app.vectorstore.base import ScoredChunk

settings = get_settings()

insights_agent = Agent(
    settings.llm_model,
    deps_type=AgentDependencies,
    output_type=TenQInsights,
    instructions=(
        "ROLE:\n\n"
        "Act as an elite equity research analyst at a top-tier investment fund. "
        "Your task is to analyze a company using both fundamental and macroeconomic perspectives. "
        "Structure your response according to the framework below.\n\n"
        "Input Section:\n"
        "- The user prompt will include:\n"
        "  • Stock Ticker / Company Name\n"
        "  • Investment Thesis\n"
        "  • Goal\n\n"
        "Instructions:\n\n"
        "Use the following structure to deliver a clear, well-reasoned equity research report:\n\n"
        "1. Fundamental Analysis\n"
        "- Analyze revenue growth, gross & net margin trends, free cash flow\n"
        "- Compare valuation metrics vs sector peers (P/E, EV/EBITDA, etc.)\n"
        "- Review insider ownership and recent insider trades\n\n"
        "2. Thesis Validation\n"
        "- Present 3 arguments supporting the thesis\n"
        "- Highlight 2 counter-arguments or key risks\n"
        "- Provide a final verdict: Bullish / Bearish / Neutral with justification\n\n"
        "3. Sector & Macro View\n"
        "- Give a short sector overview\n"
        "- Outline relevant macroeconomic trends\n"
        "- Explain company’s competitive positioning\n\n"
        "4. Catalyst Watch\n"
        "- List upcoming events (earnings, product launches, regulation, etc.)\n"
        "- Identify both short-term and long-term catalysts\n\n"
        "5. Investment Summary\n"
        "- 5-bullet investment thesis summary\n"
        "- Final recommendation: Buy / Hold / Sell\n"
        "- Confidence level (High / Medium / Low)\n"
        "- Expected timeframe (e.g. 6–12 months)\n\n"
        "✅ Formatting Requirements\n\n"
        "- Use markdown\n"
        "- Use bullet points where appropriate\n"
        "- Be concise, professional, and insight-driven\n"
        "- Do not explain your process; just deliver the analysis\n\n"
        "Tooling & grounding rules:\n"
        "- Use retrieve_tenq_chunks sparingly (max 3 calls).\n"
        "- Request only a small number of chunks per call.\n"
        "- Base claims strictly on retrieved 10-Q text; do not invent numbers.\n"
        "- If a required fact is not in the filing, say it's not disclosed or unknown."
    ),
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

from __future__ import annotations

from typing import List

from pydantic_ai import Agent, RunContext

from app.agents.dependencies import AgentDependencies
from app.agents.models import TenQInsights
from app.parsing.models import TenQChunk
from app.vectorstore.base import ScoredChunk
from app.config.settings import get_settings

settings = get_settings()

insights_agent = Agent(
    settings.llm_model,
    deps_type=AgentDependencies,
    output_type=TenQInsights,
    instructions=(
        "You are an equity research assistant analyzing 10-Q filings.\n"
        "- Use the retrieve_tenq_chunks tool sparingly (at most 3 calls).\n"
        "- Request only a small number of chunks per call.\n"
        "- Base your analysis strictly on retrieved text; do not invent numbers.\n"
        "- Summarize; do NOT paste long passages from the filing."
    ),
)

MAX_CHUNK_CHARS = 1500       # hard cap per chunk returned to the LLM
MAX_TOP_K = 5                # hard cap on how many chunks per call


@insights_agent.tool
async def retrieve_tenq_chunks(
    ctx: RunContext[AgentDependencies],
    ticker: str,
    query: str,
    top_k: int = 20,
) -> list[TenQChunk]:
    """
    Retrieve a SMALL set of chunks for a given ticker relevant to the query.

    IMPORTANT:
    - top_k is capped to avoid exceeding the model context window.
    - Each chunk's text is truncated to MAX_CHUNK_CHARS characters.
    """
    # Enforce upper bound on retrieved chunks
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
            # create a shallow copy with truncated text
            truncated = TenQChunk(
                section_name=chunk.section_name,
                section_item=chunk.section_item,
                chunk_index=chunk.chunk_index,
                text=chunk.text[:MAX_CHUNK_CHARS],
                metadata=chunk.metadata,
            )
            limited_chunks.append(truncated)
        else:
            limited_chunks.append(chunk)

    return limited_chunks

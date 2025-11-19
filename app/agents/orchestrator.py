from __future__ import annotations

from datetime import date
from pathlib import Path

from app.agents.dependencies import AgentDependencies
from app.agents.insights_agent import insights_agent
from app.agents.decision_agent import decision_agent
from app.agents.models import TenQInsights, DecisionOutput
from app.config.settings import get_settings
from app.edgar.client import EdgarHttpClient
from app.edgar.cik_resolver import CikResolver
from app.edgar.submissions import SubmissionsService
from app.edgar.downloader import FilingDownloader
from app.edgar.storage import LocalFileStorage
from app.parsing.tenq_parser import TenQParser
from app.vectorstore.embeddings import EmbeddingService
from app.vectorstore.in_memory import InMemoryVectorStore


async def build_default_deps() -> AgentDependencies:
    settings = get_settings()
    edgar_client = EdgarHttpClient()
    cik_resolver = CikResolver(edgar_client)
    submissions = SubmissionsService(edgar_client)
    storage = LocalFileStorage(Path("data"))
    filing_downloader = FilingDownloader(edgar_client, storage)
    parser = TenQParser()
    embeddings = EmbeddingService(settings.embedding_model)
    vector_store = InMemoryVectorStore()  # swap to pgvector in prod
    return AgentDependencies(
        edgar_client=edgar_client,
        cik_resolver=cik_resolver,
        submissions=submissions,
        filing_downloader=filing_downloader,
        tenq_parser=parser,
        embeddings=embeddings,
        vector_store=vector_store,
    )


async def summarize_10q_for_ticker(
    ticker: str,
    filing_period: date | None = None,
    *,
    deps: AgentDependencies | None = None,
) -> tuple[TenQInsights, DecisionOutput]:
    """
    High-level orchestration:
      * resolve ticker -> CIK
      * fetch submissions & locate 10-Q
      * download + parse + embed + upsert
      * run insights agent
      * run decision agent
    """
    deps = deps or await build_default_deps()

    # Resolve ticker
    cik_info = await deps.cik_resolver.resolve(ticker)

    # Submissions and 10-Q selection
    subs = await deps.submissions.fetch_submissions(cik_info.cik_str)
    tenq_meta = deps.submissions.select_latest_10q(subs, target_period=filing_period)
    if tenq_meta is None:
        raise RuntimeError(f"No 10-Q found for ticker {ticker}")

    # Download / parse / chunk / embed / upsert
    rel_path = await deps.filing_downloader.download_primary_html(tenq_meta)
    from pathlib import Path

    html = Path("data").joinpath(rel_path).read_text(encoding="utf-8", errors="ignore")
    chunks = deps.tenq_parser.parse_html(html, tenq_meta)
    embeddings = await deps.embeddings.embed_many([c.text for c in chunks])
    await deps.vector_store.upsert_chunks(chunks, embeddings)

    # Run insights agent
    insights_result = await insights_agent.run(
        f"Produce a structured analysis of the latest 10-Q for {ticker}.",
        deps=deps,
    )
    insights: TenQInsights = insights_result.output

    # Build a prompt that includes the structured insights as JSON
    decision_prompt = (
        "You are an equity analyst.\n\n"
        "You're given structured 10-Q insights in JSON format below.\n"
        "Based ONLY on this information, provide a Buy/Sell/Hold style view, "
        "with a clear rationale and risks. Remember this is not investment advice.\n\n"
        f"INSIGHTS_JSON:\n{insights.model_dump_json()}"
    )

    # Run decision agent (no tools kwarg)
    decision_result = await decision_agent.run(
        decision_prompt,
        deps=deps,
    )
    decision: DecisionOutput = decision_result.output

    return insights, decision

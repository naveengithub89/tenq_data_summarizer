from __future__ import annotations

from datetime import date
from pathlib import Path

from app.agents.dependencies import AgentDependencies
from app.agents.decision_agent import decision_agent
from app.agents.insights_agent import insights_agent
from app.agents.models import DecisionOutput, TenQInsights
from app.config.settings import get_settings
from app.edgar.client import EdgarHttpClient
from app.edgar.cik_resolver import CikResolver
from app.edgar.downloader import FilingDownloader
from app.edgar.metadata_cache import TenQMetadataCache
from app.edgar.storage import LocalFileStorage
from app.edgar.submissions import SubmissionsService
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
    vector_store = InMemoryVectorStore()

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
    force_refresh: bool = False,
    thesis: str | None = None,
    goal: str | None = None,
) -> tuple[TenQInsights, DecisionOutput]:
    """
    Top-level orchestration:
      * Skip SEC entirely if cached latest metadata exists AND already ingested
      * Else call SEC submissions, compare (filing_date, period_of_report)
      * Re-ingest only when metadata changed or ingestion missing
      * Run insights + decision agents

    thesis/goal are optional inputs to populate the Input Section.
    """
    deps = deps or await build_default_deps()
    cache = TenQMetadataCache()

    ticker_norm = ticker.upper()

    # Default thesis/goal if user doesn't provide
    thesis_text = thesis or (
        "Not provided. Infer a reasonable investment thesis from the latest 10-Q, "
        "company fundamentals, and typical analyst framing."
    )
    goal_text = goal or (
        "Deliver an actionable equity research view primarily grounded in the latest 10-Q."
    )

    def make_insights_prompt() -> str:
        return (
            "ROLE:\n\n"
            "Act as an elite equity research analyst at a top-tier investment fund.\n\n"
            "Input Section (Fill this in)\n\n"
            f"Stock Ticker / Company Name: {ticker_norm}\n"
            f"Investment Thesis: {thesis_text}\n"
            f"Goal: {goal_text}\n\n"
            "Follow the required structure exactly. Use retrieved 10-Q text as the primary source."
        )

    # -------- Stage A: skip SEC entirely if cache+vectors are valid --------
    cached_latest = cache.get_latest(ticker_norm)
    if cached_latest and not force_refresh:
        already_ingested = await deps.vector_store.has_accession(
            ticker_norm, cached_latest.accession_number
        )
        if already_ingested:
            insights_result = await insights_agent.run(
                make_insights_prompt(),
                deps=deps,
            )
            insights: TenQInsights = insights_result.output

            decision_prompt = (
                "You are an equity analyst.\n\n"
                "You're given structured 10-Q insights in JSON format below.\n"
                "Based ONLY on this information, provide a Buy/Sell/Hold style view, "
                "with clear rationale and risks. Remember this is not investment advice.\n\n"
                f"INSIGHTS_JSON:\n{insights.model_dump_json()}"
            )

            decision_result = await decision_agent.run(decision_prompt, deps=deps)
            decision: DecisionOutput = decision_result.output
            return insights, decision

    # -------- Otherwise: call SEC to check for updates --------

    # Resolve ticker -> CIK (may hit SEC if resolver TTL expired)
    cik_info = await deps.cik_resolver.resolve(ticker_norm)

    # Fetch submissions (SEC call)
    subs = await deps.submissions.fetch_submissions(cik_info.cik_str)

    # Select latest 10-Q
    tenq_meta = deps.submissions.select_latest_10q(subs, target_period=filing_period)
    if tenq_meta is None:
        raise RuntimeError(f"No 10-Q found for ticker {ticker_norm}")

    # -------- Stage B: compare metadata to decide re-ingest --------
    if cached_latest and not force_refresh and cached_latest.matches(tenq_meta):
        already_ingested = await deps.vector_store.has_accession(
            ticker_norm, tenq_meta.accession_number
        )
        if already_ingested:
            # Skip download/parse/embed
            insights_result = await insights_agent.run(
                make_insights_prompt(),
                deps=deps,
            )
            insights: TenQInsights = insights_result.output

            decision_prompt = (
                "You are an equity analyst.\n\n"
                "You're given structured 10-Q insights in JSON format below.\n"
                "Based ONLY on this information, provide a Buy/Sell/Hold style view.\n\n"
                f"INSIGHTS_JSON:\n{insights.model_dump_json()}"
            )

            decision_result = await decision_agent.run(decision_prompt, deps=deps)
            decision: DecisionOutput = decision_result.output
            return insights, decision

    # -------- Ingest because it's new or missing --------
    rel_path = await deps.filing_downloader.download_primary_html(tenq_meta)
    html = Path("data").joinpath(rel_path).read_text(encoding="utf-8", errors="ignore")

    chunks = deps.tenq_parser.parse_html(html, tenq_meta)
    embeddings = await deps.embeddings.embed_many([c.text for c in chunks])
    await deps.vector_store.upsert_chunks(chunks, embeddings)

    # Update cache to new "latest"
    cache.set_latest(ticker_norm, tenq_meta)

    # Run insights agent
    insights_result = await insights_agent.run(
        make_insights_prompt(),
        deps=deps,
    )
    insights: TenQInsights = insights_result.output

    # Run decision agent
    decision_prompt = (
        "You are an equity analyst.\n\n"
        "You're given structured 10-Q insights in JSON format below.\n"
        "Based ONLY on this information, provide a Buy/Sell/Hold style view.\n\n"
        f"INSIGHTS_JSON:\n{insights.model_dump_json()}"
    )
    decision_result = await decision_agent.run(decision_prompt, deps=deps)
    decision: DecisionOutput = decision_result.output

    return insights, decision

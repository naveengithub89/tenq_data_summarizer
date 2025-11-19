from __future__ import annotations

from dataclasses import dataclass

from app.edgar.client import EdgarHttpClient
from app.edgar.cik_resolver import CikResolver
from app.edgar.submissions import SubmissionsService
from app.edgar.downloader import FilingDownloader
from app.parsing.tenq_parser import TenQParser
from app.vectorstore.embeddings import EmbeddingService
from app.vectorstore.base import VectorStore


@dataclass
class AgentDependencies:
    edgar_client: EdgarHttpClient
    cik_resolver: CikResolver
    submissions: SubmissionsService
    filing_downloader: FilingDownloader
    tenq_parser: TenQParser
    embeddings: EmbeddingService
    vector_store: VectorStore

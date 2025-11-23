from __future__ import annotations

from pathlib import PurePosixPath

from app.edgar.client import EdgarHttpClient
from app.edgar.models import TenQMetadata
from app.edgar.storage import StorageBackend


class FilingDownloader:
    """
    Download primary 10-Q HTML (and optionally XBRL) and persist
    via a pluggable storage backend.
    """

    def __init__(
        self,
        client: EdgarHttpClient,
        storage: StorageBackend,
    ) -> None:
        self._client = client
        self._storage = storage
        self._archives_base = "https://www.sec.gov/Archives/edgar/data"

    def _build_primary_url(self, metadata: TenQMetadata) -> str:
        cik_no_zero = metadata.cik.lstrip("0")
        accession_nodash = metadata.accession_number.replace("-", "")
        path = PurePosixPath(cik_no_zero) / accession_nodash / metadata.primary_document
        return f"{self._archives_base}/{path}"

    async def download_primary_html(self, metadata: TenQMetadata) -> str:
        url = self._build_primary_url(metadata)
        rel_path = (
            f"filings/{metadata.ticker}/{metadata.cik}/"
            f"{metadata.form_type}/{metadata.accession_number}/"
            f"{metadata.primary_document}"
        )

        if not self._storage.exists(rel_path):
            text = await self._client.get_text(url)
            self._storage.write_bytes(rel_path, text.encode("utf-8", errors="ignore"))

        return rel_path

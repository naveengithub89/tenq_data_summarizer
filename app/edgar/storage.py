from __future__ import annotations

from pathlib import Path
from typing import Protocol


class StorageBackend(Protocol):
    def write_bytes(self, relative_path: str, data: bytes) -> str: ...
    def exists(self, relative_path: str) -> bool: ...


class LocalFileStorage(StorageBackend):
    def __init__(self, root: Path) -> None:
        self.root = root

        # ✅ Ensure root exists and is a directory (idempotent)
        if self.root.exists() and not self.root.is_dir():
            raise RuntimeError(
                f"Storage root '{self.root}' exists but is not a directory. "
                "Delete/rename it or choose a different root."
            )
        self.root.mkdir(parents=True, exist_ok=True)

    def write_bytes(self, relative_path: str, data: bytes) -> str:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        # Return a path that callers can safely re-open later.
        # If your downloader expects a *relative* path, keep this:
        return relative_path

        # If your downloader expects an *absolute* path, use this instead:
        # return str(path)

    def exists(self, relative_path: str) -> bool:
        return (self.root / relative_path).exists()

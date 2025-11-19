from __future__ import annotations

from pathlib import Path
from typing import Protocol


class StorageBackend(Protocol):
    def write_bytes(self, relative_path: str, data: bytes) -> str: ...
    def exists(self, relative_path: str) -> bool: ...


class LocalFileStorage(StorageBackend):
    def __init__(self, root: Path) -> None:
        self.root = root

    def write_bytes(self, relative_path: str, data: bytes) -> str:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def exists(self, relative_path: str) -> bool:
        return (self.root / relative_path).exists()

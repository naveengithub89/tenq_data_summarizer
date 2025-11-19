from __future__ import annotations

from typing import Iterable, List

# placeholder: use your preferred provider / client here
# Pydantic-AI will generally orchestrate LLMs, but embeddings can be a separate client.


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def embed_many(self, texts: Iterable[str]) -> List[list[float]]:
        """
        TODO: wire to actual embedding API.
        v1: just return fake low-dim vectors to keep types happy.
        """
        vectors: list[list[float]] = []
        for t in texts:
            # extremely naive stand-in: hash text to a 4-dim vector
            h = abs(hash(t))
            vectors.append(
                [
                    float(h & 0xFFFF),
                    float((h >> 16) & 0xFFFF),
                    float((h >> 32) & 0xFFFF),
                    float((h >> 48) & 0xFFFF),
                ]
            )
        return vectors

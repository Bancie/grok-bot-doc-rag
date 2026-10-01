"""CPU sentence-transformer embeddings.

Documents are encoded as plain text. Queries use the model's ``query`` prompt
when it defines one. Qwen3 embedding models without that prompt get the
instruction string published with the model.
"""

from __future__ import annotations

import os
import sys
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)


def resolve_model(flag: str | None) -> str:
    if flag and flag.strip():
        return flag.strip()
    env = os.environ.get("DOC_RAG_MODEL", "").strip()
    if env:
        return env
    return DEFAULT_MODEL


class Embedder:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        show_progress: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._model: Any = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors: list[list[float]] = []
        batch_size = max(1, self.batch_size)
        batch_count = (len(texts) + batch_size - 1) // batch_size
        for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
            if self.show_progress:
                print(
                    f"Embedding batch {batch_index}/{batch_count}",
                    file=sys.stderr,
                )
            batch = texts[start : start + batch_size]
            encoded = model.encode(
                batch,
                prompt="",
                task="document",
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            vectors.extend(row.tolist() for row in encoded)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        prompts = getattr(model, "prompts", None) or {}
        kwargs = {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if "query" in prompts:
            encoded = model.encode_query([text], **kwargs)
        elif "qwen3-embedding" in self.model_name.lower():
            encoded = model.encode(
                [f"{QWEN_QUERY_INSTRUCTION}{text}"],
                prompt="",
                task="query",
                **kwargs,
            )
        else:
            encoded = model.encode_query([text], **kwargs)
        return encoded[0].tolist()

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

"""Chroma collection plus a BM25 sidecar for one document."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
import numpy as np
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from rank_bm25 import BM25Okapi

from doc_rag.chunk import Chunk, chunk_pages
from doc_rag.pdf import parse_pdf

RRF_K = 60
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]$")
_IPV4_RE = re.compile(r"\d+\.\d+\.\d+\.\d+")
_TOKEN_RE = re.compile(r"\w+")
_UPSERT_BATCH = 256


class DocRagError(Exception):
    """Expected failure with a message for stderr."""


class NotIndexedError(DocRagError):
    pass


class ModelMismatchError(DocRagError):
    pass


@dataclass
class IndexSummary:
    doc_id: str
    title: str
    pages: int
    chunks: int
    model: str
    seconds: float
    skipped: bool


def data_home() -> Path:
    raw = os.environ.get("DOC_RAG_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".doc-rag"


def collection_name(doc_id: str) -> str:
    """Return a Chroma-safe name. ``PER-5100`` stays ``PER-5100``."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", doc_id.strip())
    cleaned = re.sub(r"\.{2,}", ".", cleaned).strip(".-_")
    if not cleaned:
        cleaned = "doc"
    if len(cleaned) < 3:
        cleaned = f"doc-{cleaned}"
    if not cleaned[0].isalnum():
        cleaned = f"d{cleaned}"
    if not cleaned[-1].isalnum():
        cleaned = f"{cleaned}0"
    if _IPV4_RE.fullmatch(cleaned):
        cleaned = f"doc-{cleaned}"
    if len(cleaned) > 63:
        suffix = hashlib.sha256(doc_id.encode()).hexdigest()[:8]
        head = cleaned[:54].rstrip(".-_")
        cleaned = f"{head}-{suffix}"
        if not cleaned[-1].isalnum():
            cleaned = f"{cleaned[:-1]}0"
    if not _NAME_RE.fullmatch(cleaned) or ".." in cleaned:
        raise DocRagError(f"cannot derive a collection name from id {doc_id!r}")
    return cleaned


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    rrf_k: int = RRF_K,
) -> dict[str, float]:
    """Fuse ranked id lists. Score is the sum of ``1 / (rrf_k + rank)``."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def hits_expected(page_start: int, page_end: int, expected_pages: set[int]) -> bool:
    return any(page_start <= page <= page_end for page in expected_pages)


def hit_at_k(results: list[dict[str, Any]], expected_pages: list[int]) -> float:
    expected = set(expected_pages)
    if any(hits_expected(row["page_start"], row["page_end"], expected) for row in results):
        return 1.0
    return 0.0


def reciprocal_rank(results: list[dict[str, Any]], expected_pages: list[int]) -> float:
    expected = set(expected_pages)
    for rank, row in enumerate(results, start=1):
        if hits_expected(row["page_start"], row["page_end"], expected):
            return 1.0 / rank
    return 0.0


def index_document(
    doc_id: str,
    pdf_path: Path,
    model: str,
    embedder: Any,
    force: bool = False,
) -> IndexSummary:
    started = _now()
    digest = file_sha256(pdf_path)
    if not force:
        current = _read_meta(doc_id)
        if (
            current
            and current.get("file_hash") == digest
            and current.get("model") == model
            and _has_collection(collection_name(doc_id))
        ):
            return IndexSummary(
                doc_id=doc_id,
                title=str(current.get("title") or ""),
                pages=int(current.get("pages") or 0),
                chunks=int(current.get("chunks") or 0),
                model=model,
                seconds=0.0,
                skipped=True,
            )

    _eprint("Parsing PDF...")
    parsed = parse_pdf(pdf_path)
    _eprint("Chunking...")
    chunks = chunk_pages(parsed.pages, doc_id)
    if not chunks:
        raise DocRagError("PDF has no extractable text")

    _eprint(f"Embedding {len(chunks)} chunks...")
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise DocRagError("embedder returned the wrong number of vectors")

    _eprint("Writing index...")
    name = collection_name(doc_id)
    _write_collection(name, doc_id, model, chunks, vectors)
    _write_sidecars(
        doc_id=doc_id,
        name=name,
        title=parsed.title,
        source_path=str(pdf_path.resolve()),
        file_hash=digest,
        pages=parsed.page_count,
        model=model,
        chunks=chunks,
        toc=parsed.toc,
    )
    return IndexSummary(
        doc_id=doc_id,
        title=parsed.title,
        pages=parsed.page_count,
        chunks=len(chunks),
        model=model,
        seconds=_now() - started,
        skipped=False,
    )


def query_document(
    doc_id: str,
    question: str,
    embedder: Any,
    k: int = 6,
    section: str | None = None,
    hybrid: bool = True,
) -> list[dict[str, Any]]:
    meta = _require_meta(doc_id)
    model = embedder.model_name
    name = collection_name(doc_id)
    collection = _client().get_collection(name, embedding_function=None)
    stored = str(meta.get("model") or "")
    collection_model = str((collection.metadata or {}).get("model") or "")
    if stored != model or (collection_model and collection_model != model):
        shown = stored or collection_model
        raise ModelMismatchError(
            f"index for {doc_id} was built with model {shown}, "
            f"but this command uses {model}. Reindex with --force."
        )

    rows = _load_rows(doc_id, collection)
    if section:
        rows = [row for row in rows if section in row["section_path"]]
    if not rows or k <= 0:
        return []

    query_vector = embedder.embed_query(question)
    vector_scores = _cosine_scores([row["embedding"] for row in rows], query_vector)
    vector_ids = _order_ids(rows, vector_scores)
    if hybrid:
        keyword_scores = _bm25_scores(rows, question)
        keyword_ids = _order_ids(rows, keyword_scores)
        fused = reciprocal_rank_fusion([vector_ids, keyword_ids])
        ranked = sorted(rows, key=lambda row: (-fused[row["id"]], row["chunk_index"]))
        score_of = fused
    else:
        score_map = {row["id"]: vector_scores[index] for index, row in enumerate(rows)}
        ranked = sorted(rows, key=lambda row: (-score_map[row["id"]], row["chunk_index"]))
        score_of = score_map

    results: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked[:k], start=1):
        results.append(
            {
                "rank": rank,
                "score": float(f"{score_of[row['id']]:.6f}"),
                "text": row["text"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "section_path": row["section_path"],
            }
        )
    return results


def list_documents() -> list[dict[str, Any]]:
    docs_dir = data_home() / "docs"
    if not docs_dir.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for meta_path in sorted(docs_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        items.append(meta)
    items.sort(key=lambda item: str(item.get("doc_id") or ""))
    return items


def document_info(doc_id: str) -> dict[str, Any]:
    return _require_meta(doc_id)


def delete_document(doc_id: str) -> None:
    meta_path = _meta_path(doc_id)
    name = collection_name(doc_id)
    if not meta_path.is_file() and not _has_collection(name):
        raise NotIndexedError(f"document {doc_id} is not indexed")
    client = _client()
    try:
        client.delete_collection(name)
    except NotFoundError:
        pass
    doc_dir = meta_path.parent
    if doc_dir.is_dir():
        for child in doc_dir.iterdir():
            child.unlink()
        doc_dir.rmdir()


def _write_collection(
    name: str,
    doc_id: str,
    model: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> None:
    client = _client()
    try:
        client.delete_collection(name)
    except NotFoundError:
        pass
    collection = client.create_collection(
        name=name,
        metadata={"model": model, "doc_id": doc_id},
        embedding_function=None,
    )
    ids = [str(chunk.chunk_index) for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "doc_id": chunk.doc_id,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_path": chunk.section_path,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]
    for start in range(0, len(ids), _UPSERT_BATCH):
        end = start + _UPSERT_BATCH
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=vectors[start:end],
        )


def _write_sidecars(
    doc_id: str,
    name: str,
    title: str,
    source_path: str,
    file_hash: str,
    pages: int,
    model: str,
    chunks: list[Chunk],
    toc: list[list],
) -> None:
    doc_dir = data_home() / "docs" / name
    doc_dir.mkdir(parents=True, exist_ok=True)
    tokens = [tokenize(chunk.text) or ["_"] for chunk in chunks]
    bm25 = {
        "ids": [str(chunk.chunk_index) for chunk in chunks],
        "tokens": tokens,
    }
    (doc_dir / "bm25.json").write_text(
        json.dumps(bm25, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    meta = {
        "doc_id": doc_id,
        "collection": name,
        "title": title,
        "source_path": source_path,
        "file_hash": file_hash,
        "pages": pages,
        "chunks": len(chunks),
        "model": model,
        "indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "toc": toc,
    }
    _meta_path(doc_id).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_rows(doc_id: str, collection: Any) -> list[dict[str, Any]]:
    got = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = list(got.get("ids") or [])
    documents = list(got.get("documents") or [])
    metadatas = list(got.get("metadatas") or [])
    embeddings = _embedding_matrix(got.get("embeddings"), len(ids))
    token_map = _token_map(doc_id)
    rows: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] or {}
        text = documents[index] or ""
        tokens = token_map.get(str(chunk_id)) or tokenize(text) or ["_"]
        rows.append(
            {
                "id": str(chunk_id),
                "text": text,
                "page_start": int(metadata.get("page_start") or 0),
                "page_end": int(metadata.get("page_end") or 0),
                "section_path": str(metadata.get("section_path") or ""),
                "chunk_index": int(metadata.get("chunk_index") or 0),
                "embedding": embeddings[index],
                "tokens": tokens,
            }
        )
    return rows


def _token_map(doc_id: str) -> dict[str, list[str]]:
    path = _meta_path(doc_id).parent / "bm25.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(chunk_id): list(tokens)
        for chunk_id, tokens in zip(payload.get("ids") or [], payload.get("tokens") or [])
    }


def _embedding_matrix(raw: Any, count: int) -> list[list[float]]:
    if raw is None:
        raise DocRagError("index is missing embeddings; reindex with --force")
    array = np.asarray(raw, dtype=np.float64)
    if count == 0:
        return []
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[0] != count:
        return [np.asarray(row, dtype=np.float64).tolist() for row in raw]
    return array.tolist()


def _cosine_scores(embeddings: list[list[float]], query_vector: list[float]) -> list[float]:
    matrix = np.asarray(embeddings, dtype=np.float64)
    query = np.asarray(query_vector, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(query))
    dots = matrix @ query
    scores = np.divide(dots, norms, out=np.zeros(len(dots), dtype=np.float64), where=norms > 0)
    return [float(score) for score in scores]


def _bm25_scores(rows: list[dict[str, Any]], question: str) -> list[float]:
    corpus = [row["tokens"] or ["_"] for row in rows]
    model = BM25Okapi(corpus)
    scores = model.get_scores(tokenize(question))
    return [float(score) for score in scores]


def _order_ids(rows: list[dict[str, Any]], scores: list[float]) -> list[str]:
    ranked = sorted(
        range(len(rows)),
        key=lambda index: (-scores[index], rows[index]["chunk_index"]),
    )
    return [rows[index]["id"] for index in ranked]


def _require_meta(doc_id: str) -> dict[str, Any]:
    meta = _read_meta(doc_id)
    if meta is None or not _has_collection(collection_name(doc_id)):
        raise NotIndexedError(f"document {doc_id} is not indexed")
    return meta


def _read_meta(doc_id: str) -> dict[str, Any] | None:
    path = _meta_path(doc_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _meta_path(doc_id: str) -> Path:
    return data_home() / "docs" / collection_name(doc_id) / "meta.json"


def _has_collection(name: str) -> bool:
    try:
        _client().get_collection(name, embedding_function=None)
    except NotFoundError:
        return False
    return True


def _client() -> Any:
    path = data_home() / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )


def _eprint(message: str) -> None:
    print(message, file=sys.stderr)


def _now() -> float:
    import time

    return time.perf_counter()

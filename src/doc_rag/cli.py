"""Command line for indexing and querying a local PDF."""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Annotated, Any

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", message=r"builtin type Swig")
warnings.filterwarnings("ignore", message=r"builtin type swigvarlink")

if "--json" in sys.argv:
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

import typer

from doc_rag.embed import Embedder, resolve_model
from doc_rag.store import (
    DocRagError,
    IndexSummary,
    delete_document,
    document_info,
    hit_at_k,
    index_document,
    list_documents,
    query_document,
    reciprocal_rank,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Local CPU-only RAG over a PDF.")


@app.command()
def index(
    doc_id: Annotated[str, typer.Option("--id", help="Document id, for example PER-5100")],
    pdf: Annotated[Path, typer.Option("--pdf", help="Path to a local PDF")],
    model: Annotated[str | None, typer.Option("--model", help="Embedding model name")] = None,
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if the file and model match")] = False,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Embedding batch size")] = 8,
) -> None:
    """Parse, chunk, embed, and store a PDF."""
    doc_id = _require_id(doc_id)
    pdf = pdf.expanduser()
    if not pdf.is_file():
        _fail(f"PDF not found: {pdf}")
    if batch_size < 1:
        _fail("batch size must be at least 1")
    model_name = resolve_model(model)
    try:
        summary = index_document(
            doc_id,
            pdf,
            model_name,
            Embedder(model_name, batch_size=batch_size, show_progress=True),
            force=force,
        )
    except (DocRagError, ValueError) as exc:
        _fail(str(exc))
    _print_index_summary(summary)


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to search for")],
    doc_id: Annotated[str, typer.Option("--id", help="Document id")],
    k: Annotated[int, typer.Option("--k", help="Number of chunks to return")] = 6,
    json_output: Annotated[bool, typer.Option("--json", help="Print a JSON array on stdout")] = False,
    section: Annotated[str | None, typer.Option("--section", help="Keep chunks whose section path contains this text")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Must match the model used to index")] = None,
    hybrid: Annotated[bool, typer.Option("--hybrid/--no-hybrid", help="Fuse BM25 with vector search")] = True,
) -> None:
    """Return the top matching chunks with page numbers."""
    if json_output:
        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)
    doc_id = _require_id(doc_id)
    if not question.strip():
        _fail("question is required")
    if k < 1:
        _fail("k must be at least 1")
    section_filter = section.strip() if section and section.strip() else None
    model_name = resolve_model(model)
    try:
        rows = query_document(
            doc_id,
            question.strip(),
            Embedder(model_name, show_progress=False),
            k=k,
            section=section_filter,
            hybrid=hybrid,
        )
    except DocRagError as exc:
        _fail(str(exc))
    if json_output:
        sys.stdout.write(json.dumps(rows, ensure_ascii=False, indent=2))
        sys.stdout.write("\n")
        return
    _print_hits(rows)


@app.command("list")
def list_cmd() -> None:
    """List indexed documents."""
    items = list_documents()
    if not items:
        print("No documents indexed.")
        return
    for item in items:
        print(
            f"{item.get('doc_id')}\t"
            f"title={item.get('title')}\t"
            f"pages={item.get('pages')}\t"
            f"chunks={item.get('chunks')}\t"
            f"model={item.get('model')}\t"
            f"indexed={item.get('indexed_at')}"
        )


@app.command()
def info(
    doc_id: Annotated[str, typer.Option("--id", help="Document id")],
) -> None:
    """Show details and the table of contents for one document."""
    doc_id = _require_id(doc_id)
    try:
        meta = document_info(doc_id)
    except DocRagError as exc:
        _fail(str(exc))
    print(f"id: {meta.get('doc_id')}")
    print(f"title: {meta.get('title')}")
    print(f"source: {meta.get('source_path')}")
    print(f"sha256: {meta.get('file_hash')}")
    print(f"pages: {meta.get('pages')}")
    print(f"chunks: {meta.get('chunks')}")
    print(f"model: {meta.get('model')}")
    print(f"indexed: {meta.get('indexed_at')}")
    print("toc:")
    toc = meta.get("toc") or []
    if not toc:
        print("  (none)")
        return
    for entry in toc:
        level, title, page = entry[0], entry[1], entry[2]
        indent = "  " * int(level)
        print(f"{indent}- {title} (p. {page})")


@app.command()
def delete(
    doc_id: Annotated[str, typer.Option("--id", help="Document id")],
) -> None:
    """Delete one document index."""
    doc_id = _require_id(doc_id)
    try:
        delete_document(doc_id)
    except DocRagError as exc:
        _fail(str(exc))
    print(f"Deleted {doc_id}")


@app.command()
def eval(
    doc_id: Annotated[str, typer.Option("--id", help="Document id")],
    questions: Annotated[Path, typer.Option("--questions", help="JSONL file of questions and expected pages")],
    k: Annotated[int, typer.Option("--k", help="Cutoff for hit@k")] = 6,
    model: Annotated[str | None, typer.Option("--model", help="Must match the indexed model")] = None,
    hybrid: Annotated[bool, typer.Option("--hybrid/--no-hybrid")] = True,
) -> None:
    """Report mean hit@k and MRR for a JSONL question file."""
    doc_id = _require_id(doc_id)
    if k < 1:
        _fail("k must be at least 1")
    questions = questions.expanduser()
    try:
        items = _load_questions(questions)
    except ValueError as exc:
        _fail(str(exc))
    model_name = resolve_model(model)
    embedder = Embedder(model_name, show_progress=False)
    hits: list[float] = []
    ranks: list[float] = []
    try:
        for item in items:
            rows = query_document(
                doc_id,
                item["q"],
                embedder,
                k=k,
                hybrid=hybrid,
            )
            hits.append(hit_at_k(rows, item["expected_pages"]))
            ranks.append(reciprocal_rank(rows, item["expected_pages"]))
    except DocRagError as exc:
        _fail(str(exc))
    count = len(items)
    print(f"questions: {count}")
    print(f"k: {k}")
    print(f"hit@{k}: {sum(hits) / count:.3f}")
    print(f"MRR: {sum(ranks) / count:.3f}")


def _print_index_summary(summary: IndexSummary) -> None:
    if summary.skipped:
        print(
            f"Already indexed {summary.doc_id} (same file and model). Use --force to rebuild."
        )
    else:
        print(f"Indexed {summary.doc_id}")
    print(f"title: {summary.title}")
    print(f"pages: {summary.pages}")
    print(f"chunks: {summary.chunks}")
    print(f"model: {summary.model}")
    if not summary.skipped:
        print(f"seconds: {summary.seconds:.2f}")


def _print_hits(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No matching chunks.")
        return
    blocks: list[str] = []
    for row in rows:
        if row["page_start"] == row["page_end"]:
            pages = str(row["page_start"])
        else:
            pages = f"{row['page_start']}-{row['page_end']}"
        section = row["section_path"] or "(no section)"
        blocks.append(
            f"[{row['rank']}] score={row['score']:.6f} pages {pages} | {section}\n{row['text']}"
        )
    print("\n\n".join(blocks))


def _load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"questions file not found: {path}")
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid questions file at line {line_number}: {exc}") from exc
        if not isinstance(obj, dict) or "q" not in obj or "expected_pages" not in obj:
            raise ValueError(
                f"invalid questions file at line {line_number}: expected q and expected_pages"
            )
        question = obj["q"]
        pages = obj["expected_pages"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"invalid questions file at line {line_number}: q must be a string")
        if not isinstance(pages, list) or not all(
            isinstance(page, int) and not isinstance(page, bool) for page in pages
        ):
            raise ValueError(
                f"invalid questions file at line {line_number}: expected_pages must be a list of integers"
            )
        items.append({"q": question.strip(), "expected_pages": pages})
    if not items:
        raise ValueError("questions file has no questions")
    return items


def _require_id(doc_id: str) -> str:
    cleaned = doc_id.strip()
    if not cleaned:
        _fail("document id is required")
    return cleaned


def _fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

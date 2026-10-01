# doc-rag

Local, CPU-only retrieval over a PDF. An agent on another machine can clone this repo and call `doc-rag` to answer questions about a book or paper with page citations. The tool only reads a PDF path you already have. It does not call Linear or Google Drive. The only network use is the first download of an embedding model from Hugging Face.

## Install

From a fresh Linux box, Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Install the CPU wheel first. `torch` is pinned to 2.9.1, and `2.9.1+cpu` from that index satisfies the pin, so the second command does not replace it with a larger PyPI build.

Development tests:

```bash
pip install -e ".[dev]"
pytest
```

## Environment

| Variable | Default | Role |
| --- | --- | --- |
| `DOC_RAG_HOME` | `~/.doc-rag` | Index data. Not stored in the repo. |
| `DOC_RAG_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model when `--model` is omitted. |

Indexes live under `$DOC_RAG_HOME`:

- `chroma/` — one Chroma collection per document id
- `docs/<collection>/meta.json` — title, page count, hash, model, TOC
- `docs/<collection>/bm25.json` — tokens for keyword search

The collection records the model name. Querying with a different model exits non-zero and asks you to reindex. Indexing again with the same file hash and model does nothing unless you pass `--force`. A different file or model replaces the index.

Other models to compare:

- `BAAI/bge-small-en-v1.5`
- `ibm-granite/granite-embedding-97m-multilingual-r2`

Queries use the model's `query` prompt when it defines one. Documents are embedded as plain text. For `Qwen/Qwen3-Embedding-0.6B` that prompt is:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:
```

## Commands

```bash
doc-rag index --id PER-5100 --pdf /path/book.pdf
doc-rag index --id PER-5100 --pdf /path/book.pdf --model BAAI/bge-small-en-v1.5 --batch-size 8 --force

doc-rag query --id PER-5100 "How does retrieval use embeddings?"
doc-rag query --id PER-5100 "Chương 8 nói gì về retrieval?" --k 6 --section "Ch 8"
doc-rag query --id PER-5100 "How does BM25 work?" --no-hybrid

doc-rag list
doc-rag info --id PER-5100
doc-rag delete --id PER-5100

doc-rag eval --id PER-5100 --questions eval.jsonl --k 6
```

`index` prints a summary to stdout (pages, chunks, seconds) and progress to stderr. The same PDF and model are skipped until `--force`.

`query` returns the top `--k` chunks (default 6). `--hybrid` is on by default: vector rank and BM25 rank are fused with reciprocal rank fusion (`1 / (60 + rank)`). `--no-hybrid` ranks by cosine similarity only. `--section` keeps chunks whose section path contains that text, for example `Ch 8` matches `Ch 8 > 8.2 Retrieval`.

Chunks stay inside one TOC section, about 400 words each, at most 500, with a 340-word step (about 15% overlap). A short section is a single chunk. Page numbers are 1-based PDF pages. If the PDF has no outline, `section_path` is empty.

`eval` reads JSONL. Each line is `{"q": "...", "expected_pages": [12, 13]}`. A hit is a returned chunk whose page range overlaps `expected_pages`. The command prints the mean hit@k and mean reciprocal rank.

Exit code `0` means success. Missing PDFs, unknown ids, a model mismatch, and a bad questions file exit non-zero with `error: ...` on stderr.

## `query --json`

`--json` writes one JSON array to stdout and nothing else. Progress and errors stay on stderr.

```json
[
  {
    "rank": 1,
    "score": 0.016393,
    "text": "chunk text",
    "page_start": 12,
    "page_end": 13,
    "section_path": "Ch 8 > 8.2 Retrieval"
  }
]
```

| Field | Type | Meaning |
| --- | --- | --- |
| `rank` | integer | 1-based position after fusion or cosine sort |
| `score` | number | Reciprocal-rank-fusion score when hybrid is on, otherwise cosine similarity |
| `text` | string | Chunk text |
| `page_start` | integer | First PDF page in the chunk, 1-based |
| `page_end` | integer | Last PDF page in the chunk, 1-based |
| `section_path` | string | Outline path, or `""` when the PDF has no outline |

Example:

```bash
doc-rag query --id PER-5100 --json --k 6 "How does retrieval use embeddings?"
```

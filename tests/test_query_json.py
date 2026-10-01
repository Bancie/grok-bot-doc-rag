import json
import re
from pathlib import Path

import pymupdf
import pytest
from typer.testing import CliRunner

from doc_rag.cli import app

runner = CliRunner()


class FakeEmbedder:
    document_calls = 0

    def __init__(self, model_name: str, batch_size: int = 8, show_progress: bool = False) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        type(self).document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * 16
        for token in re.findall(r"\w+", text.lower()):
            vector[hash(token) % 16] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    FakeEmbedder.document_calls = 0
    directory = tmp_path / "rag-home"
    monkeypatch.setenv("DOC_RAG_HOME", str(directory))
    monkeypatch.setattr("doc_rag.cli.Embedder", FakeEmbedder)
    return directory


def _write_pdf(path: Path) -> None:
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 72), "Retrieval finds relevant passages in a book. " * 8)
    second = document.new_page()
    second.insert_text((72, 72), "Keywords match exact terms in chapter nine. " * 8)
    document.set_toc([[1, "Ch 8", 1], [1, "Ch 9", 2]])
    document.save(path)
    document.close()


def _index(pdf: Path, doc_id: str = "PER-5100") -> None:
    result = runner.invoke(
        app,
        ["index", "--id", doc_id, "--pdf", str(pdf), "--model", "fake-model"],
    )
    assert result.exit_code == 0, result.output


def test_query_json_shape(tmp_path: Path, home: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _write_pdf(pdf)
    _index(pdf)
    result = runner.invoke(
        app,
        ["query", "--id", "PER-5100", "--json", "--model", "fake-model", "what is retrieval?"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert result.stdout.strip() == json.dumps(payload, ensure_ascii=False, indent=2)
    assert payload
    keys = {"rank", "score", "text", "page_start", "page_end", "section_path"}
    for index, row in enumerate(payload, start=1):
        assert set(row) == keys
        assert row["rank"] == index
        assert isinstance(row["score"], float)
        assert isinstance(row["text"], str) and row["text"]
        assert isinstance(row["page_start"], int)
        assert isinstance(row["page_end"], int)
        assert row["page_start"] <= row["page_end"]
        assert isinstance(row["section_path"], str)
    assert "Embedding" not in result.stdout


def test_section_filter_and_second_index_skips_embed(tmp_path: Path, home: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _write_pdf(pdf)
    _index(pdf)
    assert FakeEmbedder.document_calls == 1
    again = runner.invoke(
        app,
        ["index", "--id", "PER-5100", "--pdf", str(pdf), "--model", "fake-model"],
    )
    assert again.exit_code == 0, again.output
    assert "Already indexed" in again.stdout
    assert FakeEmbedder.document_calls == 1

    filtered = runner.invoke(
        app,
        [
            "query",
            "--id",
            "PER-5100",
            "--json",
            "--model",
            "fake-model",
            "--section",
            "Ch 9",
            "keywords",
        ],
    )
    assert filtered.exit_code == 0, filtered.output
    rows = json.loads(filtered.stdout)
    assert rows
    assert all("Ch 9" in row["section_path"] for row in rows)


def test_errors_and_eval(tmp_path: Path, home: Path) -> None:
    missing = runner.invoke(app, ["index", "--id", "PER-1", "--pdf", str(tmp_path / "no.pdf")])
    assert missing.exit_code != 0
    assert "PDF not found" in (missing.stderr or missing.output)

    corrupt = tmp_path / "bad.pdf"
    corrupt.write_bytes(b"not a pdf")
    broken = runner.invoke(app, ["index", "--id", "PER-1", "--pdf", str(corrupt), "--model", "fake-model"])
    assert broken.exit_code != 0
    assert "failed to read PDF" in (broken.stderr or broken.output)

    unknown = runner.invoke(app, ["query", "--id", "PER-404", "--json", "hello"])
    assert unknown.exit_code != 0
    assert "not indexed" in (unknown.stderr or unknown.output)

    pdf = tmp_path / "book.pdf"
    _write_pdf(pdf)
    _index(pdf)
    mismatch = runner.invoke(
        app,
        ["query", "--id", "PER-5100", "--json", "--model", "other-model", "retrieval"],
    )
    assert mismatch.exit_code != 0
    assert "reindex" in (mismatch.stderr or mismatch.output).lower()

    eval_pdf = tmp_path / "eval.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Retrieval finds relevant passages.")
    document.save(eval_pdf)
    document.close()
    _index(eval_pdf, "PER-EVAL")

    questions = tmp_path / "eval.jsonl"
    questions.write_text(
        '{"q": "retrieval", "expected_pages": [1]}\n'
        '{"q": "missing topic", "expected_pages": [99]}\n',
        encoding="utf-8",
    )
    evaluated = runner.invoke(
        app,
        [
            "eval",
            "--id",
            "PER-EVAL",
            "--questions",
            str(questions),
            "--model",
            "fake-model",
            "--k",
            "6",
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output
    assert "hit@6: 0.500" in evaluated.stdout
    assert "MRR: 0.500" in evaluated.stdout

    info = runner.invoke(app, ["info", "--id", "PER-5100"])
    assert info.exit_code == 0, info.output
    assert "Ch 8" in info.stdout
    assert "Ch 9" in info.stdout

    listed = runner.invoke(app, ["list"])
    assert listed.exit_code == 0
    assert "PER-5100" in listed.stdout
    assert "fake-model" in listed.stdout

    deleted = runner.invoke(app, ["delete", "--id", "PER-5100"])
    assert deleted.exit_code == 0
    gone = runner.invoke(app, ["info", "--id", "PER-5100"])
    assert gone.exit_code != 0

from pathlib import Path

import pymupdf

from doc_rag.chunk import STEP_WORDS, TARGET_WORDS, chunk_pages
from doc_rag.pdf import PageText, clean_page_texts, dehyphenate, parse_pdf, section_paths


def test_section_paths_follow_the_outline_stack() -> None:
    toc = [[1, "Ch 8", 1], [2, "8.2 Retrieval", 2], [1, "Ch 9", 4]]
    assert section_paths(toc, 4) == [
        "Ch 8",
        "Ch 8 > 8.2 Retrieval",
        "Ch 8 > 8.2 Retrieval",
        "Ch 9",
    ]
    assert section_paths([], 2) == ["", ""]


def test_dehyphenate_and_header_footer_cleanup() -> None:
    assert dehyphenate("retriev-\nal") == "retrieval"
    pages = ["Hands-On LLMs\n\nretriev-\nal ranks documents.\n12"] * 4
    cleaned = clean_page_texts(pages)
    assert cleaned[0] == "retrieval ranks documents."

    unique = [f"Unique title {index}\n\nBody text stays.\n{index}" for index in range(4)]
    kept = clean_page_texts(unique)
    assert kept[0] == "Unique title 0\n\nBody text stays."


def test_short_section_is_one_chunk_and_pages_span() -> None:
    pages = [
        PageText(1, " ".join(f"a{i}" for i in range(100)), "Ch 8"),
        PageText(2, " ".join(f"b{i}" for i in range(100)), "Ch 8"),
    ]
    chunks = chunk_pages(pages, "PER-5100")
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert chunks[0].section_path == "Ch 8"
    assert chunks[0].chunk_index == 0
    assert chunks[0].doc_id == "PER-5100"


def test_chunks_do_not_cross_sections() -> None:
    pages = [
        PageText(1, " ".join(["alpha"] * 400), "Ch 8"),
        PageText(2, " ".join(["beta"] * 400), "Ch 9"),
    ]
    chunks = chunk_pages(pages, "PER-1")
    assert [chunk.section_path for chunk in chunks] == ["Ch 8", "Ch 9"]
    assert "beta" not in chunks[0].text
    assert "alpha" not in chunks[1].text


def test_long_section_uses_400_word_windows_with_overlap() -> None:
    assert STEP_WORDS == 340
    words = [f"w{index}" for index in range(600)]
    chunks = chunk_pages([PageText(1, " ".join(words), "S")], "doc")
    assert [len(chunk.text.split()) for chunk in chunks] == [400, 260]
    assert chunks[0].text.split()[-60:] == chunks[1].text.split()[:60]
    assert all(len(chunk.text.split()) <= 500 for chunk in chunks)
    assert chunks[0].page_start == chunks[1].page_end == 1

    five_hundred = chunk_pages([PageText(3, " ".join(["w"] * 500), "")], "doc")
    assert len(five_hundred) == 1
    assert five_hundred[0].page_start == 3

    longer = [f"w{index}" for index in range(501)]
    split = chunk_pages([PageText(1, " ".join(longer), "S")], "doc")
    assert len(split[0].text.split()) == TARGET_WORDS
    assert all(len(chunk.text.split()) <= 500 for chunk in split)


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_pages([PageText(1, "   ", "")], "doc") == []


def test_parse_pdf_without_toc_and_with_outline(tmp_path: Path) -> None:
    plain = tmp_path / "note.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "A short note about retrieval.")
    document.save(plain)
    document.close()
    parsed = parse_pdf(plain)
    assert parsed.page_count == 1
    assert parsed.toc == []
    assert parsed.pages[0].section_path == ""
    assert "retrieval" in parsed.pages[0].text
    assert parsed.title == "note"

    outlined = tmp_path / "book.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Chapter text.")
    document.set_toc([[1, "Ch 8", 1], [2, "8.2 Retrieval", 1]])
    document.set_metadata({"title": "Hands-On Large Language Models"})
    document.save(outlined)
    document.close()
    parsed = parse_pdf(outlined)
    assert parsed.title == "Hands-On Large Language Models"
    assert parsed.pages[0].section_path == "Ch 8 > 8.2 Retrieval"

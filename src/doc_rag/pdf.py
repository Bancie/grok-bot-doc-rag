"""Parse a PDF into cleaned pages tagged with a TOC section path."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_PAGE_NUMBER = re.compile(r"(?i)(?:page\s+)?[\-–—\s]*\d+[\-–—\s]*")
_HEADER_MAX_LEN = 120


@dataclass
class PageText:
    page: int
    text: str
    section_path: str


@dataclass
class ParsedDocument:
    title: str
    page_count: int
    pages: list[PageText]
    toc: list[list] = field(default_factory=list)


def parse_pdf(path: Path) -> ParsedDocument:
    """Read *path* and return cleaned page text plus the outline."""
    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise ValueError(f"failed to read PDF: {exc}") from exc
    try:
        metadata = document.metadata or {}
        title = str(metadata.get("title") or "").strip() or path.stem
        toc = [list(entry) for entry in document.get_toc()]
        raw_pages = [page.get_text("text") or "" for page in document]
        page_count = document.page_count
    finally:
        document.close()
    cleaned = clean_page_texts(raw_pages)
    paths = section_paths(toc, page_count)
    pages = [
        PageText(page=index, text=text, section_path=paths[index - 1])
        for index, text in enumerate(cleaned, start=1)
    ]
    return ParsedDocument(title=title, page_count=page_count, pages=pages, toc=toc)


def section_paths(toc: list[list], page_count: int) -> list[str]:
    """Return one section path per page. An empty outline yields empty paths."""
    paths: list[str] = []
    stack: list[tuple[int, str]] = []
    cursor = 0
    for page_number in range(1, page_count + 1):
        while cursor < len(toc) and int(toc[cursor][2]) <= page_number:
            level = int(toc[cursor][0])
            title = str(toc[cursor][1]).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            if title:
                stack.append((level, title))
            cursor += 1
        paths.append(" > ".join(title for _, title in stack))
    return paths


def clean_page_texts(pages: list[str]) -> list[str]:
    """Drop repeated running headers/footers, then dehyphenate and tidy space."""
    banned = _repeated_edge_lines(pages)
    cleaned: list[str] = []
    for page in pages:
        stripped = _strip_edges(page, banned)
        cleaned.append(collapse_whitespace(dehyphenate(stripped)))
    return cleaned


def dehyphenate(text: str) -> str:
    """Join words split across a line break, including soft hyphens."""
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _repeated_edge_lines(pages: list[str]) -> set[str]:
    if len(pages) < 3:
        return set()
    first_counts: dict[str, int] = {}
    last_counts: dict[str, int] = {}
    for page in pages:
        edges = _edge_lines(page)
        if edges[0]:
            first_counts[edges[0]] = first_counts.get(edges[0], 0) + 1
        if edges[1]:
            last_counts[edges[1]] = last_counts.get(edges[1], 0) + 1
    threshold = max(3, (len(pages) * 4 + 9) // 10)  # at least 40% of pages
    banned: set[str] = set()
    for line, count in first_counts.items():
        if count >= threshold and len(line) <= _HEADER_MAX_LEN:
            banned.add(line)
    for line, count in last_counts.items():
        if count >= threshold and len(line) <= _HEADER_MAX_LEN:
            banned.add(line)
    return banned


def _edge_lines(page: str) -> tuple[str | None, str | None]:
    lines = [line.strip() for line in page.splitlines() if line.strip()]
    if not lines:
        return None, None
    if len(lines) == 1:
        return lines[0], None
    return lines[0], lines[-1]


def _strip_edges(page: str, banned: set[str]) -> str:
    lines = page.splitlines()

    def noise(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if stripped in banned:
            return True
        return _PAGE_NUMBER.fullmatch(stripped) is not None

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and noise(lines[0]):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and noise(lines[-1]):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines)

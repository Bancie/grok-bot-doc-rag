"""Split cleaned pages into section-bounded word chunks."""

from __future__ import annotations

from dataclasses import dataclass

from doc_rag.pdf import PageText

TARGET_WORDS = 400
MAX_WORDS = 500
OVERLAP_RATIO = 0.15
STEP_WORDS = TARGET_WORDS - int(TARGET_WORDS * OVERLAP_RATIO)  # 340


@dataclass
class Chunk:
    doc_id: str
    text: str
    page_start: int
    page_end: int
    section_path: str
    chunk_index: int


def chunk_pages(pages: list[PageText], doc_id: str) -> list[Chunk]:
    """Chunk *pages* without crossing a section path.

    A section of at most 500 words is one chunk. Longer sections use a
    400-word window and a 340-word step (about 15% overlap).
    """
    chunks: list[Chunk] = []
    for group in _section_groups(pages):
        words = [
            (word, page.page)
            for page in group
            for word in page.text.split()
        ]
        if not words:
            continue
        section_path = group[0].section_path
        for start, end in _windows(len(words)):
            piece = words[start:end]
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    text=" ".join(word for word, _ in piece),
                    page_start=piece[0][1],
                    page_end=piece[-1][1],
                    section_path=section_path,
                    chunk_index=len(chunks),
                )
            )
    return chunks


def _section_groups(pages: list[PageText]) -> list[list[PageText]]:
    groups: list[list[PageText]] = []
    current: list[PageText] = []
    current_path: str | None = None
    for page in pages:
        if current and page.section_path != current_path:
            groups.append(current)
            current = []
        current.append(page)
        current_path = page.section_path
    if current:
        groups.append(current)
    return groups


def _windows(word_count: int) -> list[tuple[int, int]]:
    if word_count <= MAX_WORDS:
        return [(0, word_count)]
    windows: list[tuple[int, int]] = []
    start = 0
    while start < word_count:
        remaining = word_count - start
        if remaining <= MAX_WORDS:
            windows.append((start, word_count))
            break
        windows.append((start, start + TARGET_WORDS))
        start += STEP_WORDS
    return windows

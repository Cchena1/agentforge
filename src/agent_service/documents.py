from __future__ import annotations

import asyncio
import csv
import hashlib
import importlib.metadata
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pypdf import PdfReader

from .document_models import CANONICAL_DOCUMENT_SCHEMA_VERSION


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


@dataclass(slots=True)
class ParsedBlock:
    text: str
    page: int | None = None
    kind: str = "paragraph"
    locator: str | None = None


@dataclass(slots=True)
class ParsedDocument:
    source_name: str
    blocks: list[ParsedBlock]
    parser: str
    warnings: list[str] = field(default_factory=list)
    schema_version: int = CANONICAL_DOCUMENT_SCHEMA_VERSION


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    page: int | None
    locator: str | None
    metadata: dict[str, Any]


class DocumentParser(Protocol):
    @property
    def profile_id(self) -> str: ...

    async def parse(self, path: Path) -> ParsedDocument: ...


class BuiltinDocumentParser:
    profile_id = f"builtin-v1:pypdf={_package_version('pypdf')}"
    TEXT_EXTENSIONS: ClassVar[set[str]] = {".txt", ".md", ".rst", ".py", ".json", ".yaml", ".yml", ".log"}
    TABLE_EXTENSIONS: ClassVar[set[str]] = {".csv", ".tsv"}

    async def parse(self, path: Path) -> ParsedDocument:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return await asyncio.to_thread(self._parse_pdf, path)
        if suffix in self.TABLE_EXTENSIONS:
            return await asyncio.to_thread(self._parse_delimited, path)
        if suffix in self.TEXT_EXTENSIONS or not suffix:
            text = await asyncio.to_thread(_read_text, path)
            return ParsedDocument(path.name, _text_blocks(text), "builtin-text")
        raise ValueError(f"Unsupported document type: {suffix or '<none>'}")

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        reader = PdfReader(str(path))
        blocks: list[ParsedBlock] = []
        empty_pages = 0
        for page_number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if not text:
                empty_pages += 1
                continue
            for index, block in enumerate(_text_blocks(text), 1):
                block.page = page_number
                block.locator = f"page:{page_number}:block:{index}"
                blocks.append(block)
        warnings: list[str] = []
        if reader.pages and empty_pages / len(reader.pages) >= 0.5:
            warnings.append(
                "SCANNED_PDF_SUSPECTED: at least half of pages contain no extractable text; "
                "install the 'documents' extra to enable Docling OCR/table extraction."
            )
        return ParsedDocument(path.name, blocks, "pypdf", warnings)

    def _parse_delimited(self, path: Path) -> ParsedDocument:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        text = _read_text(path)
        rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
        if not rows:
            return ParsedDocument(path.name, [], "builtin-table")
        header = rows[0]
        blocks = []
        for index, row in enumerate(rows[1:], 2):
            pairs = [f"{header[col] if col < len(header) else f'column_{col + 1}'}={value}" for col, value in enumerate(row)]
            blocks.append(ParsedBlock(text=" | ".join(pairs), kind="table_row", locator=f"row:{index}"))
        return ParsedDocument(path.name, blocks, "builtin-table")


class DoclingDocumentParser:
    """Optional high-fidelity parser for scans, layout and tables."""

    profile_id = f"docling-markdown-v1:docling={_package_version('docling')}"

    async def parse(self, path: Path) -> ParsedDocument:
        try:
            from docling.document_converter import (
                DocumentConverter,
            )
        except ImportError as exc:
            raise RuntimeError("Docling is not installed; run `uv sync --extra documents`") from exc

        def convert() -> ParsedDocument:
            result = DocumentConverter().convert(str(path))
            markdown = result.document.export_to_markdown()
            return ParsedDocument(path.name, _text_blocks(markdown), "docling")

        return await asyncio.to_thread(convert)


class CompositeDocumentParser:
    def __init__(self, prefer_docling: bool = True) -> None:
        self.prefer_docling = prefer_docling
        self.builtin = BuiltinDocumentParser()
        self.docling = DoclingDocumentParser()

    @property
    def profile_id(self) -> str:
        return (
            f"composite-v1:prefer_docling={self.prefer_docling}:"
            f"builtin={self.builtin.profile_id}:docling={self.docling.profile_id}"
        )

    async def parse(self, path: Path) -> ParsedDocument:
        if self.prefer_docling and path.suffix.lower() in {".pdf", ".docx", ".pptx", ".xlsx", ".html"}:
            try:
                return await self.docling.parse(path)
            except (RuntimeError, ImportError):
                if path.suffix.lower() != ".pdf":
                    raise
        return await self.builtin.parse(path)


class SemanticChunker:
    """Respects headings/paragraphs/table rows, then applies bounded overlap."""

    def __init__(self, target_chars: int = 1800, max_chars: int = 2800, overlap_chars: int = 240) -> None:
        if not (0 <= overlap_chars < target_chars <= max_chars):
            raise ValueError("expected 0 <= overlap < target <= max")
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[ParsedBlock] = []
        size = 0
        for block in document.blocks:
            block_text = block.text.strip()
            if not block_text:
                continue
            boundary = block.kind == "heading" or (buffer and size + len(block_text) > self.target_chars)
            if boundary:
                chunks.extend(self._flush(document.source_name, buffer, len(chunks)))
                overlap = _tail_overlap(buffer, self.overlap_chars)
                buffer = [overlap] if overlap else []
                size = len(overlap.text) if overlap else 0
            if len(block_text) > self.max_chars:
                for part in _hard_split(block_text, self.max_chars, self.overlap_chars):
                    split_block = ParsedBlock(part, block.page, block.kind, block.locator)
                    chunks.extend(self._flush(document.source_name, [split_block], len(chunks)))
                continue
            buffer.append(block)
            size += len(block_text)
        chunks.extend(self._flush(document.source_name, buffer, len(chunks)))
        return chunks

    def _flush(self, source_name: str, blocks: list[ParsedBlock], index: int) -> list[Chunk]:
        if not blocks:
            return []
        text = "\n\n".join(block.text for block in blocks).strip()
        if not text:
            return []
        page = blocks[0].page if all(block.page == blocks[0].page for block in blocks) else None
        locator = blocks[0].locator if len(blocks) == 1 else f"chunk:{index + 1}"
        digest = hashlib.sha256(f"{source_name}:{index}:{text}".encode()).hexdigest()[:24]
        return [Chunk(digest, text, page, locator, {"block_kinds": sorted({b.kind for b in blocks})})]


def resolve_workspace_file(workspace_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace_root.resolve())
    except ValueError as exc:
        raise PermissionError("file_path must stay inside AI_AGENT_WORKSPACE_ROOT") from exc
    if not resolved.is_file():
        raise FileNotFoundError(raw_path)
    return resolved


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _text_blocks(text: str) -> list[ParsedBlock]:
    sections = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    blocks = []
    for index, section in enumerate(sections, 1):
        normalized = section.strip()
        if not normalized:
            continue
        first_line = normalized.splitlines()[0].strip()
        kind = "heading" if re.match(r"^#{1,6}\s+", first_line) else "table" if "|" in normalized and "\n" in normalized else "paragraph"
        blocks.append(ParsedBlock(normalized, kind=kind, locator=f"block:{index}"))
    return blocks


def _tail_overlap(blocks: list[ParsedBlock], limit: int) -> ParsedBlock | None:
    if not blocks or limit <= 0:
        return None
    text = "\n\n".join(block.text for block in blocks)
    tail = text[-limit:]
    boundary = tail.find(" ")
    if boundary >= 0:
        tail = tail[boundary + 1 :]
    last = blocks[-1]
    return ParsedBlock(tail, last.page, "overlap", last.locator) if tail else None


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    step = size - overlap
    return [text[start : start + size] for start in range(0, len(text), step)]

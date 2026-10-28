# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Physical-page PDF Sources with whole-document parsing and citation search."""

from __future__ import annotations

import hashlib
import io
import re
from urllib.parse import quote

from pypdf import PdfReader
from pydantic import BaseModel

from powercontext.artifacts import MemoryCitation
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TextEvidence

SOURCE_NAME = "pdf-page"
SOURCE_VERSION = "1"
IDENTITY_PREFIX = "pdf:"
SEARCH_PAGE_LIMIT = 400
_FUSION_RANK_OFFSET = 60
_NEIGHBOUR_PENALTY = 2.0
_NEIGHBOUR_DEPTH = 2
_QUERY_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-]*|\d+(?:\.\d+)?")
_GENERIC_QUERY_WORDS = frozenset(
    """
    a according an and answer are as assume at be been being by calculate can could define
    defined detail details did do does for from give given has have how i if in into is it
    its me most my number of on or our please provide question relying respond response
    show shown state stated term that the their them there these they this those to us use
    used using value was we were what when where which who why will with within would you
    your
    """.split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


class PDFDocumentInput(BaseModel):
    """Complete original PDF bytes plus the caller-supplied document identity."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical PDF page with its document identity and extractable text."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """A captured physical page identified by document, revision and page number."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the stable, collision-free identity of one physical PDF page."""

    document = _require_document_id(document_id)
    digest = _require_sha256(document_sha256)
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError("page_number must be a positive integer")
    token = quote(document, safe="")
    candidate = f"{IDENTITY_PREFIX}{token}:{digest}:{page_number}"
    if len(candidate) <= MAX_SOURCE_ID_LENGTH:
        return candidate
    compressed = hashlib.sha256(document.encode("utf-8")).hexdigest()[:32]
    candidate = f"{IDENTITY_PREFIX}{compressed}:{digest}:{page_number}"
    if len(candidate) > MAX_SOURCE_ID_LENGTH:  # pragma: no cover - digest and page number are bounded
        raise ValueError(f"pdf page source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return candidate


def parse_pdf_document(value: PDFDocumentInput, /) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one complete PDF or fail without partial results."""

    document_id = _require_document_id(value.document_id)
    content = value.content
    if not isinstance(content, bytes):
        raise ValueError("content must be bytes")
    if not content:
        raise ValueError("pdf content must not be empty")
    if b"%PDF-" not in content[:1024]:
        raise ValueError("content is not a PDF document")

    digest = hashlib.sha256(content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        if reader.is_encrypted:
            outcome = reader.decrypt("")
            if not outcome:
                raise ValueError("pdf document requires a user password")
        page_count = len(reader.pages)
        if page_count < 1:
            raise ValueError("pdf document must contain at least one page")
        pages: list[PDFPageInput] = []
        for index in range(page_count):
            page = reader.pages[index]
            try:
                text = page.extract_text() or ""
            except Exception as error:  # noqa: BLE001 - any page failure aborts the whole document
                raise ValueError(f"pdf page {index + 1} cannot be parsed: {error}") from error
            pages.append(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=digest,
                    page_number=index + 1,
                    text=text,
                )
            )
    except ValueError:
        raise
    except Exception as error:  # noqa: BLE001 - every parser failure is an atomic rejection
        raise ValueError(f"pdf document cannot be parsed: {error}") from error

    return tuple(pages)


async def search_pdf_pages(
    memory: object,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Return up to ``limit`` distinct exact page citations for one natural-language question.

    Retrieval fuses a verbatim reading of the question with a focus reading that drops
    generic instruction wording, then admits the physical pages adjacent to every strong
    hit so that prose spanning a page boundary stays reachable.
    """

    from powercontext.builtin.runtime import GetMemoryEntryRequest, SearchMemoryRequest

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if limit < 1:
        return ()

    query = " ".join(question.split())
    page_citation, entry_page = await _page_catalogue(memory)
    if not page_citation:
        return ()

    fused: dict[int, float] = {}
    for reading in _readings(query):
        page = await memory.search(SearchMemoryRequest(query=reading, mode="fts", limit=SEARCH_PAGE_LIMIT))
        for rank, hit in enumerate(page.hits, start=1):
            number = entry_page.get((hit.entry_id, hit.entry_version_id))
            if number is None:
                citation = MemoryCitation(
                    memory_ref=hit.memory_ref,
                    entry_id=hit.entry_id,
                    entry_version_id=hit.entry_version_id,
                )
                record = await memory.get(GetMemoryEntryRequest(citation=citation))
                number = _entry_page_number(getattr(record, "entry", record))
            if number is None:
                continue
            fused[number] = fused.get(number, 0.0) + 1.0 / (_FUSION_RANK_OFFSET + rank)

    if not fused:
        return ()

    priority: dict[int, float] = {}
    for rank, number in enumerate(sorted(fused, key=lambda page: (-fused[page], page)), start=1):
        for distance in range(_NEIGHBOUR_DEPTH + 1):
            candidates = (number,) if distance == 0 else (number - distance, number + distance)
            for candidate in candidates:
                if candidate not in page_citation:
                    continue
                value = rank + _NEIGHBOUR_PENALTY * distance
                if candidate not in priority or value < priority[candidate]:
                    priority[candidate] = value

    ordered = sorted(priority, key=lambda page: (priority[page], page))
    return tuple(page_citation[number] for number in ordered[:limit])


async def _page_catalogue(memory: object, /) -> tuple[dict[int, MemoryCitation], dict[tuple[str, str], int]]:
    """Map every persisted page of this scope to its citation and hit keys."""

    listing = await memory.list()
    page_citation: dict[int, MemoryCitation] = {}
    entry_page: dict[tuple[str, str], int] = {}
    for record in getattr(listing, "entries", ()):
        number = _entry_page_number(getattr(record, "entry", None))
        if number is None:
            continue
        citation = record.citation
        page_citation[number] = citation
        entry_page[(citation.entry_id, citation.entry_version_id)] = number
    return page_citation, entry_page


def _readings(query: str) -> tuple[str, ...]:
    """Return the verbatim reading plus focus readings without generic instructions."""

    readings = [query]
    focused = " ".join(
        token for token in _QUERY_TOKEN.findall(query) if token.casefold() not in _GENERIC_QUERY_WORDS
    )
    if focused and focused.casefold() != query.casefold():
        readings.append(focused)
    return tuple(readings)


def _entry_page_number(entry: object) -> int | None:
    for reference in getattr(entry, "sources", ()) or ():
        source_id = getattr(reference, "source_id", None)
        if isinstance(source_id, str) and source_id.startswith(IDENTITY_PREFIX):
            head, _, tail = source_id.rpartition(":")
            if head and tail.isdigit():
                return int(tail)
    return None


def _citation_page_key(entry: object) -> str | None:
    sources = getattr(entry, "sources", ())
    for reference in sources:
        source_id = getattr(reference, "source_id", None)
        if isinstance(source_id, str) and source_id.startswith(IDENTITY_PREFIX):
            parts = source_id.rsplit(":", 1)
            if len(parts) == 2:
                return source_id
    return None


def _require_document_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("document_id must be a non-empty string")
    if not value.strip():
        raise ValueError("document_id must not be blank")
    return value


def _require_sha256(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("document_sha256 must be a lowercase hexadecimal sha256 digest")
    if value.lower() != value or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("document_sha256 must be a lowercase hexadecimal sha256 digest")
    return value


class _PDFPageAdapter:
    """Resolve and read physical page Sources without touching a catalog."""

    name = SOURCE_NAME
    input_class = PDFPageInput
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class _PDFPageTextEvidenceProjection:
    """Project one page into the standard ``powercontext.text-evidence`` shape."""

    name = "powercontext.text-evidence"
    version = "1"
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = (
    AdapterSourceDefinition(
        adapter=_PDFPageAdapter(),
        version=SOURCE_VERSION,
        projections=(_PDFPageTextEvidenceProjection(),),
    )
)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]

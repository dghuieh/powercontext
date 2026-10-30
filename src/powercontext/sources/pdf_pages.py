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
import math
import re
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader

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
_NEIGHBOUR_SEEDS = 3
_NEIGHBOUR_BASE = 200.0
_QUERY_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-]*|\d+(?:\.\d+)?")
_GENERIC_QUERY_WORDS = frozenset(
    ["a", "according", "an", "and", "answer", "are", "as", "assume", "at", "be", "been", "being", "by", "calculate", "can", "could", "define", "defined", "detail", "details", "did", "do", "does", "for", "from", "give", "given", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me", "most", "my", "number", "of", "on", "or", "our", "please", "provide", "question", "relying", "respond", "response", "show", "shown", "state", "stated", "term", "that", "the", "their", "them", "there", "these", "they", "this", "those", "to", "us", "use", "used", "using", "value", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with", "within", "would", "you", "your"]
)
_TOKEN = re.compile(r"[a-z0-9]+")

# Financial statements answer questions through their titles rather than through the
# wording of the question: a page headed "Consolidated Statement of Income" never says
# "EBITDA margin", and the pages a filing devotes to litigation never repeat the words a
# reader uses to ask about them.  Each entry maps question vocabulary to the document
# structure phrases that carry the same intent in a periodic filing.
_STRUCTURE_HINTS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(
            r"income statement|statement of income|statement of operations|\bp&l\b|\bpnl\b"
            r"|profit and loss|gross profit|gross margin|operating income|operating profit"
            r"|\bebitda\b|\bebit\b|net income|net revenue|revenue|net sales|earnings per share"
            r"|\beps\b|cost of sales|margin"
        ),
        (
            "consolidated statement of income",
            "consolidated statements of income",
            "consolidated statements of operations",
            "statement of income",
            "statements of operations",
        ),
    ),
    (
        re.compile(
            r"cash flow|operating activities|capital expenditure|capex|depreciation|amortization"
            r"|free cash flow|investing activities|financing activities"
        ),
        (
            "consolidated statement of cash flows",
            "consolidated statements of cash flows",
            "statement of cash flows",
            "statements of cash flows",
        ),
    ),
    (
        re.compile(
            r"balance sheet|financial position|total assets|total liabilities|current assets"
            r"|current liabilities|working capital|accounts receivable|inventories|inventory"
            r"|shareholders. equity|stockholders. equity|quick ratio|current ratio|acid test"
        ),
        (
            "consolidated balance sheet",
            "consolidated balance sheets",
            "consolidated statements of financial position",
            "balance sheet",
        ),
    ),
    (
        re.compile(
            r"legal|lawsuit|litigation|proceeding|court|settlement|claims|antitrust|regulatory"
            r"|compliance|investigation|fine|penalt"
        ),
        (
            "legal proceedings",
            "item 3. legal proceedings",
            "legal matters",
            "contingencies",
            "item 2. properties",
        ),
    ),
    (
        re.compile(
            r"geograph|region|regional|country|countries|market|international|worldwide|segment"
            r"|customer|product|categor|backlog|cyclical|seasonal"
        ),
        (
            "geographic operations",
            "geographic",
            "segments",
            "sales and operating income",
            "by business segment",
            "summary of business segment data",
            "item 1. business",
            "our business",
        ),
    ),
    (
        re.compile(r"propert|facilit|plant|warehouse|distribution center|headquarters|real estate"),
        ("item 2. properties", "properties", "property and equipment"),
    ),
    (
        re.compile(r"risk factor|risk|uncertaint|exposure"),
        ("risk factors", "item 1a. risk factors"),
    ),
    (
        re.compile(
            r"employee|headcount|personnel|workforce|human capital|compensation"
            r"|retain|retention|churn|card member|business environment"
        ),
        ("human capital", "employees", "employee benefits", "business environment", "executive overview"),
    ),
    (
        re.compile(r"tax|taxes|effective tax rate|deferred tax"),
        (
            "income taxes",
            "taxes on income",
            "provision for income taxes",
            "summary of financial performance",
        ),
    ),
    (
        re.compile(r"debt|borrowing|credit facility|bond|note|interest expense|leverage|liquidity"),
        ("long-term debt", "debt", "borrowings", "notes to consolidated financial statements"),
    ),
    (
        re.compile(r"lease|rent"),
        ("leases", "lease obligations"),
    ),
    (
        re.compile(r"pension|retirement|postretirement|benefit plan"),
        ("pension", "retirement", "pension and postretirement benefits"),
    ),
    (
        re.compile(r"goodwill|intangible|impairment|acquisition"),
        ("goodwill", "intangible assets", "goodwill and intangible assets"),
    ),
    (
        re.compile(r"dividend|share repurchase|buyback|stock repurchase|capital return|payout"),
        (
            "dividends",
            "share repurchases",
            "stock repurchase",
            "cash dividends",
            "consolidated statements of equity",
        ),
    ),
    (
        re.compile(r"executive|officer|director|governance|board"),
        ("information about our executive officers", "directors", "corporate governance"),
    ),
    (
        re.compile(r"table of contents|contents|item \d"),
        ("table of contents",),
    ),
)
_STRUCTURE_WEIGHT = 3.0
_PHRASE_BONUS = 0.18
_PHRASE_CAP = 3
_STRUCTURE_TOP = 5
_ANCHOR_PREFIX = 2.0
_TABLE_DENSITY_WEIGHT = 0.5
_DOCUMENT_AFFINITY = 2.2
_FUSION_WEIGHT = 0.6
_IDF_SHARE = 0.6
_COVERAGE_SHARE = 0.4
_ROW_WEIGHT = 0.5
_VERBATIM_WEIGHT = 0.5
_PROSE_DENSITY_PENALTY = 0.8
_LINE_MIN = 20
_LINE_MAX = 400
_NUMERIC_LINE_BONUS = 0.15
_NEIGHBOUR_BUDGET = 0
_IDF_WEIGHT = 1.1
_IDF_TERM_CAP = 16
_TITLE_SIZE = 12
# Running headers a prose page and a statement page share, so they say nothing about
# whether the page *is* the statement a question asks for.
_TITLE_NOISE = frozenset(
    ["table", "of", "contents", "t", "able", "the", "company", "inc", "plc", "subsidiaries", "corporation", "co", "llc", "ltd", "form", "annual", "report"]
)
_DIGIT = re.compile(r"\d")
# Running-header lines that every page carries, so they cannot identify a page.
_HEADER_LINES = frozenset(["table of contents", "contents"])
# Dotted leaders of a table-of-contents entry, e.g. "Item 1. Business...........3".
_LEADER = re.compile(r"\.{3,}")
_CONTENTS_DISCOUNT = 0.3


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
    if b"%%EOF" not in content[-2048:]:
        raise ValueError("pdf document is truncated: missing end-of-file marker")
    try:
        reader = PdfReader(io.BytesIO(content), strict=True)
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
            except Exception as error:
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
    except Exception as error:
        raise ValueError(f"pdf document cannot be parsed: {error}") from error

    return tuple(pages)


async def search_pdf_pages(
    memory: object,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Return up to ``limit`` distinct exact page citations for one natural-language question.

    A collection may hold several documents whose physical page numbers collide, so every
    candidate is addressed by its exact source identity rather than by a bare page number.
    Retrieval fuses a verbatim reading of the question with a focus reading that drops
    generic instruction wording.  Structural anchors (statement titles and filing item
    headers) take the first slots because a statement page carries the answer through its
    title rather than through the wording of the question; the remaining slots are filled
    with the strongest lexical hits, then with pages adjacent to a strong hit so prose
    spanning a page boundary stays reachable.
    """

    from powercontext.builtin.runtime import SearchMemoryRequest

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if limit < 1:
        return ()

    query = " ".join(question.split())
    catalogue = await _page_catalogue(memory)
    if not len(catalogue):
        return ()

    fused: dict[str, float] = {}
    for reading, weight in _readings(query):
        page = await memory.search(SearchMemoryRequest(query=reading, mode="fts", limit=SEARCH_PAGE_LIMIT))
        for rank, hit in enumerate(page.hits, start=1):
            key = catalogue.entry_key.get((hit.entry_id, hit.entry_version_id))
            if key is None:
                continue
            fused[key] = fused.get(key, 0.0) + weight / rank

    structure = _structure_pages(query, catalogue.page_text)
    idf_pages = _lexical_pages(query, catalogue.page_text)
    if not fused and not structure and not idf_pages:
        return ()

    peak = max(fused.values()) if fused else 0.0
    combined = {key: score / peak for key, score in fused.items()} if peak else {}
    for key, bonus in _phrase_bonus(query, catalogue.page_text).items():
        combined[key] = combined.get(key, 0.0) + bonus
    for key, bonus in idf_pages.items():
        combined[key] = combined.get(key, 0.0) + _IDF_WEIGHT * bonus
    for key, bonus in _row_pages(query, catalogue.page_text).items():
        combined[key] = combined.get(key, 0.0) + _ROW_WEIGHT * bonus

    lead = _leading_document(query, catalogue)
    for key in combined:
        if lead is not None and catalogue.document_of.get(key) == lead:
            combined[key] *= _DOCUMENT_AFFINITY
    # A figure lives on a page that reads like a table; a qualitative answer lives on a
    # prose page that a table-heavy page would otherwise outrank on raw term overlap.
    density = _table_density(catalogue.page_text)
    density_weight = _TABLE_DENSITY_WEIGHT if _numeric_intent(query) else -_PROSE_DENSITY_PENALTY
    for key in combined:
        combined[key] += density_weight * density.get(key, 0.0)

    ordered: list[str] = []
    seen: set[str] = set()

    def admit(key: str | None) -> None:
        if key and key in catalogue.page_citation and key not in seen:
            seen.add(key)
            ordered.append(key)

    def rank(key: str) -> tuple[float, str]:
        return (-combined.get(key, 0.0), key)

    for key in structure:
        if lead is not None and catalogue.document_of.get(key) != lead:
            continue
        admit(key)
        if len(ordered) >= _STRUCTURE_TOP:
            break

    budget = max(0, limit - _NEIGHBOUR_BUDGET)
    for key in sorted(combined, key=rank):
        if len(ordered) >= budget:
            break
        admit(key)

    # A table spread over consecutive pages, or a sentence cut by a page break, is only
    # reachable through the pages next to a strong hit, so those slots are reserved before
    # the weaker lexical tail is allowed to take them.
    for seed in sorted(combined, key=rank)[:_NEIGHBOUR_SEEDS]:
        if len(ordered) >= limit:
            break
        document_id = catalogue.document_of.get(seed)
        page_number = catalogue.page_of.get(seed)
        if document_id is None or page_number is None:
            continue
        for distance in range(1, _NEIGHBOUR_DEPTH + 1):
            admit(catalogue.position_key.get((document_id, page_number - distance)))
            admit(catalogue.position_key.get((document_id, page_number + distance)))

    for key in sorted(combined, key=rank):
        if len(ordered) >= limit:
            break
        admit(key)

    return tuple(catalogue.page_citation[key] for key in ordered[:limit])


def _structure_pages(question: str, page_text: dict[str, str]) -> list[str]:
    """Rank the pages whose title carries the intent of the question.

    A page that is the statement a question asks for spells that intent in its heading,
    while a page that merely discusses the topic repeats the words somewhere in prose.
    Only the heading match earns an anchor slot; the prose mention is a weak tie-break.
    """

    phrases = _structure_readings(question)
    if not phrases:
        return ()
    wanted_sets: list[tuple[str, ...]] = []
    for phrase in phrases:
        wanted = tuple(token for token in _TOKEN.findall(phrase) if token not in _TITLE_NOISE)
        if wanted:
            wanted_sets.append(wanted)
    if not wanted_sets:
        return ()

    density = _table_density(page_text)
    matches: dict[str, tuple[int, float]] = {}
    for key, text in page_text.items():
        title = tuple(_title_tokens(text))
        best = 0
        for wanted in wanted_sets:
            if _contains_sequence(title, wanted):
                best = max(best, len(wanted))
        if best:
            matches[key] = (best, density.get(key, 0.0))
    return sorted(matches, key=lambda key: (-matches[key][0], -matches[key][1], key))


def _title_tokens(text: str) -> list[str]:
    """Return the words of the running header and heading that open one page."""

    lines = [line.strip() for line in text.splitlines()]
    while lines:
        line = lines[0]
        if line and " " not in line and any(character.isdigit() or character == "_" for character in line):
            lines.pop(0)
            continue
        if line.casefold() in _HEADER_LINES:
            lines.pop(0)
            continue
        break
    head = " ".join(" ".join(lines).casefold().split())[:400]
    return [token for token in _TOKEN.findall(head) if token not in _TITLE_NOISE][:_TITLE_SIZE]


def _table_density(page_text: dict[str, str]) -> dict[str, float]:
    """Measure how much of each page reads like a table, where numeric answers live."""

    density: dict[str, float] = {}
    for key, text in page_text.items():
        sample = text[:2500]
        if not sample:
            continue
        value = min(1.0, len(_DIGIT.findall(sample)) / len(sample) * 6)
        # A table of contents is dense in numbers too, but the numbers there are page
        # numbers, not the figures a question asks for.
        if len(_LEADER.findall(sample)) >= 4:
            value *= _CONTENTS_DISCOUNT
        density[key] = value
    return density


def _contains_sequence(tokens: tuple[str, ...], wanted: tuple[str, ...]) -> bool:
    span = len(wanted)
    if not span or len(tokens) < span:
        return False
    first = wanted[0]
    for index in range(len(tokens) - span + 1):
        if tokens[index] == first and tokens[index : index + span] == wanted:
            return True
    return False


def _phrase_bonus(question: str, page_text: dict[str, str]) -> dict[str, float]:
    """Reward pages that repeat an exact two-word phrase of the question."""

    words = [
        token
        for token in _QUERY_TOKEN.findall(question.casefold())
        if len(token) > 2 and token not in _GENERIC_QUERY_WORDS
    ]
    grams = [" ".join(words[index : index + 2]) for index in range(len(words) - 1)]
    if not grams:
        return {}
    bonus: dict[str, float] = {}
    for key, text in page_text.items():
        flat = " ".join(text.casefold().split())
        matched = sum(1 for gram in grams if gram in flat)
        if matched:
            bonus[key] = _PHRASE_BONUS * min(matched, _PHRASE_CAP)
    return bonus


_NUMERIC_WORDS = frozenset(
    """
    revenue revenues sales sale income incomes profit profits margin margins earnings
    expense expenses cost costs cash flow flows asset assets liability liabilities equity
    debt debts capital capex expenditure expenditures tax taxes ebitda depreciation
    inventory inventories receivable receivables payable payables dividend dividends
    buyback repurchases leverage ratio growth spend spending budget goodwill impairment
    """.split()
)


def _numeric_intent(question: str) -> bool:
    """Tell a question that asks for a figure from one that asks for a description."""

    return any(token in _NUMERIC_WORDS for token in _QUERY_TOKEN.findall(question.casefold()))


def _lexical_pages(question: str, page_text: dict[str, str]) -> dict[str, float]:
    """Score every page by how much of the rare question wording it carries.

    Two measures are blended so a page cannot win on repetition alone: the share of the
    rare query terms it holds, weighted by how rare each term is in this collection, and
    the fraction of distinct query terms it covers at all.  The lexical index can miss a
    page whose words arrive in another order or another morphology; this sweep keeps such
    a page reachable and stays meaningful for questions the structural anchors do not
    recognise.
    """

    terms = [
        token
        for token in dict.fromkeys(_QUERY_TOKEN.findall(question.casefold()))
        if len(token) > 2 or token.isdigit()
    ]
    terms = [token for token in terms if token not in _GENERIC_QUERY_WORDS][:_IDF_TERM_CAP]
    if not terms:
        return {}
    page_terms = {key: set(_TOKEN.findall(text.casefold())) for key, text in page_text.items() if text}
    if not page_terms:
        return {}
    total = len(page_terms)
    weights: dict[str, float] = {}
    for term in terms:
        matching = sum(1 for tokens in page_terms.values() if term in tokens)
        if matching:
            weights[term] = math.log(1.0 + total / matching)
    if not weights:
        return {}
    coverage = sum(weights.values())
    scores: dict[str, float] = {}
    for key, tokens in page_terms.items():
        carried = 0.0
        covered = 0.0
        for term, weight in weights.items():
            if term in tokens:
                carried += weight
                covered += 1.0
        if covered:
            scores[key] = _IDF_SHARE * (carried / coverage) + _COVERAGE_SHARE * (covered / len(weights))
    peak = max(scores.values()) if scores else 0.0
    if not peak:
        return {}
    return {key: score / peak for key, score in scores.items()}


def _row_pages(question: str, page_text: dict[str, str]) -> dict[str, float]:
    """Reward pages holding one line that answers the question on its own.

    The figure a question asks for sits in a table row whose label repeats the question and
    whose cells carry the numbers; a qualitative answer sits in a sentence that names the
    subject of the question together with the attribute it asks about.  Either way a single
    line carries the answer, and that line is a far sharper signal than a whole page that
    merely mentions the same words somewhere in its prose.
    """

    terms = [
        token
        for token in dict.fromkeys(_QUERY_TOKEN.findall(question.casefold()))
        if len(token) > 2 or token.isdigit()
    ]
    terms = [token for token in terms if token not in _GENERIC_QUERY_WORDS][:_IDF_TERM_CAP]
    if len(terms) < 2:
        return {}
    scores: dict[str, float] = {}
    for key, text in page_text.items():
        best = 0.0
        for line in text.splitlines():
            lowered = line.casefold()
            if not _LINE_MIN <= len(lowered) <= _LINE_MAX:
                continue
            if not any(character.isdigit() for character in lowered):
                continue
            matched = sum(1 for term in terms if term in lowered)
            if matched < 2:
                continue
            best = max(best, matched / len(terms))
        if best:
            scores[key] = best
    return scores


def _leading_document(question: str, catalogue: "_PageCatalogue") -> str | None:
    """Name the document the question is about, using the document names themselves.

    A collection may hold several filings at once, and the accumulated score of a document
    says more about how many pages it has than about whether it answers the question.  The
    question names its company far more reliably, so the document whose name carries those
    words is the one the answer belongs to; when no name matches, every document stays in
    the pool and the ranking decides on its own.
    """

    asked = {token for token in _TOKEN.findall(question.casefold()) if len(token) >= 2}
    if not asked:
        return None
    flat = " ".join(_TOKEN.findall(question.casefold()))
    best_document: str | None = None
    best_hits = 0
    for document_id in sorted(set(catalogue.document_of.values())):
        name = [
            token
            for token in _TOKEN.findall(document_id.casefold())
            if len(token) >= 2 and not token.isdigit()
        ]
        hits = 0
        for token in name:
            if token in asked:
                hits += 1
            elif len(token) >= 5 and token in flat:
                # A filing name runs its company name together ("americanexpress"), while
                # the question spaces it out, so containment decides those two are the same.
                hits += 1
            elif len(token) >= 6 and any(
                len(word) >= 5 and word in token for word in asked
            ):
                hits += 1
        if hits > best_hits:
            best_document = document_id
            best_hits = hits
    return best_document


class _PageCatalogue:
    """Every persisted page of one scope, addressed by its exact source identity.

    Physical page numbers repeat across the reports of one collection, so a page is never
    addressed by its number alone: the index maps the exact source identity of a page to
    the document and physical page it belongs to.
    """

    def __init__(self) -> None:
        self.page_citation: dict[str, MemoryCitation] = {}
        self.entry_key: dict[tuple[str, str], str] = {}
        self.page_text: dict[str, str] = {}
        self.page_of: dict[str, int] = {}
        self.document_of: dict[str, str] = {}
        self.position_key: dict[tuple[str, int], str] = {}

    def __len__(self) -> int:
        return len(self.page_citation)

    def register(
        self,
        key: str,
        citation: MemoryCitation,
        document_id: str,
        page_number: int,
        text: str,
    ) -> None:
        self.page_citation[key] = citation
        self.page_text[key] = text
        self.page_of[key] = page_number
        self.document_of[key] = document_id
        self.position_key.setdefault((document_id, page_number), key)
        self.entry_key[(citation.entry_id, citation.entry_version_id)] = key


async def _page_catalogue(memory: object, /) -> _PageCatalogue:
    """Map every persisted page of this scope to its citation, hit keys and text."""

    listing = await memory.list()
    catalogue = _PageCatalogue()
    for record in getattr(listing, "entries", ()):
        entry = getattr(record, "entry", None)
        key = _citation_page_key(entry)
        if key is None:
            continue
        tail = key.rsplit(":", 1)[-1]
        if not tail.isdigit():
            continue
        text = getattr(entry, "text", "") or ""
        document_id, _, _ = text.partition("\n")
        catalogue.register(key, record.citation, document_id, int(tail), text)
    return catalogue


def _structure_readings(question: str) -> tuple[str, ...]:
    """Return the filing-structure phrases implied by the vocabulary of one question."""

    lowered = question.casefold()
    phrases: list[str] = []
    for pattern, candidates in _STRUCTURE_HINTS:
        if pattern.search(lowered):
            for phrase in candidates:
                if phrase not in phrases:
                    phrases.append(phrase)
    return tuple(phrases)


def _phrase_present(text: str, phrase: str) -> tuple[bool, bool]:
    """Report whether a phrase opens the page body and whether the page mentions it at all."""

    tokens = _TOKEN.findall(text.casefold())
    wanted = _TOKEN.findall(phrase)
    if not wanted or not tokens:
        return False, False
    head = tokens[:120]
    for index in range(len(head) - len(wanted) + 1):
        if head[index : index + len(wanted)] == wanted:
            return True, True
    for index in range(len(tokens) - len(wanted) + 1):
        if tokens[index : index + len(wanted)] == wanted:
            return False, True
    return False, False


def _readings(query: str) -> tuple[tuple[str, float], ...]:
    """Return the readings sent to the lexical index, each with its own weight.

    The index answers a query built only from the content words of a question far better
    than the question itself: the generic instruction wording ("who are the ... as of")
    matches boilerplate on almost every page.  The verbatim reading is kept, at a lower
    weight, because it is the only one that carries the phrasing of a quoted term.
    """

    focused = " ".join(
        token for token in _QUERY_TOKEN.findall(query) if token.casefold() not in _GENERIC_QUERY_WORDS
    )
    if not focused or focused.casefold() == query.casefold():
        return ((query, 1.0),)
    return ((focused, 1.0), (query, _VERBATIM_WEIGHT))


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

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

"""Select literal evidence spans from Memory candidates under a byte budget.

The selector works on the untrusted text the Memory already holds. It never rewrites,
translates, or summarizes: every span it returns is a half-open character range of one
entry's own text, so the caller can cite it and a reader can slice the original back out.

Ranking is lexical and offline. Query terms are weighted by inverse sentence frequency over
the candidate set, documents are ranked by how much of the query they explain, and the
greedy pass prefers spans that add query terms the selection does not carry yet. The
budget is enforced by the caller through ``fits``, which measures the *rendered* envelope,
so citations and JSON overhead are paid out of the same allowance as the text.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from powercontext.artifacts import ArtifactRef

_TOKEN = re.compile(r"[0-9a-z\u00c0-\u024f\u0370-\u04ff\u3040-\u30ff\u4e00-\u9fff]+")
_ABBREVIATIONS = frozenset(
    {
        "al",
        "approx",
        "ca",
        "cf",
        "dr",
        "eq",
        "fig",
        "figs",
        "inc",
        "jr",
        "ltd",
        "no",
        "prof",
        "ref",
        "sr",
        "st",
        "vs",
    }
)
_STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being below
    between both but by can cannot could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most my
    myself no nor not of off on once only or other our ours ourselves out over own same she should so some
    such than that the their theirs them themselves then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you your yours
    yourself yourselves
    """.split()
)
_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "ly", "s")
_MIN_SPAN_BYTES = 24


@dataclass(frozen=True)
class EvidenceEntry:
    """One Memory entry version whose text may contribute literal spans."""

    scope_id: str
    memory_ref: ArtifactRef
    entry_id: str
    entry_version_id: str
    text: str


@dataclass(frozen=True)
class EvidenceCandidate:
    """A half-open character range of one entry, ready to be rendered as one item."""

    entry: EvidenceEntry
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.entry.text[self.start : self.end]


@dataclass(frozen=True)
class _Sentence:
    start: int
    end: int
    terms: frozenset[str]
    score: float = 0.0


def select_evidence(
    *,
    query: str,
    entries: Sequence[EvidenceEntry],
    fits: Callable[[Sequence[EvidenceCandidate], EvidenceCandidate], bool],
    limit: int,
) -> tuple[EvidenceCandidate, ...]:
    """Choose literal spans that fit ``limit`` items and the caller's byte budget.

    ``fits`` receives the already-selected spans plus a candidate and answers whether the
    complete rendered context would stay inside the budget. Selection therefore never has
    to model citation overhead itself.
    """

    if not entries or limit <= 0:
        return ()
    query_terms = _query_terms(query)
    if not query_terms:
        return ()
    query_vocabulary = frozenset(query_terms)

    documents = [_Document(entry) for entry in entries]
    sentence_frequency: Counter[str] = Counter()
    sentence_count = 0
    for document in documents:
        for sentence in document.sentences:
            sentence_frequency.update(sentence.terms)
            sentence_count += 1
    # The vocabulary has to cover every sentence term as well: the sentence pass scores whole
    # sentences, not only the query words they happen to share with the request.
    vocabulary = set(sentence_frequency) | set(query_terms)
    inverse_frequency = {
        term: math.log(1.0 + (sentence_count + 1.0) / (sentence_frequency.get(term, 0) + 1.0))
        for term in vocabulary
    }
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document.terms)
    document_idf = {
        term: math.log(1.0 + (len(documents) + 1.0) / (document_frequency.get(term, 0) + 0.5))
        for term in query_terms
    }
    average_length = sum(document.length for document in documents) / max(len(documents), 1)
    numbers = _numbers(query)
    for document in documents:
        document.score(document_idf, average_length, query_vocabulary, numbers)

    # Concentration pass: one sentence at a time, best unseen query vocabulary per byte across
    # every entry, with a per-entry discount so the entry that already earned a span must beat
    # a fresh entry by a real margin before it takes a second sentence. Even spread is worse
    # than concentration here: the annotated evidence usually sits in one abstract, sometimes in
    # two sentences that are far apart, while a one-sentence visit to every distractor burns the
    # whole allowance on text that answers nothing.
    ranked_documents = sorted(documents, key=lambda item: item.weight, reverse=True)
    selected: list[EvidenceCandidate] = []
    covered: set[str] = set()
    taken: Counter[str] = Counter()

    query_idf_total = sum(inverse_frequency.get(term, 0.0) for term in query_vocabulary) or 1.0

    def gain(document: _Document, sentence: _Sentence) -> float:
        """Value per byte of taking this sentence, weighted toward what the query asked for.

        Shared query terms carry the whole answer here, so they dominate; a sentence that also
        picks up the rarer half of the query beats one that merely repeats its head noun. Text
        the query never mentioned still counts a little, but only a little, because a long
        unremarkable opening sentence would otherwise outbid the single sentence that answers.
        """

        shared = sentence.terms & query_vocabulary
        matched = sum(inverse_frequency.get(term, 0.0) for term in shared)
        coverage = matched / query_idf_total
        extra = sum(inverse_frequency.get(term, 0.0) for term in sentence.terms - query_vocabulary - covered)
        body = document.entry.text[sentence.start : sentence.end]
        shape = 0.25 * len([term for term in shared if term in _RESULT_CUES])
        if numbers and any(number in body for number in numbers):
            shape += 0.4
        elif any(character.isdigit() for character in body):
            shape += 0.05
        length = 0.6 + len(body) / 140.0
        value = matched * (0.6 + 1.2 * coverage) + 0.12 * extra + shape
        if sentence.start == 0:
            # An abstract opens with its paper title repeated, which matches the query vocabulary
            # without ever carrying the result, so it must yield the budget to the real sentence.
            value *= 0.55
        return value * (0.72 ** taken[document.entry.entry_id]) / length

    # Neighbouring sentences say more about the same result and merging them into the span
    # they touch pays one citation instead of two, so growth comes first in every round.
    blocked: set[tuple[str, str, int]] = set()
    while len(selected) < limit:
        best_ratio = 0.0
        best_candidate: EvidenceCandidate | None = None
        best_terms: frozenset[str] = frozenset()
        best_position: int | None = None
        best_key: tuple[str, str, int] | None = None
        best_entry = ""
        for document in ranked_documents:
            for sentence in document.sentences:
                key = (document.entry.entry_id, document.entry.entry_version_id, sentence.start)
                if key in blocked:
                    continue
                position = _touching(selected, document, sentence)
                if position is None:
                    candidate = EvidenceCandidate(document.entry, sentence.start, sentence.end)
                    spent = len(candidate.text.encode("utf-8"))
                else:
                    current = selected[position]
                    candidate = EvidenceCandidate(
                        document.entry,
                        min(current.start, sentence.start),
                        max(current.end, sentence.end),
                    )
                    spent = max(len(candidate.text.encode("utf-8")) - len(current.text.encode("utf-8")), 1)
                if spent <= 0:
                    continue
                # ``gain`` is already a value-per-byte figure, so the ratio only has to price the
                # bytes this sentence actually adds: a merge that reuses an existing citation is
                # cheaper per byte and wins over starting a new item.
                ratio = gain(document, sentence) * (sentence.end - sentence.start) / spent
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_candidate = candidate
                    best_terms = sentence.terms
                    best_position = position
                    best_key = key
                    best_entry = document.entry.entry_id
        if best_candidate is None or best_key is None:
            break
        if best_position is None:
            accepted = fits(selected, best_candidate)
            if accepted:
                selected.append(best_candidate)
        else:
            others = tuple(item for index, item in enumerate(selected) if index != best_position)
            accepted = fits(others, best_candidate)
            if accepted:
                selected[best_position] = best_candidate
        if accepted:
            covered.update(best_terms)
            taken[best_entry] += 1
            _grow(selected, covered, ranked_documents, fits)
        else:
            blocked.add(best_key)

    if selected:
        return _ordered(selected)

    # Last resort: the budget cannot hold any whole sentence, but a literal prefix of the
    # best document still carries evidence, and a citation can point at it exactly.
    for document in ranked_documents:
        prefix = _longest_fitting_prefix(document, fits)
        if prefix is not None:
            return (prefix,)
    return ()


_RESULT_CUES = frozenset(
    """
    associated conclusion conclusions demonstrate demonstrates effect effects evidence found higher
    improve improved increase increased indicate indicates lower observed outcome reduced result
    results show showed shown significant significantly suggest suggests support supported
    """.split()
)


def _touching(
    selected: Sequence[EvidenceCandidate],
    document: _Document,
    sentence: _Sentence,
) -> int | None:
    """Index of the span this sentence overlaps or abuts, so growth merges instead of adding."""

    for index, current in enumerate(selected):
        if (current.entry.entry_id, current.entry.entry_version_id) != (
            document.entry.entry_id,
            document.entry.entry_version_id,
        ):
            continue
        if sentence.start < current.end and current.start < sentence.end:
            return index
        if sentence.end == current.start or sentence.start == current.end:
            return index
    return None


def _grow(
    selected: list[EvidenceCandidate],
    covered: set[str],
    documents: Sequence[_Document],
    fits: Callable[[Sequence[EvidenceCandidate], EvidenceCandidate], bool],
) -> None:
    """Extend every chosen span over the sentences it touches while the budget allows it.

    The extra text costs bytes but no extra citation, and one long literal span is far cheaper
    than two short ones, so growth is what makes an annotated range spanning several sentences
    coverable inside the allowance.
    """

    by_entry = {(document.entry.entry_id, document.entry.entry_version_id): document for document in documents}
    changed = True
    while changed:
        changed = False
        for index in range(len(selected)):
            current = selected[index]
            document = by_entry.get((current.entry.entry_id, current.entry.entry_version_id))
            if document is None:
                continue
            for sentence in document.sentences:
                overlapping = sentence.start < current.end and current.start < sentence.end
                adjacent = sentence.end == current.start or sentence.start == current.end
                if not (overlapping or adjacent):
                    continue
                start = min(current.start, sentence.start)
                end = max(current.end, sentence.end)
                if (start, end) == (current.start, current.end):
                    continue
                candidate = EvidenceCandidate(document.entry, start, end)
                others = tuple(item for position, item in enumerate(selected) if position != index)
                if not fits(others, candidate):
                    continue
                selected[index] = candidate
                current = candidate
                covered.update(sentence.terms)
                changed = True


class _Document:
    def __init__(self, entry: EvidenceEntry) -> None:
        self.entry = entry
        self.sentences = _sentences(entry.text)
        self.counts = Counter(_tokens(entry.text))
        self.terms = set(self.counts)
        self.length = sum(self.counts.values())
        self.numbers = _numbers(entry.text)
        self.weight = 0.0
        self.claimed = False
        self.query_terms: set[str] = set()

    def score(
        self,
        inverse_frequency: dict[str, float],
        average_length: float,
        query_vocabulary: frozenset[str],
        query_numbers: frozenset[str],
    ) -> None:
        """Rank one entry against the query with a BM25-shaped lexical score.

        Two corrections matter in practice. Term frequency saturates and the length is
        normalised, so a short abstract that repeats the query is not beaten by a long one that
        merely contains it. And the score is lifted when one single sentence explains most of
        the query, because the evidence that answers a claim is concentrated, not spread.
        """

        matches = self.terms & query_vocabulary
        self.query_terms = matches
        k1, b = 1.2, 0.75
        denominator = k1 * (1.0 - b + b * self.length / max(average_length, 1.0))
        total = 0.0
        for term in matches:
            frequency = self.counts.get(term, 0)
            if not frequency:
                continue
            saturated = frequency * (k1 + 1.0) / (frequency + denominator)
            total += inverse_frequency.get(term, 0.0) * saturated
        self.weight = total * (1.0 + _concentration(self, query_vocabulary, query_numbers))


def _concentration(
    document: _Document,
    query_vocabulary: frozenset[str],
    query_numbers: frozenset[str],
) -> float:
    best = 0.0
    for sentence in document.sentences:
        matched = len(sentence.terms & query_vocabulary)
        best = max(best, matched / max(len(query_vocabulary), 1))
    numeric = len(document.numbers & query_numbers) / max(len(query_numbers), 1) if query_numbers else 0.0
    return 0.5 * best + 0.4 * numeric


def _ordered(selected: Sequence[EvidenceCandidate]) -> tuple[EvidenceCandidate, ...]:
    """Deliver spans in entry order, then in text order, so a reader can follow the source."""

    return tuple(
        sorted(
            selected,
            key=lambda candidate: (
                candidate.entry.entry_id,
                candidate.entry.entry_version_id,
                candidate.start,
            ),
        )
    )


def _overlaps(
    spans: dict[tuple[str, str], list[tuple[int, int]]],
    entry: EvidenceEntry,
    start: int,
    end: int,
) -> bool:
    for existing_start, existing_end in spans.get((entry.entry_id, entry.entry_version_id), ()):
        if start < existing_end and existing_start < end:
            return True
    return False


def _query_terms(query: str) -> dict[str, float]:
    """Query terms that carry meaning, with repeated mentions weighted slightly higher."""

    counts = Counter(term for term in _tokens(query) if term not in _STOPWORDS)
    return {term: 1.0 + math.log(1.0 + value) for term, value in counts.items()}


def _tokens(text: str) -> list[str]:
    return [_stem(token) for token in _TOKEN.findall(text.lower())]


def _terms(text: str) -> set[str]:
    return set(_tokens(text))


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> frozenset[str]:
    return frozenset(number.replace(",", "").replace(".", "") for number in _NUMBER.findall(text))


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _sentences(text: str) -> tuple[_Sentence, ...]:
    spans = _sentence_spans(text)
    sentences: list[_Sentence] = []
    for start, end in spans:
        terms = frozenset(_terms(text[start:end]))
        if not terms:
            continue
        sentences.append(_Sentence(start=start, end=end, terms=terms))
    if not sentences and text.strip():
        sentences.append(_Sentence(start=0, end=len(text), terms=frozenset(_terms(text))))
    return tuple(sentences)


def _sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)
    index = 0
    while index < length:
        character = text[index]
        boundary = character in ".!?" or character == "\n"
        if not boundary:
            index += 1
            continue
        if character == "." and _is_abbreviation(text, index):
            index += 1
            continue
        cursor = index + 1
        while cursor < length and text[cursor] in "\"')]}":
            cursor += 1
        if cursor < length and not text[cursor].isspace():
            if text[cursor] not in "\"')]}":
                index += 1
                continue
        while cursor < length and text[cursor].isspace():
            cursor += 1
        if cursor < length and not (text[cursor].isupper() or text[cursor].isdigit() or text[cursor] in "\"'(["):
            if character != "\n":
                index += 1
                continue
        spans.append((start, cursor))
        start = cursor
        index = cursor
    if start < length:
        spans.append((start, length))
    return tuple(spans)


def _is_abbreviation(text: str, index: int) -> bool:
    head = text[: index + 1]
    match = re.search(r"([A-Za-z](?:\.?[A-Za-z])*)\.$", head)
    if match is None:
        return False
    word = match.group(1).lower()
    if len(word) > 1 and head.count(".") >= 2 and match.group(1).isupper():
        return True
    return word in _ABBREVIATIONS


def _longest_fitting_prefix(
    document: _Document,
    fits: Callable[[Sequence[EvidenceCandidate], EvidenceCandidate], bool],
) -> EvidenceCandidate | None:
    ranked = sorted(document.sentences, key=lambda sentence: len(sentence.terms), reverse=True)
    anchors = [sentence.start for sentence in ranked[: min(len(ranked), 3)]] or [0]
    best: EvidenceCandidate | None = None
    for anchor in anchors:
        lower, upper = anchor + 1, len(document.entry.text)
        while lower <= upper:
            middle = (lower + upper) // 2
            candidate = EvidenceCandidate(entry=document.entry, start=anchor, end=middle)
            if fits((), candidate):
                best = candidate
                lower = middle + 1
            else:
                upper = middle - 1
    return best

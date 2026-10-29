"""Context packs and prefix deltas for incremental context transfer.

The regular ``prepare`` call returns the whole body on every turn. When a client already
holds the previous body, only the part that changed has to travel: this module turns a body
into an ordered list of chunks and expresses the next body as "keep the first N chunks of
the pack you already have, append these chunks". Reassembly is pure data work, so a receiver
can rebuild the body from bytes alone.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass

__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})
_DIGEST_PREFIX = "sha256:"
_DEFAULT_CHUNK_BYTES = 512


class ContextPackError(ValueError):
    """Raised when a pack or delta breaks the transfer protocol."""


def _canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_chunk(chunk: bytes) -> str:
    return base64.b64encode(chunk).decode("ascii")


def _decode_chunk(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ContextPackError("chunk must be a non-empty base64 string")
    if not isinstance(value, str) or any(
        character.isspace() for character in value
    ):  # pragma: no cover - defensive
        raise ContextPackError("chunk must be a base64 string without whitespace")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as error:
        raise ContextPackError("chunk is not valid base64") from error


def _pack_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": PACK_SCHEMA, "items": [_encode_chunk(item) for item in items]}
    return _DIGEST_PREFIX + hashlib.sha256(_canonical(unsigned)).hexdigest()


def _parse(raw: bytes) -> dict[str, object]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError("payload is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError("payload must be a JSON object")
    return payload


def _require_fields(payload: dict[str, object], expected: frozenset[str]) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ContextPackError(f"unexpected payload fields missing={missing} extra={extra}")


def _split_chunks(data: bytes, chunk_bytes: int) -> tuple[bytes, ...]:
    """Split UTF-8 data on natural boundaries, keeping every chunk under the caller cap."""

    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes <= 0:
        raise ContextPackError("chunk_bytes must be a positive integer")
    if not data:
        return ()
    chunks: list[bytes] = []
    cursor = 0
    length = len(data)
    while cursor < length:
        if length - cursor <= chunk_bytes:
            tail = data[cursor:]
            try:
                tail.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ContextPackError("chunk is not valid UTF-8") from error
            chunks.append(tail)
            break
        end = _chunk_end(data, cursor, chunk_bytes)
        chunk = data[cursor:end]
        try:
            chunk.decode("utf-8")
        except UnicodeDecodeError as error:  # pragma: no cover - guarded by _chunk_end
            raise ContextPackError("chunk is not valid UTF-8") from error
        chunks.append(chunk)
        cursor = end
    return tuple(chunks)


# Cut points are tried from strongest to weakest so that a chunk usually ends at a place where
# the body itself changes shape (a record boundary, a sentence end), which keeps neighbouring
# bodies chunked the same way and therefore reusable as a shared prefix.
_CUT_MARKERS = (b"\n", b"},", b"; ", b". ", b" ")


def _chunk_end(data: bytes, start: int, chunk_bytes: int) -> int:
    limit = min(start + chunk_bytes, len(data))
    floor = start + max(chunk_bytes // 2, 1)
    for marker in _CUT_MARKERS:
        position = data.rfind(marker, floor, limit)
        if position >= 0:
            candidate = position + len(marker)
            while candidate > start:
                try:
                    data[start:candidate].decode("utf-8")
                except UnicodeDecodeError:
                    candidate -= 1
                    continue
                return candidate
    candidate = limit
    while candidate > start:
        try:
            data[start:candidate].decode("utf-8")
        except UnicodeDecodeError:
            candidate -= 1
            continue
        return candidate
    raise ContextPackError("chunk_bytes cannot hold a single character")



@dataclass(frozen=True)
class ContextPack:
    """A complete body, chunked, with a content digest."""

    items: tuple[bytes, ...]
    digest: str

    @classmethod
    def build(cls, items: tuple[bytes, ...] | list[bytes]) -> "ContextPack":
        chunks = tuple(bytes(item) for item in items)
        return cls(items=chunks, digest=_pack_digest(chunks))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> "ContextPack":
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ContextPackError("content is not valid UTF-8") from error
        if not isinstance(content, str):
            raise ContextPackError("content must be text")
        return cls.build(_split_chunks(content.encode("utf-8"), chunk_bytes))

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        payload: dict[str, object] = {
            "schema": PACK_SCHEMA,
            "items": [_encode_chunk(item) for item in self.items],
            "digest": self.digest,
        }
        size = 0
        while True:
            payload["byte_count"] = size
            raw = _canonical(payload)
            if len(raw) == size:
                return raw
            size = len(raw)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ContextPack":
        payload = _parse(raw)
        _require_fields(payload, _PACK_FIELDS)
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("unknown pack schema")
        items = payload["items"]
        if not isinstance(items, list):
            raise ContextPackError("items must be a list")
        chunks = tuple(_decode_chunk(item) for item in items)
        digest = payload["digest"]
        if not isinstance(digest, str):
            raise ContextPackError("digest must be a string")
        expected = _pack_digest(chunks)
        if digest != expected:
            raise ContextPackError("pack digest mismatch")
        byte_count = payload["byte_count"]
        if not isinstance(byte_count, int) or isinstance(byte_count, bool):
            raise ContextPackError("byte_count must be an integer")
        if byte_count != len(raw):
            raise ContextPackError("byte_count mismatch")
        return cls(items=chunks, digest=digest)


@dataclass(frozen=True)
class ContextPackDelta:
    """A body expressed as a retained prefix of a base pack plus an appended suffix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> "ContextPackDelta":
        shared = 0
        for left, right in zip(base.items, target.items):
            if left != right:
                break
            shared += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=shared,
            appended=tuple(target.items[shared:]),
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _canonical(
            {
                "schema": DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": [_encode_chunk(item) for item in self.appended],
            }
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ContextPackDelta":
        payload = _parse(raw)
        _require_fields(payload, _DELTA_FIELDS)
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("unknown delta schema")
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        if not isinstance(base_digest, str) or not isinstance(target_digest, str):
            raise ContextPackError("digests must be strings")
        prefix = payload["prefix_items"]
        if not isinstance(prefix, int) or isinstance(prefix, bool) or prefix < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")
        appended = payload["appended"]
        if not isinstance(appended, list):
            raise ContextPackError("appended must be a list")
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix,
            appended=tuple(_decode_chunk(item) for item in appended),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from a base pack and a delta, verifying both digests."""

    if base.digest != delta.base_digest:
        raise ContextPackError("base digest does not match the delta")
    if not isinstance(delta.prefix_items, int) or isinstance(delta.prefix_items, bool):
        raise ContextPackError("prefix_items must be an integer")
    if delta.prefix_items < 0 or delta.prefix_items > len(base.items):
        raise ContextPackError("prefix_items is outside the base pack")
    rebuilt = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("target digest does not match the rebuilt pack")
    return rebuilt

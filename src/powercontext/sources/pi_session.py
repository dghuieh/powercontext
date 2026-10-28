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

"""Pi session branch capture with durable, resumable event ingestion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TextEvidence

SOURCE_NAME = "pi-event"
SOURCE_VERSION = "1"
CONNECTOR_NAME = "pi-session-history"
CONNECTOR_VERSION = "1"
IDENTITY_PREFIX = "pi:"
CHECKPOINT_VERSION = 1

_CONTROL_ONLY_TYPES = frozenset()

_MESSAGE_ROLES_WITH_STRUCTURED_CONTENT = frozenset({"toolResult", "bashExecution"})


class PiSessionInput(BaseModel):
    """Complete Pi v3 session archive bytes and the branch leaf to materialize."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible Pi session event on the selected branch, with its whole record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A captured Pi event identified by session and event identity."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiSessionConnector:
    """Re-read one Pi session file and ingest only newly visible branch events."""

    name = CONNECTOR_NAME
    version = CONNECTOR_VERSION
    source_definitions = frozenset({SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        raw = self._path.read_bytes()
        session_id, records, order = _parse_archive(raw)
        events = _select_branch(session_id, records, order, self._leaf_id)

        checkpoint = session.checkpoint
        imported: dict[str, str] = {}
        if isinstance(checkpoint, dict) and checkpoint.get("session_id") == session_id:
            stored = checkpoint.get("imported")
            if isinstance(stored, dict):
                imported = {str(key): str(value) for key, value in stored.items()}

        updated = dict(imported)
        for event in events:
            digest = _event_digest(event.record)
            previous = imported.get(event.event_id)
            if previous is not None:
                if previous != digest:
                    raise ValueError(f"pi event {event.event_id!r} was rewritten after ingestion")
                continue
            await session.submit(event.event_id, SOURCE_NAME, event)
            updated[event.event_id] = digest

        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={
                "version": CHECKPOINT_VERSION,
                "session_id": session_id,
                "imported": updated,
            },
        )

    def __repr__(self) -> str:
        return f"PiSessionConnector(path={str(self._path)!r}, leaf_id={self._leaf_id!r})"


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the stable, delimiter-safe identity of one Pi event Source."""

    session = _require_identity_text("session_id", session_id)
    event = _require_identity_text("event_id", event_id)
    source_id = f"{IDENTITY_PREFIX}{quote(session, safe='')}:{quote(event, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"pi event source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Return the visible events of the selected branch from root to leaf."""

    if not isinstance(value.content, bytes):
        raise ValueError("content must be bytes")
    session_id, records, order = _parse_archive(value.content)
    return _select_branch(session_id, records, order, value.leaf_id)


def _parse_archive(content: bytes) -> tuple[str, dict[str, dict[str, Any]], list[str]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"pi session archive must be UTF-8: {error}") from error

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("pi session archive must not be empty")

    rows: list[Any] = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"pi session archive line is not JSON: {error}") from error

    header = rows[0]
    if not isinstance(header, dict):
        raise ValueError("pi session archive must start with a session header")
    if header.get("type") != "session":
        raise ValueError("pi session archive must start with a session header")
    if header.get("version") != 3:
        raise ValueError("pi session archive must declare version 3")
    session_id = _require_identity_text("session id", header.get("id"))

    records: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows[1:]:
        if not isinstance(row, dict):
            raise ValueError("pi session records must be JSON objects")
        event_id = _require_identity_text("event id", row.get("id"))
        if event_id in records:
            raise ValueError(f"pi session record id {event_id!r} is duplicated")
        parent_id = row.get("parentId")
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id:
                raise ValueError("pi session parentId must be a non-empty string or null")
            if parent_id == event_id:
                raise ValueError("pi session records must not be their own parent")
            if parent_id not in records:
                raise ValueError(f"pi session record {event_id!r} references a missing parent")
        records[event_id] = row
        order.append(event_id)

    return session_id, records, order


def _select_branch(
    session_id: str,
    records: dict[str, dict[str, Any]],
    order: list[str],
    leaf_id: str | None,
) -> tuple[PiEventInput, ...]:
    if not order:
        raise ValueError("pi session archive must contain at least one record")
    if leaf_id is None:
        leaf = order[-1]
    else:
        leaf = _require_identity_text("leaf_id", leaf_id)
        if leaf not in records:
            raise ValueError(f"pi session leaf {leaf!r} does not exist")

    chain: list[dict[str, Any]] = []
    current: str | None = leaf
    while current is not None:
        record = records[current]
        chain.append(record)
        parent = record.get("parentId")
        current = parent if isinstance(parent, str) else None
    chain.reverse()

    events: list[PiEventInput] = []
    for record in chain:
        event = _visible_event(session_id, record)
        if event is not None:
            events.append(event)
    return tuple(events)


def _visible_event(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    event_id = str(record["id"])
    parent_id = record.get("parentId")
    event_type = record.get("type")
    if not isinstance(event_type, str) or not event_type:
        return None

    if event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return None
        role, body = event_type, summary
    elif event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        body = _message_body(role, message)
        if body is None:
            return None
    elif event_type == "custom_message":
        body = _content_text(record.get("content"))
        if body is None or not body.strip():
            return None
        role = "custom"
    else:
        return None

    pi_event_source_id(session_id, event_id)
    return PiEventInput(
        session_id=session_id,
        event_id=event_id,
        parent_id=parent_id if isinstance(parent_id, str) else None,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_body(role: str, message: dict[str, Any]) -> str | None:
    if role == "toolResult":
        tool_name = message.get("toolName")
        call_id = message.get("toolCallId")
        text = _content_text(message.get("content")) or ""
        prefix = f"[tool_result {_tool_token(tool_name)} {_tool_token(call_id)}]"
        return f"{prefix} {text}"
    if role == "bashExecution":
        command = message.get("command")
        command_text = command if isinstance(command, str) else ""
        output = message.get("output")
        body = f"$ {command_text}"
        if isinstance(output, str) and output:
            body = f"{body}\n{output}"
        return body
    if role in _MESSAGE_ROLES_WITH_STRUCTURED_CONTENT:  # pragma: no cover - handled above
        return None
    return _content_text(message.get("content"))


def _content_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif kind == "toolCall":
            parts.append(_tool_call_text(block))
    if not parts:
        return None
    return "\n".join(parts)


def _tool_call_text(block: dict[str, Any]) -> str:
    name = _tool_token(block.get("name"))
    call_id = _tool_token(block.get("id"))
    arguments = block.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    serialized = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"[tool_call {name} {call_id}] {serialized}"


def _tool_token(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return "unknown"


def _event_digest(record: dict[str, JsonValue]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_identity_text(label: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"pi session {label} must be a non-empty string")
    if not value.strip():
        raise ValueError(f"pi session {label} must not be blank")
    return value


class _PiEventAdapter:
    """Resolve and read Pi event Sources without touching a catalog."""

    name = SOURCE_NAME
    input_class = PiEventInput
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class _PiEventTextEvidenceProjection:
    """Project one Pi event into the standard ``powercontext.text-evidence`` shape."""

    name = "powercontext.text-evidence"
    version = "1"
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = (
    AdapterSourceDefinition(
        adapter=_PiEventAdapter(),
        version=SOURCE_VERSION,
        projections=(_PiEventTextEvidenceProjection(),),
    )
)


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]

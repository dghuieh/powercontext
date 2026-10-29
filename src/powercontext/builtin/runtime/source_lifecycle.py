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

"""Propagate Source lifecycle state through the per-scope evidence DAG.

Callers register the evidence graph explicitly: Sources are the only writable
roots, Memory entries and Topic Memory artifacts declare the parents that
support them, and every derived node carries one ``all`` / ``any`` policy.
Lifecycle events move a Source between available and withdrawn; the state is
then recomputed for the whole graph and projected onto the native Memory and
Topic Memory read paths so stale evidence leaves search and Context while
retaining its immutable body versions for audit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import insert, select, update

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.builtin.sources import validate_scope_id
from powercontext.builtin.statistics import ModelUsagePurpose
from powercontext.errors import ArtifactNotFoundError
from powercontext.sources.models import SourceRef

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

    from powercontext.builtin.artifacts.memory import Memory
    from powercontext.builtin.runtime.application import BuiltinRuntime

Family = Literal["source", "memory", "topic"]
Policy = Literal["all", "any"]
Purpose = Literal["generation", "audit"]

_FAMILIES: tuple[Family, ...] = ("source", "memory", "topic")
_POLICIES: tuple[Policy, ...] = ("all", "any")
_PURPOSES: tuple[Purpose, ...] = ("generation", "audit")
_LIFECYCLE_REASON = "source-lifecycle"


class SourceLifecycleError(ValueError):
    """Reject an invalid lifecycle request without changing stored state."""


class SourceLifecycleEvent(BaseModel):
    """One validated Source lifecycle transition supplied by the caller."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: str
    revision: int = Field(ge=1)
    sequence: int = Field(ge=1)
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def _require_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be non-blank text")
        return value


@dataclass(frozen=True)
class _Node:
    node_id: str
    family: Family
    ordinal: int
    reference: BaseModel
    parents: tuple[str, ...]
    policy: Policy
    revision: int | None
    sequence: int | None
    available: bool
    reason: str | None


async def _head_or_none(service: Any, artifact_id: str) -> Memory | None:
    try:
        return await service.head(artifact_id)
    except ArtifactNotFoundError:
        return None


def _require_node_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError("node-id")
    return value


def _require_parents(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise SourceLifecycleError("parents")
    parents = tuple(_require_node_id(parent) for parent in value)
    if not parents:
        raise SourceLifecycleError("parents")
    if len(set(parents)) != len(parents):
        raise SourceLifecycleError("duplicate-parent")
    return parents


def _require_policy(value: object) -> Policy:
    if value not in _POLICIES:
        raise SourceLifecycleError("policy")
    return value


def _require_reference(node: object, expected: type[BaseModel]) -> BaseModel:
    if not isinstance(node, expected):
        raise SourceLifecycleError("reference")
    return node


def _reference_from_row(row: dict[str, Any]) -> BaseModel:
    family = row["family"]
    payload = row["reference"]
    if family == "source":
        return SourceRef.model_validate_json(payload)
    if family == "memory":
        return MemoryCitation.model_validate_json(payload)
    if family == "topic":
        return ArtifactRef.model_validate_json(payload)
    raise SourceLifecycleError("family")


def _row_to_node(row: dict[str, Any]) -> _Node:
    return _Node(
        node_id=row["node_id"],
        family=row["family"],
        ordinal=row["ordinal"],
        reference=_reference_from_row(row),
        parents=tuple(json.loads(row["parents"])),
        policy=row["policy"],
        revision=row["revision"],
        sequence=row["sequence"],
        available=bool(row["available"]),
        reason=row["reason"],
    )


def _propagate(nodes: dict[str, _Node]) -> dict[str, bool]:
    """Recompute availability from the Sources down, honouring each policy."""

    available = {node_id: node.available for node_id, node in nodes.items() if node.family == "source"}
    derived = sorted((node for node in nodes.values() if node.family != "source"), key=lambda node: node.ordinal)
    pending = True
    while pending:
        pending = False
        for node in derived:
            if node.node_id in available:
                continue
            supports = [available.get(parent, False) for parent in node.parents]
            available[node.node_id] = all(supports) if node.policy == "all" else any(supports)
            pending = True
    return available


class SourceLifecycleApplication:
    """Runtime entry point exposing per-scope lifecycle services."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, scope_id)


class ScopedSourceLifecycle:
    """Register evidence nodes and propagate Source lifecycle state in one scope."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = validate_scope_id(scope_id)

    # ---------------------------------------------------------------- registration

    async def register_source(self, node_id: str, reference: SourceRef, *, revision: int) -> None:
        source = _require_reference(reference, SourceRef)
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise SourceLifecycleError("revision")
        identity = _require_node_id(node_id)
        await self._register(
            identity,
            family="source",
            reference=source,
            parents=(),
            policy="all",
            revision=revision,
            sequence=0,
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: Policy,
    ) -> None:
        reference = _require_reference(citation, MemoryCitation)
        identity = _require_node_id(node_id)
        bound_parents = _require_parents(parents)
        bound_policy = _require_policy(policy)
        await self._assert_memory_reference(citation)
        await self._register(
            identity,
            family="memory",
            reference=reference,
            parents=bound_parents,
            policy=bound_policy,
            revision=None,
            sequence=None,
        )

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: Policy,
    ) -> None:
        artifact = _require_reference(reference, ArtifactRef)
        identity = _require_node_id(node_id)
        bound_parents = _require_parents(parents)
        bound_policy = _require_policy(policy)
        await self._assert_topic_reference(artifact)
        await self._register(
            identity,
            family="topic",
            reference=artifact,
            parents=bound_parents,
            policy=bound_policy,
            revision=None,
            sequence=None,
        )

    async def _register(
        self,
        node_id: str,
        *,
        family: Family,
        reference: BaseModel,
        parents: tuple[str, ...],
        policy: Policy,
        revision: int | None,
        sequence: int | None,
    ) -> None:
        database = self._require_database()
        payload = reference.model_dump_json()
        async with self._runtime._scoped_operation(self.scope_id):
            async with self._runtime._locked(self.scope_id):
                async with database.transaction() as connection:
                    nodes = await self._load(connection)
                    existing = nodes.get(node_id)
                    if existing is not None:
                        if (
                            existing.family == family
                            and existing.reference.model_dump_json() == payload
                            and existing.parents == parents
                            and existing.policy == policy
                            and existing.revision == revision
                        ):
                            return
                        raise SourceLifecycleError("conflict")
                    if family != "source":
                        if node_id in parents:
                            raise SourceLifecycleError("cycle")
                        if any(parent not in nodes for parent in parents):
                            raise SourceLifecycleError("parent")
                    ordinal = max((node.ordinal for node in nodes.values()), default=-1) + 1
                    node = _Node(
                        node_id=node_id,
                        family=family,
                        ordinal=ordinal,
                        reference=reference,
                        parents=parents,
                        policy=policy,
                        revision=revision,
                        sequence=sequence,
                        available=family == "source",
                        reason=None,
                    )
                    nodes[node_id] = node
                    available = _propagate(nodes)
                    node = replace(node, available=available[node_id])
                    nodes[node_id] = node
                    await connection.execute(
                        insert(SOURCE_LIFECYCLE_NODES_TABLE).values(
                            scope_id=self.scope_id,
                            node_id=node.node_id,
                            family=node.family,
                            ordinal=node.ordinal,
                            reference=payload,
                            parents=json.dumps(list(node.parents)),
                            policy=node.policy,
                            revision=node.revision,
                            sequence=node.sequence,
                            available=node.available,
                            reason=node.reason,
                        )
                    )
                    for other in nodes.values():
                        if other.node_id == node_id or other.available == available[other.node_id]:
                            continue
                        await self._store_availability(connection, other.node_id, available[other.node_id])
            await self._sync_memory(nodes, available)

    async def _assert_memory_reference(self, citation: MemoryCitation) -> None:
        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            if citation.memory_ref.artifact_id != context.artifacts.memory_artifact_id:
                raise SourceLifecycleError("reference")
            if await _head_or_none(service, context.artifacts.memory_artifact_id) is None:
                raise SourceLifecycleError("reference")
            try:
                await service.validate_citation(citation)
            except (ArtifactNotFoundError, ValueError) as error:
                raise SourceLifecycleError("reference") from error

    async def _assert_topic_reference(self, reference: ArtifactRef) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                reference.family,
                reference.artifact_id,
                reference.revision,
            )
        except Exception as error:  # noqa: BLE001 - every miss is an unreadable citation
            raise SourceLifecycleError("reference") from error

    # -------------------------------------------------------------------- lifecycle

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event")
        database = self._require_database()
        nodes: dict[str, _Node] = {}
        available: dict[str, bool] = {}
        async with self._runtime._context(self.scope_id, embedding_purpose=ModelUsagePurpose.MEMORY_INDEXING):
            async with self._runtime._locked(self.scope_id):
                async with database.transaction() as connection:
                    nodes = await self._load(connection)
                    source = nodes.get(event.node_id)
                    if source is None or source.family != "source":
                        raise SourceLifecycleError("unknown-source")
                    if event.revision != source.revision:
                        raise SourceLifecycleError("revision")
                    current_sequence = 0 if source.sequence is None else source.sequence
                    if event.sequence < current_sequence:
                        raise SourceLifecycleError("sequence")
                    if event.sequence == current_sequence and (
                        event.available != source.available or event.reason != source.reason
                    ):
                        raise SourceLifecycleError("sequence")
                    if event.sequence > current_sequence:
                        nodes[event.node_id] = replace(
                            source,
                            sequence=event.sequence,
                            available=event.available,
                            reason=event.reason,
                        )
                        await connection.execute(
                            update(SOURCE_LIFECYCLE_NODES_TABLE)
                            .where(
                                SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self.scope_id,
                                SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == event.node_id,
                            )
                            .values(sequence=event.sequence, available=event.available, reason=event.reason)
                        )
                    available = _propagate(nodes)
                    for node in nodes.values():
                        if node.available == available[node.node_id]:
                            continue
                        await self._store_availability(connection, node.node_id, available[node.node_id])
                await self._sync_memory(nodes, available)
        return sorted(node_id for node_id, state in available.items() if state)

    async def resolve(self, node_id: str, *, purpose: Purpose = "generation") -> BaseModel:
        if purpose not in _PURPOSES:
            raise SourceLifecycleError("purpose")
        identity = _require_node_id(node_id)
        database = self._require_database()
        async with database.transaction() as connection:
            node = (await self._load(connection)).get(identity)
        if node is None:
            raise SourceLifecycleError("unknown-node")
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError("unavailable")
        return node.reference

    # -------------------------------------------------------------------- projection

    async def _sync_memory(self, nodes: dict[str, _Node], available: dict[str, bool]) -> None:
        """Project recomputed availability onto the native Memory entry states."""

        targets = {
            node_id: available.get(node_id, False) for node_id, node in nodes.items() if node.family == "memory"
        }
        if not targets:
            return
        async with self._runtime._context(self.scope_id, embedding_purpose=ModelUsagePurpose.MEMORY_INDEXING) as context:
            service = context.artifacts.memory
            current = await _head_or_none(service, context.artifacts.memory_artifact_id)
            if current is None:
                return
            manifest = {entry.entry_id: entry for entry in current.content.manifest.entries}
            versions = {(version.entry_id, version.entry_version_id): version for version in await service.entries(current)}
            deactivate = []
            activate = []
            for node_id, desired in targets.items():
                citation = nodes[node_id].reference
                if not isinstance(citation, MemoryCitation):
                    continue
                entry = manifest.get(citation.entry_id)
                if entry is None or entry.entry_version_id != citation.entry_version_id:
                    continue
                if (entry.state == "active") == desired:
                    continue
                version = versions.get((citation.entry_id, citation.entry_version_id))
                if version is None:
                    continue
                (activate if desired else deactivate).append(version)
            if deactivate:
                current = await service.forget(current, entries=tuple(deactivate), reason=_LIFECYCLE_REASON)
            if activate:
                await service.reactivate(current, entries=tuple(activate), reason=_LIFECYCLE_REASON)

    async def unavailable_topic_ids(self) -> frozenset[str]:
        """Return Topic artifacts whose supporting evidence is withdrawn."""

        database = self._require_database()
        async with database.transaction() as connection:
            nodes = await self._load(connection)
        return frozenset(
            node.reference.artifact_id
            for node in nodes.values()
            if node.family == "topic" and isinstance(node.reference, ArtifactRef) and not node.available
        )

    async def filter_topic_hits(self, result: Any) -> Any:
        """Hide withdrawn Topic Memory from search and Context projections."""

        hidden = await self.unavailable_topic_ids()
        if not hidden:
            return result
        hits = tuple(hit for hit in result.hits if hit.artifact_ref.artifact_id not in hidden)
        if len(hits) == len(result.hits):
            return result
        return result.model_copy(update={"hits": hits})

    # ------------------------------------------------------------------------- storage

    def _require_database(self) -> Any:
        database = self._runtime._database
        if database is None:
            raise SourceLifecycleError("persistence")
        return database

    async def _load(self, connection: AsyncConnection) -> dict[str, _Node]:
        statement = (
            select(SOURCE_LIFECYCLE_NODES_TABLE)
            .where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self.scope_id)
            .order_by(SOURCE_LIFECYCLE_NODES_TABLE.c.ordinal)
        )
        rows = (await connection.execute(statement)).mappings().all()
        return {node.node_id: node for node in (_row_to_node(dict(row)) for row in rows)}

    async def _store_availability(self, connection: AsyncConnection, node_id: str, available: bool) -> None:
        await connection.execute(
            update(SOURCE_LIFECYCLE_NODES_TABLE)
            .where(
                SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self.scope_id,
                SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
            )
            .values(available=available)
        )


__all__ = [
    "ScopedSourceLifecycle",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
]

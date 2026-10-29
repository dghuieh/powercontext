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

"""OceanBase single-statement Memory hybrid candidate query.

``mode="hybrid"`` used to issue one FULLTEXT statement and one vector statement
and merge the two channel rankings in the application. The same two native
indexes can serve both candidate sets from one statement, so the backend keeps
the filtering, head validation and channel identities while asking the database
once.
"""

from __future__ import annotations

from typing import Any

from pyobvector import VECTOR
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import (
    CapabilityNotSupportedError,
    EmbeddingProfile,
    MemoryCapabilities,
    MemoryProjection,
    MemorySearchChannels,
    MemorySearchRequest,
)
from powercontext.builtin.artifacts.memory.canonical import validate_embedding
from powercontext.builtin.persistence.memory_index import memory_channel_hits
from powercontext.builtin.persistence.oceanbase.memory_index import OceanBaseMemoryVectorIndex
from powercontext.builtin.persistence.tags import memory_tag_parameters

_FTS_TAG_MARKER = "__PC_FTS_TAGS__"
_VECTOR_TAG_MARKER = "__PC_VECTOR_TAGS__"
_APPROXIMATE_MARKER = "__PC_APPROXIMATE__"

# One statement, two native candidate sub-queries. The FULLTEXT branch drives
# ``TEXT RETRIEVAL SCAN`` on ``ix_pc_memory_entry_heads_fts`` and the vector
# branch drives ``VECTOR INDEX`` on the HNSW projection, so both native indexes
# serve the hybrid candidate set without a second round trip.
_HYBRID_CANDIDATE_SQL = f"""
(SELECT
    'fts' AS channel,
    h.memory_artifact_id AS memory_artifact_id,
    h.head_revision AS head_revision,
    h.entry_id AS entry_id,
    h.entry_version_id AS entry_version_id,
    v.text AS text,
    NULL AS distance,
    MATCH (h.searchable_text) AGAINST (:analyzed_query) AS fts_score
 FROM pc_memory_entry_heads AS h
 JOIN pc_memory_entry_versions AS v
   ON v.scope_id = h.scope_id
  AND v.memory_artifact_id = h.memory_artifact_id
  AND v.entry_version_id = h.entry_version_id
 WHERE h.scope_id = :scope_id
   AND h.memory_artifact_id IN :memory_artifact_ids
   {_FTS_TAG_MARKER}
   AND MATCH (h.searchable_text) AGAINST (:analyzed_query)
 ORDER BY MATCH (h.searchable_text) AGAINST (:analyzed_query) DESC,
          h.memory_artifact_id,
          h.entry_id,
          h.entry_version_id
 LIMIT :candidate_limit)
UNION ALL
(SELECT
    'vector' AS channel,
    m.memory_artifact_id AS memory_artifact_id,
    m.head_revision AS head_revision,
    m.entry_id AS entry_id,
    m.entry_version_id AS entry_version_id,
    v.text AS text,
    l2_distance(m.embedding, :query_vector) AS distance,
    NULL AS fts_score
 FROM pc_memory_vector_entries AS m
 JOIN pc_memory_entry_versions AS v
   ON v.scope_id = m.scope_id
  AND v.memory_artifact_id = m.memory_artifact_id
  AND v.entry_version_id = m.entry_version_id
 WHERE m.scope_id = :scope_id
   AND m.memory_artifact_id IN :memory_artifact_ids
   {_VECTOR_TAG_MARKER}
 ORDER BY l2_distance(m.embedding, :query_vector) {_APPROXIMATE_MARKER}
 LIMIT :candidate_limit)
"""


def _tag_predicate_sql(alias: str) -> str:
    """A parameterized tag predicate for the native candidate SQL."""

    return f"""
   AND (SELECT COUNT(*) FROM pc_artifact_tags AS tags
        WHERE tags.scope_id = {alias}.scope_id
          AND tags.family = 'memory'
          AND tags.artifact_id = {alias}.memory_artifact_id
          AND tags.target_type = 'memory_entry'
          AND tags.target_id = {alias}.entry_id
          AND tags.tag_key_hash IN :tag_hashes
          AND tags.tag_key IN :tag_keys) >= :tag_minimum
    """


class OceanBaseMemoryHybridIndex:
    """Serve both candidate channels of one hybrid search from one statement."""

    def __init__(self, vector_index: OceanBaseMemoryVectorIndex) -> None:
        profile: EmbeddingProfile = vector_index.profile
        self.profile = profile
        self.capabilities = MemoryCapabilities(
            tag_filter=True,
            fts=True,
            vector=True,
            hybrid=True,
            embedding_profile=profile,
        )
        self.tables: tuple[Any, ...] = ()
        # The hybrid statement reuses the projection tables and indexes that the
        # FULLTEXT and vector strategies already create, own and maintain.
        self._vector = vector_index
        self._untagged = self._statement(approximate=True, tagged=False)
        self._tagged = self._statement(approximate=False, tagged=True)

    def _statement(self, *, approximate: bool, tagged: bool) -> Any:
        sql = _HYBRID_CANDIDATE_SQL
        sql = sql.replace(_FTS_TAG_MARKER, _tag_predicate_sql("h") if tagged else "")
        sql = sql.replace(_VECTOR_TAG_MARKER, _tag_predicate_sql("m") if tagged else "")
        sql = sql.replace(_APPROXIMATE_MARKER, "APPROXIMATE" if approximate else "")
        parameters = [
            bindparam("memory_artifact_ids", expanding=True),
            bindparam("query_vector", type_=VECTOR(self.profile.dimension)),
        ]
        if tagged:
            parameters.extend(
                (
                    bindparam("tag_keys", expanding=True),
                    bindparam("tag_hashes", expanding=True),
                )
            )
        return text(sql).bindparams(*parameters)

    async def initialize(self, connection: AsyncConnection, /) -> None:
        if connection.dialect.name != "mysql":
            raise CapabilityNotSupportedError("oceanbase-hybrid")

    async def replace(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        del connection, scope_id, memory_ref, projections

    async def delete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        entry_ids: tuple[str, ...],
        /,
    ) -> None:
        del connection, scope_id, memory_ref, entry_ids

    async def upsert(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        del connection, scope_id, memory_ref, projections

    async def search(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
        /,
    ) -> MemorySearchChannels:
        if request.mode != "hybrid":
            return MemorySearchChannels()
        if request.embedding_profile != self.profile or request.query_vector is None:
            raise CapabilityNotSupportedError("embedding-profile")
        if not await self.vector_complete(connection, scope_id, request.memories, self.profile):
            raise CapabilityNotSupportedError("vector")
        arguments: dict[str, Any] = {
            "scope_id": scope_id,
            "memory_artifact_ids": tuple(memory.artifact_id for memory in request.memories),
            "analyzed_query": request.analyzed_query,
            "query_vector": validate_embedding(request.query_vector, dimension=self.profile.dimension),
            "candidate_limit": request.candidate_limit,
            **memory_tag_parameters(request.tag_filter),
        }
        statement = self._untagged if request.tag_filter is None else self._tagged
        rows = (await connection.execute(statement, arguments)).mappings().all()
        return MemorySearchChannels(
            fts=memory_channel_hits(_ordered(rows, "fts"), request.memories),
            vector=memory_channel_hits(_ordered(rows, "vector"), request.memories),
        )

    async def vector_complete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memories: tuple[ArtifactRef, ...],
        profile: EmbeddingProfile,
        /,
    ) -> bool:
        return await self._vector.vector_complete(connection, scope_id, memories, profile)

    async def hydrate(
        self,
        connection: AsyncConnection,
        scope_id: str,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> tuple[MemoryProjection, ...]:
        del connection, scope_id
        return projections


def _ordered(rows: Any, channel: str) -> tuple[dict[str, Any], ...]:
    """Restore channel order and break ties deterministically.

    ``UNION ALL`` does not promise to preserve the order of its branches, so the
    native rank key each branch selects is re-applied here before fusion.
    """

    selected = [dict(row) for row in rows if str(row["channel"]) == channel]
    if channel == "fts":
        selected.sort(key=lambda row: (-float(row["fts_score"]), _tie_break(row)))
    else:
        selected.sort(key=lambda row: (float(row["distance"]), _tie_break(row)))
    return tuple(selected)


def _tie_break(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["memory_artifact_id"]), str(row["entry_id"]), str(row["entry_version_id"]))


__all__ = ["OceanBaseMemoryHybridIndex"]

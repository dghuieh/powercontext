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

"""HTTP cross-encoder reranking for coarse Memory search candidates."""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import (
    MemoryRerankDecision,
    _validate_bounds,
)
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score coarse candidates through one batched HTTP scoring service."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("endpoint must be a non-empty string")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("policy_id must be a non-empty string")
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return the globally ordered ranks chosen from service scores."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests_made = 0
        total = len(candidates)
        for start in range(0, total, self._batch_size):
            batch = tuple(
                enumerate(candidates[start : start + self._batch_size], start=start + 1)
            )
            payload = {
                "query": query,
                "candidates": [{"rank": rank, "text": candidate.text} for rank, candidate in batch],
            }
            response = await self._client.post(self._endpoint, json=payload)
            requests_made += 1
            response.raise_for_status()
            expected = {rank for rank, _candidate in batch}
            scores.update(_parse_scores(response, expected))
        ordered = sorted(range(1, total + 1), key=lambda rank: (-scores[rank], rank))
        return MemoryRerankDecision(
            selected_ranks=tuple(ordered[:limit]),
            usage=InferenceUsage(requests=requests_made),
        )


def _parse_scores(response: httpx.Response, expected: frozenset[int] | set[int]) -> dict[int, float]:
    """Validate one batch response and map its scores back to global ranks."""

    try:
        body: Any = response.json()
    except ValueError as error:
        raise ValueError("reranker response is not valid JSON") from error
    if not isinstance(body, dict) or set(body) != {"scores"}:
        raise ValueError("reranker response must contain exactly one 'scores' field")
    entries = body["scores"]
    if not isinstance(entries, list):
        raise ValueError("reranker 'scores' must be a list")
    parsed: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"rank", "score"}:
            raise ValueError("each score must contain exactly 'rank' and 'score'")
        rank = entry["rank"]
        score = entry["score"]
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError("score rank must be an integer")
        if rank in parsed:
            raise ValueError("score rank appears more than once in one response")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("score value must be a finite number")
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("score value must be a finite number")
        parsed[rank] = value
    if set(parsed) != set(expected):
        raise ValueError("reranker response must cover every requested rank exactly once")
    return parsed


__all__ = ["DEFAULT_HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]

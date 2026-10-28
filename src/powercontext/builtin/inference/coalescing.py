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

"""Cross-call embedding request coalescing and a bounded vector cache.

The wrapper lets several concurrent ``embed`` calls share one underlying model
request when they ask for the same text inside the same coalescing window, keeps
late arrivals attached to a request that is already running, and reuses successful
vectors through an LRU cache. Cancelling one caller never cancels the request others
still wait for, and each caller only pays usage for the distinct provider batches it
actually waited on.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from powercontext.builtin.inference.models import (
    EmbeddingResult,
    EmbeddingVector,
    InferenceUsage,
)
from powercontext.builtin.inference.protocols import EmbeddingModel

_QUEUE_SETTLE = 0.0005


@dataclass(frozen=True, slots=True)
class _Delivered:
    """One settled provider vector together with the batch that produced it."""

    vector: EmbeddingVector
    batch_id: int
    usage: InferenceUsage


@dataclass(frozen=True, slots=True)
class _Failed:
    """A provider failure handed to every caller that waited on the failed batch."""

    error: BaseException


class CoalescingEmbeddingModel:
    """Wrap an embedding model with request coalescing and a bounded LRU vector cache."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if not isinstance(coalesce_ms, (int, float)) or isinstance(coalesce_ms, bool) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache_size must not be negative")
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000.0
        self._cache_limit = cache_size
        self.profile = getattr(model, "profile", None)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._waiters: dict[str, asyncio.Future[Any]] = {}
        self._queued: list[str] = []
        self._flush: asyncio.Task[None] | None = None
        self._batch_seq = 0

    # ------------------------------------------------------------------ embed

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Embed one ordered batch, sharing duplicate and concurrent work."""

        items = tuple(texts)
        if not items:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._adopt_loop(loop)

        resolved: dict[int, EmbeddingVector] = {}
        waiting: list[tuple[int, asyncio.Future[Any]]] = []
        for index, text in enumerate(items):
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                resolved[index] = cached
                continue
            waiter = self._waiters.get(text)
            if waiter is None:
                waiter = loop.create_future()
                self._waiters[text] = waiter
                self._queued.append(text)
                self._schedule_flush(loop)
            waiting.append((index, waiter))

        batches: dict[int, InferenceUsage] = {}
        for index, waiter in waiting:
            delivered = await asyncio.shield(waiter)
            if isinstance(delivered, _Failed):
                raise delivered.error
            resolved[index] = delivered.vector
            batches.setdefault(delivered.batch_id, delivered.usage)

        return EmbeddingResult(
            vectors=tuple(resolved[index] for index in range(len(items))),
            usage=_merge_usage(batches.values()),
        )

    async def embed_query(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Embed retrieval queries through the wrapped model when it distinguishes them."""

        embed_query = getattr(self._model, "embed_query", None)
        if embed_query is None:
            return await self.embed(texts)
        return await embed_query(texts)

    # ------------------------------------------------------- internal plumbing

    def _adopt_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._waiters.clear()
        self._queued.clear()
        self._flush = None

    def _schedule_flush(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._flush is not None:
            return
        self._flush = loop.create_task(self._flush_queue())

    async def _flush_queue(self) -> None:
        delay = self._coalesce_seconds if self._coalesce_seconds > 0 else _QUEUE_SETTLE
        if delay > 0:
            await asyncio.sleep(delay)
        queued = self._queued
        self._queued = []
        self._flush = None
        for start in range(0, len(queued), self._max_batch_size):
            batch = tuple(queued[start : start + self._max_batch_size])
            self._batch_seq += 1
            asyncio.get_running_loop().create_task(self._run_batch(batch, self._batch_seq))

    async def _run_batch(self, batch: tuple[str, ...], batch_id: int) -> None:
        try:
            result = await self._model.embed(batch)
        except asyncio.CancelledError:  # pragma: no cover - shutdown of the host loop
            self._settle(batch, _Failed(RuntimeError("embedding request cancelled")))
            raise
        except BaseException as error:  # noqa: BLE001 - failures are forwarded to every waiter
            self._settle(batch, _Failed(error))
            return

        vectors = tuple(result.vectors)
        if len(vectors) != len(batch):
            self._settle(
                batch,
                _Failed(
                    ValueError(
                        f"embedding model returned {len(vectors)} vectors for {len(batch)} texts"
                    )
                ),
            )
            return

        usage = result.usage
        for text, vector in zip(batch, vectors, strict=True):
            resolved = tuple(float(component) for component in vector)
            self._remember(text, resolved)
            self._settle_text(text, _Delivered(resolved, batch_id, usage))

    def _settle(self, batch: Iterable[str], outcome: _Delivered | _Failed) -> None:
        for text in batch:
            self._settle_text(text, outcome)

    def _settle_text(self, text: str, outcome: _Delivered | _Failed) -> None:
        self._queued = [queued for queued in self._queued if queued != text]
        waiter = self._waiters.pop(text, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(outcome)

    def _remember(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_limit <= 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        model = self.__dict__.get("_model")
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)


def _merge_usage(batches: Iterable[InferenceUsage]) -> InferenceUsage:
    requests = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    for usage in batches:
        requests += usage.requests
        if usage.input_tokens is not None:
            input_tokens = (input_tokens or 0) + usage.input_tokens
        if usage.output_tokens is not None:
            output_tokens = (output_tokens or 0) + usage.output_tokens
    return InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)


__all__ = ["CoalescingEmbeddingModel"]

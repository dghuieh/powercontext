#!/usr/bin/env python3
"""PowerContext 读写链路自检：验证 800513 应用依赖的两个 REST 端点可用。

两种模式
--------
1) 真实服务模式（默认）—— 连一台已启动的 PowerContext
       export PC_BASE_URL=http://localhost:8848
       export PC_API_KEY=changeme
       export PC_USER_ID=dify-player-1
       python scripts/smoke_test.py

2) 离线契约模式（--mock）—— 零依赖、不起服务，用与服务端同构的桩响应
   验证本脚本的判定逻辑（重点覆盖 data 为数组的返回形态）
       python scripts/smoke_test.py --mock

契约适配（按服务端实际返回）
--------------------------
  * POST /api/v1/memories         -> 服务端 data 为 **数组**（形如 [{memory_id, ...}]）
                                    本脚本同时兼容 data 为对象
  * POST /api/v1/memories/search  -> data 为 {results|items|memories: [...]} 或直接是数组

退出码：0=PASS，1=FAILED。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("PC_BASE_URL", "http://localhost:8848").rstrip("/")
KEY = os.environ.get("PC_API_KEY", "changeme")
USER = os.environ.get("PC_USER_ID", "dify-player-1")

ID_KEYS = ("memory_id", "memoryId", "id", "uuid")
LIST_KEYS = ("results", "items", "memories", "records", "list", "data")
CONTENT_KEYS = ("content", "text", "memory", "value")
SCORE_KEYS = ("score", "similarity", "rerank_score", "distance")

MOCK_MODE = False


# --------------------------------------------------------------------------- 传输层
def _http(method: str, path: str, payload: dict | None, timeout: int = 20):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json", "X-API-Key": KEY},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"HTTP {exc.code} {method} {path}: {detail}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"非 JSON 响应 {method} {path}: {raw[:200]}") from exc


def _mock(method: str, path: str, payload: dict | None, timeout: int = 20):
    """与真实服务端同构的桩：data 为数组，search 结果放在 data.results。"""
    if path == "/api/v1/system/health":
        return {"code": 0, "message": "ok", "data": {"status": "ok", "backend": "contract-mock"}}
    if path == "/api/v1/memories" and method == "POST":
        record = {
            "memory_id": "mem_mock_0001",
            "content": (payload or {}).get("content", ""),
            "user_id": (payload or {}).get("user_id"),
        }
        return {"code": 0, "message": "ok", "data": [record]}  # 数组形态
    if path == "/api/v1/memories/search" and method == "POST":
        return {
            "code": 0,
            "message": "ok",
            "data": {
                "results": [
                    {
                        "memory_id": "mem_mock_0001",
                        "content": "我今天说要每天跑步，从明天开始",
                        "score": 0.9312,
                    }
                ]
            },
        }
    raise RuntimeError(f"mock 未实现 {method} {path}")


def call(method: str, path: str, payload: dict | None = None, timeout: int = 20):
    return _mock(method, path, payload, timeout) if MOCK_MODE else _http(method, path, payload, timeout)


# --------------------------------------------------------------------------- 结构解析
def unwrap(payload):
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _first_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return None


def pick(container, keys, default=None):
    node = _first_dict(container)
    if node is None:
        return default
    for key in keys:
        value = node.get(key)
        if value not in (None, ""):
            return value
    return default


def extract_mem_id(added) -> str:
    for candidate in (unwrap(added), added):
        found = pick(candidate, ID_KEYS)
        if found:
            return str(found)
    return "?"


def extract_results(found) -> list[dict]:
    data = unwrap(found)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in LIST_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if any(k in data for k in CONTENT_KEYS):
            return [data]
    return []


# --------------------------------------------------------------------------- 主流程
def run() -> int:
    label = "离线契约桩" if MOCK_MODE else BASE
    print(f"  target = {label}  user = {USER}")

    print("[1/3] health ", end="", flush=True)
    try:
        call("GET", "/api/v1/system/health")
        print("................ ok")
    except Exception as exc:  # noqa: BLE001
        print(f"................ FAIL ({exc})")
        print("      服务未启动或 PC_BASE_URL 不可达；离线自检可加 --mock")
        return 1

    content = "我今天说要每天跑步，从明天开始"
    print("[2/3] add memory ", end="", flush=True)
    try:
        added = call(
            "POST",
            "/api/v1/memories",
            {
                "content": content,
                "user_id": USER,
                "scope": "user",
                "memory_type": "fact",
                "infer": True,
                "metadata": {"source": "dify-roast-memory"},
            },
        )
        mem_id = extract_mem_id(added)
        print(f"............ ok (id={mem_id})")
    except Exception as exc:  # noqa: BLE001
        print(f"............ FAIL ({exc})")
        return 1

    time.sleep(0.5 if MOCK_MODE else 1)  # 给向量索引一点落盘时间

    print("[3/3] search memory ", end="", flush=True)
    try:
        found = call(
            "POST",
            "/api/v1/memories/search",
            {"query": "我要开始跑步了", "user_id": USER, "limit": 5},
        )
        results = extract_results(found)
        if not results:
            print("......... FAIL (无召回，检查 embedder 是否就绪)")
            print(f"      原始响应: {json.dumps(found, ensure_ascii=False)[:300]}")
            return 1
        top = results[0]
        print(
            f"......... ok (hit={len(results)}, "
            f"top_score={pick(top, SCORE_KEYS, 'n/a')})"
        )
        print(f"      top: {pick(top, CONTENT_KEYS, '<无 content 字段>')}")
    except Exception as exc:  # noqa: BLE001
        print(f"......... FAIL ({exc})")
        return 1

    print("PASS")
    print(f"      写入的记忆 {mem_id} 可被立即检索；跨会话复用时带同一 user_id={USER} 即可召回")
    return 0


def main(argv: list[str]) -> int:
    global MOCK_MODE
    if "--mock" in argv:
        MOCK_MODE = True
    return run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

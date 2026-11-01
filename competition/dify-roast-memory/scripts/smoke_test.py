#!/usr/bin/env python3
"""PowerContext 读写链路自检：验证 800513 应用依赖的两个 REST 端点可用。

用法：
    export PC_BASE_URL=http://localhost:8848
    export PC_API_KEY=changeme
    export PC_USER_ID=dify-player-1
    python scripts/smoke_test.py
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("PC_BASE_URL", "http://localhost:8848").rstrip("/")
KEY = os.environ.get("PC_API_KEY", "changeme")
USER = os.environ.get("PC_USER_ID", "dify-player-1")


def call(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "X-API-Key": KEY},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def main():
    failed = 0

    print("[1/3] health ", end="", flush=True)
    try:
        call("GET", "/api/v1/system/health")
        print("................ ok")
    except Exception as exc:  # noqa: BLE001
        print(f"................ FAIL ({exc})")
        print("      服务未启动或 PC_BASE_URL 不可达")
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
        mem_id = (added.get("data") or {}).get("memory_id", "?")
        print(f"............ ok (id={mem_id})")
    except Exception as exc:  # noqa: BLE001
        print(f"............ FAIL ({exc})")
        return 1

    time.sleep(1)  # 给向量索引一点落盘时间

    print("[3/3] search memory ", end="", flush=True)
    try:
        found = call("POST", "/api/v1/memories/search",
                     {"query": "我要开始跑步了", "user_id": USER, "limit": 5})
        results = ((found.get("data") or {}).get("results") or [])
        if not results:
            print("......... FAIL (无召回，检查 embedder 是否就绪)")
            failed = 1
        else:
            top = results[0]
            print(f"......... ok (hit={len(results)}, top_score={top.get('score')})")
            print(f"      top: {top.get('content')}")
    except Exception as exc:  # noqa: BLE001
        print(f"......... FAIL ({exc})")
        failed = 1

    print("PASS" if not failed else "FAILED")
    return failed


if __name__ == "__main__":
    sys.exit(main())

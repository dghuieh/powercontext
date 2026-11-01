#!/usr/bin/env python3
"""高德开放平台 Web 服务最小客户端（仅标准库，无第三方依赖）。

用法:
    export AMAP_KEY=<你的高德Web服务Key>          # Windows: set AMAP_KEY=xxx
    python gaode_client.py geocode "西湖" 杭州
    python gaode_client.py around 120.153576 30.287459 咖啡 1500 1
    python gaode_client.py text "杭州 亲子餐厅"
    python gaode_client.py walking 120.15,30.28 120.16,30.29

设计原则:
- 只透传高德真实字段，不做任何估算或补全，缺失即为空；
- 所有 POI 输出统一带 poi_id，便于上层做反幻觉校验；
- 接口报错原样返回 infocode/info，由调用方如实告知用户。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

BASE = "https://restapi.amap.com"
TIMEOUT = 12


def _key() -> str:
    key = os.environ.get("AMAP_KEY", "").strip()
    if not key:
        sys.exit(json.dumps({
            "error": "AMAP_KEY_MISSING",
            "message": "未配置环境变量 AMAP_KEY，无法访问高德开放平台，禁止编造地点数据。",
        }, ensure_ascii=False))
    return key


def _get(path: str, **params) -> dict:
    params = {k: v for k, v in params.items() if v not in (None, "")}
    params["key"] = _key()
    params.setdefault("output", "JSON")
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if str(data.get("status")) != "1":
        data.setdefault("error", "AMAP_API_ERROR")
    return data


def _poi(raw: dict) -> dict:
    """把高德 POI 压成稳定契约，字段缺失一律置空，不做推断。"""
    biz = raw.get("biz_ext") or {}
    if isinstance(biz, list):
        biz = biz[0] if biz else {}
    return {
        "poi_id": raw.get("id", ""),
        "name": raw.get("name", ""),
        "type": raw.get("type", ""),
        "address": raw.get("address", "") if isinstance(raw.get("address"), str) else "",
        "lng_lat": raw.get("location", ""),
        "distance_m": raw.get("distance", ""),
        "tel": raw.get("tel", "") if isinstance(raw.get("tel"), str) else "",
        "city": raw.get("cityname", ""),
        "district": raw.get("adname", ""),
        "rating": (biz.get("rating") or ""),
        "cost": (biz.get("cost") or ""),
        "open_time": (biz.get("open_time") or ""),
        "booking": (biz.get("booking") or ""),
        "tag": raw.get("tag", "") if isinstance(raw.get("tag"), str) else "",
        "photo": (raw.get("photos") or [{}])[0].get("url", "") if isinstance(raw.get("photos"), list) else "",
    }


def geocode(address: str, city: str = "") -> dict:
    data = _get("/v3/geocode/geo", address=address, city=city)
    geos = data.get("geocodes") or []
    if not geos:
        return {"error": "NO_GEOCODE", "query": address, "infocode": data.get("infocode")}
    g = geos[0]
    return {
        "formatted_address": g.get("formatted_address", ""),
        "lng_lat": g.get("location", ""),
        "city": g.get("city", "") if isinstance(g.get("city"), str) else city,
        "district": g.get("district", ""),
        "adcode": g.get("adcode", ""),
    }


def around(lng: float, lat: float, keyword: str, radius: int = 2000, page: int = 1) -> dict:
    data = _get("/v3/place/around", location=f"{lng},{lat}", keywords=keyword,
                radius=radius, offset=25, page=page, extensions="all", sortrule="distance")
    pois = data.get("pois") or []
    return {
        "count": data.get("count", "0"),
        "page": page,
        "keyword": keyword,
        "radius": radius,
        "pois": [_poi(p) for p in pois],
    }


def text(keywords: str, city: str = "", page: int = 1) -> dict:
    data = _get("/v3/place/text", keywords=keywords, city=city, offset=25,
                page=page, extensions="all")
    pois = data.get("pois") or []
    return {"count": data.get("count", "0"), "pois": [_poi(p) for p in pois]}


def walking(origin: str, destination: str) -> dict:
    data = _get("/v3/direction/walking", origin=origin, destination=destination)
    paths = ((data.get("route") or {}).get("paths") or [])
    if not paths:
        return {"error": "NO_ROUTE", "infocode": data.get("infocode")}
    p = paths[0]
    return {"distance_m": int(p.get("distance", 0) or 0),
            "duration_s": int(p.get("duration", 0) or 0)}


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print(json.dumps({"usage": __doc__}, ensure_ascii=False, indent=2))
        return
    cmd, args = argv[1], argv[2:]
    try:
        if cmd == "geocode":
            out = geocode(args[0], args[1] if len(args) > 1 else "")
        elif cmd == "around":
            out = around(float(args[0]), float(args[1]), args[2],
                         int(args[3]) if len(args) > 3 else 2000,
                         int(args[4]) if len(args) > 4 else 1)
        elif cmd == "text":
            out = text(args[0], args[1] if len(args) > 1 else "")
        elif cmd == "walking":
            out = walking(args[0], args[1])
        else:
            out = {"error": "UNKNOWN_COMMAND", "command": cmd}
    except (IndexError, ValueError) as exc:
        out = {"error": "BAD_ARGUMENTS", "detail": str(exc)}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv)

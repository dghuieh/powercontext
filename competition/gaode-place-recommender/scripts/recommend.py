#!/usr/bin/env python3
"""周边去哪儿 · 端到端推荐器。

流程: 锚点地理编码 -> 关键词抽取 -> 周边检索(最多2页) -> 硬过滤 -> 加权打分 -> 输出 Top-N

用法:
    set AMAP_KEY=xxx
    python recommend.py --anchor "西湖" --city 杭州 \
        --intent "下雨天想找个安静的咖啡馆坐一下午" --top 5

所有输出字段均来自高德接口；缺失字段以空串或 "未提供" 呈现，绝不估算。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaode_client import around, geocode  # noqa: E402

# 场景权重（可调）：距离 / 口碑 / 场景 / 价格 / 数据完整度
WEIGHTS = {"distance": 0.30, "rating": 0.25, "scene": 0.25, "cost": 0.10, "complete": 0.10}

SCENE_RULES = {
    "雨天": {
        "keywords": ["咖啡", "书店", "商场", "博物馆", "展览", "影院", "茶馆"],
        "boost_types": ["购物中心", "博物馆", "咖啡厅", "书店", "电影院"],
        "penalty_types": ["公园", "景点", "烧烤", "露营地"],
    },
    "带娃": {
        "keywords": ["亲子", "儿童", "公园", "科技馆", "博物馆"],
        "boost_types": ["公园", "博物馆", "科技馆", "儿童", "亲子"],
        "penalty_types": ["酒吧", "夜店", "网吧", "KTV"],
    },
    "聚餐": {
        "keywords": ["餐厅", "火锅", "烤肉"],
        "boost_types": ["餐厅", "中餐厅", "火锅"],
        "penalty_types": ["便利店", "快照", "小吃"],
    },
    "安静": {
        "keywords": ["咖啡", "书店", "茶馆"],
        "boost_types": ["咖啡厅", "书店", "茶馆"],
        "penalty_types": ["KTV", "酒吧", "电玩", "夜市"],
    },
    "夜间": {
        "keywords": ["深夜食堂", "酒吧", "夜宵"],
        "boost_types": ["餐厅", "酒吧"],
        "penalty_types": [],
    },
    "预算": {
        "keywords": ["快餐", "小吃", "食堂", "平价"],
        "boost_types": ["快餐", "小吃", "中餐"],
        "penalty_types": ["西餐", "酒店", "会所"],
    },
}

TYPE_HINT = {
    "餐饮": ["餐厅", "火锅", "小吃", "快餐", "烤肉", "咖啡", "茶"],
    "购物": ["商场", "购物中心", "超市", "便利店"],
    "文体": ["博物馆", "科技馆", "书店", "图书馆", "展览", "影院"],
    "休闲": ["公园", "景点", "游乐园", "茶馆", "酒吧"],
}


def parse_intent(intent: str) -> dict:
    """从自由文本抽取关键词与场景标签，纯规则实现，避免引入模型依赖。"""
    scenes = [tag for tag in SCENE_RULES if tag in intent]
    found = re.findall(r"想(?:去|找|吃|喝)?([\u4e00-\u9fa5]{2,6})", intent)
    keywords = list(dict.fromkeys(found))
    for tag in scenes:
        for kw in SCENE_RULES[tag]["keywords"]:
            if kw in intent:
                keywords.append(kw)
    if not keywords:
        for hint, words in TYPE_HINT.items():
            if hint in intent:
                keywords.extend(words[:2])
    keywords = list(dict.fromkeys(k for k in keywords if len(k) >= 2))
    radius = 2000
    m = re.search(r"(\d+(?:\.\d+)?)\s*(公里|km|千米)", intent)
    if m:
        radius = int(float(m.group(1)) * 1000)
    else:
        m = re.search(r"(\d+)\s*米", intent)
        if m:
            radius = int(m.group(1))
    budget = None
    m = re.search(r"(\d+)\s*(?:元|块)\s*(?:以内|以下|左右)?", intent)
    if m:
        budget = int(m.group(1))
    return {"scenes": scenes, "keywords": keywords[:4] or ["餐厅"], "radius": radius, "budget": budget}


def _open_state(open_time: str, now: dt.datetime) -> tuple[bool, str]:
    """open_time 形如 '09:00-22:00' / '10:00-22:00,10:00-22:00'。无法解析时返回未知。"""
    if not open_time:
        return True, "营业时间未提供"
    spans = re.findall(r"(\d{1,2}:\d{2})-(\d{1,2}:\d{2})", open_time)
    if not spans:
        return True, f"营业时间 {open_time}"
    cur = now.hour * 60 + now.minute
    for start, end in spans:
        sh, sm = map(int, start.split(":"))
        eh, em = map(int, end.split(":"))
        s, e = sh * 60 + sm, eh * 60 + em
        if e <= s:  # 跨零点
            if cur >= s or cur <= e:
                return True, f"营业至次日 {end}"
        elif s <= cur <= e:
            return True, f"营业至 {end}"
    return False, f"营业时间 {open_time}（当前已打烊）"


def score(poi: dict, req: dict, now: dt.datetime) -> dict:
    detail = {}
    dist = poi.get("distance_m")
    dist = int(dist) if str(dist).isdigit() else None
    detail["distance"] = 1.0 if dist is None else max(0.0, 1 - dist / max(req["radius"], 1))

    try:
        rating = float(poi.get("rating") or 0)
    except ValueError:
        rating = 0.0
    detail["rating"] = min(rating / 5.0, 1.0)

    ptype = f"{poi.get('type','')} {poi.get('tag','')} {poi.get('name','')}"
    scene_hit = 0.5
    for tag in req["scenes"]:
        rule = SCENE_RULES[tag]
        if any(t in ptype for t in rule["boost_types"]):
            scene_hit = 1.0
        if any(t in ptype for t in rule["penalty_types"]):
            scene_hit = 0.05
    if any(k in ptype for k in req["keywords"]):
        scene_hit = min(1.0, scene_hit + 0.15)
    detail["scene"] = scene_hit

    try:
        cost = float(poi.get("cost") or 0)
    except ValueError:
        cost = 0.0
    if req.get("budget"):
        detail["cost"] = 1.0 if 0 < cost <= req["budget"] else (0.4 if cost == 0 else 0.1)
    else:
        detail["cost"] = 0.6 if cost == 0 else 0.8

    fields = ("rating", "cost", "open_time", "tel", "address")
    detail["complete"] = sum(1 for f in fields if str(poi.get(f) or "").strip()) / len(fields)

    _, open_note = _open_state(poi.get("open_time", ""), now)
    total = sum(WEIGHTS[k] * v for k, v in detail.items())
    return {"total": round(total, 4), "detail": {k: round(v, 3) for k, v in detail.items()},
            "open_note": open_note}


def recommend(anchor: str, city: str, intent: str, top: int) -> dict:
    geo = geocode(anchor, city)
    if geo.get("error"):
        return {"error": "ANCHOR_NOT_FOUND", "anchor": anchor, "detail": geo}
    lng, lat = geo["lng_lat"].split(",")
    req = parse_intent(intent)
    now = dt.datetime.now()

    seen, rows = set(), []
    for kw in req["keywords"]:
        for page in (1, 2):
            res = around(float(lng), float(lat), kw, req["radius"], page)
            if res.get("error"):
                return res
            if not res["pois"]:
                break
            for poi in res["pois"]:
                pid = poi.get("poi_id")
                if not pid or pid in seen:  # 反幻觉：无 POI ID 丢弃
                    continue
                seen.add(pid)
                sc = score(poi, req, now)
                if not sc["open_note"].endswith("（当前已打烊）"):
                    rows.append({**poi, "score": sc["total"],
                                 "score_detail": sc["detail"], "open_note": sc["open_note"]})

    rows.sort(key=lambda r: r["score"], reverse=True)
    for r in rows:
        dist = r.get("distance_m")
        r["walk_min"] = (round(int(dist) / 80) if str(dist).isdigit() else "")
    return {"anchor": geo, "intent": intent, "parsed": req,
            "candidates": len(rows), "top": rows[:top]}


def main() -> None:
    ap = argparse.ArgumentParser(description="周边去哪儿 · 高德地点推荐")
    ap.add_argument("--anchor", required=True, help="锚点地名或 'lng,lat'")
    ap.add_argument("--city", default="", help="城市名，提升地理编码命中率")
    ap.add_argument("--intent", required=True, help="用户原话")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    print(json.dumps(recommend(args.anchor, args.city, args.intent, args.top),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

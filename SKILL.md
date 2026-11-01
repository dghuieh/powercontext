---
name: gaode-place-recommender
description: 基于高德开放平台真实 POI 数据的「周边去哪儿」本地生活推荐 Skill。当用户询问附近有什么好吃的、好玩的、下雨天能去哪、带孩子去哪、夜里还能去哪等周边场景推荐，或在给定地点周边挑选餐厅/咖啡/景点/亲子场所时使用。所有候选地点均来自高德 Web 服务实时检索，禁止臆造 POI。
---

# 周边去哪儿 · 基于高德的地点推荐助手

> 赛题 800511 / questionId 600074。实现说明与脚本见 `competition/gaode-place-recommender/`。

## 使用前提

需要环境变量 `AMAP_KEY`（高德开放平台 Web 服务 Key）。缺失时**必须**明确告知
"未配置 AMAP_KEY，无法获取真实地点数据"，**禁止**凭记忆编造地点名称、地址、评分。

## 工作流（严格按序）

1. **锚点解析**：把"我在 XXX / 从 XXX 出发"转成经纬度 + 城市。
   有明确地名 → `python competition/gaode-place-recommender/scripts/gaode_client.py geocode "<地名>" "<城市>"`；
   锚点含糊（"公司楼下""宿舍附近"）→ 先追问一句，或用 IP 定位兜底。
2. **意图结构化**：抽取关键词、半径（步行 1000m / 默认 2000m / 开车 5000m）、预算、场景约束。
3. **周边检索**：`gaode_client.py around <lng> <lat> <关键词> [半径] [页数]`，每页 25 条，最多 2 页。
4. **筛选打分**：`scripts/recommend.py --anchor <地名> --city <城市> --intent "<用户原话>" --top 5`

   | 维度 | 权重 | 数据来源 |
   |---|---|---|
   | 距离可达性 | 30% | 高德 `distance` + 步行耗时 |
   | 口碑评分 | 25% | 高德 `rating` |
   | 场景匹配 | 25% | 关键词 / `type` / `tag` |
   | 价格适配 | 10% | `cost` 是否在预算内 |
   | 数据完整度 | 10% | 是否有营业时间、电话、评分 |

5. **营业状态校验**：用 `biz_ext.open_time` 判断此刻是否营业，已打烊必须标注且不推荐。
6. **呈现**：Top 3~5，每条给 `名称 / 分类 / 距你 X 米（步行 Y 分钟）/ 评分 / 人均 / 营业至 / 一句话理由 / 高德 POI ID`。
   理由必须由真实字段推出（"评分 4.6、距你 380m、营业到 23:00"），不写无依据的形容词。

## 场景规则

- **带娃**：优先 公园 / 博物馆 / 科技馆 / 亲子餐厅，过滤 bar、夜店、网吧。
- **雨天**：优先室内（商场 / 博物馆 / 书店 / 影院），理由中给出"室内"依据。
- **聚餐**：优先有 `cost` 的餐厅，`biz_ext.booking` 为真时标注"可订座"。
- **夜间**：必须校验 `open_time` 覆盖当前时刻。
- 多备选对比时用表格，不写长段落。

## 反幻觉硬约束

- 任何地点必须带高德 `poi_id`，无 `poi_id` 一律不输出。
- 接口返回空或超额时，如实说明"未返回结果 / 超出配额"并建议扩大半径或换词，**不得**用记忆里的老店名填充。
- 距离、评分、人均必须来自接口字段；接口未给写"未提供"，不估算。

## 运行示例

```bash
set AMAP_KEY=你的Key
python competition/gaode-place-recommender/scripts/recommend.py \
    --anchor "西湖" --city 杭州 --intent "下雨天想找个安静的咖啡馆坐一下午" --top 5
```

输出为 JSON，字段固定为
`poi_id / name / type / address / lng_lat / distance_m / rating / cost / open_time / tag`
外加 `score / score_detail / open_note / walk_min`。

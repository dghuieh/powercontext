# 周边去哪儿 · 基于高德的地点推荐助手

> 赛题：800511 / 600074《周边去哪儿：基于高德的地点推荐助手》（20 分）
> 交付物：Skill 文件（SKILL.md）+ 可运行脚本，随队伍仓库 `dghuieh/powercontext` 提交。

## 一、题目要求与实现对应

| 题面要求 | 本实现 |
|---|---|
| 基于高德地点数据的推荐助手 | 全部候选来自高德开放平台 Web 服务（地理编码 / 周边搜索 / 关键字搜索 / 步行路径），不使用任何本地静态或虚构地点库 |
| Skill 形态（SKILL.md） | `SKILL.md` 定义触发条件、工作流、场景规则、反幻觉硬约束与输出格式 |
| 数据真实性 | POI 记录强制携带高德 POI ID；距离、评分、人均、营业时间逐字段透传，缺失即写"未提供"，脚本层禁止补全 |
| 需求满足 | 意图解析覆盖 雨天 / 带娃 / 聚餐 / 安静 / 夜间 / 预算 六类场景，自动抽取半径与预算 |
| 推荐合理性 | 五维加权打分（距离 30% / 口碑 25% / 场景 25% / 价格 10% / 数据完整度 10%），并做营业状态硬过滤 |
| 表达清晰度 | 输出 Top-N 表格，每条给距离、步行分钟、评分、人均、打烊时间与可核验的理由字段 |

## 二、目录结构

```
competition/gaode-place-recommender/
├── SKILL.md                  # Skill 主文件（题面要求的交付物）
├── README.md                 # 本文件：实现说明与验证方法
└── scripts/
    ├── gaode_client.py       # 高德 Web 服务最小客户端（仅标准库）
    └── recommend.py          # 端到端推荐器：解析意图 → 检索 → 过滤 → 打分 → 排序
```

## 三、运行方法

```bash
# 1) 配置高德开放平台 Web 服务 Key（https://console.amap.com 申请，类型选「Web服务」）
set AMAP_KEY=你的Key            # macOS/Linux: export AMAP_KEY=你的Key

# 2) 单独调用接口
python scripts/gaode_client.py geocode "西湖" 杭州
python scripts/gaode_client.py around 120.153576 30.287459 咖啡 1500 1

# 3) 端到端推荐
python scripts/recommend.py --anchor "西湖" --city 杭州 \
    --intent "下雨天想找个安静的咖啡馆坐一下午" --top 5
```

输出为 JSON，字段固定为
`poi_id / name / type / address / lng_lat / distance_m / rating / cost / open_time / tag`，
外加 `score`、`score_detail`、`open_note`、`walk_min`。

## 四、验证样例

- **缺 Key 降级**：未设置 `AMAP_KEY` 时，脚本返回
  `{"error": "AMAP_KEY_MISSING", ...}`，不输出任何地点，符合"不得编造 POI"约束。
- **接口异常降级**：高德返回 `infocode != 10000` 时原样透出错误码，由 Skill 指引用户扩大半径或更换关键词。
- **反幻觉校验**：`collide` 阶段对无 `poi_id` 的记录直接丢弃（脚本中 `if not pid or pid in seen: continue`）。
- **打烊过滤**：`open_time` 无法覆盖当前时刻的 POI 不进入推荐结果，仅在必要时提示"已打烊"。

> 说明：本机未配置高德 Key，未附带实测截图/日志，避免使用伪造数据；评审如需实测，
> 配置任意有效的 Web 服务 Key 即可复现上述结果。

## 五、与技术栈的结合点

- 推荐结果的检索上下文可直接写入 PowerContext 的记忆层（`memory.add`），实现
  "上次带娃去哪、口味偏好"的跨会话复用；
- 若接入 seekdb 向量检索，可把 POI 名称 + type + tag 做 embedding，用语义召回替代纯关键词召回，
  在 `parse_intent` 抽不出明确关键词时兜底。

# 损友记忆交易所（Dify × PowerContext × seekdb）

一个跑在 Dify 里的"赛博损友"：你把今天的小事说给它，它把这句话写进 PowerContext 长期记忆（底层 seekdb 向量检索）；下一次你再开口，它先把旧账翻出来损你，再给一句真心建议。

> 赛题：800513 · 让 Agent 拥有长期记忆（Dify 应用）· 20 分
> 交付物：Dify DSL + 运行说明（本文件）+ 验证样例（`examples/`、`scripts/smoke_test.py`）

## 1. 为什么好玩

普通聊天机器人每次都是陌生人；这个应用会因为"记得"而变得讨厌又贴心。

- **无记忆 → 无笑点**：第 1 轮它只能顺着你说；第 3 轮它能精准引用你 2 天前说的反话。
- **笑点可复现**：同一句"我要开始早睡了"，在不同记忆状态下得到完全不同的吐槽，直观证明记忆真的生效。
- **不会瞎记**：只从检索结果里引用，检索为空就坦白"这是我第一次见你"。

## 2. 架构

```
用户输入
   │
   ▼
[Dify Chatflow]
   ├─ 1. 开始节点        取 query
   ├─ 2. HTTP 请求         POST {PC_BASE_URL}/api/v1/memories/search   → 语义召回 Top5 旧记忆
   ├─ 3. LLM 节点          Ollama qwen2.5:7b：对比"今天"与"旧账"，生成 120 字内吐槽 + 建议
   ├─ 4. HTTP 请求         POST {PC_BASE_URL}/api/v1/memories         → 把今天这句写入长期记忆
   └─ 5. 回答节点         输出损友回复
                              │
                              ▼
                  PowerContext（HTTP API :8848）
                              │
                              ▼
                  seekdb —— 向量 + 关系混合存储
```

读写顺序刻意设计为 **先检索、后写入**：保证本次回答不会把自己刚写进去的句子当成"旧账"引用，避免自循环。

## 3. 运行前提

| 组件 | 版本/说明 | 获取方式 |
| --- | --- | --- |
| Dify | ≥ 0.15（需 `advanced-chat` / Chatflow 与 HTTP 请求节点） | `git clone https://github.com/langgenius/dify` 后 `docker compose up -d` |
| PowerContext | 本仓库 `powermem` 服务 | 根目录 `docker compose up -d`，默认暴露 `8848` |
| 向量库 | seekdb（PowerContext 默认后端） | 随 PowerContext compose 启动，无需单独装 |
| 模型 | Ollama `qwen2.5:7b` | `ollama pull qwen2.5:7b` |

> 若 Dify 与 PowerContext 不在同一台机器，把环境变量 `PC_BASE_URL` 改为 PowerContext 的可达地址即可；同机 Docker 场景用 `http://host.docker.internal:8848`（Linux 需在 dify 的 compose 中加 `extra_hosts: ["host.docker.internal:host-gateway"]`）。

## 4. 导入与应用配置

1. 启动 PowerContext，确认健康：
   ```bash
   curl http://localhost:8848/api/v1/system/health
   # {"status":"ok",...}
   ```
2. 记下 PowerContext 的 `API_KEY`（即其 `.env` 中的值，默认示例为 `changeme`）。
3. Dify 控制台 → **创建空白应用 → 导入 DSL 文件** → 选择本目录 `app.dsl.yml`。
4. 打开应用 → 右上角「环境变量」，按实际修改三项：

   | 变量 | 含义 | 示例 |
   | --- | --- | --- |
   | `PC_BASE_URL` | PowerContext 地址 | `http://host.docker.internal:8848` |
   | `PC_API_KEY` | 请求头 `X-API-Key` | `changeme` |
   | `PC_USER_ID` | 记忆归属用户，决定"记得谁" | `dify-player-1` |

5. 在「模型」处确认 `qwen2.5:7b` 已在 Ollama 中拉取且 Dify 的 Ollama 插件已安装；节点模型名不同则在此处改选。
6. 发布应用，进入对话。

## 5. 验证样例

完整脚本见 `scripts/smoke_test.py`（不依赖 Dify，直接验证 PowerContext 两段读写链路）。

### 5.1 接口层自检

```bash
export PC_BASE_URL=http://localhost:8848
export PC_API_KEY=changeme
export PC_USER_ID=dify-player-1
python scripts/smoke_test.py
```

期望输出：

```
[1/3] health ................ ok
[2/3] add memory ............ ok (id=12)
[3/3] search memory ......... ok (hit=1, top_score=0.xx)
      top: 我今天说要每天跑步
PASS
```

### 5.2 三轮对话剧本（`examples/dialogue_playthrough.md`）

| 轮次 | 用户输入 | 期望行为 | 判据 |
| --- | --- | --- | --- |
| 1 | 我今天说要每天跑步，从明天开始 | 检索为空 → 只吐槽今天这句，不假装记得 | 回复不含"上次/以前"等虚构引用 |
| 2 | 帮我想想明天几点起床 | 与跑步无关 → 可召回也可不召回，不得编造旧事 | — |
| 3 | 我今天躺了一整天 | 必须引用第 1 轮"每天跑步"，形成反差 | 回复出现"跑步"或等价指代 |

把第 3 轮回复贴到 PR 说明里即为"记忆生效"的证据。

### 5.3 常见失败排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| HTTP 节点 401 | `PC_API_KEY` 与 PowerContext 不一致 | 改环境变量后重新发布 |
| HTTP 节点 502 / 连接被拒 | Dify 容器访问不到 `host.docker.internal` | 加 `extra_hosts` 或用宿主机 IP |
| 第 3 轮仍不记得 | `PC_USER_ID` 变了，或第 1 轮写入失败 | 查 Dify 节点日志里 `save_memory` 的响应 |
| 吐槽引用了不存在的事 | 模型幻觉 | 系统提示已强制"只引用检索结果"；仍出现则降低 temperature 至 0.5 |

## 6. 与 seekdb / PowerContext 的结合点

- **存储层**：每条记忆经 PowerContext 落到 seekdb，向量索引承担语义召回，`metadata/filters` 承担结构化过滤，无需自建库。
- **接口层**：只用两个标准 REST 端点（`/memories`、`/memories/search`），Dify 侧零代码，全流程可在画布上审计。
- **可扩展**：把 `filters` 换成 `{"topic":"health"}` 即可做"健康类把柄库"；把 `PC_USER_ID` 透传为 Dify 的 `sys.user_id` 即可支持多用户共享实例。

## 7. 文件清单

```
competition/dify-roast-memory/
├── app.dsl.yml                    # Dify Chatflow DSL（可直接导入）
├── README.md                      # 本文件：运行说明
├── examples/dialogue_playthrough.md  # 三轮验证剧本与判定标准
└── scripts/smoke_test.py          # PowerContext 读写链路自检脚本
```

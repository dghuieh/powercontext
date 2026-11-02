# StepFun × seekdb 端侧迷你知识库

浏览器内完成向量化的迷你知识库：**embedding 100% 在用户浏览器里跑**（Transformers.js + ONNX Runtime Web，优先 WebGPU，自动降级 WASM），向量、原文与元数据写入 **seekdb**，查询时的「向量 + 全文」混合检索由 seekdb 一条 SQL 完成并直接排序返回。

```
浏览器                         本地 Node 服务                seekdb
┌──────────────────────┐      ┌──────────────────────┐     ┌────────────────────┐
│ Transformers.js      │      │ /api/ingest          │     │ kb_xxx 表          │
│ all-MiniLM-L6-v2 INT8│─向量─▶│  只写入，不推理      │────▶│ VECTOR(384) + HNSW │
│ WebGPU ⇄ WASM 降级   │◀─结果─│ /api/search          │◀────│ FULLTEXT 索引      │
└──────────────────────┘      │  混合检索单条 SQL    │     └────────────────────┘
```

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `web/` | 前端单页应用（无打包步骤，原生 ESM），含离线模型目录 `web/models/all-MiniLM-L6-v2/` |
| `web/vendor/transformers.min.js` | Transformers.js 3.0.2 浏览器 bundle |
| `server/index.js` | Express 服务：写入 / 检索 / 静态资源，**不含任何推理逻辑** |
| `server/seekdb-js.js` | seekdb-js 数据访问层：集合建表、向量索引与全文索引、upsert、混合检索 SQL |
| `server/schema.sql` | 表结构与索引定义（可手工建表） |
| `benchmark/quant_report.json` | 量化前后体积、压缩比、向量一致性与延迟实测 |
| `tools/quantize_int8.py` | 由 FP32 ONNX 动态量化出 INT8 模型的脚本（与本仓库 int8 产物一致） |
| `docs/技术说明.md` | 一页技术说明（模型与量化 / 端侧运行 / seekdb 使用 / 优化与数据） |

## 启动步骤

### 1. 启动 seekdb

```bash
docker run -d --name seekdb -p 2881:2881 oceanbase/seekdb:latest
```

或使用本地 `seekdb.exe`：

```bash
seekdb.exe --base-dir <空目录> --port 2881 --nodaemon --parameter memory_limit=6G
```

连接参数通过环境变量覆盖：`SEEKDB_HOST` / `SEEKDB_PORT` / `SEEKDB_USER` / `SEEKDB_PASSWORD` / `SEEKDB_DB`。

### 2. 安装依赖并启动应用

```bash
cd stepfun
npm install
# 浏览器端 Transformers.js bundle（若仓库未带该文件，执行一次；离线环境可从 node_modules 拷贝同版本 dist）
curl -L -o web/vendor/transformers.min.js \
  https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2/dist/transformers.min.js
npm start            # http://127.0.0.1:8787
```

浏览器要求：Chrome / Edge 最新版（WebGPU 默认开启；不支持时自动降级 WASM，界面右上角标签会显示当前实际后端）。

### 3. 使用

1. 在「导入语料」区填入文本或选择 `.txt/.md` 文件 → 点「导入并向量化」；进度条与分阶段耗时实时更新。
2. 在「查询」区输入问题、拖动向量权重 α → 点「检索」，结果列表展示 `score / vec_score / txt_score`。
3. 切换集合名即可隔离不同知识库（增量导入不会串库）。

> 首次运行前若 `web/models/` 为空，先执行模型下载与量化（约 1 分钟）：
> ```bash
> python tools/fetch_model.py --out web/models/all-MiniLM-L6-v2
> python tools/quantize_int8.py --src web/models/all-MiniLM-L6-v2/onnx/model.onnx \
>     --dst web/models/all-MiniLM-L6-v2/onnx/model_quantized.onnx \
>     --tokenizer web/models/all-MiniLM-L6-v2/tokenizer.json \
>     --report benchmark/quant_report.json
> ```
> 下载源为魔搭镜像（国内可达），量化产物即 `benchmark/quant_report.json` 中记录的 INT8 权重。

## 模型获取与量化

- 模型：`sentence-transformers/all-MiniLM-L6-v2`（Apache-2.0），384 维，仓库内 `web/models/all-MiniLM-L6-v2/` 已带 tokenizer 与 INT8 ONNX。
- 原始 FP32 ONNX：`onnx/model.onnx`（86.2 MB），从魔搭镜像 `AI-ModelScope/all-MiniLM-L6-v2` 获取。
- 量化：`python tools/quantize_int8.py`（onnxruntime `quantize_dynamic`，`QuantType.QUInt8`）→ `onnx/model_quantized.onnx`（21.8 MB）。

## 实测数据（详见 `docs/技术说明.md`）

| 配置 | 模型体积 | 压缩比 | 与 FP32 向量余弦均值 | Embedding 延迟（CPU 4 线程参考） |
| --- | --- | --- | --- | --- |
| FP32 | 86.2 MB | 1.00× | 1.0000 | 27.2 ms / 段 |
| INT8 | 21.8 MB | 3.95× | 0.9888 | 27.4 ms / 段（首次含预热） |

## 验证与已知限制

- 本机 Windows 上 `seekdb.exe` 因缺少运行时 DLL 无法启动（`STATUS_0xC0000135`），故本次提交的检索链路按 docker 方式复现；服务在 seekdb 不可用时会返回 503 并提示启动方式（`/api/health` 可查状态）。
- 浏览器端 WebGPU/WASM 的绝对延迟取决于评测机显卡与驱动，说明文档给出测量口径与分阶段计时项，应用内已内置计时表。

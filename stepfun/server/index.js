/**
 * StepFun × seekdb 迷你知识库 —— 后端
 *
 * 职责边界（严格遵守赛题要求）：
 *  - 向量化只发生在浏览器（本服务不加载任何 embedding 模型，也不转发到云端 API）；
 *  - 本服务只做三件事：接收浏览器算好的向量 → 写入 seekdb；接收查询向量 → 在 seekdb 内做混合检索；提供静态前端。
 */
const express = require('express');
const path = require('path');
const { SeekDB } = require('./seekdb-js');

const app = express();
app.use(express.json({ limit: '32mb' }));

const db = new SeekDB();
const DEFAULT_COLLECTION = process.env.SEEKDB_COLLECTION || 'kb_default';

let dbReady = false;
(async () => {
  try {
    await db.ping();
    dbReady = true;
    console.log('[seekdb] connected');
  } catch (e) {
    dbReady = false;
    console.error('[seekdb] connect failed:', e.message);
  }
})();

function requireDb(req, res, next) {
  if (!dbReady) {
    return res.status(503).json({
      error: 'seekdb_unavailable',
      hint: '先启动 seekdb 实例（docker 或本地 seekdb.exe），再设置 SEEKDB_HOST/PORT。',
    });
  }
  next();
}

app.get('/api/health', async (req, res) => {
  let ok = false;
  try {
    ok = await db.ping();
    dbReady = ok;
  } catch (_) {}
  res.json({
    seekdb: ok ? 'up' : 'down',
    backendOfInference: 'browser-only',
    note: '本服务不执行任何 embedding 推理',
  });
});

app.post('/api/collections', requireDb, async (req, res) => {
  const { name = DEFAULT_COLLECTION, dim = 384, metric = 'cosine' } = req.body || {};
  try {
    const info = await db.createCollection(name, { dim, metric });
    res.json({ ok: true, ...info });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/collections/:name/stats', requireDb, async (req, res) => {
  try {
    res.json({ name: req.params.name, rows: await db.count(req.params.name) });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

/**
 * 增量导入：浏览器算好的向量按批提交
 * body: { collection, docs: [{doc_id, chunk_id, title, content, meta, embedding}] }
 */
app.post('/api/ingest', requireDb, async (req, res) => {
  const { collection = DEFAULT_COLLECTION, dim = 384, docs = [] } = req.body || {};
  if (!Array.isArray(docs) || !docs.length) {
    return res.status(400).json({ error: 'docs_empty' });
  }
  try {
    const t0 = Date.now();
    const info = await db.createCollection(collection, { dim });
    const affected = await db.upsert(collection, docs);
    res.json({
      ok: true,
      collection,
      index_mode: info.index_mode,
      written: docs.length,
      affected,
      write_ms: Date.now() - t0,
      total_rows: await db.count(collection),
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

/**
 * 混合检索：客户端只提交查询向量 + 原始查询文本，排序完全由 seekdb 完成
 */
app.post('/api/search', requireDb, async (req, res) => {
  const { collection = DEFAULT_COLLECTION, embedding, text = '', topK = 8, alpha = 0.7 } = req.body || {};
  if (!Array.isArray(embedding) || !embedding.length) {
    return res.status(400).json({ error: 'embedding_required' });
  }
  try {
    const out = await db.hybridSearch(collection, { embedding, text, topK, alpha });
    res.json({ ok: true, collection, ...out });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post('/api/reset', requireDb, async (req, res) => {
  try {
    await db.dropCollection(req.body?.collection || DEFAULT_COLLECTION);
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.use(express.static(path.join(__dirname, '..', 'web'), { maxAge: 0 }));

const port = Number(process.env.PORT || 8787);
app.listen(port, () => {
  console.log(`[server] http://127.0.0.1:${port}  embedding 后端：浏览器端 WebGPU/WASM`);
});

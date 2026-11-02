/**
 * 端侧迷你知识库前端
 *
 * 关键点（对应赛题验收项）：
 *  - 文档向量化与查询向量化 100% 在浏览器内完成（Transformers.js + ONNX Runtime Web），不调用任何云端推理 API；
 *  - 优先请求 WebGPU，不可用时自动降级 WASM，并在界面明确提示当前后端；
 *  - 向量、原文、元数据全部写入 seekdb，混合检索（向量 + 全文）由 seekdb 的 SQL 完成，前端只渲染结果。
 */
import { pipeline, env } from './vendor/transformers.min.js';

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = '/models/';
env.backends.onnx.wasm.wasmPaths = '/vendor/';

const MODEL_ID = 'all-MiniLM-L6-v2';
const DIM = 384;
const BATCH = 8;

const $ = (id) => document.getElementById(id);
const bad = (el, text) => { el.className = 'badge badge-warn'; el.textContent = text; };
const okBadge = (el, text) => { el.className = 'badge badge-ok'; el.textContent = text; };

let extractor = null;
let backendName = '未加载';
const timings = {};

function stage(name, ms) {
  if (ms == null) return;
  timings[name] = (timings[name] || []).concat(ms);
  renderTimings();
}

function median(arr) {
  const a = [...arr].sort((x, y) => x - y);
  return a[Math.floor(a.length / 2)];
}

function renderTimings() {
  const tbody = document.querySelector('#timing tbody');
  tbody.innerHTML = Object.entries(timings)
    .map(([k, v]) => `<tr><td>${k}</td><td>${median(v).toFixed(1)} ms（n=${v.length}）</td></tr>`)
    .join('');
}

function setProgress(barId, ratio) {
  $(barId).style.width = `${Math.round(ratio * 100)}%`;
}

async function probeDb() {
  const el = $('badge-db');
  try {
    const r = await fetch('/api/health').then((x) => x.json());
    if (r.seekdb === 'up') okBadge(el, 'seekdb：已连接');
    else bad(el, 'seekdb：未连接（先启动实例）');
  } catch (e) {
    bad(el, 'seekdb：后端不可用');
  }
}

async function loadModel() {
  const badge = $('badge-backend');
  const t0 = performance.now();
  const hasGPU = !!(navigator.gpu && navigator.gpu.requestAdapter);
  const tryInit = async (device) => pipeline('feature-extraction', MODEL_ID, { device, dtype: 'q8' });

  if (hasGPU) {
    try {
      extractor = await tryInit('webgpu');
      backendName = 'WebGPU（INT8）';
    } catch (e) {
      console.warn('WebGPU 初始化失败，降级 WASM：', e);
    }
  }
  if (!extractor) {
    extractor = await tryInit('wasm');
    backendName = 'WASM（INT8）';
  }
  okBadge(badge, `推理后端：${backendName}`);
  okBadge($('badge-model'), `模型：${MODEL_ID} · INT8`);
  stage('模型下载/加载', performance.now() - t0);
}

async function embed(texts, barId) {
  const out = [];
  for (let i = 0; i < texts.length; i += BATCH) {
    const chunk = texts.slice(i, i + BATCH);
    const t0 = performance.now();
    const res = await extractor(chunk, { pooling: 'mean', normalize: true });
    out.push(...res.tolist());
    stage(i === 0 ? '文档向量化（含预热）' : '文档向量化（预热后）', performance.now() - t0);
    setProgress(barId, Math.min(1, (i + chunk.length) / texts.length));
  }
  return out;
}

function splitCorpus(raw) {
  return raw
    .split(/\n{2,}|(?<=\。)\s*|\n/)
    .map((s) => s.trim())
    .filter((s) => s.length >= 8);
}

async function ingest() {
  const btn = $('btn-ingest');
  btn.disabled = true;
  $('log-ingest').textContent = '准备模型…';
  try {
    if (!extractor) await loadModel();
    const pieces = splitCorpus($('corpus').value);
    if (!pieces.length) throw new Error('语料为空');
    $('log-ingest').textContent = `本地向量化 ${pieces.length} 段文本…`;
    const tVec = performance.now();
    const vectors = await embed(pieces, 'bar-ingest');
    const vecMs = performance.now() - tVec;

    const collection = $('collection').value.trim() || 'kb_default';
    const docs = pieces.map((content, i) => ({
      doc_id: `doc-${Date.now()}-${i}`,
      chunk_id: i,
      title: content.slice(0, 30),
      content,
      meta: { source: 'manual', batch: Date.now() },
      embedding: vectors[i],
    }));

    $('log-ingest').textContent = '写入 seekdb…';
    const t0 = performance.now();
    const r = await fetch('/api/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ collection, dim: DIM, docs }),
    }).then((x) => x.json());
    stage('写入 seekdb', performance.now() - t0);
    stage('文档向量化合计（本轮）', vecMs);

    if (r.error) throw new Error(r.error);
    $('log-ingest').textContent =
      `已写入集合 ${collection}：${r.written} 段（索引模式 ${r.index_mode}，库内总行数 ${r.total_rows}）` +
      `\n写入耗时 ${r.write_ms} ms · 当前推理后端 ${backendName}`;
  } catch (e) {
    $('log-ingest').textContent = `导入失败：${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

async function search() {
  const btn = $('btn-search');
  const q = $('query').value.trim();
  if (!q) return;
  btn.disabled = true;
  $('results').innerHTML = '';
  try {
    if (!extractor) await loadModel();
    $('log-query').textContent = '本地生成查询向量…';
    const t0 = performance.now();
    const res = await extractor([q], { pooling: 'mean', normalize: true });
    const vec = res.tolist()[0];
    stage('查询向量化', performance.now() - t0);

    const t1 = performance.now();
    const out = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        collection: $('collection').value.trim() || 'kb_default',
        embedding: vec,
        text: q,
        topK: 8,
        alpha: Number($('alpha').value),
      }),
    }).then((x) => x.json());
    stage('端到端查询（含检索）', performance.now() - t1 + (performance.now() - t0));
    if (out.error) throw new Error(out.error);
    stage('seekdb 混合检索', out.ms);
    setProgress('bar-query', 1);

    $('log-query').textContent =
      `检索模式 ${out.mode} · 库内耗时 ${out.ms} ms · 向量权重 α=${out.alpha} · 后端 ${backendName}`;
    $('results').innerHTML = out.rows
      .map(
        (r) => `<li>
          <div class="head"><span>${r.doc_id}#${r.chunk_id}</span>
          <span>score <b>${r.score}</b> · vec ${r.vec_score} · txt ${r.txt_score}</span></div>
          <div class="content">${escapeHtml(r.content)}</div>
        </li>`
      )
      .join('');
  } catch (e) {
    $('log-query').textContent = `检索失败：${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

$('btn-ingest').addEventListener('click', ingest);
$('btn-search').addEventListener('click', search);
$('alpha').addEventListener('input', (e) => ($('alpha-val').textContent = Number(e.target.value).toFixed(2)));
$('btn-sample').addEventListener('click', () => {
  $('corpus').value = [
    'seekdb 是 OceanBase 推出的嵌入式数据库，支持向量类型与 HNSW 向量索引，可直接在应用进程内运行。',
    'HNSW 通过分层小世界图实现近似最近邻检索，查询复杂度近似 O(log N)。',
    'Transformers.js 基于 ONNX Runtime Web，可在浏览器中执行 embedding 模型，支持 WebGPU 与 WASM 两种后端。',
    'INT8 动态量化把 FP32 权重映射到 8 位整数，模型体积约缩小为原来的四分之一。',
    '混合检索把向量相似度与全文相关性加权求和，能同时覆盖语义相近与关键词精确匹配的查询。',
    '浏览器首次加载模型需要下载权重文件，之后可从 HTTP 缓存读取，建议区分冷启动与预热后延迟。',
  ].join('\n\n');
});
$('files').addEventListener('change', async (e) => {
  const files = [...e.target.files];
  const texts = await Promise.all(files.map((f) => f.text()));
  $('corpus').value = texts.join('\n\n');
});

probeDb();
$('alpha-val').textContent = '0.70';
loadModel().catch((e) => bad($('badge-backend'), `模型加载失败：${e.message}`));

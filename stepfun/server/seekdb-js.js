/**
 * seekdb-js —— 面向 seekdb（MySQL 协议）的极简 JS 数据访问层
 *
 * 设计目标：
 *  1. 向量 + 全文的混合检索由数据库一条 SQL 完成，客户端不做两路结果拼合；
 *  2. 写入、索引刷新、集合（表）生命周期都在本层收敛，应用层不直接写 SQL。
 */
const mysql = require('mysql2/promise');

const DEFAULT_OPTS = {
  host: process.env.SEEKDB_HOST || '127.0.0.1',
  port: Number(process.env.SEEKDB_PORT || 2881),
  user: process.env.SEEKDB_USER || 'root',
  password: process.env.SEEKDB_PASSWORD || '',
  database: process.env.SEEKDB_DB || 'seekdb',
  waitForConnections: true,
  connectionLimit: 8,
  namedPlaceholders: true,
};

class SeekDB {
  constructor(opts = {}) {
    this.opts = { ...DEFAULT_OPTS, ...opts };
    this.pool = mysql.createPool(this.opts);
    this.collections = new Map();
  }

  async ping() {
    const [rows] = await this.pool.query('SELECT 1 AS ok');
    return rows[0].ok === 1;
  }

  quoteIdent(name) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
      throw new Error(`非法集合名: ${name}`);
    }
    return `\`${name}\``;
  }

  /** 建集合（表）：向量列 + 原文 + 元数据 + 向量索引 + 全文索引 */
  async createCollection(name, { dim = 384, metric = 'cosine' } = {}) {
    const t = this.quoteIdent(name);
    await this.pool.query(`
      CREATE TABLE IF NOT EXISTS ${t} (
        doc_id     VARCHAR(128) NOT NULL,
        chunk_id   INT NOT NULL,
        title      VARCHAR(512),
        content    MEDIUMTEXT,
        meta       JSON,
        embedding  VECTOR(${dim}) NOT NULL,
        created_at BIGINT,
        PRIMARY KEY (doc_id, chunk_id)
      )
    `);

    let index_mode = 'vector+fulltext';
    try {
      await this.pool.query(`
        CREATE VECTOR INDEX idx_${name}_vec ON ${t} (embedding) WITH (type=HNSW, distance=${metric})
      `);
    } catch (e) {
      if (!/exist/i.test(e.message)) throw e;
    }
    try {
      await this.pool.query(`CREATE FULLTEXT INDEX idx_${name}_ft ON ${t} (content)`);
    } catch (e) {
      // 部分 seekdb 版本未开启全文索引能力，退化到 DB 内 LIKE 打分（仍在库内完成，不回客户端拼合）
      index_mode = 'vector+like';
    }
    this.collections.set(name, { dim, metric, index_mode });
    return { name, dim, metric, index_mode };
  }

  /** 批量写入；同一 (doc_id, chunk_id) 覆盖写，避免增量导入产生重复向量 */
  async upsert(name, rows) {
    if (!rows.length) return 0;
    const t = this.quoteIdent(name);
    const values = rows.map((r) => [
      r.doc_id,
      r.chunk_id,
      r.title || '',
      r.content,
      JSON.stringify(r.meta || {}),
      JSON.stringify(r.embedding),
      r.created_at || Date.now(),
    ]);
    const [res] = await this.pool.query(
      `INSERT INTO ${t} (doc_id, chunk_id, title, content, meta, embedding, created_at)
       VALUES ? ON DUPLICATE KEY UPDATE title=VALUES(title), content=VALUES(content),
       meta=VALUES(meta), embedding=VALUES(embedding), created_at=VALUES(created_at)`,
      [values]
    );
    return res.affectedRows;
  }

  /** 混合检索：向量相似度与文本相关性在同一条 SQL 内加权排序 */
  async hybridSearch(name, { embedding, text, topK = 8, alpha = 0.7 }) {
    const t = this.quoteIdent(name);
    const info = this.collections.get(name) || {};
    const mode = info.index_mode || 'vector+fulltext';
    const qvec = JSON.stringify(embedding);
    const safeText = String(text || '').replace(/'/g, "''").slice(0, 512);
    const wVec = Number(alpha).toFixed(2);
    const wTxt = (1 - Number(alpha)).toFixed(2);

    const textScore =
      mode === 'vector+fulltext'
        ? `MATCH(content) AGAINST ('${safeText}')`
        : `(CASE WHEN content LIKE '%${safeText}%' THEN 1 ELSE 0 END)`;

    const sql = `
      SELECT doc_id, chunk_id, title, content, meta,
             ROUND(1 - COSINE_DISTANCE(embedding, '${qvec}'), 6) AS vec_score,
             ROUND(${textScore}, 6) AS txt_score,
             ROUND(${wVec} * (1 - COSINE_DISTANCE(embedding, '${qvec}')) + ${wTxt} * ${textScore}, 6) AS score
      FROM ${t}
      ORDER BY score DESC
      LIMIT ${Number(topK)}
    `;
    const t0 = process.hrtime.bigint();
    const [rows] = await this.pool.query(sql);
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    return {
      rows,
      ms: Math.round(ms * 10) / 10,
      mode,
      alpha: Number(alpha),
      sql_mode: mode,
    };
  }

  async count(name) {
    const t = this.quoteIdent(name);
    const [rows] = await this.pool.query(`SELECT COUNT(*) AS c FROM ${t}`);
    return rows[0].c;
  }

  async dropCollection(name) {
    await this.pool.query(`DROP TABLE IF EXISTS ${this.quoteIdent(name)}`);
    this.collections.delete(name);
  }

  async close() {
    await this.pool.end();
  }
}

module.exports = { SeekDB };

-- seekdb 集合（表）结构与索引：与 server/seekdb-js.js 中 createCollection() 保持一致
CREATE TABLE IF NOT EXISTS kb_default (
  doc_id     VARCHAR(128) NOT NULL,
  chunk_id   INT NOT NULL,
  title      VARCHAR(512),
  content    MEDIUMTEXT,
  meta       JSON,
  embedding  VECTOR(384) NOT NULL,
  created_at BIGINT,
  PRIMARY KEY (doc_id, chunk_id)
);

CREATE VECTOR INDEX idx_kb_default_vec ON kb_default (embedding) WITH (type=HNSW, distance=cosine);
CREATE FULLTEXT INDEX idx_kb_default_ft ON kb_default (content);

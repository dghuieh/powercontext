"""由 FP32 ONNX 动态量化出 INT8 模型，并输出体积/压缩比/向量一致性报告。

用法:
    python tools/quantize_int8.py --src web/models/all-MiniLM-L6-v2/onnx/model.onnx \
                                 --dst web/models/all-MiniLM-L6-v2/onnx/model_quantized.onnx \
                                 --tokenizer web/models/all-MiniLM-L6-v2/tokenizer.json \
                                 --report benchmark/quant_report.json
"""
import argparse, json, os, time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def embed(session, tok, texts, max_len=256):
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=max_len, pad_id=0, pad_token="[PAD]")
    encs = tok.encode_batch(texts)
    ids = np.array([e.ids for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    feeds = {}
    for inp in session.get_inputs():
        if inp.name == "input_ids":
            feeds[inp.name] = ids
        elif inp.name == "attention_mask":
            feeds[inp.name] = mask
        elif inp.name == "token_type_ids":
            feeds[inp.name] = np.zeros_like(ids, dtype=np.int64)
    out = session.run(None, feeds)[0]
    if out.ndim == 3:
        out = out[:, 0, :]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--report", default="benchmark/quant_report.json")
    args = ap.parse_args()

    report = {}
    if not os.path.exists(args.dst):
        t0 = time.time()
        quantize_dynamic(args.src, args.dst, weight_type=QuantType.QUInt8)
        report["quantize_seconds"] = round(time.time() - t0, 1)

    fp32_size, int8_size = os.path.getsize(args.src), os.path.getsize(args.dst)
    report.update(
        fp32_mb=round(fp32_size / 1048576, 1),
        int8_mb=round(int8_size / 1048576, 1),
        compression_ratio=round(fp32_size / int8_size, 2),
    )

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    docs = [
        "OceanBase is a distributed relational database that supports HTAP workloads.",
        "seekdb 是一个可以嵌入到应用中的轻量数据库，支持向量检索。",
        "Transformer models convert text into dense vector embeddings.",
        "浏览器端推理需要在 WebGPU 或 WASM 后端执行 ONNX 模型。",
        "The weather in Changsha is rainy and humid in spring.",
        "向量数据库使用 HNSW 索引加速近似最近邻搜索。",
        "Kubernetes schedules containers across a cluster of machines.",
        "量化可以把 FP32 权重压缩成 INT8，减少模型体积与内存占用。",
    ]
    queries = [
        "什么是向量数据库的索引方式",
        "How does a relational database store data?",
        "端侧推理的后端选择",
        "container orchestration",
    ]

    def sess(path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    res = {}
    for tag, path in (("fp32", args.src), ("int8", args.dst)):
        s = sess(path)
        embed(s, tok, docs[:2])  # 预热
        t0 = time.perf_counter(); dv = embed(s, tok, docs); t_doc = time.perf_counter() - t0
        t0 = time.perf_counter(); qv = embed(s, tok, queries); t_q = time.perf_counter() - t0
        ranks = np.argsort(-(qv @ dv.T), axis=1).tolist()
        res[tag] = dict(dv=dv, ranks=ranks,
                        doc_ms_per_text=round(t_doc * 1000 / len(docs), 1),
                        query_ms_per_text=round(t_q * 1000 / len(queries), 1))

    cos = (res["fp32"]["dv"] * res["int8"]["dv"]).sum(axis=1)
    report["embedding_cosine_fp32_vs_int8"] = [round(float(x), 4) for x in cos]
    report["mean_cosine"] = round(float(cos.mean()), 4)
    report["top1_same"] = [r[0] for r in res["fp32"]["ranks"]] == [r[0] for r in res["int8"]["ranks"]]
    report["top3_same"] = [r[:3] for r in res["fp32"]["ranks"]] == [r[:3] for r in res["int8"]["ranks"]]
    report["fp32_latency"] = {k: res["fp32"][k] for k in ("doc_ms_per_text", "query_ms_per_text")}
    report["int8_latency"] = {k: res["int8"][k] for k in ("doc_ms_per_text", "query_ms_per_text")}
    report["cpu_threads"] = 4

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""从魔搭镜像拉取 all-MiniLM-L6-v2 的 tokenizer 与 ONNX 权重到 web/models/ 下。

用法:
    python tools/fetch_model.py --out web/models/all-MiniLM-L6-v2
    python tools/quantize_int8.py --src web/models/all-MiniLM-L6-v2/onnx/model.onnx \
        --dst web/models/all-MiniLM-L6-v2/onnx/model_quantized.onnx \
        --tokenizer web/models/all-MiniLM-L6-v2/tokenizer.json \
        --report benchmark/quant_report.json
"""
import argparse, os, sys
import urllib.request

BASE = "https://modelscope.cn/models/AI-ModelScope/all-MiniLM-L6-v2/resolve/master/"
FILES = [
    ("config.json", "config.json"),
    ("tokenizer.json", "tokenizer.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("special_tokens_map.json", "special_tokens_map.json"),
    ("vocab.txt", "vocab.txt"),
    ("onnx/model.onnx", "onnx/model.onnx"),
]


def download(rel_src, dst):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"skip {dst}")
        return
    url = BASE + rel_src
    print(f"GET {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:
        f.write(r.read())
    print(f"  -> {dst} ({os.path.getsize(dst)} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/models/all-MiniLM-L6-v2")
    args = ap.parse_args()
    for src, rel in FILES:
        try:
            download(src, os.path.join(args.out, rel))
        except Exception as e:  # 网络不可达时明确报错，不静默跳过权重
            print(f"FAILED {src}: {e}", file=sys.stderr)
            if rel.endswith(".onnx"):
                raise
    print("模型文件就绪，下一步执行 tools/quantize_int8.py 生成 INT8 权重。")


if __name__ == "__main__":
    main()

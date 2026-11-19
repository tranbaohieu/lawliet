"""Build the ANN benchmark dataset: Vietnamese Wikipedia passages embedded by Triton.

Downloads one shard of wikimedia/wikipedia (20231101.vi) from Hugging Face, splits
articles into passages, and embeds them with a Triton model (gRPC). Writes to --out:

    corpus.f32     n × dim float32 (little-endian, row-major)
    queries.f32    q × dim float32
    corpus.jsonl   one {"id", "title", "text"} per corpus row
    queries.jsonl  one {"title", "text"} per query row
    info.json      sizes, model, timing

Queries are held-out passages shortened to their first sentence (short-question-like
text); they never appear in the corpus. Embedding is resumable: progress is
checkpointed in --out/partial_*.f32.

Usage:
    python prepare_data.py --out data --n 100000 --queries 1000 --triton 127.0.0.1:6101
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "wikimedia/wikipedia"
CONFIG = "20231101.vi"


def load_passages(min_words: int, max_words: int, need: int, seed: int) -> list[dict]:
    shards = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith(CONFIG + "/"))
    path = hf_hub_download(REPO, shards[0], repo_type="dataset")
    print(f"shard {shards[0]} ({len(shards)} total)", flush=True)
    table = pq.read_table(path, columns=["title", "text"])
    titles, texts = table.column("title").to_pylist(), table.column("text").to_pylist()
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)

    passages, seen = [], set()
    for i in order:
        for para in texts[i].split("\n"):
            words = para.split()
            if len(words) < min_words:
                continue
            text = " ".join(words[:max_words])
            if text in seen:
                continue
            seen.add(text)
            passages.append({"title": titles[i], "text": text})
            break  # at most one passage per article: more topics, fewer near-duplicates
        if len(passages) >= need:
            break
    if len(passages) < need:
        sys.exit(f"only {len(passages)} passages found, need {need}")
    return passages


def first_sentence(text: str, max_words: int = 25) -> str:
    s = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return " ".join(s.split()[:max_words])


class Embedder:
    def __init__(self, url: str, model: str, input_name: str, output_name: str):
        import tritonclient.grpc as grpcclient

        self.g = grpcclient
        self.local = threading.local()
        self.url, self.model, self.input_name, self.output_name = url, model, input_name, output_name

    def client(self):
        if not hasattr(self.local, "c"):
            self.local.c = self.g.InferenceServerClient(self.url)
        return self.local.c

    def __call__(self, text: str) -> np.ndarray:
        arr = np.array([[text.encode()]], dtype=object)
        inp = self.g.InferInput(self.input_name, arr.shape, "BYTES")
        inp.set_data_from_numpy(arr)
        for attempt in range(5):
            try:
                out = self.client().infer(self.model, [inp])
                return out.as_numpy(self.output_name)[0].astype(np.float32)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)
        raise AssertionError


def embed_all(emb: Embedder, texts: list[str], dim: int, path: str, workers: int) -> np.ndarray:
    """Embed texts into path (float32 rows), resuming from a partial file."""
    partial = path + ".partial"
    done = os.path.getsize(partial) // (4 * dim) if os.path.exists(partial) else 0
    if os.path.exists(path):
        return np.fromfile(path, dtype="<f4").reshape(-1, dim)
    t0, chunk = time.time(), 2000
    with open(partial, "ab") as f, ThreadPoolExecutor(workers) as ex:
        f.truncate(done * 4 * dim)
        for start in range(done, len(texts), chunk):
            vecs = list(ex.map(emb, texts[start : start + chunk]))
            np.stack(vecs).astype("<f4").tofile(f)
            f.flush()
            n = start + len(vecs)
            rate = (n - done) / (time.time() - t0)
            print(f"  {os.path.basename(path)}: {n}/{len(texts)}  {rate:.0f}/s  "
                  f"eta {(len(texts) - n) / rate / 60:.1f} min", flush=True)
    os.replace(partial, path)
    return np.fromfile(path, dtype="<f4").reshape(-1, dim)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--queries", type=int, default=1_000)
    ap.add_argument("--triton", default="127.0.0.1:6101")
    ap.add_argument("--model", default="embedding")
    ap.add_argument("--input", default="text")
    ap.add_argument("--output", default="embeddings")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    corpus_jsonl = os.path.join(args.out, "corpus.jsonl")
    queries_jsonl = os.path.join(args.out, "queries.jsonl")
    if not os.path.exists(corpus_jsonl):
        passages = load_passages(40, 120, args.n + args.queries, args.seed)
        corpus, held = passages[: args.n], passages[args.n :]
        with open(corpus_jsonl, "w") as f:
            for i, p in enumerate(corpus):
                f.write(json.dumps({"id": f"p{i}", **p}, ensure_ascii=False) + "\n")
        with open(queries_jsonl, "w") as f:
            for p in held:
                f.write(json.dumps({"title": p["title"], "text": first_sentence(p["text"])}, ensure_ascii=False) + "\n")
    corpus = [json.loads(l)["text"] for l in open(corpus_jsonl)]
    queries = [json.loads(l)["text"] for l in open(queries_jsonl)]
    print(f"{len(corpus)} passages, {len(queries)} queries", flush=True)

    emb = Embedder(args.triton, args.model, args.input, args.output)
    t0 = time.time()
    q = embed_all(emb, queries, args.dim, os.path.join(args.out, "queries.f32"), args.workers)
    x = embed_all(emb, corpus, args.dim, os.path.join(args.out, "corpus.f32"), args.workers)
    info = {
        "source": f"{REPO} {CONFIG}", "n": len(x), "queries": len(q), "dim": args.dim,
        "model": args.model, "embed_seconds": round(time.time() - t0, 1),
        "norm_mean": float(np.linalg.norm(x[:1000], axis=1).mean()),
    }
    json.dump(info, open(os.path.join(args.out, "info.json"), "w"), indent=2)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()

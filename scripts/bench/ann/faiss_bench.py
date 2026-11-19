"""Faiss baseline for the ANN benchmark, same dataset and settings as main.go.

Writes the exact ground truth (gt.i32: queries × 100 ids, inner product on
normalized vectors) used by both engines, then for each configuration measures
build time (train + add, all threads), memory (serialized index size), and per
search setting: recall@10, single-thread single-query latency, all-thread batch
throughput, and filtered recall (10% of ids, IDSelector).

Quantized configs are wrapped in IndexRefineFlat with k_factor = Lawliet's
RERANK, because Lawliet always re-scores the final candidates with the full
vectors it stores (RERANK 1 = exact scores for the top 10 only).

    python faiss_bench.py --data data --out results-faiss.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import faiss
import numpy as np

K = 10


def load(data: str):
    info = json.load(open(os.path.join(data, "info.json")))
    d = info["dim"]
    x = np.fromfile(os.path.join(data, "corpus.f32"), dtype="<f4").reshape(-1, d)
    q = np.fromfile(os.path.join(data, "queries.f32"), dtype="<f4").reshape(-1, d)
    faiss.normalize_L2(x)
    faiss.normalize_L2(q)
    return x, q


def ground_truth(data: str, x: np.ndarray, q: np.ndarray) -> np.ndarray:
    path = os.path.join(data, "gt.i32")
    if os.path.exists(path):
        return np.fromfile(path, dtype="<i4").reshape(len(q), -1)
    idx = faiss.IndexFlatIP(x.shape[1])
    idx.add(x)
    _, I = idx.search(q, 100)
    I.astype("<i4").tofile(path)
    return I


def recall(I: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(I[:, :K], gt[:, :K])]))


def configs(d: int, nlist: int):
    ip = faiss.METRIC_INNER_PRODUCT

    def ivf(make):
        def f():
            return make(faiss.IndexFlatIP(d))
        return f

    def hnsw(m, efc):
        def f():
            i = faiss.IndexHNSWFlat(d, m, ip)
            i.hnsw.efConstruction = efc
            return i
        return f

    def ivfsq8(qz):
        i = faiss.IndexIVFScalarQuantizer(qz, d, nlist, faiss.ScalarQuantizer.QT_8bit, ip, False)
        return i  # by_residual=False, like Lawliet

    return [
        ("flat", lambda: faiss.IndexFlatIP(d), None, [{}]),
        ("flat-sq8", lambda: faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, ip), "refine",
         [{"k_factor": 1}, {"k_factor": 4}]),
        ("flat-pq64", lambda: faiss.IndexPQ(d, 64, 8, ip), "refine",
         [{"k_factor": 1}, {"k_factor": 4}, {"k_factor": 10}]),
        ("hnsw16", hnsw(16, 100), None, [{"efSearch": e} for e in (16, 32, 64, 128, 256)]),
        ("hnsw32", hnsw(32, 200), None, [{"efSearch": e} for e in (16, 32, 64, 128, 256)]),
        ("ivf", ivf(lambda qz: faiss.IndexIVFFlat(qz, d, nlist, ip)), None,
         [{"nprobe": p} for p in (1, 4, 8, 16, 32, 64)]),
        ("ivf-sq8", ivf(ivfsq8), "refine", [{"nprobe": p, "k_factor": 1} for p in (8, 16, 32)]),
        ("ivf-pq64", ivf(lambda qz: faiss.IndexIVFPQ(qz, d, nlist, 64, 8, ip)), "refine",
         [{"nprobe": p, "k_factor": kf} for kf in (1, 10) for p in (16, 32)]),
    ]


def search_params(p: dict, sel=None):
    base = None
    if "efSearch" in p:
        base = faiss.SearchParametersHNSW(efSearch=p["efSearch"])
    elif "nprobe" in p:
        base = faiss.SearchParametersIVF(nprobe=p["nprobe"])
    if sel is not None:
        if base is None:
            base = faiss.SearchParameters()
        base.sel = sel
    if "k_factor" in p:
        return faiss.IndexRefineSearchParameters(k_factor=p["k_factor"], base_index_params=base)
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results-faiss.json")
    ap.add_argument("--configs", default="")
    ap.add_argument("--nlist", type=int, default=316)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    x, q = load(args.data)
    gt = ground_truth(args.data, x, q)
    n, d = x.shape
    print(f"dataset: {n} × {d}, {len(q)} queries; faiss {faiss.__version__}", flush=True)

    # Filtered run: ids with i % 10 == 3 (same as Lawliet's FILTER 'b == 3').
    sub = np.arange(3, n, 10)
    sel = faiss.IDSelectorBatch(sub.astype("int64"))
    sims = q @ x[sub].T
    fgt = sub[np.argsort(-sims, axis=1)[:, :K]]

    only = set(filter(None, args.configs.split(",")))
    with open(args.out, "a") as out:
        for name, make, wrap, sweep in configs(d, args.nlist):
            if only and name not in only:
                continue
            faiss.omp_set_num_threads(args.threads)
            index = make()
            if wrap == "refine":
                index = faiss.IndexRefineFlat(index)
            t0 = time.time()
            if not index.is_trained:
                index.train(x)
            index.add(x)
            build = time.time() - t0
            size = len(faiss.serialize_index(index))
            res = {"engine": "faiss", "name": name, "describe": type(index).__name__, "build_s": build,
                   "serialized_bytes": size, "build_threads": args.threads, "searches": []}
            print(f"{name}: built in {build:.1f}s ({args.threads} threads), {size / 2**20:.1f} MiB", flush=True)

            for p in sweep:
                sp = search_params(p)
                faiss.omp_set_num_threads(1)
                lat = []
                I1 = np.empty((len(q), K), dtype="int64")
                for i in range(len(q)):
                    t = time.perf_counter()
                    _, I1[i : i + 1] = index.search(q[i : i + 1], K, params=sp)
                    lat.append((time.perf_counter() - t) * 1000)
                faiss.omp_set_num_threads(args.threads)
                t = time.perf_counter()
                for _ in range(3):
                    index.search(q, K, params=sp)
                qps_all = 3 * len(q) / (time.perf_counter() - t)

                # Filtered search; some indexes (e.g. IndexPQ) reject search parameters.
                fsp = search_params(p, sel)
                flat = []
                faiss.omp_set_num_threads(1)
                IF = np.empty((len(q), K), dtype="int64")
                try:
                    for i in range(len(q)):
                        t = time.perf_counter()
                        _, IF[i : i + 1] = index.search(q[i : i + 1], K, params=fsp)
                        flat.append((time.perf_counter() - t) * 1000)
                    frec, fp50 = recall(IF, fgt), float(np.percentile(flat, 50))
                except RuntimeError as e:
                    print(f"  filtered search not supported: {str(e).splitlines()[-1][:100]}", flush=True)
                    frec = fp50 = None

                lat = np.array(lat)
                sr = {
                    "params": p, "recall_at_10": recall(I1, gt),
                    "p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99)),
                    "qps_1_thread": 1000 / float(lat.mean()), "qps_all_threads": qps_all,
                    "threads": args.threads,
                    "filtered_recall_at_10": frec, "filtered_p50_ms": fp50,
                }
                res["searches"].append(sr)
                print(f"  {json.dumps(p):34s} recall@10 {sr['recall_at_10']:.4f}  p50 {sr['p50_ms']:6.2f} ms  "
                      f"p99 {sr['p99_ms']:6.2f} ms  qps(1) {sr['qps_1_thread']:7.0f}  qps({args.threads}) {qps_all:7.0f}  "
                      f"filtered recall {frec if frec is None else round(frec, 3)}", flush=True)
            out.write(json.dumps(res) + "\n")
            out.flush()


if __name__ == "__main__":
    main()

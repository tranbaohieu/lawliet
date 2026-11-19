"""End-to-end ANN benchmark over RESP: a real lawliet server and redis-py.

For each configuration: start the server, VCREATE, VADD all vectors in pipelined
batches (FP32 blobs), then VSEARCH every query from one client (latency and
recall@10 against gt.i32) and from several client threads (throughput).

    python resp_bench.py --binary ./lawliet --data data --out results-resp.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import threading
import time

import numpy as np
import redis

CONFIGS = [
    ("flat", ["INDEX", "flat"], [[]]),
    ("hnsw16", ["INDEX", "hnsw", "M", 16, "EF_CONSTRUCTION", 100], [["EF", 64], ["EF", 128]]),
    ("ivf", ["INDEX", "ivf", "NLIST", 316], [["NPROBE", 16], ["NPROBE", 32]]),
    ("ivf-pq64", ["INDEX", "ivf", "NLIST", 316, "QUANT", "pq", "PQ_M", 64],
     [["NPROBE", 32, "RERANK", 10]]),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./lawliet")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results-resp.json")
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--configs", default="")
    args = ap.parse_args()

    info = json.load(open(os.path.join(args.data, "info.json")))
    d = info["dim"]
    x = np.fromfile(os.path.join(args.data, "corpus.f32"), dtype="<f4").reshape(-1, d)
    q = np.fromfile(os.path.join(args.data, "queries.f32"), dtype="<f4").reshape(-1, d)
    gt = np.fromfile(os.path.join(args.data, "gt.i32"), dtype="<i4").reshape(len(q), -1)[:, :10]
    qb = [v.astype("<f4").tobytes() for v in q]
    only = set(filter(None, args.configs.split(",")))

    for name, create, sweep in CONFIGS:
        if only and name not in only:
            continue
        port = free_port()
        proc = subprocess.Popen([args.binary, "--port", str(port)], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            r = redis.Redis(port=port)
            for _ in range(100):
                try:
                    r.ping()
                    break
                except redis.ConnectionError:
                    time.sleep(0.05)
            r.execute_command("VCREATE", "b", "DIM", d, "METRIC", "cosine", *create)
            t0 = time.time()
            pipe = r.pipeline(transaction=False)
            for i, v in enumerate(x):
                pipe.execute_command("VADD", "b", str(i), "FP32", v.astype("<f4").tobytes())
                if (i + 1) % 500 == 0:
                    pipe.execute()
            pipe.execute()
            insert_s = time.time() - t0
            vinfo = r.execute_command("VINFO", "b")
            vinfo = vinfo if isinstance(vinfo, dict) else dict(zip(vinfo[::2], vinfo[1::2]))
            res = {"engine": "lawliet-resp", "name": name, "insert_s": insert_s,
                   "insert_per_s": len(x) / insert_s,
                   "index_state": vinfo[b"index_state"].decode(), "searches": []}
            print(f"{name}: {len(x)} VADD in {insert_s:.1f}s ({len(x) / insert_s:.0f}/s), {res['index_state']}",
                  flush=True)

            for extra in sweep:
                lat, hits = [], 0
                for i, b in enumerate(qb):
                    t = time.perf_counter()
                    out = r.execute_command("VSEARCH", "b", "FP32", b, "K", 10, *extra)
                    lat.append((time.perf_counter() - t) * 1000)
                    hits += len({int(h[0]) for h in out} & set(gt[i].tolist()))

                total, lock, nxt = 3 * len(qb), threading.Lock(), [0]

                def worker():
                    c = redis.Redis(port=port)
                    while True:
                        with lock:
                            i = nxt[0]
                            nxt[0] += 1
                        if i >= total:
                            return
                        c.execute_command("VSEARCH", "b", "FP32", qb[i % len(qb)], "K", 10, *extra)

                ths = [threading.Thread(target=worker) for _ in range(args.clients)]
                t = time.perf_counter()
                for th in ths:
                    th.start()
                for th in ths:
                    th.join()
                qps = total / (time.perf_counter() - t)
                lat = np.array(lat)
                sr = {"params": " ".join(map(str, extra)), "recall_at_10": hits / (10 * len(qb)),
                      "p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99)),
                      "qps_clients": qps, "clients": args.clients}
                res["searches"].append(sr)
                print(f"  {sr['params'] or '-':24s} recall@10 {sr['recall_at_10']:.4f}  p50 {sr['p50_ms']:6.2f} ms  "
                      f"p99 {sr['p99_ms']:6.2f} ms  qps({args.clients} clients) {qps:6.0f}", flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")
        finally:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    main()

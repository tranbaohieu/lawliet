"""Merge results-*.json (JSON lines) into Markdown tables, Lawliet next to Faiss.

    python report.py results-lawliet.json results-faiss.json [results-resp.json] > report.md
"""

from __future__ import annotations

import json
import sys

KEYS = {"EfSearch": "ef", "NProbe": "nprobe", "Rerank": "rerank",
        "efSearch": "ef", "nprobe": "nprobe", "k_factor": "rerank"}


def key(params: dict) -> str:
    parts = [f"{KEYS[k]}={v}" for k, v in sorted(params.items(), key=lambda kv: KEYS[kv[0]]) if v]
    return " ".join(parts) or "-"


def load(paths):
    rows, churn = {}, []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r.get("churn"):
                churn.append(r)  # churn runs repeat a config: keep them apart
            else:
                rows[(r["engine"], r["name"])] = r
    return rows, churn


def mib(b):
    return f"{b / 2**20:.0f}"


def main() -> None:
    rows, churn = load(sys.argv[1:])
    names = [n for (e, n) in rows if e == "lawliet"]
    vec_mib = 100_000 * 1024 * 4 / 2**20

    print("### Build\n")
    print("| index | Lawliet build (1 thread, VADD path) | max single insert | Lawliet memory: vectors + index MiB "
          "| Faiss build (train+add, all threads) | Faiss index MiB |")
    print("|---|---:|---:|---:|---:|---:|")
    for n in names:
        lw, fa = rows[("lawliet", n)], rows.get(("faiss", n))
        print(f"| {n} | {lw['build_s']:.1f} s | {lw['max_insert_ms']:.0f} ms | "
              f"{vec_mib:.0f} + {mib(lw['index_bytes'])} | "
              f"{fa['build_s']:.1f} s | {mib(fa['serialized_bytes'])} |" if fa else
              f"| {n} | {lw['build_s']:.1f} s | {lw['max_insert_ms']:.0f} ms | {vec_mib:.0f} + {mib(lw['index_bytes'])} | - | - |")

    print("\n### Search (K=10, 1000 queries)\n")
    print("| index | params | recall@10 Lawliet | recall@10 Faiss | p50 ms Lawliet | p50 ms Faiss "
          "| QPS all threads Lawliet | QPS all threads Faiss (batch) | filtered recall L / F |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for n in names:
        lw, fa = rows[("lawliet", n)], rows.get(("faiss", n))
        fs = {key(s["params"]): s for s in fa["searches"]} if fa else {}
        for s in lw["searches"]:
            k = key(s["params"])
            f = fs.get(k)
            fr = s.get("filtered_recall_at_10")
            ffr = f.get("filtered_recall_at_10") if f else None
            print(f"| {n} | {k} | {s['recall_at_10']:.3f} | {f['recall_at_10']:.3f} | " if f else
                  f"| {n} | {k} | {s['recall_at_10']:.3f} | - | ", end="")
            print(f"{s['p50_ms']:.2f} | {f['p50_ms']:.2f} | {s['qps_all_threads']:.0f} | {f['qps_all_threads']:.0f} | "
                  if f else f"{s['p50_ms']:.2f} | - | {s['qps_all_threads']:.0f} | - | ", end="")
            fmt = lambda v: "-" if v is None else f"{v:.3f}"
            print(f"{fmt(fr)} / {fmt(ffr)} |")

    if churn:
        print("\n### Churn (delete a fraction of records and insert them again, same ids and vectors)\n")
        print("| index | params | rounds × fraction | recall@10 before | recall@10 after | time |")
        print("|---|---|---|---:|---:|---:|")
        for r in churn:
            c = r["churn"]
            print(f"| {r['name']} | {key(c['params'])} | {c['rounds']} × {c['fraction']:.0%} | "
                  f"{c['recall_before']:.3f} | {c['recall_after']:.3f} | {c['seconds']:.0f} s |")

    resp = [r for (e, _), r in rows.items() if e == "lawliet-resp"]
    if resp:
        print("\n### End to end over RESP (redis-py, one server)\n")
        print("| index | VADD/s (pipelined) | params | recall@10 | p50 ms | p99 ms | QPS (clients) |")
        print("|---|---:|---|---:|---:|---:|---:|")
        for r in resp:
            for s in r["searches"]:
                print(f"| {r['name']} | {r['insert_per_s']:.0f} | {s['params'] or '-'} | {s['recall_at_10']:.3f} | "
                      f"{s['p50_ms']:.2f} | {s['p99_ms']:.2f} | {s['qps_clients']:.0f} ({s['clients']}) |")


if __name__ == "__main__":
    main()

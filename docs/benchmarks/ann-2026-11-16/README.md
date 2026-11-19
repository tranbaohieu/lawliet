# ANN indexes: Lawliet vs Faiss (2026-11-16)

Lawliet's new indexes (`hnsw`, `ivf`, `QUANT sq8|pq`; DESIGN.md §7) against Faiss 1.16.0 (faiss-cpu
from PyPI) on the same data, parameters and ground truth.

## Setup

| | |
|---|---|
| Data | 100,000 passages from Vietnamese Wikipedia (`wikimedia/wikipedia` 20231101.vi, first shard, one paragraph of 40–120 words per article) |
| Queries | 1,000 held-out passages cut to their first sentence (≤ 25 words); never in the corpus |
| Embeddings | Triton model `embedding` (Jina-based ensemble), 1024 dims, L2-normalized; metric cosine / inner product |
| Ground truth | exact top-100 (Faiss `IndexFlatIP`), `gt.i32`; recall@10 = overlap of the top 10 |
| Machine | Intel Core i5-12500 (6 cores / 12 threads), 15 GB RAM, Linux; Go 1.27.4, Python 3.12 |
| Lawliet in-process | `scripts/bench/ann/main.go`: records go through `store.Collection` (VADD/VSEARCH path without RESP), inserted one by one |
| Faiss | `scripts/bench/ann/faiss_bench.py`: train + add with 12 threads; quantized indexes wrapped in `IndexRefineFlat` (k_factor = `RERANK`), IVF-SQ8 non-residual, to match Lawliet |
| Over RESP | `scripts/bench/ann/resp_bench.py`: real server, redis-py, pipelined VADD, 1 client for latency, 8 for QPS |
| Filtered | `FILTER 'b == 3'` (10% of records); Faiss uses `IDSelectorBatch` |

Latency = single query, single thread (Faiss with 1 OpenMP thread). "QPS all threads": Lawliet
runs 12 goroutines issuing single queries; Faiss searches the 1,000 queries as one batch with 12
threads (its strongest mode: BLAS for flat). Numbers are from one run on one machine.

Reproduce: `prepare_data.py` → `faiss_bench.py` (writes `gt.i32`) → `go run ./scripts/bench/ann` →
`resp_bench.py` → `report.py`. Raw results: `results-*.json` in this directory.

## Update: AVX2 kernels and HNSW delete repair (same day)

Second run after two changes: AVX2+FMA distance kernels (`GOEXPERIMENT=simd`, DESIGN §7; opt-in
with `make build SIMD=1`, release binaries are scalar and match the first run) and
HNSW deletes that repair the graph instead of leaving tombstones (TD-3). Same machine, data
and harness; raw results in `results-*-simd.json`. Recall is unchanged to 4 decimals (the
SIMD kernels compute the same distances).

| index | params | recall@10 | p50 before → after (ms) | Faiss p50 | QPS 12 thr before → after | build before → after | max insert before → after |
|---|---|---:|---:|---:|---:|---:|---:|
| flat | - | 1.000 | 30.96 → **19.80** (1.6×) | 18.09 | 132 → 370 | 0.3 → 0.3 s | 1 → 1 ms |
| flat-sq8 | rerank=1 | 0.994 | 41.51 → **9.23** (4.5×) | 6.09 | 139 → 553 | 1.3 → 1.3 s | 333 → 328 ms |
| flat-pq64 | rerank=10 | 0.926 | 4.03 → **4.05** (1.0×) | 1.37 | 1519 → 1513 | 26.4 → 28.5 s | 7672 → 8358 ms |
| hnsw16 | ef=64 | 0.979 | 0.60 → **0.44** (1.4×) | 0.33 | 4028 → 4498 | 108.4 → 63.7 s | 4 → 2 ms |
| hnsw32 | ef=64 | 0.990 | 0.91 → **0.66** (1.4×) | 0.51 | 2581 → 2894 | 229.2 → 149.8 s | 6 → 4 ms |
| ivf | nprobe=16 | 0.942 | 2.44 → **1.74** (1.4×) | 1.00 | 852 → 997 | 14.5 → 3.5 s | 3832 → 863 ms |
| ivf-sq8 | nprobe=16 rerank=1 | 0.940 | 2.33 → **0.53** (4.4×) | 0.37 | 2448 → 5183 | 24.7 → 6.5 s | 5306 → 1381 ms |
| ivf-pq64 | nprobe=16 rerank=10 | 0.865 | 0.52 → **0.44** (1.2×) | 0.21 | 12895 → 15609 | 47.4 → 34.6 s | 12576 → 9575 ms |

Kernel microbenchmarks (1024 dims, `go test -bench ./internal/vecmath`): Dot 281 → 60 ns
(4.7×), L2Sq 293 → 57 ns (5.1×). The kernels must end with `VZEROUPPER`: without it the
AVX→SSE transition penalty made the index tests 3× slower than the scalar build.

Over RESP: HNSW M=16 ef=64 p50 0.69 → **0.52 ms**, VADD 917 → **1,541/s**; IVF VADD
6,362 → 22,112/s; flat p50 32.6 → 21.0 ms.

Churn (3 × 30% delete + re-insert, ef=64): HNSW recall 0.979 → **0.971** (was → 0.954 with
tombstones); the graph still ends with 8% fewer links (heuristic pruning during repair).
IVF unchanged (0.942).

2026-11-19 (review of PR #3): `connect` now also drops links to removed nodes. Same churn run:
0.979 → **0.9717** (`results-churn-fix.json`), i.e. unchanged; the graph still ends with 8%
fewer links (2,126,118 vs 2,314,641), which is where the remaining loss comes from (TD-3).

Gap to Faiss per query is now 1.1× (flat), 1.3× (HNSW), 1.4–1.5× (SQ8), 1.7× (IVF-Flat). PQ
did not move (2–3×): ADC table lookups and PQ training run on 16-dim sub-vectors, below the
32-float SIMD threshold.

## Findings (first run, scalar kernels, tombstones)

1. **Recall matches Faiss** for every index and setting (within ±0.03, most within ±0.005): the
   algorithms behave like Faiss's.
2. **Faiss is faster per query**: 1.7–2× for HNSW, 2.4–2.6× for IVF, 2.7–3× for PQ, 5.7–6.8× for
   SQ8, 1.7× for flat single queries and 22× for flat batches (BLAS). Lawliet's distance loops are scalar Go
   (no SIMD); SQ8 decoding is slower than float math without SIMD, so `flat QUANT sq8` is slower
   than `flat` in Lawliet.
3. **HNSW is the best fit for Lawliet**: at ef=64, recall 0.979 with 0.60 ms p50 (50× faster than
   flat), 0.69 ms over RESP, no training pause. Builds are slow (sequential VADD path: 917 VADD/s
   for M=16; Faiss builds in 29 s with 12 threads).
4. **IVF trains inside a VADD**: the insert that crosses a training threshold blocks for 3.8 s
   (IVF-Flat), 5.3 s (IVF-SQ8), 12.6 s (IVF-PQ). Acceptable for batch loads, not for a live cache.
5. **IVF-SQ8 beats IVF-Flat on throughput** (2,448 vs 852 QPS at nprobe 16, same recall):
   contiguous codes per cell vs vectors scattered in the slab.
6. **PQ64 caps recall** near 0.87–0.93 even with RERANK 10 (same in Faiss). It saves scan
   bandwidth only: Lawliet keeps full vectors anyway, so PQ adds memory instead of saving it.
7. **Filters**: Lawliet HNSW keeps recall under a 10% filter (0.997 at ef=64) by traversing until
   `ef` matches are found, at 5–6× the latency; Faiss keeps latency and loses recall (0.886).
   At equal filtered recall the cost is similar.
8. **Churn degrades HNSW**: after deleting and re-inserting 30% of records three times, recall at
   ef=64 drops 0.979 → 0.954 (stale links); IVF does not change. Matches the DESIGN §16.3 concern.

### Build

| index | Lawliet build (1 thread, VADD path) | max single insert | Lawliet memory: vectors + index MiB | Faiss build (train+add, all threads) | Faiss index MiB |
|---|---:|---:|---:|---:|---:|
| flat | 0.3 s | 1 ms | 391 + 0 | 0.2 s | 391 |
| flat-sq8 | 1.3 s | 333 ms | 391 + 119 | 0.3 s | 488 |
| flat-pq64 | 26.4 s | 7672 ms | 391 + 9 | 5.2 s | 398 |
| hnsw16 | 108.4 s | 4 ms | 391 + 18 | 28.9 s | 404 |
| hnsw32 | 229.2 s | 6 ms | 391 + 30 | 74.8 s | 417 |
| ivf | 14.5 s | 3832 ms | 391 + 3 | 1.7 s | 393 |
| ivf-sq8 | 24.7 s | 5306 ms | 391 + 113 | 1.8 s | 490 |
| ivf-pq64 | 47.4 s | 12576 ms | 391 + 11 | 6.9 s | 400 |

### Search (K=10, 1000 queries)

| index | params | recall@10 Lawliet | recall@10 Faiss | p50 ms Lawliet | p50 ms Faiss | QPS all threads Lawliet | QPS all threads Faiss (batch) | filtered recall L / F |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| flat | - | 1.000 | 1.000 | 30.96 | 18.09 | 132 | 2910 | 1.000 / 1.000 |
| flat-sq8 | rerank=1 | 0.994 | 0.994 | 41.51 | 6.09 | 139 | 970 | 0.995 / 0.996 |
| flat-sq8 | rerank=4 | 1.000 | 1.000 | 41.51 | 6.08 | 138 | 961 | 1.000 / 1.000 |
| flat-pq64 | rerank=1 | 0.543 | 0.535 | 3.92 | 1.30 | 1539 | 3347 | 0.567 / - |
| flat-pq64 | rerank=4 | 0.835 | 0.831 | 3.95 | 1.32 | 1535 | 3296 | 0.900 / - |
| flat-pq64 | rerank=10 | 0.926 | 0.919 | 4.03 | 1.37 | 1519 | 3141 | 0.978 / - |
| hnsw16 | ef=16 | 0.884 | 0.875 | 0.24 | 0.12 | 10872 | 12462 | 0.971 / 0.598 |
| hnsw16 | ef=32 | 0.947 | 0.942 | 0.36 | 0.20 | 6739 | 7620 | 0.990 / 0.759 |
| hnsw16 | ef=64 | 0.979 | 0.976 | 0.60 | 0.33 | 4028 | 4606 | 0.997 / 0.886 |
| hnsw16 | ef=128 | 0.991 | 0.989 | 1.02 | 0.58 | 2358 | 2694 | 0.999 / 0.952 |
| hnsw16 | ef=256 | 0.995 | 0.995 | 1.74 | 1.02 | 1371 | 1557 | 1.000 / 0.982 |
| hnsw32 | ef=16 | 0.919 | 0.924 | 0.34 | 0.18 | 7113 | 8137 | 0.986 / 0.682 |
| hnsw32 | ef=32 | 0.971 | 0.970 | 0.54 | 0.30 | 4372 | 4974 | 0.996 / 0.827 |
| hnsw32 | ef=64 | 0.990 | 0.989 | 0.91 | 0.51 | 2581 | 2919 | 0.999 / 0.932 |
| hnsw32 | ef=128 | 0.996 | 0.996 | 1.55 | 0.87 | 1509 | 1699 | 1.000 / 0.979 |
| hnsw32 | ef=256 | 0.998 | 0.999 | 2.60 | 1.50 | 883 | 991 | 1.000 / 0.994 |
| ivf | nprobe=1 | 0.538 | 0.568 | 0.26 | 0.10 | 10859 | 14946 | 0.436 / 0.449 |
| ivf | nprobe=4 | 0.798 | 0.817 | 0.73 | 0.29 | 3095 | 4121 | 0.705 / 0.722 |
| ivf | nprobe=8 | 0.888 | 0.895 | 1.31 | 0.53 | 1637 | 2154 | 0.818 / 0.830 |
| ivf | nprobe=16 | 0.942 | 0.945 | 2.44 | 1.00 | 852 | 1123 | 0.903 / 0.904 |
| ivf | nprobe=32 | 0.970 | 0.971 | 4.76 | 1.89 | 432 | 579 | 0.950 / 0.948 |
| ivf | nprobe=64 | 0.987 | 0.987 | 9.63 | 3.75 | 215 | 292 | 0.977 / 0.977 |
| ivf-sq8 | nprobe=8 rerank=1 | 0.889 | 0.891 | 1.26 | 0.22 | 4568 | 10333 | 0.816 / 0.828 |
| ivf-sq8 | nprobe=16 rerank=1 | 0.940 | 0.941 | 2.33 | 0.37 | 2448 | 5561 | 0.897 / 0.902 |
| ivf-sq8 | nprobe=32 rerank=1 | 0.968 | 0.966 | 4.50 | 0.68 | 1273 | 2907 | 0.947 / 0.946 |
| ivf-pq64 | nprobe=16 rerank=1 | 0.527 | 0.534 | 0.43 | 0.16 | 14357 | 37077 | 0.545 / 0.554 |
| ivf-pq64 | nprobe=32 rerank=1 | 0.527 | 0.534 | 0.65 | 0.23 | 9905 | 24004 | 0.546 / 0.556 |
| ivf-pq64 | nprobe=16 rerank=10 | 0.865 | 0.862 | 0.52 | 0.21 | 12895 | 21406 | 0.890 / 0.894 |
| ivf-pq64 | nprobe=32 rerank=10 | 0.872 | 0.868 | 0.70 | 0.28 | 9279 | 16561 | 0.920 / 0.925 |

### Churn (delete a fraction of records and insert them again, same ids and vectors)

| index | params | rounds × fraction | recall@10 before | recall@10 after | time |
|---|---|---|---:|---:|---:|
| hnsw16 | ef=64 | 3 × 30% | 0.979 | 0.954 | 100 s |
| ivf | nprobe=16 | 3 × 30% | 0.942 | 0.942 | 8 s |
| ivf-pq64 | nprobe=16 rerank=10 | 3 × 30% | 0.865 | 0.865 | 20 s |

### End to end over RESP (redis-py, one server)

| index | VADD/s (pipelined) | params | recall@10 | p50 ms | p99 ms | QPS (clients) |
|---|---:|---|---:|---:|---:|---:|
| flat | 78827 | - | 1.000 | 32.61 | 36.72 | 98 (8) |
| hnsw16 | 917 | EF 64 | 0.979 | 0.69 | 0.93 | 3729 (8) |
| hnsw16 | 917 | EF 128 | 0.991 | 1.13 | 1.50 | 2209 (8) |
| ivf | 6362 | NPROBE 16 | 0.942 | 2.69 | 3.93 | 819 (8) |
| ivf | 6362 | NPROBE 32 | 0.970 | 5.04 | 6.69 | 415 (8) |
| ivf-pq64 | 2017 | NPROBE 32 RERANK 10 | 0.872 | 0.79 | 0.93 | 6323 (8) |

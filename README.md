# Lawliet

**An in-memory vector cache that speaks the Redis protocol.** One binary, zero
config, ready in milliseconds — for testing, prototyping and demoing RAG and LLM
apps without running a full vector database.

```bash
./bin/lawliet                    # listening on 127.0.0.1:6380
```

```python
import redis, numpy as np

r = redis.Redis(port=6380)       # any Redis client works, default settings
embedding = np.random.rand(384)  # your embedding model's output

r.execute_command("VCREATE", "docs", "DIM", 384)
r.execute_command("VADD", "docs", "doc:1", "FP32", embedding.astype("<f4").tobytes(),
                  "TEXT", "Employees get 12 days of paid leave.", "META", '{"source":"hr.md"}')
hits = r.execute_command("VSEARCH", "docs", "FP32", embedding.astype("<f4").tobytes(),
                         "K", 5, "FILTER", 'source == "hr.md"', "WITHTEXT")
# [[b'doc:1', 0.99999976, b'Employees get 12 days of paid leave.']]  (float32 rounding)
```

> **Status:** v0.2.0 — MVP plus semantic cache commands (`VCACHE.*`). Exact (flat) search only; see
> [Limitations](#limitations).

## Why

Vector databases such as Milvus are built for production scale: object storage,
segments, lazy loading. For a test suite, a notebook or a chatbot demo that is a lot
of moving parts. Lawliet keeps everything in RAM, speaks RESP2/RESP3 so existing Redis
clients work unchanged, and adds the features a RAG prototype needs:

- **Semantic cache commands** — `VCACHE.GET` returns a cached answer when a similar
  question was already answered (threshold in one round trip), `VCACHE.SET` stores it with a
  default TTL, plus hit/miss statistics and invalidation by filter. Skip search + generation
  for repeated questions.
- **Exact vector search** — cosine, L2, inner product.
- **Text and metadata** stored with each vector, returned with results.
- **Filters** — `(source == "hr.md" or page >= 2) and not tags contains "draft"`.
- **TTL per record** and **eviction policies per collection** (`noeviction`,
  `allkeys-lru`, `volatile-ttl`) — documents and a semantic cache in one server.
- **Memory limits** (`--maxmemory`, `MAX_VECTORS`).
- **Optional snapshots** to one local file, restored bit-exact on restart.

## Quick start

### Python: `pip install`

The [`lawliet` Python package](python/README.md) bundles the server binary: install it and
start the server from your script, notebook or test — nothing else to download or run.

```bash
pip install lawliet-server        # import lawliet
```

```python
import lawliet

r = lawliet.start()                  # server on a free port, returns redis.Redis
r.execute_command("VCREATE", "docs", "DIM", 384)
...
lawliet.stop()                       # also stopped automatically at exit

with lawliet.server() as r:          # fresh in-memory server for one block (e.g. a test)
    ...
```

Wheels are built for the same platforms as the binaries below and attached to each
[release](https://github.com/tranbaohieu/lawliet/releases) from the release after v0.2.0
on; `pip install lawliet-server` works once they are published to PyPI. The package also installs a
`lawliet` command (`lawliet --port 6380`).

### Download a binary

Prebuilt binaries for macOS, Linux and Windows are on the
[Releases](https://github.com/tranbaohieu/lawliet/releases) page — one file, nothing else to install.

```bash
# macOS Apple Silicon. Other builds: darwin_amd64, linux_amd64, linux_arm64, windows_amd64 (.zip)
curl -L https://github.com/tranbaohieu/lawliet/releases/latest/download/lawliet_darwin_arm64.tar.gz | tar xz
./lawliet_darwin_arm64/lawliet --version
./lawliet_darwin_arm64/lawliet          # listening on 127.0.0.1:6380
```

`checksums.txt` in each release lists SHA-256 sums. If macOS blocks a binary downloaded with a
browser, remove the quarantine flag: `xattr -d com.apple.quarantine lawliet`.

### From source (Go 1.27.4+)

```bash
git clone https://github.com/tranbaohieu/lawliet && cd lawliet
make build
./bin/lawliet --port 6380
redis-cli -p 6380 ping          # PONG
```

### Docker

```bash
docker build -t lawliet .
docker run -p 6380:6380 lawliet

# with persistence
docker run -p 6380:6380 -v lawliet-data:/data lawliet --bind 0.0.0.0 --dir /data
```

### Try it with redis-cli

```text
> VCREATE demo DIM 3
OK
> VADD demo d1 VALUES 3 0.1 0.2 0.3 TEXT "first" META '{"lang":"en"}'
(integer) 1
> VADD demo d2 VALUES 3 0.9 0.1 0.0 TEXT "second" META '{"lang":"vi"}' EX 60
(integer) 1
> VSEARCH demo VALUES 3 0.1 0.2 0.25 K 2 FILTER 'lang == "en"' WITHTEXT
1) 1) "d1"
   2) "0.9960239"
   3) "first"
```

### RAG example

[`examples/python/rag_demo.py`](examples/python/rag_demo.py) ingests documents, retrieves
with filters, and puts a semantic cache (`VCACHE.*`, TTL + LRU) in front of the LLM. It runs without
model downloads or API keys; swap `embed()` and `generate()` for real models.

```bash
pip install -r examples/python/requirements.txt
python examples/python/rag_demo.py
```

## Commands

| Command | Purpose |
|---|---|
| `VCREATE col DIM n [METRIC cosine\|l2\|ip] [MAX_VECTORS n] [POLICY p] [CACHE_THRESHOLD s] [CACHE_TTL sec]` | create a collection |
| `VDROP col` · `VLIST` · `VINFO col` | drop, list, inspect |
| `VCONFIG col GET` · `VCONFIG col SET POLICY p MAX_VECTORS n CACHE_THRESHOLD s CACHE_TTL sec` | change limits / cache defaults at runtime |
| `VADD col id (FP32 blob \| VALUES n v…) [TEXT s] [META json] [EX s \| PX ms] [NX \| XX]` | insert or replace a record |
| `VGET col id [WITHVEC]` · `VDEL col id…` · `VEXISTS col id` · `VCOUNT col` | read, delete, check, count |
| `VSEARCH col (FP32 blob \| VALUES n v…) [K n] [FILTER expr] [WITHTEXT] [WITHMETA] [WITHVEC]` | nearest neighbours |
| `VEXPIRE col id s` · `VTTL col id` · `VPERSIST col id` | TTL management |
| `VCACHE.GET col (FP32 blob \| VALUES n v…) [THRESHOLD s] [FILTER expr] [WITHMETA]` | cached answer if score ≥ threshold, else null |
| `VCACHE.SET col id (FP32 blob \| VALUES n v…) TEXT answer [META json] [EX s \| PX ms]` | store an answer (default TTL = `CACHE_TTL`) |
| `VCACHE.STATS col` · `VCACHE.INVALIDATE col FILTER expr` | hit/miss stats, remove entries by metadata |
| `SAVE` · `INFO [section]` · `PING` · `HELLO [2\|3]` | server |

Details — reply formats, filter grammar, errors — are in [DESIGN.md](DESIGN.md).

**Vectors:** send `FP32` blobs (float32, little-endian): `vec.astype("<f4").tobytes()`.
`VALUES` is for typing by hand. **Scores** are similarities: higher is better
(L2 returns the negated distance).

## Server options

| Flag | Default | |
|---|---|---|
| `--bind` | `127.0.0.1` | there is no authentication: keep it local or behind a firewall |
| `--port` | `6380` | |
| `--maxmemory` | `0` (off) | e.g. `512mb`, `2gb` — see [Limitations](#limitations) |
| `--maxmemory-policy` | `noeviction` | default policy for new collections |
| `--max-collections` | `0` (off) | |
| `--dir` | empty (off) | enable snapshots in this existing directory |
| `--dbfilename` | `dump.lwdb` | |
| `--save "sec changes"` | `3600 1`, `300 100`, `60 10000` | repeatable; `--save ""` disables auto-save |
| `--exit-on-stdin-eof` | off | shut down when stdin closes (used by the Python package) |

Snapshots are written on save rules, on `SAVE`, and on shutdown (`SIGINT`/`SIGTERM`, or stdin
closed with `--exit-on-stdin-eof`),
and loaded at startup. A corrupt file stops the server instead of loading partial data.

## Performance

End-to-end over RESP with redis-py (`scripts/bench/bench.py`), cosine, K=10,
flat index. Measured once on an Apple Silicon laptop (macOS, 10 CPUs); numbers on
your machine will differ.

| Records | Dim | Insert | Search p50 | Search p99 | Filtered p50 (10% match) | QPS (8 clients) |
|---|---|---|---|---|---|---|
| 10,000 | 384 | 123k vec/s | 1.1 ms | 1.2 ms | 0.4 ms | 3,702 |
| 100,000 | 384 | 132k vec/s | 11.1 ms | 11.7 ms | 7.3 ms | 462 |
| 10,000 | 1536 | 86k vec/s | 5.4 ms | 6.0 ms | 0.9 ms | 984 |
| 100,000 | 1536 | 88k vec/s | 54.3 ms | 56.4 ms | 12.0 ms | 105 |

Search time grows linearly with the collection size (exact search). Lawliet targets
up to roughly 100k vectors per collection; beyond that an approximate index (HNSW, on the
roadmap) is needed.

## Limitations

- **Exact search only** (flat index); one query uses one CPU core.
- **`--maxmemory` limits the estimated record size, not process memory.** Freed vector
  slots are kept for reuse and the Go GC needs headroom; measured process memory can be
  1.5–3× the limit. Tracked as [TD-1](TECH_DEBT.md).
- **No authentication or TLS**, no replication or clustering.
- Snapshots block writes to a collection while it is being written (about 60 ms for
  10k × 384-dim records); data written after the last snapshot is lost on a crash.
- Filters scan metadata linearly (no secondary indexes).

## Development

```bash
make test          # unit + integration tests
make race          # with the race detector
make e2e           # end-to-end checks against a real server (needs python3, redis, numpy)
make bench         # benchmark
make docker        # build the image
make py-wheel      # Python wheel for this machine (needs: pip install build)
make py-test       # Python package tests against bin/lawliet (needs: pip install -e python pytest)
```

Layout: `cmd/lawliet` (main), `internal/resp` (protocol), `internal/server`,
`internal/command` (V* commands), `internal/store` (collections, TTL, eviction),
`internal/index` (flat search), `internal/filter`, `internal/meta`, `internal/snapshot`,
`python/` (pip package bundling the binary).
Design decisions live in [DESIGN.md](DESIGN.md); known debts in [TECH_DEBT.md](TECH_DEBT.md).

## Roadmap

Lawliet aims to be a **semantic cache and vector store for LLM apps** rather than a
general-purpose vector database ([DESIGN.md §1.3](DESIGN.md)). In priority order
(drafts in [DESIGN.md §16](DESIGN.md)):

1. ✅ **`VCACHE.*` semantic cache commands** — threshold lookup, default TTL, hit/miss stats,
   invalidation by filter. Next: optimizations (sliding TTL, per-entry hit counts, ...).
2. **LangChain and LlamaIndex adapters** — vector store and semantic cache.
3. **Approximate index** — spike IVF trained with e-means vs HNSW, then implement the winner.
4. **Server-side embedding** — send text, the server computes vectors (Ollama /
   OpenAI-compatible).
5. **Slab compaction** so `--maxmemory` bounds real memory ([TD-1](TECH_DEBT.md)).

Later: HTTP/REST API · batch insert and delete by filter · metadata indexes · quantization
(int8, float16) · parallel search and SIMD · AUTH/TLS.

## License

[Apache-2.0](LICENSE)

# Lawliet — Design Document

> In-memory vector embedding cache, speaking the Redis protocol (RESP).
> Status: **Draft v0** · License: **Apache-2.0** · Language: **Go**

---

## 1. Overview

Lawliet is a lightweight, single-binary, in-memory vector database designed for
**fast testing, prototyping and demoing RAG / LLM applications**.

Heavy vector databases (e.g. Milvus) need object storage, segment management,
lazy loading, etc. Lawliet takes the opposite approach: keep everything in RAM,
speak RESP so any existing Redis client works, and start with one command.

```bash
lawliet --port 6380
```

```python
import redis
r = redis.Redis(port=6380)
r.execute_command("VCREATE", "docs", "DIM", 1536, "METRIC", "cosine")
```

### 1.1 Goals

- Zero-config, single binary, starts in milliseconds.
- RESP2 + RESP3 wire compatibility: usable from `redis-cli`, `redis-py`, `ioredis`, `go-redis`, ... via raw commands,
  with each client's default protocol settings.
- Exact (Flat) vector search with metadata filtering.
- Cache-oriented features: per-record TTL, memory limits, per-collection eviction policies.
- Optional snapshot persistence to a single local file.

### 1.2 Non-goals (MVP)

- Distributed mode, replication, clustering.
- Datasets larger than RAM.
- Object storage, segments, WAL/AOF.
- Authentication / TLS (bind to `127.0.0.1` by default).
- HTTP/REST API (planned later).

### 1.3 Positioning

Lawliet is a **semantic cache and vector store for LLM applications**, not a general-purpose
vector database. It does not try to beat Chroma, FAISS or RediSearch on scale or recall; it
aims to be the simplest way to put a semantic cache and a small document store next to an
LLM app.

Differentiators to build on:

- **Semantic cache commands** (`VCACHE.*`, §5.7): threshold lookup in one round trip,
  default TTL, server-side hit/miss statistics, invalidation by metadata filter — usable
  from any language through a plain Redis client.
- Per-collection eviction policies. (Redis can approximate "documents vs. cache" with a
  `volatile-*` policy when only cache keys have a TTL; Lawliet lets each collection pick its
  own policy.)
- One command stores vector + text + metadata + TTL (`VADD`), no schema to declare.
- Single Apache-2.0 binary, used through existing Redis clients.
- Planned (§16): LangChain / LlamaIndex adapters, server-side embedding.

Known gaps vs. those projects: no approximate index yet, no SIMD, no ecosystem integrations.

---

## 2. Architecture

```
┌──────────────────────────────────────────────┐
│ TCP listener (net)  — 1 goroutine / conn     │
├──────────────────────────────────────────────┤
│ RESP2/RESP3 reader / writer                  │
├──────────────────────────────────────────────┤
│ Command dispatcher  (V*, SAVE, courtesy cmds)│
├──────────────────────────────────────────────┤
│ Store                                        │
│  ├─ Collection (RWMutex each)                │
│  │   ├─ VectorSlab (chunked []float32)       │
│  │   ├─ ID map, text, metadata, TTL, LRU     │
│  │   └─ Index (interface) → FlatIndex        │
│  ├─ MemTracker (estimated bytes)             │
│  └─ Evictor / Expirer                        │
├──────────────────────────────────────────────┤
│ Snapshotter (optional, --dir)                │
└──────────────────────────────────────────────┘
```

### 2.1 Repository layout

```
lawliet/
├── cmd/lawliet/main.go        # flags, wiring, signal handling
├── internal/
│   ├── config/                # flag parsing, size parsing ("2gb")
│   ├── resp/                  # RESP parser + writer (RESP2/RESP3)
│   ├── server/                # TCP server, connection loop, dispatch
│   ├── command/               # one file per command group
│   ├── store/                 # Store, Collection, slab, memtracker, eviction, ttl
│   ├── index/                 # Index interface, flat
│   ├── vecmath/               # dot, l2, normalize
│   ├── filter/                # lexer, parser, evaluator
│   ├── snapshot/              # .lwdb writer / reader
│   └── server/servertest/     # test server + RESP client for integration tests
├── python/                    # pip package: bundles the binary, start()/stop()/Server
│   └── hatch_build.py         # build hook: binary into wheel, platform tag per GOOS/GOARCH
├── examples/python/           # RAG demo (no model download needed)
├── scripts/e2e/               # end-to-end checks; run_all.sh runs them all
├── scripts/bench/             # end-to-end benchmark
├── .github/workflows/ci.yml   # gofmt, vet, race tests, e2e, docker build + smoke test
├── .github/workflows/release.yml # on tag v*: test, build 5 OS/arch binaries + Python wheels,
│                              #   GitHub Release; PyPI when vars.PYPI_PUBLISH == 'true'
├── Dockerfile                 # distroless, non-root, listens on 0.0.0.0 in the container
├── DESIGN.md
├── README.md
└── LICENSE
```

---

## 3. Data Model

```
Server
 └── Collection  (name, dim, metric, index, max_vectors, policy)
      └── Record
           ├── ID        string, user-defined, upsert on duplicate
           ├── Vector    []float32 of length dim
           ├── Text      string (optional, RAG chunk content)
           ├── Metadata  flat typed key/value (optional)
           └── ExpireAt  unix ms, 0 = no TTL
```

No Redis-style numbered databases. `SELECT 0` returns `OK`; other indexes return an error.

### 3.1 Collection config

| Field         | Values                                       | Mutable |
|---------------|----------------------------------------------|---------|
| `name`        | string, unique                               | no      |
| `dim`         | int > 0                                      | no      |
| `metric`      | `cosine` (default) \| `l2` \| `ip`           | no      |
| `index`       | `flat` (default) \| `hnsw` \| `ivf` + options (§7) | no      |
| `max_vectors` | int, 0 = unlimited                           | yes     |
| `policy`      | `noeviction` \| `allkeys-lru` \| `volatile-ttl` (default from server flag) | yes |
| `cache_threshold` | float, unset by default; default `THRESHOLD` for `VCACHE.GET` | yes |
| `cache_ttl`   | seconds, 0 = none; default TTL for `VCACHE.SET`             | yes     |

Lowering `max_vectors` below the current count **only blocks new inserts**; existing records are kept.

### 3.2 IDs and slots

- External ID: string chosen by the user (e.g. `doc:42`).
- Internal slot: `uint32`, index into the vector slab and parallel arrays.
- Deleted slots become tombstones and are pushed to a free list for reuse.

### 3.3 Vector storage

- Vectors are stored in a **chunked slab**: `[][]float32`, each chunk holds `chunkSize` (1024) vectors.
- Chunks contain no pointers → the Go GC does not scan their contents.
- Growth allocates a new chunk; existing data is never copied.
- For `cosine`, vectors are **L2-normalized on insert**. Returned vectors (`WITHVEC`) are the normalized ones.

### 3.4 Metadata

Passed as a **flat JSON object**. Types are inferred:

| JSON type          | Internal kind | Filter ops                      |
|--------------------|---------------|---------------------------------|
| string             | string        | `== != in`                      |
| integer number     | int64         | `== != < <= > >=`               |
| non-integer number | float64       | `== != < <= > >=`               |
| bool               | bool          | `== !=`                         |
| array of strings   | []string      | `contains`                      |

Nested objects, `null`, arrays of non-strings and empty field names are rejected with
`-ERR invalid metadata`. Integers that do not fit in int64 become float64.
An empty object `{}` is stored as "no metadata".

Metadata is returned as a JSON object string with **sorted keys**. Floats always keep a
decimal point or exponent (`3.0`, not `3`), so they parse back as float.

A stored metadata map is immutable: an update replaces the whole map, so readers can share it
without copying.

### 3.5 Go structures (sketch)

```go
type Collection struct {
    mu  sync.RWMutex
    cfg CollectionConfig

    vectors    VectorSlab
    idToSlot   map[string]uint32
    slotToID   []string
    texts      []string
    meta       []map[string]Value
    expireAt   []int64
    lastAccess []atomic.Int64
    deleted    []bool
    freeList   []uint32
    ttlSlots   map[uint32]struct{} // slots having a TTL, for active expiry / volatile-ttl
    liveCount  int

    index Index
}

type Index interface {
    Add(slot uint32, vec []float32)
    Remove(slot uint32)
    Search(q []float32, k int, accept func(slot uint32) bool) []Hit
}
```

---

## 4. Wire Format

### 4.1 Protocol

RESP2 by default; a connection switches to RESP3 with `HELLO 3` (and back with `HELLO 2`).

Why RESP3 is required: redis-py 8.1.0 defaults to RESP3, sends `HELLO 3` on connect,
does not fall back to RESP2 on `-NOPROTO`, and checks `proto == 3` in the reply
(verified against redis-py 8.1.0 source).

The protocol is stored per connection in the reply writer. Command handlers use
version-neutral writer methods; only these types are encoded differently:

| Type   | RESP2                     | RESP3      |
|--------|---------------------------|------------|
| null   | `$-1` / `*-1`             | `_`        |
| map    | flat array `*2n`          | `%n`       |
| double | bulk string               | `,<float>` |

Not implemented (not needed yet): sets, pushes, attributes, big numbers, verbatim strings,
client-side caching (`CLIENT TRACKING`).

### 4.2 Vector encoding

Two equivalent forms, accepted wherever a vector is expected:

| Form                          | Use                    |
|-------------------------------|------------------------|
| `FP32 <blob>`                 | Code. `dim × 4` bytes, IEEE-754 float32, **little-endian** |
| `VALUES <dim> v1 v2 ... vn`   | Manual testing with `redis-cli` |

Blob length must be a positive multiple of 4 (`-ERR invalid FP32 blob`) and equal `dim × 4`
(`-ERR dimension mismatch`). Values must be finite (no NaN/Inf). With `cosine`, the zero vector
is rejected because it cannot be normalized.

Python client:

```python
blob = emb.astype("<f4").tobytes()   # float32, little-endian
```

### 4.3 Scores

Always **similarity, higher is better**:

| Metric   | Score returned            |
|----------|---------------------------|
| `cosine` | cosine similarity [-1, 1] |
| `ip`     | dot product               |
| `l2`     | `-euclidean_distance`     |

Scores are written with `WriteDouble`: RESP3 double, or bulk string in RESP2.

---

## 5. Commands

### 5.1 Collections

```
VCREATE <col> DIM <n> [METRIC cosine|l2|ip] [INDEX flat|hnsw|ivf]
        [M <n>] [EF_CONSTRUCTION <n>] [EF_SEARCH <n>]                  (hnsw)
        [NLIST <n>] [NPROBE <n>]                                       (ivf)
        [QUANT none|sq8|pq] [PQ_M <n>] [RERANK <n>]                    (flat, ivf)
        [MAX_VECTORS <n>] [POLICY noeviction|allkeys-lru|volatile-ttl]
        [CACHE_THRESHOLD <score>] [CACHE_TTL <sec>]                     → +OK
VDROP   <col>                                                           → :1 | :0
VLIST                                                                   → array of names
VINFO   <col>                                                           → flat key/value array
VCONFIG <col> GET                                                       → flat key/value array
VCONFIG <col> SET POLICY <p>                                            → +OK
VCONFIG <col> SET MAX_VECTORS <n>                                       → +OK
VCONFIG <col> SET CACHE_THRESHOLD <score|NONE>                          → +OK
VCONFIG <col> SET CACHE_TTL <sec>                                       → +OK
```

`VINFO` reply is a map (flat array in RESP2). Fields: `name, dim, metric, index, count, expires,
max_vectors, policy, cache_threshold` (null if unset), `cache_ttl, memory_bytes` (estimated,
§8.2), `vector_bytes` (allocated slab), then `index_params, index_state, index_bytes` (§7.1; added
last so positional RESP2 parsers are not affected). `VCONFIG GET` returns `max_vectors, policy,
cache_threshold, cache_ttl`.

`VCONFIG <col> SET` accepts several `param value` pairs; all are validated before any change is
applied. Unknown or immutable parameters (`DIM`, `METRIC`, `INDEX`) are rejected.
`CACHE_THRESHOLD NONE` clears the default threshold.

### 5.2 Records

```
VADD <col> <id> (FP32 <blob> | VALUES <dim> v...)
     [TEXT <str>] [META <json>] [EX <sec> | PX <ms>] [NX | XX]   → :1 inserted | :0 updated | nil (NX/XX not met)
VGET    <col> <id> [WITHVEC]       → [text, meta_json, (vector blob)] | nil   (text/meta are null until M2)
VDEL    <col> <id> [<id> ...]      → :<deleted count>
VEXISTS <col> <id>                 → :1 | :0
VCOUNT  <col>                      → :<n>
```

- No flag: upsert. `NX`: only insert if absent. `XX`: only update if present.
- An update **replaces the whole record**: `TEXT`/`META` not given are cleared, and an update
  without `EX/PX` removes an existing TTL (same as Redis `SET`).
- In `VGET`/`VSEARCH` replies, empty text and missing metadata are null.

### 5.3 TTL

```
VEXPIRE  <col> <id> <sec>   → :1 | :0
VTTL     <col> <id>         → seconds | -1 (no TTL) | -2 (not found)
VPERSIST <col> <id>         → :1 | :0
```

- `EX`/`PX` on `VADD` must be positive; TTLs are capped at ~100 years (`-ERR invalid expire time`).
  `EX` and `PX` together → syntax error.
- `VEXPIRE` with a value ≤ 0 deletes the record immediately and returns 1 (same as Redis `EXPIRE`).
- `VTTL` rounds the remaining time to the nearest second, like Redis `TTL`.
- `VPERSIST` returns 0 if the record has no TTL.
- An expired record behaves as missing for every command (`NX` inserts, `XX` skips,
  `VDEL` counts 0, `VEXPIRE`/`VPERSIST` return 0).

### 5.4 Search

```
VSEARCH <col> (FP32 <blob> | VALUES <dim> v...)
        [K <n>] [FILTER <expr>] [WITHTEXT] [WITHMETA] [WITHVEC]
        [EF <n>] [NPROBE <n>] [RERANK <n>]
```

`EF` (1–4096), `NPROBE` (1–65536) and `RERANK` (1–100) override the index search options for
this query (§7); they are ignored by indexes they do not apply to.

`K` defaults to 10, range 1–10000. Reply: array of hits, best first. Each hit is an array:

```
[id, score, (text), (meta_json), (vector blob)]
```

Optional parts always appear in this fixed order, whatever the order of the flags.
`FILTER` is applied **before** ranking: the reply holds the best K records among those
matching the filter (fewer if fewer match).

Records returned in the top-K have their `lastAccess` updated.

### 5.5 Persistence

```
SAVE   → +OK | -ERR snapshot disabled: start server with --dir
```

Synchronous. Serialized with automatic snapshots via a mutex.

### 5.6 Courtesy commands (client compatibility)

| Command                           | Behavior                               |
|-----------------------------------|----------------------------------------|
| `PING [msg]`                      | `+PONG` or echo                        |
| `ECHO msg`                        | echo                                   |
| `HELLO [2\|3] [SETNAME n]`        | switch protocol; map with server, version, proto, id, mode, role, modules; other versions → `-NOPROTO` |
| `SELECT 0`                        | `+OK`; other → error                   |
| `CLIENT SETNAME/SETINFO/GETNAME`  | `+OK` / name                           |
| `COMMAND`, `COMMAND DOCS`         | empty array                            |
| `INFO [section]`                  | `server`, `memory`, `stats`, `persistence` |
| `QUIT`                            | `+OK`, close connection                |

### 5.7 Semantic cache (`VCACHE.*`)

A semantic cache is an ordinary collection; `allkeys-lru` or `volatile-ttl` with
`MAX_VECTORS` is recommended. The commands add a threshold lookup, a default TTL,
statistics and invalidation.

```
VCACHE.GET <col> (FP32 <blob> | VALUES <n> v...) [THRESHOLD <s>] [FILTER <expr>] [WITHMETA]
    → hit:  [id, score, text, (meta_json)]   best entry with score >= threshold
    → miss: null
VCACHE.SET <col> <id> (FP32 <blob> | VALUES <n> v...) TEXT <answer>
           [META <json>] [EX <sec> | PX <ms>] [NX | XX]
    → :1 inserted | :0 updated | null (NX/XX not met)
VCACHE.STATS <col>
    → map: hits, misses, hit_rate, avg_hit_score, entries
VCACHE.INVALIDATE <col> FILTER <expr>
    → number of entries removed
```

- **Threshold**: `THRESHOLD` on the call, else the collection's `CACHE_THRESHOLD`; if neither
  is set the call fails. There is no implicit default because the right value depends on
  the embedding model, and a wrong one serves wrong answers. Scores follow §4.3 (higher is
  better).
- **TTL**: `VCACHE.SET` uses `EX`/`PX` if given, else `CACHE_TTL` (0 = no TTL). `TEXT` is
  required. Otherwise it behaves like `VADD` (update replaces the whole record).
- **LRU**: only a hit counts as an access. A miss does not touch the nearest entry; otherwise
  entries that are often "almost" matched would never be evicted.
- **Scoping**: `FILTER` restricts lookups, e.g. `model == "m1" and prompt_version == 3`, so
  answers produced under another model or prompt are never served.
- **Statistics** are per collection, in memory only: they reset on restart and are not in
  snapshots. `hit_rate` and `avg_hit_score` are 0 before the first lookup / hit.
- **Invalidation** removes matching entries in one command under the collection write lock;
  expired entries met on the way are reclaimed but not counted.

Measured (scripts/e2e/vcache.py, dim 384, 200 entries, threshold 0.95): paraphrased queries
hit the right entry 100/100, unrelated queries miss 100/100, `VCACHE.GET` p50 0.14 ms.

---

## 6. Filter Language

### 6.1 Grammar (EBNF)

```
expr       = or_expr ;
or_expr    = and_expr { "or" and_expr } ;
and_expr   = not_expr { "and" not_expr } ;
not_expr   = [ "not" ] primary ;
primary    = "(" expr ")" | comparison ;
comparison = IDENT op literal
           | IDENT "in" "[" literal { "," literal } "]"
           | IDENT "contains" STRING ;
op         = "==" | "!=" | "<" | "<=" | ">" | ">=" ;
literal    = STRING | NUMBER | "true" | "false" ;
STRING     = '"' { char } '"' ;
```

Lexical rules:

- Keywords (`and or not in contains true false`) are case-insensitive.
- `IDENT` = `[A-Za-z_][A-Za-z0-9_.]*` (e.g. `user.id`). Field names with other characters
  cannot be filtered on yet.
- Strings use double or single quotes, so filters are easy to pass from a shell:
  `FILTER 'source == "hr.md"'`. Escapes: `\" \' \\ \n \t \r`.
- `NUMBER` = `[+-]digits[.digits][e[+-]digits]`. Without `.`/`e` it is an int64, else a float64.
- `not` may be repeated (`not not x == 1`).

Precedence: `not` > `and` > `or`.

Limits: 16 KiB per expression, nesting depth 64, 1024 values per `in` list.

### 6.2 Semantics

- Missing field → comparison is `false`, for `!=` too (so `not field == x` is `true`,
  but `field != x` is `false` when the field is missing).
- Type mismatch at evaluation (e.g. `page == "3"` when `page` is int) → `false`, not an error.
- Type errors visible at parse time are errors: `<`, `<=`, `>`, `>=` need a number literal.
- int and float compare numerically (`page == 3.0` matches `3`).
- `contains` only matches string-list fields; `in` only matches scalar fields.
- Parse errors report the byte position: `-ERR invalid filter: <reason> at position <n>`.
- Parsed once per `VSEARCH`, evaluated per record (brute force) **before** distance computation.

Example:

```
(source == "hr.md" or source in ["it.md", "ops.md"]) and page >= 2 and not tags contains "draft"
```

---

## 7. Search and Indexes

All indexes sit behind `index.Index` (`internal/index`), read full vectors from the collection
slab, and see deletions only through the `accept` function (tombstones, expired records,
`FILTER`). Scores are always exact: quantized candidates are re-scored with the stored vectors.

### 7.1 Kinds and options

| `INDEX` | Faiss equivalent | Options (default) |
|---|---|---|
| `flat` | `IndexFlat` | — |
| `flat QUANT sq8` | `IndexScalarQuantizer` (8 bit) + refine | `RERANK` (1) |
| `flat QUANT pq` | `IndexPQ` (8 bit codes) + refine | `PQ_M` (DIM/16, must divide DIM), `RERANK` (1) |
| `hnsw` | `IndexHNSWFlat` | `M` (16), `EF_CONSTRUCTION` (100), `EF_SEARCH` (64) |
| `ivf` | `IndexIVFFlat` | `NLIST` (auto ≈ √n), `NPROBE` (auto = NLIST/16) |
| `ivf QUANT sq8` | `IndexIVFScalarQuantizer` (non-residual) + refine | + `RERANK` |
| `ivf QUANT pq` | `IndexIVFPQ` (residual) + refine | + `PQ_M`, `RERANK` |

Options that do not apply to the kind are rejected by `VCREATE`. `VINFO` reports
`index_params` (effective options), `index_state` and `index_bytes` (memory on top of vectors,
not counted by `maxmemory`).

Distances (`internal/vecmath`) use AVX2 + FMA when built with `GOEXPERIMENT=simd` on amd64
and the CPU has them (`INFO server` → `vector_simd`); otherwise unrolled scalar loops.
`GOEXPERIMENT=simd` is an experimental toolchain flag, so it is **opt-in**: release binaries,
wheels and the Docker image are scalar by default; `make build SIMD=1`,
`docker build --build-arg SIMD=1` or `GOEXPERIMENT=simd` for the wheel hook build the AVX2
path. CI tests both builds. Each kernel ends with `VZEROUPPER`
(`archsimd.ClearAVXUpperBits`): without it Go's SSE scalar code after an AVX kernel pays the
AVX→SSE transition penalty (measured 3× slower overall). Results can differ from the scalar
path in the last bits.

### 7.2 Flat

- Iterate all live, non-expired slots; apply filter; compute score.
- Keep top-K with a **min-heap of size K**.
- Cosine on normalized vectors = dot product.

### 7.3 HNSW

- Graph per Malkov & Yashunin: random level per node (mL = 1/ln M), up to `M` links on upper
  levels and `2M` on level 0, neighbors chosen with the diversity heuristic.
- Inserts are incremental (each `VADD` under the collection write lock). A delete (or the
  removal half of an update) takes the node out and repairs the graph: each neighbor that
  linked to it re-selects its links with the heuristic among its other links and the removed
  node's links (as hnswlib does). Links from nodes the removed one did not link to are left
  dangling and skipped while searching; they are dropped the next time that list gets a new
  link (`connect`), so they neither occupy slots nor block live candidates in the heuristic.
- Search: greedy descent, then a beam of `max(EF, K)` on level 0. `FILTER` is applied to results
  while traversing; if fewer than K matches are found the query falls back to an exact scan.

### 7.4 IVF and quantizers

- Training needs data: the index is exact until the collection holds 1000 records (10,000 for
  PQ; 39 × `NLIST` if `NLIST` is set), then trains k-means (10 Lloyd iterations, ≤ 64 points per
  cell) and assigns every record. It retrains when the count doubles while the sample is not
  saturated (always with auto `NLIST`). Training runs inside the `VADD` that crosses the
  threshold: that call can take seconds (see §16.3).
- Cells store codes contiguously (slot ids; 1 byte/dim for sq8; `PQ_M` bytes of residual for pq).
- Search scans the `NPROBE` best cells; quantized scores pick `K × RERANK` candidates, re-scored
  exactly. Fewer than K matches (selective filter) → exact scan.
- Index structures are not persisted: snapshots store options (§10.4), indexes are rebuilt on load.

---

## 8. Memory Limits & Eviction

### 8.1 Limits

| Limit            | Scope      | Flag / option               |
|------------------|------------|-----------------------------|
| `maxmemory`      | server     | `--maxmemory 2gb` (0 = off) |
| `max-collections`| server     | `--max-collections <n>`     |
| `max_vectors`    | collection | `VCREATE ... MAX_VECTORS n` |

### 8.2 Memory accounting

Memory is **estimated**, not measured: each insert/delete adds/subtracts

```
dim*4 + len(id)*2 + len(text) + metadata (48 + Σ(88 + len(key) + len(value))) + 96 overhead
```

Upsert reserves only the delta.

**`maxmemory` limits the logical size of the records, not the process memory.**
Measured with dim 1536 and `--maxmemory 20mb` (scripts/e2e/m4_limits.py):

| Figure (`INFO memory`)       | Value    | What it is                                   |
|------------------------------|----------|----------------------------------------------|
| `used_memory_estimated`      | 20.0 MiB | the limited figure                           |
| `used_memory_vector_slab`    | 30.0 MiB | allocated slab chunks                        |
| `used_memory_heap_live`      | 31.1 MiB | Go heap in use after the last GC             |
| `used_memory_runtime`        | 57.5 MiB | all memory mapped by the Go runtime          |

Sources of the difference:

1. **Chunk granularity**: slabs grow by 1024 vectors (6 MiB at dim 1536) per collection.
2. **Freed slots stay allocated**: deleted, expired and evicted records leave slots that only
   the same collection reuses; chunks are never returned. Evicting a cache to make room for
   another collection frees *estimated* memory but not slab memory.
3. **Go GC headroom**: by default the heap may grow to about 2× the live heap between GCs.
   Users can set the standard `GOMEMLIMIT` environment variable to make the GC stricter.

Tracked as **TD-1** in [TECH_DEBT.md](TECH_DEBT.md): return empty chunks / compact slabs, then
optionally account allocated memory.

### 8.3 Policies (per collection)

| Policy          | Victim                                   |
|-----------------|------------------------------------------|
| `noeviction`    | none → error                             |
| `allkeys-lru`   | lowest `lastAccess` among N random samples |
| `volatile-ttl`  | lowest `expireAt` among N random samples with TTL |

`N = 5`. Approximate (sampling) by design, similar to Redis. Collections with ≤ 5 records
consider every record (exact). Expired records in a sample are always chosen first.
With 2000 records and 10% hot records, sampled LRU kept 198–200 of 200 hot records in tests.

"Access" = `VADD`, `VGET`, or being returned in `VSEARCH` top-K.

### 8.4 Flow on insert of a new record

```
VADD
 1. need = size(new record) − size(existing record, if any)
 2. Store.freeMemory(need)          (no collection lock held)
      need > maxmemory              → -OOM (nothing is evicted)
      while used + need > maxmemory:
        a. reclaim expired records in any collection (no data loss)
        b. pick a random collection whose policy != noeviction, evict one record
        nothing left to reclaim or evict → -OOM
 3. lock collection
 4. new record and count >= max_vectors:
        reclaim one expired record (sampled), else evict by policy,
        else -ERR max_vectors reached
 5. insert, unlock
```

Never call `freeMemory` while holding a collection lock (deadlock risk).
Another writer may use freed memory between steps 2 and 5, so the limit can be exceeded by a
few records under concurrency. Deletes, reads, `VEXPIRE`, `VPERSIST` and `VCONFIG` are always
allowed at the limit.

`INFO` also reports `maxmemory`, `maxmemory_policy` (default for new collections),
`evicted_keys` (stats) and a `keyspace` section with `collections` and `max_collections`.

---

## 9. TTL Expiry

TTLs are stored as absolute unix milliseconds per slot (`expireAt`); `ttlSlots` holds the
slots that have one.

- **Visibility**: a record is expired as soon as `now >= expireAt`. Every read (`VGET`,
  `VEXISTS`, `VTTL`, `VSEARCH`) and write treats it as missing immediately.
- **Lazy reclaim**: reads hold only the read lock, so they note expired records and delete
  them right after releasing it (re-checking expiry under the write lock).
- **Active reclaim**: a background goroutine runs every 100 ms. For each collection it
  samples up to 20 entries of `ttlSlots` (Go map iteration order is randomized, which gives
  a cheap sample), deletes the expired ones, and repeats while > 25% of the sample was
  expired. Each cycle may use at most 25 ms. A small residue of expired records can remain
  after one cycle; later cycles or reads reclaim it.
- **Counts**: `VCOUNT` / `VINFO count` include expired records not yet reclaimed, like
  Redis `DBSIZE`. The active cycle keeps the difference small.
- The clock is injectable (`Store.SetClock`) so tests do not sleep.

---

## 10. Persistence (Snapshot)

Disabled unless `--dir` is given. `--dir` must be an existing directory; `--dbfilename`
(default `dump.lwdb`) must be a plain file name.

### 10.1 Triggers

- Save rules: `--save "<seconds> <changes>"` (repeatable). Defaults when `--dir` is set:
  `3600 1`, `300 100`, `60 10000`. `--save ""` disables auto-save. Rules are checked every
  second; after a failed save the next attempt waits at least 5 s.
- `SAVE` command (synchronous; one save at a time).
- Graceful shutdown (`SIGINT`, `SIGTERM`): the server first stops accepting and closes all
  connections, **then** saves, so no write can land after the final snapshot.
- Load on startup if the file exists, before the server listens. Limits (`maxmemory`,
  `max_vectors`, `max-collections`) are not enforced while loading, like Redis loading an RDB.
  Records that expired while the server was down are skipped.
- A file that fails to load (bad magic, version, checksum, truncation) stops the server with
  exit code 1 and a log message; it is never partially loaded into a running server.

`dirty` counter increments on every mutation; after a successful snapshot it is decreased
by the value captured at snapshot start (mutations during the snapshot are kept).

### 10.2 Consistency

Collections are snapshotted one at a time, each under its **read lock**:
reads continue, writes to that collection wait. (Go cannot safely `fork()` like Redis.)
Each collection is consistent; the snapshot is not atomic across collections.

Measured: 10 000 records × 384 dims with text and metadata → 15.4 MiB written in ~60 ms.

### 10.3 Safe write

```
write  <dir>/dump.lwdb.tmp
fsync
rename dump.lwdb.tmp → dump.lwdb   (atomic on POSIX, same filesystem)
```

### 10.4 File format `.lwdb` (little-endian)

```
magic        "LWDB"           4B
version      uint16
n_coll       uint32
repeat n_coll:
  name       str
  dim        uint32
  metric     uint8
  index      uint8            0 flat, 1 hnsw, 2 ivf
  max_vec    uint32
  policy     uint8
  [v2+] has_cache_threshold uint8
  [v2+] cache_threshold     float64
  [v2+] cache_ttl_sec       int64
  [v3+] quant               uint8
  [v3+] m, ef_construction, ef_search, nlist, nprobe, pq_m, rerank   uint32 each
  n_rec      uint32
  repeat n_rec:
    id       str
    vector   dim × float32
    text     str
    meta     str (JSON)
    expireAt int64
crc32        uint32 (IEEE, over everything above)

str = uint32 length + bytes
```

Tombstones and expired records are skipped (natural compaction).
`lastAccess` is not persisted; set to load time.
Vectors are written in their stored form and restored **bit-exact**: cosine vectors are not
normalized again on load (re-normalizing changes the float bits of ~50% of vectors and thus
the scores).
Bad magic / version / CRC → refuse to start with a clear error.

Versions: **1** = v0.1.0; **2** = adds the cache defaults per collection; **3** = adds index
kinds and options (index structures are rebuilt on load). The server writes the lowest
version that holds the data — v2 while every collection is `flat` with default options, so
a v0.2.x binary can still read it; v3 otherwise — and reads every version up to the current
one (a v1 file loads with no cache defaults; `internal/snapshot/testdata/v1.lwdb` guards
this).

`INFO persistence`: `snapshot_enabled`, `snapshot_path`, `rdb_changes_since_last_save`,
`rdb_last_save_time`, `rdb_last_save_status`, `rdb_last_save_duration_ms`.

---

## 11. Concurrency

- One goroutine per connection.
- `Store.mu` (RWMutex) guards the collection map.
- `Collection.mu` (RWMutex): `VSEARCH`/`VGET` take read lock; mutations take write lock.
- `lastAccess` uses atomics so readers can update it under the read lock.
- `MemTracker` uses atomic CAS.
- Lock order: `Store.mu` → `Collection.mu`. Never hold two collection locks at once.

---

## 12. Server Flags

| Flag                  | Default        |
|-----------------------|----------------|
| `--bind`              | `127.0.0.1`    |
| `--port`              | `6380`         |
| `--maxmemory`         | `0` (unlimited)|
| `--maxmemory-policy`  | `noeviction` (default for new collections) |
| `--max-collections`   | `0` (unlimited)|
| `--dir`               | empty → persistence off |
| `--dbfilename`        | `dump.lwdb`    |
| `--save`              | see §10.1      |
| `--exit-on-stdin-eof` | off: when set, graceful shutdown once stdin reaches EOF (parent process gone; used by `python/`) |

---

## 13. Errors

| Error                                                        | When                          |
|--------------------------------------------------------------|-------------------------------|
| `-ERR unknown command '<cmd>'`                               | unknown command               |
| `-ERR wrong number of arguments for '<cmd>' command`         | arity                         |
| `-ERR syntax error`                                          | bad option                    |
| `-ERR collection not found`                                  | missing collection            |
| `-ERR collection already exists`                             | duplicate `VCREATE`           |
| `-ERR max collections reached`                               | `max-collections` hit         |
| `-ERR dimension mismatch: expected <d>, got <n>`             | wrong vector length           |
| `-ERR invalid FP32 blob: length <n> is not a positive multiple of 4` | malformed blob        |
| `-ERR VALUES <n> expects <n> numbers, got <m>`               | too few VALUES                |
| `-ERR invalid number in VALUES: '<x>'`                       | non-numeric value             |
| `-ERR zero vector is not allowed with cosine metric`         | zero vector, cosine           |
| `-ERR vector contains NaN or Inf`                            | non-finite value              |
| `-ERR K must be an integer between 1 and 10000`              | bad `K`                       |
| `-ERR invalid metadata: <reason>`                            | bad `META` JSON               |
| `-ERR invalid expire time`                                   | bad `EX`/`PX`/`VEXPIRE` value |
| `-ERR invalid filter: <reason> at position <n>`              | bad `FILTER` expression       |
| `-ERR max_vectors reached`                                   | limit hit, nothing evictable  |
| `-OOM command not allowed when used memory > 'maxmemory'`    | memory limit, cannot evict    |
| `-ERR snapshot disabled: start server with --dir`            | `SAVE` without `--dir`        |
| `-ERR snapshot failed: <reason>`                             | `SAVE` could not write        |
| `-ERR no threshold: pass THRESHOLD or set CACHE_THRESHOLD on the collection` | `VCACHE.GET` without a threshold |
| `-ERR VCACHE.SET requires a non-empty TEXT (the cached answer)` | `VCACHE.SET` without `TEXT` |
| `-ERR THRESHOLD must be a finite number` (also `CACHE_THRESHOLD`) | bad threshold value     |
| `-NOPROTO unsupported protocol version`                      | `HELLO` with version other than 2/3 |

---

## 14. Roadmap

### MVP

| Milestone | Scope | Done when |
|-----------|-------|-----------|
| **M0** Skeleton ✅ | RESP2/RESP3 parser/writer, TCP server, courtesy commands | `redis-cli -p 6380 ping` → `PONG`; `redis-py` connects with default settings |
| **M1** Core vectors ✅ | Collections, slab, `VADD/VGET/VDEL/VEXISTS/VCOUNT`, `VALUES`/`FP32`, Flat `VSEARCH` (no filter) | Python script inserts 10k vectors and gets correct top-K vs NumPy |
| **M2** Metadata & filter ✅ | `TEXT`, `META`, filter lexer/parser/evaluator, `WITH*` flags | Filter unit tests incl. `and/or/not/()`, `in`, `contains` |
| **M3** TTL ✅ | `EX/PX`, `VEXPIRE/VTTL/VPERSIST`, lazy + active expiry | Records disappear after TTL; `VTTL` matches Redis semantics |
| **M4** Limits & eviction ✅ | MemTracker, `maxmemory`, `max_vectors`, 3 policies, `VCONFIG`, `VINFO`, `INFO` | Eviction tests per policy; `noeviction` returns errors |
| **M5** Snapshot ✅ | `.lwdb` writer/reader, save rules, `SAVE`, shutdown save, startup load | Restart restores identical search results; corrupt file detected |
| **M6** Release ✅ | README quickstart, Python RAG example, Dockerfile, GitHub Actions CI, benchmark script | `docker run` + example works end-to-end |

### Phase 2 (after MVP)

Ordered by priority (agreed 2026-11-15). Items 1–4 have draft designs in §16; each is
finalized step by step before implementation, like the MVP commands.

| # | Item | Why this order |
|---|------|----------------|
| 1 ✅ | `VCACHE.*` semantic cache commands (§5.7; draft §16.1) | the main differentiator; builds on TTL + eviction |
| 2 | LangChain & LlamaIndex adapters (§16.2) | how users discover and try Lawliet |
| 3 | Approximate index: ✅ spike done — hnsw / ivf / sq8 / pq implemented and benchmarked (§16.3); next: TD-2, TD-3, SIMD | needed beyond ~100k vectors per collection |
| 4 | Server-side embedding on `VADD` / `VSEARCH` (§16.4) | demos without embedding code |
| 5 | Return empty slab chunks / compact slabs (TD-1, §8.2) | makes `maxmemory` bound real memory |

Later, unordered:

- HTTP/REST API
- Batch insert command; delete by filter
- Metadata secondary index for faster filtering
- Quantization (int8 / binary), float16
- Parallel Flat search, SIMD distance
- AUTH / TLS
- Chunked/incremental snapshot to reduce write blocking

---

## 15. Project Info

- Go module path: `github.com/tranbaohieu/lawliet`
- Default port: `6380`

---

## 16. Phase 2 Proposals (draft)

Status: **draft**. Names, syntax and defaults below are proposals to discuss, not decisions.
Open questions are listed per item.

### 16.1 Semantic cache commands (`VCACHE.*`) — implemented

Implemented as specified in §5.7. Decisions on the open questions of the draft:

| Question | Decision |
|---|---|
| Default threshold | none: `THRESHOLD` or `CACHE_THRESHOLD` is required |
| Refresh TTL on hit (sliding expiration) | not yet — candidate optimization |
| `VCACHE.INVALIDATE` vs. general `VDEL ... FILTER` | cache command for now, built on `Collection.DeleteWhere`, reusable for `VDEL ... FILTER` |
| Stats persistence | in memory only, reset on restart |
| Does a lookup count as an access? | only hits (`SearchOptions.NoTouch` + `Touch`) |

Optimization ideas (next, to validate against real RAG usage):

- Sliding expiration: optional `TOUCH` on hit to extend the TTL.
- Per-entry hit counters (most asked questions; cache warming).
- Stale-while-revalidate: serve an entry close to expiry and tell the client to refresh it.
- Threshold tuning help: score distribution of hits and near-misses in `VCACHE.STATS`.

### 16.2 LangChain & LlamaIndex adapters

A Python package (working name `lawliet-py`, in `clients/python/` of this repo or a separate
repo) providing:

- **LangChain**: a vector store (add texts with metadata, similarity search with scores,
  delete, metadata filter dict → `FILTER` expression) and a semantic cache backed by
  `VCACHE.*`.
- **LlamaIndex**: a vector store integration with the same mapping.
- A thin client (`Lawliet(...)`) wrapping raw commands with NumPy-friendly helpers
  (float32 little-endian encoding, RESP2/RESP3 reply normalization).

Open questions:

- Package name, location and PyPI publishing; supported LangChain / LlamaIndex versions.
- Interface names to implement must be checked against the current LangChain / LlamaIndex
  docs before starting (not verified here).
- Mapping rules from their filter syntax to Lawliet `FILTER` (unsupported operators → error).

### 16.3 Approximate index: IVF + e-means vs HNSW

**Status (2026-11-16): spike done, wider than planned.** Implemented in pure Go behind `Index`
(§7): HNSW, IVF-Flat (Lloyd k-means; e-means added later, see below), IVF-SQ8, IVF-PQ, flat SQ8/PQ. Benchmarked
against Faiss 1.16 on 100k × 1024 real embeddings:
[docs/benchmarks/ann-2026-11-16](docs/benchmarks/ann-2026-11-16/README.md).

Results in short: recall equals Faiss for every index; Faiss is 1.7–3× faster per query (SIMD;
6× for SQ8); HNSW ef=64 gives recall 0.979 at 0.6 ms (flat: 31 ms) without training pauses;
IVF/PQ training blocks one `VADD` for 4–13 s; HNSW recall drops 0.979 → 0.954 after replacing
30% of records three times, IVF does not change.

Update (same day): AVX2 kernels and HNSW delete repair. HNSW ef=64 p50 0.60 → 0.44 ms (Faiss
0.33), build 108 → 64 s, recall after churn 0.954 → 0.971; IVF-SQ8 4.4× faster; IVF training
pause 3.8 s → 0.9 s. Gap to Faiss is now 1.1–1.7× except PQ (2–3×).

Recommendation: **HNSW as the ANN index for Lawliet** (default parameters M=16, EF_SEARCH 64),
with one follow-up before calling it production-ready — closing the remaining churn loss
(TD-3: refill pruned links or rebuild in the background). Keep IVF for batch-loaded, churn-heavy collections
once training moves off the write path (TD-2). PQ is not worth it while full vectors are kept.

Update (2026-11-23): **e-means implemented, not the default.** `emeans()` in
`internal/index/kmeans.go` (port of the reference: decayed counts/sums, batch size and step size
from the scaling laws, cosine schedule, revival every 5 batches, last-epoch means). IVF training
goes through the package variables `trainCentroids` (default `lloyd`) and `trainPerCell`
(default 64 points per cell); nothing is user-facing and the snapshot format is unchanged.
Small comparison (`TestTrainersCompare`, `BenchmarkTrain`, Apple M4, synthetic clustered data):

| dim × nlist, sample | Lloyd 10 iters | e-means 5 epochs |
|---|---|---|
| 128 × 256, 16k points: train time / loss | 304 ms / 86.7 | 155 ms / 84.9 |
| 128 × 256, 65k points: train time / loss | 1.11 s / 87.7 | 0.63 s / 84.5 |
| 32 × 64, 4k points, IVF recall@10 nprobe=2 (l2 / cosine) | 0.663 / 0.686 | 0.692 / 0.728 |
| same, e-means on a 4× sample (16k points) | | 0.721 / 0.750 |

Before switching the default (and possibly raising `trainPerCell`), measure on real embeddings
at 100k–1M (`scripts/bench/ann`): recall@10 at fixed nprobe, training pause, recall after churn.
Larger-scale runs are open to contributors.

The original plan, kept for reference:

The flat index takes ~54 ms per query at 100k × 1536 (README). An approximate index is
needed beyond ~100k vectors per collection. Two candidates, behind the existing `Index`
interface (§3.5):

| | HNSW | IVF-Flat trained with e-means |
|---|---|---|
| Insert | insert into a graph | assign to nearest centroid (cheap) |
| Delete (TTL, eviction) | tombstones; the graph degrades and needs rebuilds | remove from a posting list (cheap) |
| Extra memory | graph edges | centroids + posting lists |
| Weak point | churn-heavy workloads | centroids drift as data changes → periodic retraining |

Why IVF is considered: a semantic cache has heavy churn (TTL, eviction). e-means (FAIR at
Meta, arXiv 2611.09501, code in Faiss `contrib/e_means`) reports reaching the loss of 25
Lloyd k-means iterations in 1.4–2.6 epochs, which would make periodic background retraining
cheap. Its core is simple to port to Go: per cluster a decayed count `N_j` and sum `S_j`
updated per mini-batch (`N ← γN + (1−γ)n`, `S ← γS + (1−γ)s`, centroid `= S/N`), a cosine
step-size schedule, and revival of clusters whose count falls below 10% of the mean.
Caveat: its reported gains are measured at 11M–11B vectors and mostly on GPU; gains at
Lawliet's scale (CPU, ~100k per collection) are unknown.

Plan: a **spike** before committing to one design.

1. Implement IVF-Flat (e-means training in Go) and a minimal HNSW behind `Index`.
2. Measure on 100k–1M vectors, dim 384 and 1536: recall@10, p50/p99 latency, build /
   retrain time, memory, and recall after replacing 30% of records (cache churn).
3. Pick by the numbers; record the decision and results here.

Open questions:

- `Index` interface additions: `Train` / `Rebuild` hooks, background retraining trigger
  (e.g. % of records changed since last training), snapshot of the index vs rebuild on load.
- User-facing parameters: `INDEX ivf NLIST n NPROBE p` / `INDEX hnsw M m EF ef`, and
  per-query overrides in `VSEARCH`.
- Filtered search with ANN (pre-filter vs post-filter, fallback to flat for selective filters).

### 16.4 Server-side embedding

Let the server compute embeddings so clients can send text only:

```
lawliet --embed-provider ollama --embed-url http://localhost:11434 --embed-model nomic-embed-text
        # API keys for hosted providers come from environment variables, never from flags

VCREATE docs DIM 768 EMBED default                  # collection uses the server embedder
VADD    docs doc:1 TEXT "Employees get 12 days of paid leave."   # no vector: server embeds TEXT
VSEARCH docs QUERY "how many leave days?" K 5        # QUERY instead of FP32/VALUES
VCACHE.GET answers QUERY "when is salary paid?"
```

Concerns:

- Latency and failures of a network call inside a command: timeouts, retries, error replies.
- Pipelines: batch embedding requests instead of one call per `VADD`.
- `DIM` must match the model; check on `VCREATE` (probe call) or on first use.
- Privacy: with a hosted provider, text leaves the machine; document clearly, default off.
- Snapshots store vectors as today; the embedder config is not persisted.

Open questions:

- Which providers first (Ollama local, OpenAI-compatible HTTP API)?
- Should the server also keep a small embedding cache (same text → same vector)?

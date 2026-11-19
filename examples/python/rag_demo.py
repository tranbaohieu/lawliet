"""Minimal RAG demo on Lawliet: ingest, filtered retrieval and a semantic cache (VCACHE).

Runs without model downloads or API keys: `embed()` uses a hashing
bag-of-words embedder so the example works anywhere. Replace `embed()` with a
real model (OpenAI, sentence-transformers, Ollama, ...) and `generate()` with
your LLM call for real use; nothing else changes.

Usage, with the Python package (server started by the script, nothing else to run):
    pip install lawliet-server numpy
    python examples/python/rag_demo.py

or against a server you started yourself on port 6380:
    pip install -r examples/python/requirements.txt
    ./bin/lawliet &                  # or: docker run -p 6380:6380 lawliet
    python examples/python/rag_demo.py
"""

import hashlib
import json
import re

import numpy as np
import redis

DIM = 512

DOCS = [
    ("hr.md", "leave", "Employees get 12 days of paid annual leave per year. Unused leave expires in March."),
    ("hr.md", "leave", "Sick leave requires a medical certificate after 2 consecutive days."),
    ("hr.md", "salary", "Salary is paid on the 5th of every month by bank transfer."),
    ("it.md", "vpn", "Connect to the VPN with the company app before accessing internal tools."),
    ("it.md", "password", "Passwords must be rotated every 90 days and contain at least 12 characters."),
    ("it.md", "laptop", "Report a lost laptop to IT within 24 hours to lock the device."),
    ("ops.md", "oncall", "The on-call engineer must acknowledge an incident page within 15 minutes."),
    ("ops.md", "deploy", "Production deploys are allowed Monday to Thursday before 4 pm."),
]


# --- replace these two functions for real use ----------------------------------

def embed(text: str) -> np.ndarray:
    """Hashing bag-of-words embedding. Demo only: it matches words, not meaning."""
    words = re.findall(r"\w+", text.lower())
    v = np.zeros(DIM, dtype=np.float32)
    for tok in words:
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
        v[h % DIM] += 1.0 if (h >> 63) == 0 else -1.0
    return v


def generate(question: str, context: list[str]) -> str:
    """Stub LLM: plug your model here. Returns the prompt it would send."""
    ctx = "\n".join(f"- {c}" for c in context)
    return f"[LLM answer to {question!r} using {len(context)} chunks]\n{ctx}"


# --- Lawliet usage --------------------------------------------------------------

def blob(v: np.ndarray) -> bytes:
    return v.astype("<f4").tobytes()  # float32, little-endian


def setup(r: redis.Redis) -> None:
    r.execute_command("VDROP", "docs")
    r.execute_command("VDROP", "answers")
    # Documents: never evicted.
    r.execute_command("VCREATE", "docs", "DIM", DIM, "METRIC", "cosine")
    # Semantic cache: bounded, least recently used answers are evicted first.
    # CACHE_THRESHOLD: similarity above which a cached answer is reused. 0.85 suits
    # the demo embedder; tune it for your model (often 0.9-0.95 for sentence embeddings).
    # CACHE_TTL: default lifetime of a cached answer (seconds).
    r.execute_command("VCREATE", "answers", "DIM", DIM, "METRIC", "cosine",
                      "MAX_VECTORS", 1000, "POLICY", "allkeys-lru",
                      "CACHE_THRESHOLD", 0.85, "CACHE_TTL", 3600)

    pipe = r.pipeline(transaction=False)
    for i, (source, topic, text) in enumerate(DOCS):
        pipe.execute_command("VADD", "docs", f"chunk:{i}", "FP32", blob(embed(text)),
                             "TEXT", text, "META", json.dumps({"source": source, "topic": topic}))
    pipe.execute()
    print(f"ingested {r.execute_command('VCOUNT', 'docs')} chunks\n")


def retrieve(r: redis.Redis, question: str, k: int = 3, where: str | None = None):
    args = ["VSEARCH", "docs", "FP32", blob(embed(question)), "K", k, "WITHTEXT", "WITHMETA"]
    if where:
        args += ["FILTER", where]
    return [(hit[0].decode(), float(hit[1]), hit[2].decode(), json.loads(hit[3]))
            for hit in r.execute_command(*args)]


def answer(r: redis.Redis, question: str, where: str | None = None) -> str:
    q = blob(embed(question))

    # 1. Semantic cache: a similar question answered recently? (one round trip)
    hit = r.execute_command("VCACHE.GET", "answers", "FP32", q)
    if hit:
        return f"(cache hit, similarity {float(hit[1]):.2f}) {hit[2].decode()}"

    # 2. Retrieve context, 3. generate, 4. cache the answer (TTL = CACHE_TTL).
    chunks = retrieve(r, question, where=where)
    text = generate(question, [c[2] for c in chunks])
    key = "q:" + hashlib.sha1(question.encode()).hexdigest()[:12]
    r.execute_command("VCACHE.SET", "answers", key, "FP32", q, "TEXT", text)
    return text


def connect() -> redis.Redis:
    """Embedded server from the `lawliet` package if installed, else localhost:6380."""
    try:
        import lawliet
    except ImportError:
        print("lawliet package not installed: using the server on localhost:6380")
        return redis.Redis(port=6380)
    r = lawliet.start()  # stopped automatically when the script exits
    host, port = r.connection_pool.connection_kwargs["host"], r.connection_pool.connection_kwargs["port"]
    print(f"using a server started by the lawliet package on {host}:{port} (not localhost:6380)")
    return r


def main() -> None:
    r = connect()
    setup(r)

    print("== retrieval")
    for cid, score, text, meta in retrieve(r, "how many days of annual leave do I get?"):
        print(f"{score:.3f}  {cid}  [{meta['source']}/{meta['topic']}]  {text}")

    print("\n== retrieval restricted to IT docs (FILTER)")
    for cid, score, text, meta in retrieve(r, "what should I do if I lose my laptop", where='source == "it.md"'):
        print(f"{score:.3f}  {cid}  [{meta['source']}/{meta['topic']}]  {text}")

    print("\n== RAG with semantic cache")
    print(answer(r, "When is salary paid?"))
    print()
    print(answer(r, "when is the salary paid"))  # near-duplicate question -> cache hit
    print()
    print(answer(r, "how do I connect to the VPN"))  # unrelated question -> miss

    stats = r.execute_command("VCACHE.STATS", "answers")
    stats = stats if isinstance(stats, dict) else dict(zip(stats[::2], stats[1::2]))
    print(f"\ncache: {stats[b'entries']} answers, {stats[b'hits']} hits, {stats[b'misses']} misses, "
          f"hit rate {float(stats[b'hit_rate']):.0%}")


if __name__ == "__main__":
    main()

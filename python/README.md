# Lawliet for Python

**An in-memory vector cache that speaks the Redis protocol** — bundled as a Python
package. The wheel contains the server binary: `pip install`, start it from Python,
talk to it with `redis-py`. No Docker, no separate download.

```bash
pip install lawliet-server
```

The import name is `lawliet`. Do not install it next to the unrelated `lawliet` package on
PyPI (a web framework): both provide a `lawliet` module.

```python
import lawliet
import numpy as np

r = lawliet.start()                       # server on a free port; returns redis.Redis

vec = np.random.rand(384).astype("<f4")   # your embedding model's output
r.execute_command("VCREATE", "docs", "DIM", 384)
r.execute_command("VADD", "docs", "doc:1", "FP32", vec.tobytes(),
                  "TEXT", "Employees get 12 days of paid leave.", "META", '{"source":"hr.md"}')
hits = r.execute_command("VSEARCH", "docs", "FP32", vec.tobytes(),
                         "K", 5, "FILTER", 'source == "hr.md"', "WITHTEXT")

lawliet.stop()                            # or let it stop when Python exits
```

## API

| | |
|---|---|
| `lawliet.start(port=None, **options) -> redis.Redis` | start one server, return a client |
| `lawliet.stop()` | stop it (idempotent; also runs at exit) |
| `with lawliet.server(**options) as r:` | server for the duration of a block |
| `lawliet.Server(port=None, **options)` | several servers, or full control: `.start()`, `.stop()`, `.client(**redis_kwargs)`, `.port`, `with Server() as srv:` |
| `lawliet.binary_path()` | path of the bundled binary |

`options`: `host` (default `127.0.0.1`), `data_dir` (enable snapshots in this directory),
`maxmemory` (e.g. `"512mb"`), `args` (extra server flags, e.g. `["--maxmemory-policy", "allkeys-lru"]`),
`startup_timeout` (seconds, default 10).

Without `data_dir` all data is lost when the server stops — ideal for tests:

```python
# conftest.py
import pytest, lawliet

@pytest.fixture
def vdb():
    with lawliet.server() as r:   # fresh, empty server per test
        yield r
```

## Command line

The package installs a `lawliet` command that runs the bundled server:

```bash
lawliet --port 6380               # same flags as the standalone binary
python -m lawliet --version
```

## Notes

- Wheels: macOS 12+ (arm64, x86_64), Linux (x86_64, aarch64; glibc and musl), Windows (x86_64).
- `LAWLIET_BINARY=/path/to/lawliet` makes the package use another server binary (it must
  support `--exit-on-stdin-eof`, i.e. be from the same release or newer).
- The server never outlives the Python process: it watches a pipe on its stdin and shuts down
  when the pipe closes, also when Python is killed (SIGTERM, SIGKILL, Jupyter kernel restart).
  `stop()` and normal exit shut it down gracefully on every platform (snapshot written).
- There is no authentication: keep the default `host="127.0.0.1"`.

Commands (`VSEARCH`, `VCACHE.*`, filters, TTL, eviction), limitations and design:
[project README](https://github.com/tranbaohieu/lawliet#readme) ·
[DESIGN.md](https://github.com/tranbaohieu/lawliet/blob/main/DESIGN.md).

"""Lawliet: an in-memory vector cache that speaks the Redis protocol.

The wheel bundles the server binary, so there is nothing else to install::

    import lawliet

    r = lawliet.start()              # server on a free port, returns redis.Redis
    r.execute_command("VCREATE", "docs", "DIM", 384)
    ...
    lawliet.stop()                   # also stopped automatically at exit

    with lawliet.server() as r:      # stopped when the block ends
        ...

Commands (VADD, VSEARCH, VCACHE.*, ...) are documented in the project README.
"""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING, Any, Iterator

from ._server import LawlietError, Server, binary_path
from ._version import __version__

if TYPE_CHECKING:
    import redis

__all__ = ["LawlietError", "Server", "binary_path", "server", "start", "stop", "__version__"]

_default: Server | None = None
_lock = threading.Lock()  # start() / stop() from several threads


def start(port: int | None = None, **options: Any) -> redis.Redis:
    """Start a server and return a ``redis.Redis`` client for it.

    ``options`` are passed to :class:`Server` (``host``, ``data_dir``,
    ``maxmemory``, ``args``, ...). Only one server at a time is managed this
    way; use :class:`Server` directly for more.
    """
    global _default
    with _lock:
        if _default is not None and _default.running:
            raise LawlietError(f"already started ({_default!r}); call lawliet.stop() first")
        _default = Server(port, **options).start()
        return _default.client()


def stop() -> None:
    """Stop the server started by :func:`start`, if any."""
    global _default
    with _lock:
        if _default is not None:
            _default.stop()
            _default = None


@contextlib.contextmanager
def server(port: int | None = None, **options: Any) -> Iterator[redis.Redis]:
    """Run a server for the duration of a ``with`` block and yield a client."""
    with Server(port, **options) as srv:
        client = srv.client()
        try:
            yield client
        finally:
            client.close()

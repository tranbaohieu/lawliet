"""Start and stop the bundled Lawliet server as a child process."""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import redis

_EXE = "lawliet.exe" if os.name == "nt" else "lawliet"

# Go's net.Listen error text when the port is taken (Unix, Windows).
_PORT_IN_USE = ("address already in use", "only one usage of each socket address")


class LawlietError(RuntimeError):
    """The server binary is missing, failed to start or stopped unexpectedly."""


class _PortInUse(LawlietError):
    pass


def binary_path() -> str:
    """Path of the lawliet server binary.

    ``LAWLIET_BINARY`` overrides the binary bundled in the wheel (useful for
    editable installs and for testing a local build).
    """
    override = os.environ.get("LAWLIET_BINARY")
    if override:
        if not os.path.isfile(override):
            raise LawlietError(f"LAWLIET_BINARY={override!r} is not a file")
        return override

    path = Path(__file__).resolve().parent / "bin" / _EXE
    if not path.is_file():
        raise LawlietError(
            f"lawliet server binary not found at {path}. Install a platform wheel "
            "(pip install lawliet-server), or set LAWLIET_BINARY to a lawliet binary."
        )
    if os.name != "nt" and not os.access(path, os.X_OK):
        try:
            path.chmod(0o755)
        except OSError:
            pass
    return str(path)


class Server:
    """A Lawliet server running as a child process.

    ``port=None`` picks a free port. Without ``data_dir`` everything stays in
    memory and is gone after ``stop()``; with it, snapshots are loaded at start
    and written on ``stop()``.

    The server's stdin is a pipe held by this process and the server runs with
    ``--exit-on-stdin-eof``: it shuts down when the pipe closes, so it does not
    outlive this process even when that is killed (SIGTERM, SIGKILL, a Jupyter
    kernel restart, ``docker stop``). ``stop()`` closes the pipe too, which gives
    a graceful shutdown on every platform.

    >>> with Server() as srv:
    ...     r = srv.client()
    ...     r.execute_command("VCREATE", "docs", "DIM", 384)
    """

    def __init__(
        self,
        port: int | None = None,
        *,
        host: str = "127.0.0.1",
        data_dir: str | os.PathLike[str] | None = None,
        maxmemory: str | None = None,
        args: Sequence[str] = (),
        startup_timeout: float = 10.0,
        binary: str | None = None,
    ) -> None:
        self.host = host
        self.data_dir = os.fspath(data_dir) if data_dir is not None else None
        self.maxmemory = maxmemory
        self.args = list(args)
        self.startup_timeout = startup_timeout
        self.binary = binary
        self._requested_port = port
        self.port: int | None = port
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: Any = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> Server:
        """Start the server and wait until it accepts connections. Returns ``self``."""
        if self.running:
            raise LawlietError(f"{self!r} is already running")
        binary = self.binary or binary_path()
        # A free port can be taken between choosing it and binding it: retry.
        attempts = 1 if self._requested_port is not None else 3
        for attempt in range(attempts):
            port = self._requested_port or _free_port(self.host)
            try:
                self._spawn(binary, port)
                return self
            except _PortInUse:
                if attempt == attempts - 1:
                    raise
        raise AssertionError("unreachable")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the server: close its stdin (graceful shutdown, snapshot written),
        kill it if it is still running after ``timeout`` seconds. Idempotent."""
        proc, self._proc = self._proc, None
        _live.discard(self)
        if proc is not None:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            try:
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> Server:
        if not self.running:
            self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- clients -------------------------------------------------------------

    @property
    def address(self) -> tuple[str, int]:
        """(host, port) to connect to."""
        if self.port is None:
            raise LawlietError("server not started")
        return _connect_host(self.host), self.port

    def client(self, **kwargs: Any) -> redis.Redis:
        """A ``redis.Redis`` client for this server; ``kwargs`` go to ``redis.Redis``."""
        import redis

        host, port = self.address
        return redis.Redis(host=host, port=port, **kwargs)

    def __repr__(self) -> str:
        state = "running" if self.running else "stopped"
        return f"<lawliet.Server {self.host}:{self.port} {state}>"

    # -- internals -----------------------------------------------------------

    def _command(self, binary: str, port: int) -> list[str]:
        cmd = [binary, "--exit-on-stdin-eof", "--bind", self.host, "--port", str(port)]
        if self.data_dir is not None:
            cmd += ["--dir", self.data_dir]
        if self.maxmemory is not None:
            cmd += ["--maxmemory", self.maxmemory]
        return cmd + self.args

    def _spawn(self, binary: str, port: int) -> None:
        log = tempfile.TemporaryFile()
        proc = subprocess.Popen(
            self._command(binary, port),
            stdin=subprocess.PIPE,  # closed when this process exits → server stops
            stdout=subprocess.DEVNULL,
            stderr=log,
        )
        self._proc, self._log, self.port = proc, log, port
        _live.add(self)

        host = _connect_host(self.host)
        deadline = time.monotonic() + self.startup_timeout
        while True:
            code = proc.poll()
            if code is not None:
                output = self._read_log()
                self.stop()
                if any(s in output.lower() for s in _PORT_IN_USE):
                    raise _PortInUse(f"port {port} is in use: {output}")
                raise LawlietError(f"lawliet exited with code {code}: {output or '(no output)'}")
            if _is_ours(host, port, proc.pid):
                return
            if time.monotonic() > deadline:
                output = self._read_log()
                self.stop()
                raise LawlietError(
                    f"lawliet did not answer on {host}:{port} within {self.startup_timeout}s: {output}"
                )
            time.sleep(0.02)

    def _read_log(self) -> str:
        if self._log is None:
            return ""
        self._log.seek(0)
        return self._log.read().decode(errors="replace").strip()


# Servers still running at interpreter exit are stopped gracefully (snapshot written).
# If the interpreter is killed instead, the stdin pipe closes and the server stops itself.
_live: set[Server] = set()


@atexit.register
def _stop_all() -> None:
    for srv in list(_live):
        srv.stop()


def _connect_host(bind: str) -> str:
    return "127.0.0.1" if bind in ("", "0.0.0.0", "::") else bind


def _free_port(host: str) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _is_ours(host: str, port: int, pid: int) -> bool:
    """True if the server on host:port is process ``pid``.

    Checking the PID (not just PING) avoids mistaking another server that
    already holds the port for the one being started.
    """
    try:
        with socket.create_connection((host, port), timeout=0.5) as s:
            s.sendall(b"INFO server\r\n")  # inline command, no client library needed
            reply = s.makefile("rb")
            header = reply.readline()
            if not header.startswith(b"$"):
                return False
            body = reply.read(int(header[1:]))
            return f"process_id:{pid}\r\n".encode() in body
    except (OSError, ValueError):
        return False

"""Hatch build hook: bundle the Lawliet server binary into a platform wheel.

Environment variables:

    LAWLIET_TARGET   GOOS/GOARCH the wheel is for, e.g. linux/arm64.
                     Default: this machine.
    LAWLIET_BINARY   Prebuilt lawliet binary for that target. A relative path is
                     resolved against the repository root (the parent of this
                     directory), not the working directory: build frontends
                     run this hook from python/.
                     Default: build it with `go build` from the repository
                     (needs Go and the Go sources next to this directory).
                     GOEXPERIMENT=simd in the environment builds the AVX2
                     kernels (experimental; release wheels are scalar).

Wheels only: the sdist does not contain the Go sources, so a wheel built from
it needs LAWLIET_BINARY. Editable installs stay pure Python and find the binary
through LAWLIET_BINARY at run time.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Wheel platform tags per release target (same targets as .github/workflows/release.yml).
# The binary is static (CGO_ENABLED=0), so one Linux wheel serves glibc and musl.
# macOS 12 is the oldest version recent Go releases support.
PLATFORM_TAGS = {
    "darwin/arm64": "macosx_12_0_arm64",
    "darwin/amd64": "macosx_12_0_x86_64",
    "linux/amd64": "manylinux_2_17_x86_64.manylinux2014_x86_64.musllinux_1_1_x86_64",
    "linux/arm64": "manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64",
    "windows/amd64": "win_amd64",
}

_GOOS = {"darwin": "darwin", "linux": "linux", "win32": "windows"}
_GOARCH = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "amd64", "amd64": "amd64"}

_VERSION_VAR = "github.com/tranbaohieu/lawliet/internal/version.Version"


def host_target() -> str:
    goos = _GOOS.get(sys.platform)
    goarch = _GOARCH.get(platform.machine().lower())
    if goos is None or goarch is None:
        raise RuntimeError(
            f"no Lawliet build for {sys.platform}/{platform.machine()}; "
            f"set LAWLIET_TARGET to one of {sorted(PLATFORM_TAGS)}"
        )
    return f"{goos}/{goarch}"


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        self._staging: Path | None = None
        if self.target_name != "wheel" or version == "editable":
            return

        target = os.environ.get("LAWLIET_TARGET") or host_target()
        if target not in PLATFORM_TAGS:
            raise ValueError(f"LAWLIET_TARGET={target!r}: expected one of {sorted(PLATFORM_TAGS)}")
        exe = "lawliet.exe" if target.startswith("windows/") else "lawliet"

        self._staging = Path(tempfile.mkdtemp(prefix="lawliet-wheel-"))
        staged = self._staging / exe
        prebuilt = os.environ.get("LAWLIET_BINARY")
        if prebuilt:
            path = Path(prebuilt)
            if not path.is_absolute():
                path = Path(self.root).parent / path
            if not path.is_file():
                raise FileNotFoundError(f"LAWLIET_BINARY={prebuilt!r}: {path} is not a file")
            prebuilt = str(path)
            shutil.copyfile(prebuilt, staged)
        else:
            self._go_build(target, staged)
        staged.chmod(0o755)  # the wheel keeps this mode, so the binary is executable once installed

        build_data["force_include"][str(staged)] = f"lawliet/bin/{exe}"
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{PLATFORM_TAGS[target]}"

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        if self._staging is not None:
            shutil.rmtree(self._staging, ignore_errors=True)

    def _go_build(self, target: str, out: Path) -> None:
        repo = Path(self.root).parent
        if not (repo / "cmd" / "lawliet").is_dir():
            raise RuntimeError(
                f"Go sources not found in {repo}: set LAWLIET_BINARY to a prebuilt lawliet binary"
            )
        go = shutil.which("go")
        if go is None:
            raise RuntimeError("go not found on PATH: install Go or set LAWLIET_BINARY")
        goos, goarch = target.split("/")
        env = {**os.environ, "CGO_ENABLED": "0", "GOOS": goos, "GOARCH": goarch}
        ldflags = f"-s -w -X {_VERSION_VAR}=v{self.metadata.version}"
        subprocess.run(
            [go, "build", "-trimpath", "-ldflags", ldflags, "-o", str(out), "./cmd/lawliet"],
            cwd=repo,
            env=env,
            check=True,
        )

"""`lawliet` command: run the bundled server with the given flags.

    lawliet --port 6380
    python -m lawliet --version
"""

from __future__ import annotations

import os
import subprocess
import sys

from ._server import LawlietError, binary_path


def main() -> None:
    try:
        binary = binary_path()
    except LawlietError as e:
        sys.exit(f"lawliet: {e}")
    argv = [binary, *sys.argv[1:]]
    if os.name == "nt":
        try:
            sys.exit(subprocess.call(argv))
        except KeyboardInterrupt:
            sys.exit(130)
    os.execv(binary, argv)  # replace this process: signals go straight to the server


if __name__ == "__main__":
    main()

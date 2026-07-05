"""Module entrypoint for ``python -m wings_control``."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_application_path() -> None:
    candidates = [
        Path(os.getenv("APP_WORKDIR", "/opt/wings-control")),
        Path(__file__).resolve().parent,
    ]
    for path in reversed(candidates):
        if path.exists():
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)


_bootstrap_application_path()

from wings_control.wings_control import run


if __name__ == "__main__":
    raise SystemExit(run())

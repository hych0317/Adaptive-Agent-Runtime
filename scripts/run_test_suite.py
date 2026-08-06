"""Run the full suite while proving the production Runtime DB is untouched."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from applications.research_agent.agent import DEFAULT_RESEARCH_RUNTIME_DB


def _snapshot(path: Path) -> tuple[bool, int, int, str | None]:
    if not path.exists():
        return (False, 0, 0, None)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    stat = path.stat()
    return (True, stat.st_size, stat.st_mtime_ns, digest.hexdigest())


def main() -> int:
    runtime_path = Path(DEFAULT_RESEARCH_RUNTIME_DB).resolve()
    before = _snapshot(runtime_path)
    arguments = sys.argv[1:] or ["discover", "-s", "tests", "-t", "."]
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", *arguments],
        cwd=PROJECT_ROOT,
        check=False,
    )
    after = _snapshot(runtime_path)
    if before != after:
        print(
            "ERROR: test suite changed the production Research Runtime database",
            file=sys.stderr,
        )
        return 2
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

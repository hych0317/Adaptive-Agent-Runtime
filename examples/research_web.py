"""Run the Research Runtime Console from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from applications.research_agent.web import main  # noqa: E402


if __name__ == "__main__":
    main()

"""Suite-owned temporary database paths for Research Agent tests."""

from __future__ import annotations

import atexit
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, TypeVar
from uuid import uuid4


_SUITE_DATABASES = TemporaryDirectory(prefix="adaptive-runtime-tests-")
_OPEN_RESOURCES: list[_Closable] = []


class _Closable(Protocol):
    def close(self) -> None: ...


ClosableT = TypeVar("ClosableT", bound=_Closable)


def runtime_database_path() -> Path:
    """Return a unique file database destroyed with the test process."""

    return Path(_SUITE_DATABASES.name) / f"{uuid4()}.sqlite3"


def register_test_resource(resource: ClosableT) -> ClosableT:
    """Keep a test resource reachable and close it before temp cleanup."""

    _OPEN_RESOURCES.append(resource)
    return resource


@atexit.register
def _close_test_resources() -> None:
    for resource in reversed(_OPEN_RESOURCES):
        resource.close()
    _OPEN_RESOURCES.clear()

"""Bounded asynchronous process transport for local CLI inference backends."""

from __future__ import annotations

import asyncio
import os
import signal
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.llm.json_types import LLMModel


class ProcessTransportError(Exception):
    """Base failure before a bounded child process result is available."""


class ProcessTransportTimeoutError(ProcessTransportError):
    pass


class ProcessTransportUnavailableError(ProcessTransportError):
    pass


class ProcessTransportOutputLimitError(ProcessTransportUnavailableError):
    """Raised after a child exceeds the transport's bounded output buffer."""


class ProcessResult(LLMModel):
    exit_code: int
    stdout: str = ""
    stderr: str = ""


@runtime_checkable
class AsyncProcessTransport(RuntimeModule, Protocol):
    def resolve(self, executable: str) -> str | None: ...

    async def run(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None,
        cwd: str,
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> ProcessResult: ...


class SubprocessTransport:
    module_id = "llm.process_transport.subprocess"

    def __init__(self, *, max_output_bytes: int = 4_000_000) -> None:
        if max_output_bytes < 1:
            raise ValueError("process output limit must be positive")
        self._max_output_bytes = max_output_bytes

    def resolve(self, executable: str) -> str | None:
        direct = shutil.which(executable)
        if direct is not None:
            return direct
        # GUI-launched runtimes often inherit a stripped PATH. Mirror Open
        # Design's resolver by searching conventional user toolchain roots.
        from adaptive_agent_runtime.llm.providers.cli_integration import (
            well_known_user_toolchain_bins,
        )

        for directory in well_known_user_toolchain_bins():
            resolved = shutil.which(executable, path=directory)
            if resolved is not None:
                return resolved
        candidate = Path(executable).expanduser()
        if candidate.is_absolute() and candidate.is_file():
            if os.name == "nt" or os.access(candidate, os.X_OK):
                return str(candidate)
        return None

    async def run(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None,
        cwd: str,
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> ProcessResult:
        if not argv:
            raise ValueError("process arguments cannot be empty")
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=(
                    asyncio.subprocess.PIPE
                    if stdin is not None
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=dict(environment),
                creationflags=(
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    if os.name == "nt"
                    else 0
                ),
                start_new_session=os.name != "nt",
            )
        except (OSError, ValueError) as exc:
            raise ProcessTransportUnavailableError from exc
        assert process.stdout is not None
        assert process.stderr is not None
        tasks: list[asyncio.Task[object]] = [
            asyncio.create_task(
                _read_bounded(process.stdout, self._max_output_bytes)
            ),
            asyncio.create_task(
                _read_bounded(process.stderr, self._max_output_bytes)
            ),
            asyncio.create_task(process.wait()),
        ]
        if stdin is not None:
            assert process.stdin is not None
            tasks.append(
                asyncio.create_task(
                    _write_stdin(process.stdin, stdin.encode("utf-8"))
                )
            )
        communication = asyncio.gather(*tasks)
        try:
            results = await asyncio.wait_for(
                asyncio.shield(communication),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            await _terminate_process_tree(process)
            await _cancel_tasks(tasks)
            await asyncio.gather(communication, return_exceptions=True)
            raise ProcessTransportTimeoutError from exc
        except asyncio.CancelledError:
            await _terminate_process_tree(process)
            await _cancel_tasks(tasks)
            await asyncio.gather(communication, return_exceptions=True)
            raise
        except ProcessTransportOutputLimitError:
            await _terminate_process_tree(process)
            await _cancel_tasks(tasks)
            await asyncio.gather(communication, return_exceptions=True)
            raise
        stdout = results[0]
        stderr = results[1]
        if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
            raise AssertionError("process stream reader returned invalid output")
        return ProcessResult(
            exit_code=process.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )


async def _read_bounded(
    stream: asyncio.StreamReader,
    maximum: int,
) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await stream.read(min(65_536, maximum - size + 1))
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > maximum:
            raise ProcessTransportOutputLimitError
        chunks.append(chunk)


async def _write_stdin(
    stream: asyncio.StreamWriter,
    payload: bytes,
) -> None:
    try:
        stream.write(payload)
        await stream.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        stream.close()
        try:
            await stream.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass


async def _terminate_process_tree(
    process: asyncio.subprocess.Process,
) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=5.0)
        except (OSError, TimeoutError):
            process.kill()
    else:
        try:
            kill_process_group = cast(
                Callable[[int, int], None],
                getattr(os, "killpg"),
            )
            kill_process_group(
                process.pid,
                int(getattr(signal, "SIGKILL")),
            )
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except TimeoutError:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def _cancel_tasks(tasks: Sequence[asyncio.Task[object]]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def inherited_environment(names: Sequence[str]) -> dict[str, str]:
    """Copy only explicitly allowed variables into a CLI child process."""

    return {
        name: value
        for name in names
        if (value := os.environ.get(name)) is not None
    }

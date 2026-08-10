from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest

from adaptive_agent_runtime.llm import (
    ProcessTransportOutputLimitError,
    ProcessTransportTimeoutError,
    SubprocessTransport,
)


class SubprocessTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_output_is_bounded_before_process_completion(
        self,
    ) -> None:
        transport = SubprocessTransport(max_output_bytes=128)
        with tempfile.TemporaryDirectory() as cwd:
            with self.assertRaises(ProcessTransportOutputLimitError):
                await transport.run(
                    (sys.executable, "-c", "print('x' * 1024)"),
                    stdin=None,
                    cwd=cwd,
                    environment=os.environ,
                    timeout_seconds=5.0,
                )

    async def test_timeout_terminates_the_process_group(self) -> None:
        transport = SubprocessTransport()
        with tempfile.TemporaryDirectory() as cwd:
            with self.assertRaises(ProcessTransportTimeoutError):
                await transport.run(
                    (sys.executable, "-c", "import time; time.sleep(30)"),
                    stdin=None,
                    cwd=cwd,
                    environment=os.environ,
                    timeout_seconds=0.05,
                )

    async def test_external_cancellation_terminates_the_process_group(
        self,
    ) -> None:
        transport = SubprocessTransport()
        with tempfile.TemporaryDirectory() as cwd:
            pid_path = os.path.join(cwd, "pid")
            task = asyncio.create_task(
                transport.run(
                    (
                        sys.executable,
                        "-c",
                        (
                            "import os, pathlib, sys, time; "
                            "pathlib.Path(sys.argv[1]).write_text("
                            "str(os.getpid()), encoding='utf-8'); "
                            "time.sleep(30)"
                        ),
                        pid_path,
                    ),
                    stdin=None,
                    cwd=cwd,
                    environment=os.environ,
                    timeout_seconds=30.0,
                )
            )
            for _ in range(100):
                if os.path.exists(pid_path):
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(os.path.exists(pid_path))
            with open(pid_path, encoding="utf-8") as handle:
                pid = int(handle.read())

            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()

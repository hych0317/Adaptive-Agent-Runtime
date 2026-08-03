from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()

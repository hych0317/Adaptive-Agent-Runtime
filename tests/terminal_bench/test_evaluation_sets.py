from __future__ import annotations

import unittest
from pathlib import Path


_CORE48_PATH = (
    Path(__file__).parents[2]
    / "applications"
    / "terminal_bench"
    / "evaluation_sets"
    / "core48.txt"
)


class TerminalEvaluationSetTests(unittest.TestCase):
    def test_core48_is_fixed_unique_and_nonvisual(self) -> None:
        tasks = tuple(
            line.strip()
            for line in _CORE48_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

        self.assertEqual(len(tasks), 48)
        self.assertEqual(len(set(tasks)), 48)
        self.assertEqual(tasks, tuple(sorted(tasks)))
        self.assertTrue(
            all(task.startswith("terminal-bench/") for task in tasks)
        )
        self.assertNotIn("terminal-bench/chess-best-move", tasks)
        self.assertNotIn("terminal-bench/code-from-image", tasks)
        self.assertIn("terminal-bench/pypi-server", tasks)
        self.assertIn("terminal-bench/fix-code-vulnerability", tasks)


if __name__ == "__main__":
    unittest.main()

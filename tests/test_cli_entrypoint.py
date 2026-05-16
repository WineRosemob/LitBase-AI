from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class CLIEntrypointTest(unittest.TestCase):
    def test_cli_help_is_quiet_and_lists_commands(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "litbase_ai.cli", "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("search", proc.stdout)
        self.assertIn("doctor", proc.stdout)
        self.assertNotIn("NumExpr", proc.stdout)
        self.assertNotIn("NumExpr", proc.stderr)


if __name__ == "__main__":
    unittest.main()

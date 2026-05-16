from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class WebVPNEntrypointTest(unittest.TestCase):
    def test_module_entrypoint_has_no_runpy_runtime_warning(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "litbase_ai.download.webvpn_login", "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertNotIn("RuntimeWarning", proc.stderr)


if __name__ == "__main__":
    unittest.main()

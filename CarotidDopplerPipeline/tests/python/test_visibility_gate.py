from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.python.test_contract import write_complete_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class VisibilityGateTests(unittest.TestCase):
    def test_cli_returns_machine_json_and_threshold_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "run"
            run_directory.mkdir()
            write_complete_run(run_directory)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(PROJECT_ROOT / "python")
            command = [
                sys.executable,
                "-m",
                "carotid_doppler.visibility_gate",
                "--run",
                str(run_directory),
                "--lumen-x-mm",
                "-20",
                "20",
                "--lumen-z-mm",
                "-1",
                "1",
                "--wall-x-mm",
                "-20",
                "20",
                "--wall-z-mm",
                "69",
                "71",
                "--minimum-db",
                "-6",
            ]

            passed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )
            self.assertEqual(passed.returncode, 0, passed.stderr)
            result = json.loads(passed.stdout)
            self.assertTrue(result["passed"])
            self.assertAlmostEqual(result["contrast_db"], -3.9794001, places=5)
            self.assertEqual(result["lumen"]["median_power"], 2.0)
            self.assertEqual(result["wall"]["median_power"], 5.0)

            command[-1] = "-3"
            failed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )
            self.assertEqual(failed.returncode, 2, failed.stderr)
            self.assertFalse(json.loads(failed.stdout)["passed"])


if __name__ == "__main__":
    unittest.main()

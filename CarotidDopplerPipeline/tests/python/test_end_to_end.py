from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tests.python.test_render_cli import PROJECT_ROOT, write_render_config


class SyntheticEndToEndTests(unittest.TestCase):
    def test_cuda_linear_output_flows_into_python_only_render(self) -> None:
        executable = Path(os.environ["CAROTID_DOPPLER_CUDA_EXE"])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = root / "synthetic-closed-loop"
            generated = subprocess.run(
                [
                    str(executable),
                    "--synthetic",
                    "--run-dir",
                    str(run_directory),
                    "--nx",
                    "4",
                    "--nz",
                    "3",
                    "--frames",
                    "5",
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            self.assertFalse(any(run_directory.glob("*.png")))
            self.assertFalse(any(run_directory.glob("*db*")))

            render_config = root / "shared-reference.json"
            write_render_config(render_config, reference_power=12.0)
            config = json.loads(render_config.read_text(encoding="utf-8"))
            config["reference"]["source_run_id"] = "synthetic-closed-loop"
            render_config.write_text(json.dumps(config), encoding="utf-8")

            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(PROJECT_ROOT / "python")
            rendered = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "carotid_doppler.render",
                    "--run",
                    str(run_directory),
                    "--render-config",
                    str(render_config),
                    "--render-id",
                    "shared-reference-render",
                ],
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )
            self.assertEqual(rendered.returncode, 0, rendered.stderr)

            power_db = np.load(
                run_directory
                / "renders"
                / "shared-reference-render"
                / "power_db_f32.npy"
            )
            self.assertEqual(power_db.shape, (3, 4))
            self.assertAlmostEqual(float(power_db[-1, -1]), 0.0, places=6)
            self.assertAlmostEqual(float(power_db[0, 0]), -10.791812, places=5)


if __name__ == "__main__":
    unittest.main()

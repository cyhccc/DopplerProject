from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tests.python.test_contract import write_complete_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_render_config(path: Path, reference_power: float = 6.0) -> None:
    config = {
        "schema_version": 1,
        "reference": {
            "reference_id": "fixed-ticket-1-reference",
            "reference_power": reference_power,
            "source_run_id": "contract-fixture",
            "source_artifact": "power_linear_f32.bin",
            "calibration": "explicit_test_literal",
        },
        "db": {
            "minimum": -40.0,
            "maximum": 0.0,
            "epsilon_ratio": 1e-12,
        },
        "image": {
            "colormap": "turbo",
            "dpi": 100,
            "show_colorbar": True,
            "x_unit": "mm",
            "z_unit": "mm",
        },
    }
    path.write_text(json.dumps(config), encoding="utf-8")


class RenderCliTests(unittest.TestCase):
    def test_cli_uses_fixed_reference_and_writes_separate_render_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = root / "run"
            run_directory.mkdir()
            write_complete_run(run_directory)
            render_config = root / "render.json"
            write_render_config(render_config)

            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(PROJECT_ROOT / "python")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "carotid_doppler.render",
                    "--run",
                    str(run_directory),
                    "--render-config",
                    str(render_config),
                    "--render-id",
                    "fixed-render",
                    "--x-range-mm",
                    "-10",
                    "10",
                    "--z-range-mm",
                    "0",
                    "20",
                ],
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            render_directory = run_directory / "renders" / "fixed-render"
            self.assertEqual(
                {path.name for path in render_directory.iterdir()},
                {"power_db_f32.npy", "power_doppler.png", "render_manifest.json"},
            )

            power_db = np.load(render_directory / "power_db_f32.npy")
            self.assertEqual(power_db.shape, (2, 3))
            self.assertEqual(power_db.dtype, np.dtype("<f4"))
            self.assertAlmostEqual(float(power_db[0, 0]), -7.7815125, places=5)
            self.assertAlmostEqual(float(power_db[0, 2]), -3.0103000, places=5)
            self.assertAlmostEqual(float(power_db[1, 2]), 0.0, places=6)

            manifest = json.loads(
                (render_directory / "render_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["source"]["run_id"], "contract-fixture")
            self.assertEqual(manifest["reference"]["reference_power"], 6.0)
            self.assertEqual(
                manifest["db"]["formula"],
                "10*log10(max(power_linear, epsilon)/reference_power)",
            )
            self.assertEqual(manifest["image"]["array_layout"], ["z", "x"])
            self.assertEqual(manifest["image"]["depth_direction"], "down")
            self.assertEqual(
                manifest["image"]["display_x_range_mm"], [-10.0, 10.0]
            )
            self.assertEqual(manifest["image"]["display_z_range_mm"], [0.0, 20.0])
            self.assertFalse(manifest["image"]["display_crop_changes_power_db"])
            self.assertEqual(
                (render_directory / "power_doppler.png").read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )

    def test_cli_rejects_config_without_explicit_reference_power(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = root / "run"
            run_directory.mkdir()
            write_complete_run(run_directory)
            render_config = root / "render.json"
            write_render_config(render_config)
            config = json.loads(render_config.read_text(encoding="utf-8"))
            del config["reference"]["reference_power"]
            render_config.write_text(json.dumps(config), encoding="utf-8")

            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(PROJECT_ROOT / "python")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "carotid_doppler.render",
                    "--run",
                    str(run_directory),
                    "--render-config",
                    str(render_config),
                    "--render-id",
                    "must-fail",
                ],
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("reference_power", completed.stderr)
            self.assertFalse((run_directory / "renders").exists())

    def test_cli_rejects_display_range_outside_manifest_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = root / "run"
            run_directory.mkdir()
            write_complete_run(run_directory)
            render_config = root / "render.json"
            write_render_config(render_config)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(PROJECT_ROOT / "python")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "carotid_doppler.render",
                    "--run",
                    str(run_directory),
                    "--render-config",
                    str(render_config),
                    "--render-id",
                    "outside-grid",
                    "--x-range-mm",
                    "-21",
                    "20",
                ],
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("inside manifest image grid", completed.stderr)
            self.assertFalse((run_directory / "renders").exists())

    def test_cli_rejects_non_positive_reference_power(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = root / "run"
            run_directory.mkdir()
            write_complete_run(run_directory)
            render_config = root / "render.json"
            write_render_config(render_config, reference_power=0.0)

            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(PROJECT_ROOT / "python")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "carotid_doppler.render",
                    "--run",
                    str(run_directory),
                    "--render-config",
                    str(render_config),
                    "--render-id",
                    "must-fail",
                ],
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("greater than zero", completed.stderr)
            self.assertFalse((run_directory / "renders").exists())


if __name__ == "__main__":
    unittest.main()

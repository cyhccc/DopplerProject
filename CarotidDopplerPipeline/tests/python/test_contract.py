from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from carotid_doppler.contract import ContractError, load_run  # noqa: E402


def write_complete_run(run_directory: Path) -> None:
    power = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f4"
    )
    singular_values = np.array([3.0, 2.0, 1.0], dtype="<f4")
    power.tofile(run_directory / "power_linear_f32.bin")
    singular_values.tofile(run_directory / "singular_values_f32.bin")

    manifest = {
        "schema_version": 1,
        "run_id": "contract-fixture",
        "status": "complete",
        "input": {
            "path": "synthetic://fixture",
            "dtype": "float32_le",
            "layout": ["frame", "z", "x"],
            "shape": [3, 2, 3],
            "frame_start": 0,
            "frame_count": 3,
            "fingerprint": {
                "size_bytes": 0,
                "last_write_time_utc": "1970-01-01T00:00:00Z",
                "sha256": None,
            },
        },
        "channel_mapping": {
            "source_order": "ascending_x",
            "internal_order": "ascending_x",
            "reversed": False,
        },
        "acquisition": {},
        "image_grid": {
            "nx": 3,
            "nz": 2,
            "x_min_m": -0.02,
            "x_max_m": 0.02,
            "z_min_m": 0.0,
            "z_max_m": 0.07,
            "layout": ["z", "x"],
        },
        "processing": {},
        "artifacts": {
            "power_linear": {
                "path": "power_linear_f32.bin",
                "dtype": "float32_le",
                "shape": [2, 3],
                "layout": ["z", "x"],
                "size_bytes": 24,
            },
            "singular_values": {
                "path": "singular_values_f32.bin",
                "dtype": "float32_le",
                "shape": [3],
                "layout": ["component"],
                "size_bytes": 12,
            },
            "beamformed_iq": None,
            "filtered_iq": None,
        },
        "software": {},
        "timings_ms": {},
    }
    (run_directory / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def write_complete_angle_artifacts(run_directory: Path) -> dict[str, Any]:
    manifest_path = run_directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["acquisition"]["transmit_angles_deg"] = [-6, -3, 0, 3, 6]
    manifest["processing"]["angle_diagnostics_enabled"] = True
    manifest["processing"]["dump_angle_beamformed_iq"] = False
    arrays = {
        "angle_power_linear": (
            np.ones((5, 2, 3), dtype="<f4"),
            ["angle", "z", "x"],
        ),
        "angle_singular_values": (
            np.tile(np.array([3.0, 2.0, 1.0], dtype="<f4"), (5, 1)),
            ["angle", "component"],
        ),
        "incoherent_power_linear": (
            np.ones((2, 3), dtype="<f4"),
            ["z", "x"],
        ),
        "angle_coherence": (
            np.full((2, 3), 0.5, dtype="<f4"),
            ["z", "x"],
        ),
    }
    for key, (array, layout) in arrays.items():
        filename = f"{key}_f32.bin"
        array.tofile(run_directory / filename)
        manifest["artifacts"][key] = {
            "path": filename,
            "dtype": "float32_le",
            "shape": list(array.shape),
            "layout": layout,
            "size_bytes": array.nbytes,
        }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class RunContractTests(unittest.TestCase):
    def test_complete_manifest_reconstructs_declared_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            write_complete_run(run_directory)

            run = load_run(run_directory)

            self.assertEqual(run.run_id, "contract-fixture")
            self.assertEqual(run.power.shape, (2, 3))
            self.assertEqual(run.power.dtype, np.dtype("<f4"))
            np.testing.assert_array_equal(
                run.power,
                np.array([[1, 2, 3], [4, 5, 6]], dtype="<f4"),
            )
            np.testing.assert_array_equal(
                run.singular_values, np.array([3, 2, 1], dtype="<f4")
            )

    def test_non_complete_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            write_complete_run(run_directory)
            manifest_path = run_directory / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "running"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ContractError, "status must be 'complete'"):
                load_run(run_directory)

    def test_truncated_binary_reports_expected_and_actual_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            write_complete_run(run_directory)
            power_path = run_directory / "power_linear_f32.bin"
            power_path.write_bytes(power_path.read_bytes()[:-4])

            with self.assertRaisesRegex(
                ContractError, "expected 24 bytes, actual 20 bytes"
            ):
                load_run(run_directory)

    def test_unknown_power_layout_is_rejected_without_transpose_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            write_complete_run(run_directory)
            manifest_path = run_directory / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["power_linear"]["layout"] = ["x", "z"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(
                ContractError, r"layout must be \['z', 'x'\]"
            ):
                load_run(run_directory)

    def test_angle_coherence_must_remain_in_zero_to_one_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            write_complete_run(run_directory)
            manifest = write_complete_angle_artifacts(run_directory)
            coherence = np.full((2, 3), 1.01, dtype="<f4")
            coherence.tofile(run_directory / "angle_coherence_f32.bin")
            manifest_path = run_directory / "run_manifest.json"
            manifest["artifacts"]["angle_coherence"] = {
                "path": "angle_coherence_f32.bin",
                "dtype": "float32_le",
                "shape": [2, 3],
                "layout": ["z", "x"],
                "size_bytes": coherence.nbytes,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ContractError, "less than or equal to 1.0"):
                load_run(run_directory)

    def test_angle_singular_values_must_decrease_within_each_angle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            write_complete_run(run_directory)
            manifest = write_complete_angle_artifacts(run_directory)
            values = np.tile(np.array([3.0, 2.0, 1.0], dtype="<f4"), (5, 1))
            values[2] = [3.0, 1.0, 2.0]
            values.tofile(run_directory / "angle_singular_values_f32.bin")
            manifest_path = run_directory / "run_manifest.json"
            manifest["artifacts"]["angle_singular_values"] = {
                "path": "angle_singular_values_f32.bin",
                "dtype": "float32_le",
                "shape": [5, 3],
                "layout": ["angle", "component"],
                "size_bytes": values.nbytes,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(
                ContractError, "non-increasing within every angle"
            ):
                load_run(run_directory)

    def test_enabled_angle_diagnostics_require_all_four_scalar_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            write_complete_run(run_directory)
            manifest = write_complete_angle_artifacts(run_directory)
            manifest["artifacts"]["angle_singular_values"] = None
            (run_directory / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ContractError,
                "enabled angle diagnostics require.*angle_singular_values",
            ):
                load_run(run_directory)


if __name__ == "__main__":
    unittest.main()

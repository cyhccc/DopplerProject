from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

import numpy as np

from carotid_doppler.angle_diagnostic import (
    AngleDiagnosticError,
    diagnose_angles,
)
from carotid_doppler.contract import ContractError
from tests.python.test_contract import write_complete_run
from tests.python.test_render_cli import write_render_config


def set_real_diagnostic_identity(
    run_directory: Path, manifest: dict[str, Any]
) -> None:
    manifest["input"]["frame_count"] = 381
    manifest["input"]["shape"] = [381, 5, 2816, 128]
    manifest["channel_mapping"] = {
        "source_order": "descending_x",
        "internal_order": "ascending_x",
        "reversed": True,
    }
    manifest["processing"].update(
        {
            "channel_repair": False,
            "motion_correction": False,
            "block_svd": False,
            "angle_diagnostics_enabled": True,
            "dump_angle_beamformed_iq": False,
        }
    )
    singular_values = np.linspace(381.0, 1.0, 381, dtype="<f4")
    singular_values.tofile(run_directory / "singular_values_f32.bin")
    manifest["artifacts"]["singular_values"] = {
        "path": "singular_values_f32.bin",
        "dtype": "float32_le",
        "shape": [381],
        "layout": ["component"],
        "size_bytes": singular_values.nbytes,
    }


def write_angle_artifacts(run_directory: Path) -> None:
    manifest_path = run_directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    set_real_diagnostic_identity(run_directory, manifest)
    manifest["acquisition"]["transmit_angles_deg"] = [-6, -3, 0, 3, 6]
    manifest["processing"].update(
        {
            "svd_keep_start": 121,
            "svd_keep_end": 300,
            "svd_indexing": "one_based_inclusive",
            "power_linear_semantics": (
                "coherent angle sum followed by one global temporal SVD"
            ),
        }
    )
    angle_power = np.arange(1, 31, dtype="<f4").reshape(5, 2, 3)
    incoherent = angle_power.sum(axis=0, dtype=np.float32)
    coherence = np.array([[0.1, 0.4, 0.8], [0.2, 0.6, 0.9]], dtype="<f4")
    angle_singular_values = np.tile(
        np.linspace(381.0, 1.0, 381, dtype="<f4"), (5, 1)
    )
    arrays = {
        "angle_power_linear": (
            angle_power,
            "angle_power_linear_f32.bin",
            ["angle", "z", "x"],
        ),
        "incoherent_power_linear": (
            incoherent,
            "incoherent_power_linear_f32.bin",
            ["z", "x"],
        ),
        "angle_coherence": (
            coherence,
            "angle_coherence_f32.bin",
            ["z", "x"],
        ),
        "angle_singular_values": (
            angle_singular_values,
            "angle_singular_values_f32.bin",
            ["angle", "component"],
        ),
    }
    for key, (array, filename, layout) in arrays.items():
        array.tofile(run_directory / filename)
        manifest["artifacts"][key] = {
            "path": filename,
            "dtype": "float32_le",
            "shape": list(array.shape),
            "layout": layout,
            "size_bytes": array.nbytes,
        }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


class AngleDiagnosticTests(unittest.TestCase):
    def test_explicit_cuda_power_maps_share_one_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = root / "run"
            run_directory.mkdir()
            write_complete_run(run_directory)
            write_angle_artifacts(run_directory)
            render_config = root / "render.json"
            write_render_config(render_config, reference_power=100.0)
            config = json.loads(render_config.read_text(encoding="utf-8"))
            config["image"]["colormap"] = "hot"
            render_config.write_text(json.dumps(config), encoding="utf-8")

            output = diagnose_angles(
                run_directory,
                render_config,
                "five-angle-smoke",
                x_range_mm=(-10.0, 10.0),
                z_range_mm=(0.0, 20.0),
            )

            power_db = np.load(output / "power_db_f32.npy")
            self.assertEqual(power_db.shape, (7, 2, 3))
            self.assertAlmostEqual(float(power_db[0, 0, 0]), -20.0, places=5)
            self.assertEqual(
                (output / "five_angle_comparison.png").read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )
            manifest = json.loads(
                (output / "angle_diagnostic_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(manifest["db"]["shared_by_all_power_panels"])
            self.assertFalse(
                manifest["source"]["pre_svd_angle_beamformed_iq_used_as_power"]
            )
            self.assertEqual(
                manifest["processing"]["svd_keep_start"], 121
            )
            self.assertEqual(manifest["processing"]["svd_keep_end"], 300)
            metrics = json.loads(
                (output / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metrics["svd"],
                {
                    "keep_start": 121,
                    "keep_end": 300,
                    "indexing": "one_based_inclusive",
                },
            )
            self.assertIn(
                "[121,300]", metrics["definitions"]["coherent_power"]
            )
            self.assertEqual(manifest["image"]["display_x_range_mm"], [-10.0, 10.0])
            self.assertEqual(manifest["image"]["display_z_range_mm"], [0.0, 20.0])
            self.assertFalse(manifest["image"]["display_crop_changes_arrays_or_db"])
            with self.assertRaisesRegex(AngleDiagnosticError, "inside manifest"):
                diagnose_angles(
                    run_directory,
                    render_config,
                    "outside-grid-must-fail",
                    x_range_mm=(-21.0, 20.0),
                )
            run_manifest_path = run_directory / "run_manifest.json"
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            run_manifest["channel_mapping"]["reversed"] = False
            run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
            with self.assertRaisesRegex(AngleDiagnosticError, "reversed must be True"):
                diagnose_angles(run_directory, render_config, "bad-channel-map")

            run_manifest["channel_mapping"]["reversed"] = True
            run_manifest["processing"]["motion_correction"] = True
            run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                AngleDiagnosticError, "motion_correction must be false"
            ):
                diagnose_angles(run_directory, render_config, "bad-processing")

            run_manifest["processing"]["motion_correction"] = False
            run_manifest["processing"]["svd_keep_start"] = 0
            run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                AngleDiagnosticError, "1 <= svd_keep_start"
            ):
                diagnose_angles(run_directory, render_config, "bad-svd-range")

    def test_pre_svd_iq_cannot_substitute_for_power_doppler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = root / "run"
            run_directory.mkdir()
            write_complete_run(run_directory)
            manifest_path = run_directory / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            set_real_diagnostic_identity(run_directory, manifest)
            manifest["acquisition"]["transmit_angles_deg"] = [-6, -3, 0, 3, 6]
            manifest["processing"].update(
                {
                    "svd_keep_start": 100,
                    "svd_keep_end": 220,
                    "svd_indexing": "one_based_inclusive",
                    "power_linear_semantics": (
                        "coherent angle sum followed by one global temporal SVD"
                    ),
                }
            )
            angle_iq = np.zeros((381, 5, 2, 3, 2), dtype="<f4")
            angle_iq.tofile(run_directory / "angle_beamformed_iq_c64.bin")
            manifest["artifacts"]["angle_beamformed_iq"] = {
                "path": "angle_beamformed_iq_c64.bin",
                "scalar_dtype": "float32_le",
                "complex_encoding": "interleaved_real_imag",
                "shape": [381, 5, 2, 3],
                "layout": ["frame", "angle", "z", "x"],
                "size_bytes": angle_iq.nbytes,
            }
            manifest["processing"]["dump_angle_beamformed_iq"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            render_config = root / "render.json"
            write_render_config(render_config)
            config = json.loads(render_config.read_text(encoding="utf-8"))
            config["image"]["colormap"] = "hot"
            render_config.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(
                ContractError, "enabled angle diagnostics require"
            ):
                diagnose_angles(run_directory, render_config, "must-fail")


if __name__ == "__main__":
    unittest.main()

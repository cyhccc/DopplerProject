from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


class CudaCliContractTests(unittest.TestCase):
    def test_formal_protocol_rejects_small_config_before_run_directory_creation(
        self,
    ) -> None:
        executable = Path(os.environ["CAROTID_DOPPLER_CUDA_EXE"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rf_path = root / "tiny_rf.bin"
            np.arange(4, dtype="<i2").tofile(rf_path)
            config_path = root / "tiny_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "input": {
                            "path": str(rf_path),
                            "dtype": "int16_le",
                            "layout": [
                                "frame",
                                "sequence",
                                "sample",
                                "source_channel",
                            ],
                            "sequences": 1,
                            "channels": 1,
                            "samples": 2,
                            "frame_start": 0,
                            "frame_count": 2,
                            "channel_order": "ascending_x",
                        },
                        "acquisition": {
                            "sample_rate_hz": 30_000_000,
                            "center_frequency_hz": 5_000_000,
                            "sound_speed_m_per_s": 1480,
                            "pitch_m": 0.0003,
                            "transmit_angles_deg": [0],
                            "tx_delay_offset_samples": 0,
                            "transmit_cycle_count": 2.0,
                            "tx_reference_mode": "aligned_reference",
                        },
                        "image_grid": {
                            "nx": 2,
                            "nz": 2,
                            "x_min_m": -0.001,
                            "x_max_m": 0.001,
                            "z_min_m": 0.001,
                            "z_max_m": 0.002,
                        },
                        "receive": {
                            "f_number": 1.7,
                            "minimum_aperture_m": 0.001,
                            "apodization": "rectangular",
                        },
                        "tgc": {
                            "enabled": False,
                            "alpha_coeff_db_cm_mhz_power": 0.3,
                            "frequency_power": 1.5,
                        },
                        "bandpass": {
                            "enabled": False,
                            "coefficients_path": "unused.bin",
                            "dtype": "float32_le",
                            "taps": 1,
                        },
                        "svd": {
                            "mode": "global_temporal",
                            "keep_start": 1,
                            "keep_end": 1,
                            "indexing": "one_based_inclusive",
                        },
                        "output": {
                            "run_root": "runs",
                            "dump_beamformed_iq": False,
                            "dump_filtered_iq": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            validation_only = subprocess.run(
                [str(executable), "--config", str(config_path)],
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(validation_only.returncode, 0, validation_only.stderr)

            run_directory = root / "must-not-exist"
            formal = subprocess.run(
                [
                    str(executable),
                    "--config",
                    str(config_path),
                    "--run",
                    "--run-dir",
                    str(run_directory),
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(formal.returncode, 2)
            self.assertIn("$.input.frame_count", formal.stderr)
            self.assertIn("expected 381", formal.stderr)
            self.assertFalse(run_directory.exists())

            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["input"].update(
                {
                    "sequences": 5,
                    "channels": 128,
                    "samples": 2816,
                    "frame_count": 381,
                    "channel_order": "descending_x",
                }
            )
            config["acquisition"].update(
                {
                    "transmit_angles_deg": [-6, -3, 0, 3, 6],
                    "tx_delay_offset_samples": 177,
                }
            )
            config["svd"].update({"keep_start": 121, "keep_end": 300})
            config_path.write_text(json.dumps(config), encoding="utf-8")

            configurable_rank_run_directory = root / "configurable-rank"
            configurable_rank = subprocess.run(
                [
                    str(executable),
                    "--config",
                    str(config_path),
                    "--run",
                    "--run-dir",
                    str(configurable_rank_run_directory),
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(configurable_rank.returncode, 2)
            self.assertNotIn(
                "formal carotid protocol violation", configurable_rank.stderr
            )
            self.assertNotIn("$.svd.keep_", configurable_rank.stderr)

    def test_synthetic_cli_writes_only_complete_linear_contract(self) -> None:
        executable = Path(os.environ["CAROTID_DOPPLER_CUDA_EXE"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "synthetic-run"
            completed = subprocess.run(
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

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                {path.name for path in run_directory.iterdir()},
                {
                    "power_linear_f32.bin",
                    "run_manifest.json",
                    "singular_values_f32.bin",
                },
            )

            manifest = json.loads(
                (run_directory / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["image_grid"]["layout"], ["z", "x"])
            self.assertEqual(
                manifest["artifacts"]["power_linear"],
                {
                    "path": "power_linear_f32.bin",
                    "dtype": "float32_le",
                    "shape": [3, 4],
                    "layout": ["z", "x"],
                    "size_bytes": 48,
                },
            )
            self.assertEqual(
                manifest["artifacts"]["singular_values"],
                {
                    "path": "singular_values_f32.bin",
                    "dtype": "float32_le",
                    "shape": [5],
                    "layout": ["component"],
                    "size_bytes": 20,
                },
            )

            power = np.fromfile(
                run_directory / "power_linear_f32.bin", dtype="<f4"
            ).reshape(3, 4)
            np.testing.assert_array_equal(
                power,
                np.array(
                    [[1.0, 2.0, 3.0, 4.0],
                     [2.0, 4.0, 6.0, 8.0],
                     [3.0, 6.0, 9.0, 12.0]],
                    dtype="<f4",
                ),
            )
            singular_values = np.fromfile(
                run_directory / "singular_values_f32.bin", dtype="<f4"
            )
            np.testing.assert_array_equal(
                singular_values, np.array([5, 4, 3, 2, 1], dtype="<f4")
            )

    def test_existing_run_directory_is_rejected_without_modifying_it(self) -> None:
        executable = Path(os.environ["CAROTID_DOPPLER_CUDA_EXE"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "immutable-run"
            run_directory.mkdir()
            existing_files = {
                "power_linear_f32.bin": b"existing-power",
                "singular_values_f32.bin": b"existing-singular-values",
                "run_manifest.json": b'{"existing": true}',
                "owner-note.txt": b"do not modify",
            }
            for name, content in existing_files.items():
                (run_directory / name).write_bytes(content)

            completed = subprocess.run(
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

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("must be new and immutable", completed.stderr)
            self.assertEqual(
                {path.name: path.read_bytes() for path in run_directory.iterdir()},
                existing_files,
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


def write_coordinate_encoded_rf(
    path: Path,
    *,
    frames: int = 2,
    sequences: int = 2,
    samples: int = 3,
    channels: int = 4,
) -> np.ndarray:
    rf = np.empty((frames, sequences, samples, channels), dtype="<i2")
    for frame in range(frames):
        for sequence in range(sequences):
            for sample in range(samples):
                for source_channel in range(channels):
                    rf[frame, sequence, sample, source_channel] = (
                        1000 * frame
                        + 100 * sequence
                        + 10 * sample
                        + source_channel
                    )
    rf.tofile(path)
    return rf


def write_pipeline_config(
    path: Path,
    rf_path: Path,
    *,
    channel_order: str = "descending_x",
    frame_start: int = 0,
    frame_count: int = 2,
) -> dict[str, object]:
    config: dict[str, object] = {
        "schema_version": 1,
        "input": {
            "path": str(rf_path),
            "dtype": "int16_le",
            "layout": ["frame", "sequence", "sample", "source_channel"],
            "sequences": 2,
            "channels": 4,
            "samples": 3,
            "frame_start": frame_start,
            "frame_count": frame_count,
            "channel_order": channel_order,
        },
        "acquisition": {
            "sample_rate_hz": 30_000_000,
            "center_frequency_hz": 5_000_000,
            "sound_speed_m_per_s": 1480,
            "pitch_m": 0.0003,
            "transmit_angles_deg": [-3, 3],
            "tx_delay_offset_samples": 113,
            "transmit_cycle_count": 2.0,
            "tx_reference_mode": "aperture_edge",
        },
        "image_grid": {
            "nx": 4,
            "nz": 3,
            "x_min_m": -0.02,
            "x_max_m": 0.02,
            "z_min_m": 0.0,
            "z_max_m": 0.07,
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
            "coefficients_path": "unused-for-ticket-2.bin",
            "dtype": "float32_le",
            "taps": 129,
        },
        "svd": {
            "mode": "global_temporal",
            "keep_start": 1,
            "keep_end": frame_count,
            "indexing": "one_based_inclusive",
        },
        "output": {
            "run_root": "runs",
            "dump_beamformed_iq": False,
            "dump_filtered_iq": False,
        },
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return config


class RfConfigCliTests(unittest.TestCase):
    def test_unknown_legacy_input_field_is_rejected_at_its_json_path(self) -> None:
        executable = Path(os.environ["CAROTID_DOPPLER_CUDA_EXE"])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rf_path = root / "coordinate_rf.bin"
            write_coordinate_encoded_rf(rf_path)
            config_path = root / "config.json"
            config = write_pipeline_config(config_path, rf_path)
            input_config = config["input"]
            assert isinstance(input_config, dict)
            input_config["motion_correction"] = True
            config_path.write_text(json.dumps(config), encoding="utf-8")

            completed = subprocess.run(
                [str(executable), "--config", str(config_path)],
                capture_output=True,
                check=False,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("$.input.motion_correction", completed.stderr)
            self.assertIn("unknown field", completed.stderr)


if __name__ == "__main__":
    unittest.main()

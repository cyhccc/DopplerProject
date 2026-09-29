from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


def _write_frame_major_iq(path: Path, casorati: np.ndarray) -> None:
    """Write A[pixel, frame] as contract layout [frame, pixel]."""
    np.asarray(casorati.T, dtype="<c8", order="C").tofile(path)


def _read_filtered_iq(path: Path, pixels: int, frames: int) -> np.ndarray:
    return np.fromfile(path, dtype="<c8").reshape(frames, pixels).T


class GlobalSvdDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.executable = Path(os.environ["CAROTID_GLOBAL_SVD_DIAGNOSTIC_EXE"])

    def _run(
        self,
        input_path: Path,
        output_directory: Path,
        *,
        pixels: int,
        frames: int,
        keep_start: int,
        keep_end: int,
        dump_filtered_iq: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            str(self.executable),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_directory),
            "--pixels",
            str(pixels),
            "--frames",
            str(frames),
            "--keep-start",
            str(keep_start),
            "--keep-end",
            str(keep_end),
        ]
        if dump_filtered_iq:
            command.append("--dump-filtered-iq")
        return subprocess.run(command, capture_output=True, check=False, text=True)

    def test_small_complex_matrix_matches_independent_numpy_svd(self) -> None:
        pixels = 9
        frames = 6
        keep_start = 2
        keep_end = 5
        rng = np.random.default_rng(20260718)
        casorati = (
            rng.normal(size=(pixels, frames))
            + 1j * rng.normal(size=(pixels, frames))
        ).astype(np.complex64).astype(np.complex128)

        u, singular_reference, vh = np.linalg.svd(casorati, full_matrices=False)
        kept = slice(keep_start - 1, keep_end)
        filtered_reference = (
            u[:, kept]
            @ (singular_reference[kept, np.newaxis] * vh[kept, :])
        )
        power_reference = np.sum(np.abs(filtered_reference) ** 2, axis=1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "beamformed_iq_c64.bin"
            output_directory = root / "svd-output"
            _write_frame_major_iq(input_path, casorati)

            completed = self._run(
                input_path,
                output_directory,
                pixels=pixels,
                frames=frames,
                keep_start=keep_start,
                keep_end=keep_end,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                {
                    "filtered_iq_c64.bin",
                    "power_linear_f32.bin",
                    "singular_values_f32.bin",
                },
            )

            singular_actual = np.fromfile(
                output_directory / "singular_values_f32.bin", dtype="<f4"
            )
            filtered_actual = _read_filtered_iq(
                output_directory / "filtered_iq_c64.bin", pixels, frames
            )
            power_actual = np.fromfile(
                output_directory / "power_linear_f32.bin", dtype="<f4"
            )

            np.testing.assert_allclose(
                singular_actual,
                singular_reference,
                rtol=1e-4,
                atol=1e-6 * float(singular_reference[0]),
            )
            self.assertFalse(
                np.allclose(
                    singular_actual,
                    singular_reference**2,
                    rtol=1e-4,
                    atol=1e-6,
                ),
                "diagnostic must write singular values, not covariance eigenvalues",
            )
            filtered_relative_l2 = np.linalg.norm(
                filtered_actual - filtered_reference
            ) / np.linalg.norm(filtered_reference)
            power_relative_l2 = np.linalg.norm(
                power_actual - power_reference
            ) / np.linalg.norm(power_reference)
            self.assertLess(filtered_relative_l2, 5e-4)
            self.assertLess(power_relative_l2, 5e-4)

    def test_one_based_inclusive_100_to_220_keeps_exactly_121_modes(self) -> None:
        frames = 224
        pixels = 224
        singular_reference = np.linspace(300.0, 1.0, frames, dtype=np.float32)
        casorati = np.diag(singular_reference).astype(np.complex64)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "diagonal_iq_c64.bin"
            output_directory = root / "svd-output"
            _write_frame_major_iq(input_path, casorati)

            completed = self._run(
                input_path,
                output_directory,
                pixels=pixels,
                frames=frames,
                keep_start=100,
                keep_end=220,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            singular_actual = np.fromfile(
                output_directory / "singular_values_f32.bin", dtype="<f4"
            )
            filtered_actual = _read_filtered_iq(
                output_directory / "filtered_iq_c64.bin", pixels, frames
            )
            power_actual = np.fromfile(
                output_directory / "power_linear_f32.bin", dtype="<f4"
            )

            expected_filtered = np.zeros_like(casorati)
            expected_filtered[99:220, 99:220] = np.diag(singular_reference[99:220])
            expected_power = np.sum(np.abs(expected_filtered) ** 2, axis=1)

            np.testing.assert_allclose(singular_actual, singular_reference, rtol=1e-5)
            np.testing.assert_allclose(filtered_actual, expected_filtered, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(power_actual, expected_power, rtol=1e-5, atol=1e-5)
            self.assertEqual(int(np.count_nonzero(power_actual)), 121)

    def test_filtered_iq_is_only_written_by_explicit_diagnostic_flag(self) -> None:
        pixels = 4
        frames = 4
        casorati = np.eye(pixels, frames, dtype=np.complex64)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "identity_iq_c64.bin"
            output_directory = root / "svd-output"
            _write_frame_major_iq(input_path, casorati)

            completed = self._run(
                input_path,
                output_directory,
                pixels=pixels,
                frames=frames,
                keep_start=1,
                keep_end=4,
                dump_filtered_iq=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                {"power_linear_f32.bin", "singular_values_f32.bin"},
            )


if __name__ == "__main__":
    unittest.main()

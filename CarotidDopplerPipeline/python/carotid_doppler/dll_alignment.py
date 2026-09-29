from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


class AlignmentError(ValueError):
    """Raised when DLL/CUDA alignment inputs violate the diagnostic contract."""


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlignmentError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AlignmentError(f"{field} must be finite")
    return result


def _read_manifest(path: Path) -> tuple[dict[str, Any], int, int, float, float]:
    with path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    try:
        artifact = manifest["artifact"]
        shape = artifact["shape"]
        grid = manifest["image_grid"]
        if artifact["complex_encoding"] != "interleaved_real_imag":
            raise AlignmentError("CUDA artifact must use interleaved_real_imag")
        if artifact["layout"] != ["frame", "z", "x"]:
            raise AlignmentError("CUDA artifact layout must be [frame,z,x]")
        if len(shape) != 3 or int(shape[0]) != 1:
            raise AlignmentError("alignment requires exactly one CUDA frame")
        nz, nx = int(shape[1]), int(shape[2])
        dx_m = _require_number(grid["dx_m"], "image_grid.dx_m")
        dz_m = _require_number(grid["dz_m"], "image_grid.dz_m")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, AlignmentError):
            raise
        raise AlignmentError(f"invalid DAS diagnostic manifest: {error}") from error
    if nx < 2 or nz < 2 or dx_m <= 0.0 or dz_m <= 0.0:
        raise AlignmentError("invalid image dimensions or grid spacing")
    return manifest, nz, nx, dz_m, dx_m


def _read_exact(path: Path, dtype: str, count: int, label: str) -> np.ndarray:
    expected_bytes = count * np.dtype(dtype).itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise AlignmentError(
            f"{label} byte count mismatch: {actual_bytes} != {expected_bytes}"
        )
    values = np.fromfile(path, dtype=dtype, count=count)
    if values.size != count or not np.isfinite(values).all():
        raise AlignmentError(f"{label} must contain exactly {count} finite values")
    return values


def _percentile_scale(image: np.ndarray, roi: tuple[slice, slice]) -> float:
    values = image[roi]
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise AlignmentError("alignment ROI contains no finite positive values")
    scale = float(np.percentile(positive.astype(np.float64, copy=False), 99.9))
    if not math.isfinite(scale) or scale <= 0.0:
        raise AlignmentError("invalid P99.9 envelope scale")
    return scale


def _db_image(image: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.maximum(image.astype(np.float64, copy=False) / scale, 1e-12)
    return np.clip(20.0 * np.log10(normalized), -60.0, 0.0)


def _shifted_views(
    reference: np.ndarray,
    candidate: np.ndarray,
    roi: tuple[slice, slice],
    shift_z: int,
    shift_x: int,
) -> tuple[np.ndarray, np.ndarray]:
    z_start = int(roi[0].start or 0)
    z_stop = int(roi[0].stop or reference.shape[0])
    x_start = int(roi[1].start or 0)
    x_stop = int(roi[1].stop or reference.shape[1])

    ref_z0 = z_start + max(shift_z, 0)
    ref_z1 = z_stop + min(shift_z, 0)
    cand_z0 = z_start + max(-shift_z, 0)
    cand_z1 = z_stop + min(-shift_z, 0)
    ref_x0 = x_start + max(shift_x, 0)
    ref_x1 = x_stop + min(shift_x, 0)
    cand_x0 = x_start + max(-shift_x, 0)
    cand_x1 = x_stop + min(-shift_x, 0)
    if ref_z0 >= ref_z1 or ref_x0 >= ref_x1:
        raise AlignmentError("alignment search shift leaves an empty ROI")
    return (
        reference[ref_z0:ref_z1, ref_x0:ref_x1],
        candidate[cand_z0:cand_z1, cand_x0:cand_x1],
    )


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.ravel().astype(np.float64, copy=False)
    right_flat = right.ravel().astype(np.float64, copy=False)
    left_centered = left_flat - float(left_flat.mean())
    right_centered = right_flat - float(right_flat.mean())
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator == 0.0:
        return -1.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _search(
    reference_db: np.ndarray,
    candidate_db: np.ndarray,
    roi: tuple[slice, slice],
    max_shift_z: int,
    max_shift_x: int,
) -> tuple[
    dict[str, float | int | bool],
    list[dict[str, float | int | bool]],
]:
    best: dict[str, float | int | bool] | None = None
    orientation_results: list[dict[str, float | int | bool]] = []
    for mirrored in (False, True):
        oriented = np.fliplr(candidate_db) if mirrored else candidate_db
        orientation_best: dict[str, float | int | bool] | None = None
        for shift_z in range(-max_shift_z, max_shift_z + 1):
            for shift_x in range(-max_shift_x, max_shift_x + 1):
                ref_view, candidate_view = _shifted_views(
                    reference_db, oriented, roi, shift_z, shift_x
                )
                correlation = _pearson(ref_view, candidate_view)
                result: dict[str, float | int | bool] = {
                    "mirrored": mirrored,
                    "shift_z_rows": shift_z,
                    "shift_x_columns": shift_x,
                    "log_envelope_pearson": correlation,
                }
                if best is None or correlation > float(best["log_envelope_pearson"]):
                    best = result
                if orientation_best is None or correlation > float(
                    orientation_best["log_envelope_pearson"]
                ):
                    orientation_best = result
        if orientation_best is None:
            raise AlignmentError("orientation alignment search produced no result")
        orientation_results.append(orientation_best)
    if best is None:
        raise AlignmentError("alignment search produced no result")
    return best, orientation_results


def compare_alignment(
    dll_envelope_path: Path,
    cuda_iq_path: Path,
    manifest_path: Path,
    *,
    z_min_mm: float = 7.0,
    z_max_mm: float = 65.0,
    x_abs_max_mm: float = 18.0,
    max_shift_z: int = 24,
    max_shift_x: int = 8,
) -> dict[str, Any]:
    manifest, nz, nx, dz_m, dx_m = _read_manifest(manifest_path)
    dll_envelope = _read_exact(
        dll_envelope_path, "<f4", nz * nx, "DLL envelope"
    ).reshape(nz, nx)
    cuda_iq = _read_exact(cuda_iq_path, "<c8", nz * nx, "CUDA IQ").reshape(
        nz, nx
    )
    cuda_envelope = np.abs(cuda_iq).astype(np.float32)

    grid = manifest["image_grid"]
    z0_m = _require_number(grid["z_min_m"], "image_grid.z_min_m")
    x0_m = _require_number(grid["x_min_m"], "image_grid.x_min_m")
    z_coordinates_mm = (z0_m + np.arange(nz) * dz_m) * 1e3
    x_coordinates_mm = (x0_m + np.arange(nx) * dx_m) * 1e3
    z_indices = np.flatnonzero(
        (z_coordinates_mm >= z_min_mm) & (z_coordinates_mm <= z_max_mm)
    )
    x_indices = np.flatnonzero(np.abs(x_coordinates_mm) <= x_abs_max_mm)
    if z_indices.size == 0 or x_indices.size == 0:
        raise AlignmentError("physical ROI does not intersect the image grid")
    roi = (
        slice(int(z_indices[0]), int(z_indices[-1]) + 1),
        slice(int(x_indices[0]), int(x_indices[-1]) + 1),
    )

    dll_scale = _percentile_scale(dll_envelope, roi)
    cuda_scale = _percentile_scale(cuda_envelope, roi)
    dll_db = _db_image(dll_envelope, dll_scale)
    cuda_db = _db_image(cuda_envelope, cuda_scale)
    best, orientation_results = _search(
        dll_db, cuda_db, roi, max_shift_z=max_shift_z, max_shift_x=max_shift_x
    )

    oriented_cuda = (
        np.fliplr(cuda_envelope) if bool(best["mirrored"]) else cuda_envelope
    )
    ref_linear, cuda_linear = _shifted_views(
        dll_envelope,
        oriented_cuda,
        roi,
        int(best["shift_z_rows"]),
        int(best["shift_x_columns"]),
    )
    mask = ref_linear >= dll_scale * (10.0 ** (-50.0 / 20.0))
    if not np.any(mask):
        raise AlignmentError("no DLL pixels remain above the -50 dB L2 mask")
    reference_values = ref_linear[mask].astype(np.float64, copy=False)
    cuda_values = cuda_linear[mask].astype(np.float64, copy=False)
    reference_norm = float(np.linalg.norm(reference_values))
    cuda_energy = float(np.dot(cuda_values, cuda_values))
    if reference_norm == 0.0 or cuda_energy == 0.0:
        raise AlignmentError("masked linear envelope has zero energy")
    fitted_scale = float(np.dot(reference_values, cuda_values) / cuda_energy)
    fitted_rel_l2 = float(
        np.linalg.norm(reference_values - fitted_scale * cuda_values)
        / reference_norm
    )
    raw_rel_l2 = float(
        np.linalg.norm(reference_values - cuda_values) / reference_norm
    )

    shift_z = int(best["shift_z_rows"])
    shift_x = int(best["shift_x_columns"])
    return {
        "schema_version": 1,
        "inputs": {
            "dll_envelope": str(dll_envelope_path.resolve()),
            "cuda_iq": str(cuda_iq_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
        "shape": [nz, nx],
        "roi_mm": {
            "z_min": z_min_mm,
            "z_max": z_max_mm,
            "x_abs_max": x_abs_max_mm,
        },
        "normalization": {
            "dll_envelope_p99_9": dll_scale,
            "cuda_envelope_p99_9": cuda_scale,
            "log_floor_db": -60.0,
            "l2_mask_db": -50.0,
        },
        "best_alignment": {
            **best,
            "applied_shift_z_mm": shift_z * dz_m * 1e3,
            "applied_shift_x_mm": shift_x * dx_m * 1e3,
        },
        "orientation_results": orientation_results,
        "linear_envelope": {
            "cuda_to_dll_fitted_scale": fitted_scale,
            "fitted_relative_l2": fitted_rel_l2,
            "unscaled_relative_l2": raw_rel_l2,
            "masked_pixel_count": int(mask.sum()),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare one real-DLL envelope with one CUDA DAS IQ frame."
    )
    parser.add_argument("--dll-envelope", required=True, type=Path)
    parser.add_argument("--cuda-iq", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--z-min-mm", type=float, default=7.0)
    parser.add_argument("--z-max-mm", type=float, default=65.0)
    parser.add_argument("--x-abs-max-mm", type=float, default=18.0)
    parser.add_argument("--max-shift-z", type=int, default=24)
    parser.add_argument("--max-shift-x", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        result = compare_alignment(
            arguments.dll_envelope,
            arguments.cuda_iq,
            arguments.manifest,
            z_min_mm=arguments.z_min_mm,
            z_max_mm=arguments.z_max_mm,
            x_abs_max_mm=arguments.x_abs_max_mm,
            max_shift_z=arguments.max_shift_z,
            max_shift_x=arguments.max_shift_x,
        )
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if arguments.output is not None:
            output = arguments.output.resolve()
            if output.exists():
                raise AlignmentError(f"refusing to overwrite output: {output}")
            output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    except (AlignmentError, OSError, json.JSONDecodeError) as error:
        print(f"carotid_doppler.dll_alignment: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

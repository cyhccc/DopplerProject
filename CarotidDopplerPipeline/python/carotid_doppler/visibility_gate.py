from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from .contract import ContractError, RunData, load_run, read_float32_artifact


Float32Array = npt.NDArray[np.float32]


class VisibilityGateError(ValueError):
    """Raised when the visibility-gate contract or ROI is invalid."""


_ARTIFACT_CHOICES = ("power_linear", "incoherent_power_linear", "angle_power_linear")


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VisibilityGateError(f"{field} must be a JSON object")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VisibilityGateError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise VisibilityGateError(f"{field} must be finite")
    return result


def _read_declared_power(run: RunData, artifact: str) -> Float32Array:
    if artifact == "power_linear":
        return run.power
    artifacts = _mapping(run.manifest["artifacts"], "artifacts")
    field = f"artifacts.{artifact}"
    descriptor = artifacts.get(artifact)
    if descriptor is None:
        raise VisibilityGateError(f"{field} is required")
    expected_layout = (
        ["angle", "z", "x"] if artifact == "angle_power_linear" else ["z", "x"]
    )
    grid = _mapping(run.manifest["image_grid"], "image_grid")
    spatial_shape = (grid.get("nz"), grid.get("nx"))
    expected_shape = (
        (5,) + spatial_shape
        if artifact == "angle_power_linear"
        else spatial_shape
    )
    return read_float32_artifact(
        run.run_directory,
        descriptor,
        field=field,
        expected_layout=expected_layout,
        expected_shape=expected_shape,
        require_non_negative=True,
    )


def _coordinates_mm(run: RunData) -> tuple[Float32Array, Float32Array]:
    grid = _mapping(run.manifest["image_grid"], "image_grid")
    nx = grid.get("nx")
    nz = grid.get("nz")
    if isinstance(nx, bool) or not isinstance(nx, int) or nx <= 1:
        raise VisibilityGateError("image_grid.nx must be an integer greater than one")
    if isinstance(nz, bool) or not isinstance(nz, int) or nz <= 1:
        raise VisibilityGateError("image_grid.nz must be an integer greater than one")
    x_min = _finite(grid.get("x_min_m"), "image_grid.x_min_m") * 1000.0
    x_max = _finite(grid.get("x_max_m"), "image_grid.x_max_m") * 1000.0
    z_min = _finite(grid.get("z_min_m"), "image_grid.z_min_m") * 1000.0
    z_max = _finite(grid.get("z_max_m"), "image_grid.z_max_m") * 1000.0
    if not x_min < x_max or not z_min < z_max:
        raise VisibilityGateError("image grid coordinate bounds must increase")
    x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    z = np.linspace(z_min, z_max, nz, dtype=np.float32)
    return x, z


def _roi_mask(
    x: Float32Array,
    z: Float32Array,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    field: str,
) -> npt.NDArray[np.bool_]:
    x_min, x_max = x_range
    z_min, z_max = z_range
    if not x_min < x_max or not z_min < z_max:
        raise VisibilityGateError(f"{field} ranges must be strictly increasing")
    x_selected = (x >= x_min) & (x <= x_max)
    z_selected = (z >= z_min) & (z <= z_max)
    mask = z_selected[:, None] & x_selected[None, :]
    if not np.any(mask):
        raise VisibilityGateError(f"{field} does not contain any manifest grid point")
    return mask


def evaluate_visibility(
    run_directory: str | Path,
    *,
    artifact: str = "power_linear",
    angle_index: int | None = None,
    lumen_x_mm: tuple[float, float],
    lumen_z_mm: tuple[float, float],
    wall_x_mm: tuple[float, float],
    wall_z_mm: tuple[float, float],
    minimum_db: float,
) -> tuple[dict[str, Any], bool]:
    """Evaluate median lumen-to-wall linear-power contrast in explicit ROIs."""

    if artifact not in _ARTIFACT_CHOICES:
        raise VisibilityGateError(f"unsupported artifact: {artifact!r}")
    if artifact == "angle_power_linear":
        if angle_index is None or not 0 <= angle_index < 5:
            raise VisibilityGateError(
                "--angle-index 0..4 is required for angle_power_linear"
            )
    elif angle_index is not None:
        raise VisibilityGateError(
            "--angle-index is only valid with angle_power_linear"
        )
    threshold = _finite(minimum_db, "minimum_db")
    run = load_run(run_directory)
    declared = _read_declared_power(run, artifact)
    power = declared[angle_index] if angle_index is not None else declared
    x, z = _coordinates_mm(run)
    lumen_mask = _roi_mask(x, z, lumen_x_mm, lumen_z_mm, "lumen ROI")
    wall_mask = _roi_mask(x, z, wall_x_mm, wall_z_mm, "wall ROI")
    lumen_median = float(np.median(power[lumen_mask]))
    wall_median = float(np.median(power[wall_mask]))
    if lumen_median > 0.0 and wall_median > 0.0:
        contrast_db: float | None = 10.0 * math.log10(lumen_median / wall_median)
        contrast_class = "finite"
        passed = contrast_db >= threshold
    elif lumen_median > 0.0 and wall_median == 0.0:
        contrast_db = None
        contrast_class = "positive_infinity"
        passed = True
    elif lumen_median == 0.0 and wall_median > 0.0:
        contrast_db = None
        contrast_class = "negative_infinity"
        passed = False
    else:
        contrast_db = None
        contrast_class = "undefined_zero_over_zero"
        passed = False
    angles = _mapping(run.manifest["acquisition"], "acquisition").get(
        "transmit_angles_deg"
    )
    angle_deg = (
        float(angles[angle_index])
        if angle_index is not None
        and isinstance(angles, list)
        and len(angles) > angle_index
        else None
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "gate": "median_lumen_to_wall_power_db",
        "run_id": run.run_id,
        "artifact": artifact,
        "angle_index": angle_index,
        "angle_deg": angle_deg,
        "minimum_db": threshold,
        "contrast_db": contrast_db,
        "contrast_class": contrast_class,
        "passed": passed,
        "lumen": {
            "x_mm": list(lumen_x_mm),
            "z_mm": list(lumen_z_mm),
            "pixel_count": int(np.count_nonzero(lumen_mask)),
            "median_power": lumen_median,
        },
        "wall": {
            "x_mm": list(wall_x_mm),
            "z_mm": list(wall_z_mm),
            "pixel_count": int(np.count_nonzero(wall_mask)),
            "median_power": wall_median,
        },
        "coordinate_source": "run_manifest.image_grid inclusive sample coordinates",
    }
    return result, passed


def _range(values: list[float], field: str) -> tuple[float, float]:
    return (_finite(values[0], f"{field}[0]"), _finite(values[1], f"{field}[1]"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail when median lumen-to-wall linear-power contrast is too low."
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--artifact", choices=_ARTIFACT_CHOICES, default="power_linear")
    parser.add_argument("--angle-index", type=int)
    parser.add_argument("--lumen-x-mm", nargs=2, required=True, type=float)
    parser.add_argument("--lumen-z-mm", nargs=2, required=True, type=float)
    parser.add_argument("--wall-x-mm", nargs=2, required=True, type=float)
    parser.add_argument("--wall-z-mm", nargs=2, required=True, type=float)
    parser.add_argument("--minimum-db", required=True, type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        result, passed = evaluate_visibility(
            arguments.run,
            artifact=arguments.artifact,
            angle_index=arguments.angle_index,
            lumen_x_mm=_range(arguments.lumen_x_mm, "lumen_x_mm"),
            lumen_z_mm=_range(arguments.lumen_z_mm, "lumen_z_mm"),
            wall_x_mm=_range(arguments.wall_x_mm, "wall_x_mm"),
            wall_z_mm=_range(arguments.wall_z_mm, "wall_z_mm"),
            minimum_db=arguments.minimum_db,
        )
    except (ContractError, VisibilityGateError, OSError, ValueError) as error:
        print(f"carotid_doppler.visibility_gate: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

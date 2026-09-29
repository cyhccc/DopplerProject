from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

from .contract import ContractError, load_run


class ReferenceError(ValueError):
    """Raised when a shared power reference cannot be calibrated safely."""


_CALIBRATION_PERCENTILE = 99.9
_EPSILON_RATIO = 1e-12


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ReferenceError(f"{field} must be finite")
    return result


def _reference_id(value: str | None, run_id: str) -> str:
    reference_id = value if value is not None else f"{run_id}-power-p999"
    if not reference_id:
        raise ReferenceError("reference_id must be a non-empty string")
    return reference_id


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ReferenceError(f"refusing to overwrite shared reference: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise ReferenceError(f"partial reference output already exists: {partial}")

    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            raise ReferenceError(f"refusing to overwrite shared reference: {path}")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def create_reference_config(
    run_directory: str | Path,
    output_path: str | Path,
    *,
    reference_id: str | None = None,
    percentile: float = _CALIBRATION_PERCENTILE,
) -> Path:
    """Calibrate and persist one immutable shared-reference render config."""

    calibrated_percentile = _finite_number(percentile, "percentile")
    if calibrated_percentile != _CALIBRATION_PERCENTILE:
        raise ReferenceError(
            "percentile must be 99.9 for the v1 shared-reference contract"
        )

    run = load_run(run_directory)
    positive_power = run.power[np.isfinite(run.power) & (run.power > 0.0)]
    if positive_power.size == 0:
        raise ReferenceError(
            "baseline linear power contains no finite positive values"
        )

    reference_power = float(
        np.percentile(
            positive_power.astype(np.float64, copy=False),
            calibrated_percentile,
        )
    )
    if not math.isfinite(reference_power) or reference_power <= 0.0:
        raise ReferenceError(
            "calibrated reference_power must be finite and greater than zero"
        )

    power_descriptor = run.manifest["artifacts"]["power_linear"]
    source_artifact = power_descriptor.get("path")
    if not isinstance(source_artifact, str) or not source_artifact:
        raise ContractError(
            "artifacts.power_linear.path must be a non-empty relative path"
        )

    config: dict[str, Any] = {
        "schema_version": 1,
        "reference": {
            "reference_id": _reference_id(reference_id, run.run_id),
            "reference_power": reference_power,
            "source_run_id": run.run_id,
            "source_artifact": source_artifact,
            "calibration": "finite_positive_percentile_99.9",
        },
        "db": {
            "minimum": -40.0,
            "maximum": 0.0,
            "epsilon_ratio": _EPSILON_RATIO,
        },
        "image": {
            "colormap": "hot",
            "dpi": 200,
            "show_colorbar": True,
            "x_unit": "mm",
            "z_unit": "mm",
        },
    }

    output = Path(output_path).resolve()
    _write_json_atomic(output, config)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a persistent shared power reference."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser(
        "create",
        help="create a render config from a complete baseline CUDA run",
    )
    create.add_argument("--run", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--reference-id")
    create.add_argument(
        "--percentile",
        type=float,
        default=_CALIBRATION_PERCENTILE,
        help="v1 calibration percentile; must be 99.9",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.command != "create":
            raise ReferenceError(f"unsupported command: {arguments.command!r}")
        output = create_reference_config(
            arguments.run,
            arguments.output,
            reference_id=arguments.reference_id,
            percentile=arguments.percentile,
        )
    except (ContractError, ReferenceError, OSError) as error:
        print(f"carotid_doppler.reference: {error}", file=sys.stderr)
        return 2
    print(f"complete shared reference config: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

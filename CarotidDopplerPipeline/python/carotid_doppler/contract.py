from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


Float32Array = npt.NDArray[np.float32]


class ContractError(ValueError):
    """Raised when a run directory violates the declared data contract."""


@dataclass(frozen=True)
class RunData:
    run_directory: Path
    run_id: str
    manifest: dict[str, Any]
    power: Float32Array
    singular_values: Float32Array


_REQUIRED_TOP_LEVEL = {
    "schema_version",
    "run_id",
    "status",
    "input",
    "channel_mapping",
    "acquisition",
    "image_grid",
    "processing",
    "artifacts",
    "software",
    "timings_ms",
}


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{field} must be a JSON object")
    return value


def _required(mapping: dict[str, Any], key: str, field: str) -> Any:
    if key not in mapping:
        raise ContractError(f"{field}.{key} is required")
    return mapping[key]


def _positive_dimensions(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{field} must be a non-empty dimension array")
    dimensions: list[int] = []
    for index, dimension in enumerate(value):
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise ContractError(f"{field}[{index}] must be a positive integer")
        dimensions.append(dimension)
    return tuple(dimensions)


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer")
    return value


def _artifact_path(run_directory: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field}.path must be a non-empty relative path")
    relative_path = Path(value)
    if relative_path.is_absolute():
        raise ContractError(f"{field}.path must remain inside the run directory")

    resolved_run = run_directory.resolve()
    resolved_artifact = (resolved_run / relative_path).resolve()
    if not resolved_artifact.is_relative_to(resolved_run):
        raise ContractError(f"{field}.path escapes the run directory: {value!r}")
    return resolved_artifact


def read_float32_artifact(
    run_directory: Path,
    descriptor_value: Any,
    *,
    field: str,
    expected_layout: list[str],
    expected_shape: tuple[int, ...] | None = None,
    require_finite: bool = True,
    require_non_negative: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Float32Array:
    """Read one manifest-declared float32 artifact through a strict contract."""

    descriptor = _mapping(descriptor_value, field)
    dtype = _required(descriptor, "dtype", field)
    if dtype != "float32_le":
        raise ContractError(f"{field}.dtype must be 'float32_le', got {dtype!r}")

    layout = _required(descriptor, "layout", field)
    if layout != expected_layout:
        raise ContractError(
            f"{field}.layout must be {expected_layout!r}, got {layout!r}"
        )

    shape = _positive_dimensions(
        _required(descriptor, "shape", field), f"{field}.shape"
    )
    if expected_shape is not None and shape != expected_shape:
        raise ContractError(
            f"{field}.shape must be {list(expected_shape)!r}, "
            f"got {list(shape)!r}"
        )
    expected_size = math.prod(shape) * np.dtype("<f4").itemsize
    declared_size = _required(descriptor, "size_bytes", field)
    if isinstance(declared_size, bool) or not isinstance(declared_size, int):
        raise ContractError(f"{field}.size_bytes must be an integer")
    if declared_size != expected_size:
        raise ContractError(
            f"{field}.size_bytes declares {declared_size}, expected {expected_size} "
            f"from dtype and shape"
        )

    artifact_path = _artifact_path(
        run_directory, _required(descriptor, "path", field), field
    )
    if not artifact_path.is_file():
        raise ContractError(f"{field}.path does not exist: {artifact_path}")
    actual_size = artifact_path.stat().st_size
    if actual_size != expected_size:
        raise ContractError(
            f"{field} file size mismatch: expected {expected_size} bytes, "
            f"actual {actual_size} bytes"
        )

    array = np.fromfile(artifact_path, dtype="<f4").reshape(shape)
    if require_finite and not np.all(np.isfinite(array)):
        raise ContractError(f"{field} must contain only finite values")
    if require_non_negative and np.any(array < 0.0):
        raise ContractError(f"{field} must contain only non-negative values")
    if minimum is not None and np.any(array < minimum):
        raise ContractError(
            f"{field} values must be greater than or equal to {minimum}"
        )
    if maximum is not None and np.any(array > maximum):
        raise ContractError(f"{field} values must be less than or equal to {maximum}")
    array.setflags(write=False)
    return array


def _validate_optional_complex_float32_artifact(
    run_directory: Path,
    descriptor_value: Any,
    *,
    field: str,
    expected_shape: tuple[int, ...],
    expected_layout: list[str],
) -> None:
    if descriptor_value is None:
        return

    descriptor = _mapping(descriptor_value, field)
    scalar_dtype = _required(descriptor, "scalar_dtype", field)
    if scalar_dtype != "float32_le":
        raise ContractError(
            f"{field}.scalar_dtype must be 'float32_le', got {scalar_dtype!r}"
        )

    complex_encoding = _required(descriptor, "complex_encoding", field)
    if complex_encoding != "interleaved_real_imag":
        raise ContractError(
            f"{field}.complex_encoding must be 'interleaved_real_imag', "
            f"got {complex_encoding!r}"
        )

    layout = _required(descriptor, "layout", field)
    if layout != expected_layout:
        raise ContractError(
            f"{field}.layout must be {expected_layout!r}, got {layout!r}"
        )

    shape = _positive_dimensions(
        _required(descriptor, "shape", field), f"{field}.shape"
    )
    if shape != expected_shape:
        raise ContractError(
            f"{field}.shape must equal the declared logical dimensions "
            f"{list(expected_shape)!r}, got {list(shape)!r}"
        )

    expected_size = math.prod(shape) * 2 * np.dtype("<f4").itemsize
    declared_size = _required(descriptor, "size_bytes", field)
    if isinstance(declared_size, bool) or not isinstance(declared_size, int):
        raise ContractError(f"{field}.size_bytes must be an integer")
    if declared_size != expected_size:
        raise ContractError(
            f"{field}.size_bytes declares {declared_size}, expected "
            f"{expected_size} from scalar dtype, complex encoding and shape"
        )

    artifact_path = _artifact_path(
        run_directory, _required(descriptor, "path", field), field
    )
    if not artifact_path.is_file():
        raise ContractError(f"{field}.path does not exist: {artifact_path}")
    actual_size = artifact_path.stat().st_size
    if actual_size != expected_size:
        raise ContractError(
            f"{field} file size mismatch: expected {expected_size} bytes, "
            f"actual {actual_size} bytes"
        )


def _load_manifest(run_directory: Path) -> dict[str, Any]:
    manifest_path = run_directory / "run_manifest.json"
    if not manifest_path.is_file():
        raise ContractError(f"run_manifest.json is missing from {run_directory}")
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(
            f"run_manifest.json is not valid UTF-8 JSON: {error}"
        ) from error

    manifest = _mapping(manifest_value, "manifest")
    missing = sorted(_REQUIRED_TOP_LEVEL.difference(manifest))
    if missing:
        raise ContractError(
            f"run manifest is missing required fields: {', '.join(missing)}"
        )
    if manifest["schema_version"] != 1:
        raise ContractError(
            f"schema_version must be 1, got {manifest['schema_version']!r}"
        )
    if manifest["status"] != "complete":
        raise ContractError(f"status must be 'complete', got {manifest['status']!r}")
    if not isinstance(manifest["run_id"], str) or not manifest["run_id"]:
        raise ContractError("run_id must be a non-empty string")
    for object_field in (
        "input",
        "channel_mapping",
        "acquisition",
        "image_grid",
        "processing",
        "artifacts",
        "software",
        "timings_ms",
    ):
        _mapping(manifest[object_field], object_field)
    return manifest


def load_run(run_directory: str | Path) -> RunData:
    """Read and validate one complete CUDA run through its public contract."""

    directory = Path(run_directory).resolve()
    if not directory.is_dir():
        raise ContractError(f"run directory does not exist: {directory}")

    manifest = _load_manifest(directory)
    input_config = _mapping(manifest["input"], "input")
    frame_count = _positive_integer(
        _required(input_config, "frame_count", "input"), "input.frame_count"
    )
    image_grid = _mapping(manifest["image_grid"], "image_grid")
    if image_grid.get("layout") != ["z", "x"]:
        raise ContractError(
            "image_grid.layout must be ['z', 'x'], "
            f"got {image_grid.get('layout')!r}"
        )
    nx = _positive_integer(
        _required(image_grid, "nx", "image_grid"), "image_grid.nx"
    )
    nz = _positive_integer(
        _required(image_grid, "nz", "image_grid"), "image_grid.nz"
    )

    artifacts = _mapping(manifest["artifacts"], "artifacts")
    processing = _mapping(manifest["processing"], "processing")
    angle_diagnostics_enabled = processing.get("angle_diagnostics_enabled", False)
    dump_angle_beamformed_iq = processing.get("dump_angle_beamformed_iq", False)
    if not isinstance(angle_diagnostics_enabled, bool):
        raise ContractError("processing.angle_diagnostics_enabled must be boolean")
    if not isinstance(dump_angle_beamformed_iq, bool):
        raise ContractError("processing.dump_angle_beamformed_iq must be boolean")
    if dump_angle_beamformed_iq and not angle_diagnostics_enabled:
        raise ContractError(
            "processing.dump_angle_beamformed_iq requires enabled angle diagnostics"
        )
    angle_diagnostic_keys = (
        "angle_power_linear",
        "angle_singular_values",
        "incoherent_power_linear",
        "angle_coherence",
    )
    present_angle_diagnostics = [
        key for key in angle_diagnostic_keys if artifacts.get(key) is not None
    ]
    if angle_diagnostics_enabled and len(present_angle_diagnostics) != 4:
        missing = [
            key for key in angle_diagnostic_keys if artifacts.get(key) is None
        ]
        raise ContractError(
            "enabled angle diagnostics require non-null artifacts: "
            + ", ".join(missing)
        )
    if not angle_diagnostics_enabled and present_angle_diagnostics:
        raise ContractError(
            "disabled angle diagnostics require null artifacts: "
            + ", ".join(present_angle_diagnostics)
        )
    angle_iq_present = artifacts.get("angle_beamformed_iq") is not None
    if dump_angle_beamformed_iq != angle_iq_present:
        raise ContractError(
            "processing.dump_angle_beamformed_iq must match the non-null "
            "artifacts.angle_beamformed_iq descriptor"
        )

    power = read_float32_artifact(
        directory,
        _required(artifacts, "power_linear", "artifacts"),
        field="artifacts.power_linear",
        expected_layout=["z", "x"],
        expected_shape=(nz, nx),
        require_non_negative=True,
    )

    singular_values = read_float32_artifact(
        directory,
        _required(artifacts, "singular_values", "artifacts"),
        field="artifacts.singular_values",
        expected_layout=["component"],
        expected_shape=(frame_count,),
        require_non_negative=True,
    )
    if np.any(singular_values[1:] > singular_values[:-1]):
        raise ContractError("singular values must be non-increasing")

    expected_iq_shape = (frame_count, nz, nx)
    _validate_optional_complex_float32_artifact(
        directory,
        _required(artifacts, "beamformed_iq", "artifacts"),
        field="artifacts.beamformed_iq",
        expected_shape=expected_iq_shape,
        expected_layout=["frame", "z", "x"],
    )
    _validate_optional_complex_float32_artifact(
        directory,
        _required(artifacts, "filtered_iq", "artifacts"),
        field="artifacts.filtered_iq",
        expected_shape=expected_iq_shape,
        expected_layout=["frame", "z", "x"],
    )

    angle_artifact_keys = (
        "angle_beamformed_iq",
        "angle_power_linear",
        "angle_singular_values",
        "incoherent_power_linear",
        "angle_coherence",
    )
    if any(artifacts.get(key) is not None for key in angle_artifact_keys):
        acquisition = _mapping(manifest["acquisition"], "acquisition")
        angles = acquisition.get("transmit_angles_deg")
        expected_angles = [-6, -3, 0, 3, 6]
        if angles != expected_angles:
            raise ContractError(
                "acquisition.transmit_angles_deg must be exactly "
                "[-6, -3, 0, 3, 6] for five-angle artifacts"
            )

    _validate_optional_complex_float32_artifact(
        directory,
        artifacts.get("angle_beamformed_iq"),
        field="artifacts.angle_beamformed_iq",
        expected_shape=(frame_count, 5, nz, nx),
        expected_layout=["frame", "angle", "z", "x"],
    )
    if artifacts.get("angle_power_linear") is not None:
        read_float32_artifact(
            directory,
            artifacts["angle_power_linear"],
            field="artifacts.angle_power_linear",
            expected_layout=["angle", "z", "x"],
            expected_shape=(5, nz, nx),
            require_non_negative=True,
        )
    if artifacts.get("angle_singular_values") is not None:
        angle_singular_values = read_float32_artifact(
            directory,
            artifacts["angle_singular_values"],
            field="artifacts.angle_singular_values",
            expected_layout=["angle", "component"],
            expected_shape=(5, frame_count),
            require_non_negative=True,
        )
        if np.any(angle_singular_values[:, 1:] > angle_singular_values[:, :-1]):
            raise ContractError(
                "angle singular values must be non-increasing within every angle"
            )
    if artifacts.get("incoherent_power_linear") is not None:
        read_float32_artifact(
            directory,
            artifacts["incoherent_power_linear"],
            field="artifacts.incoherent_power_linear",
            expected_layout=["z", "x"],
            expected_shape=(nz, nx),
            require_non_negative=True,
        )
    if artifacts.get("angle_coherence") is not None:
        read_float32_artifact(
            directory,
            artifacts["angle_coherence"],
            field="artifacts.angle_coherence",
            expected_layout=["z", "x"],
            expected_shape=(nz, nx),
            minimum=0.0,
            maximum=1.0,
        )

    return RunData(
        run_directory=directory,
        run_id=manifest["run_id"],
        manifest=manifest,
        power=power,
        singular_values=singular_values,
    )

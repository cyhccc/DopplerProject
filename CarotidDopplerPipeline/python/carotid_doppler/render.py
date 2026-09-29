from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from . import __version__
from .contract import ContractError, RunData, load_run


class RenderConfigError(ValueError):
    """Raised when a Python-only render configuration is invalid."""


_TOP_LEVEL_KEYS = {"schema_version", "reference", "db", "image"}
_REFERENCE_KEYS = {
    "reference_id",
    "reference_power",
    "source_run_id",
    "source_artifact",
    "calibration",
}
_DB_KEYS = {"minimum", "maximum", "epsilon_ratio"}
_IMAGE_KEYS = {"colormap", "dpi", "show_colorbar", "x_unit", "z_unit"}
_SAFE_RENDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EPSILON_RATIO = 1e-12


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RenderConfigError(f"{field} must be a JSON object")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected.difference(mapping))
    unknown = sorted(set(mapping).difference(expected))
    if missing:
        raise RenderConfigError(f"{field} missing fields: {', '.join(missing)}")
    if unknown:
        raise RenderConfigError(f"{field} unknown fields: {', '.join(unknown)}")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderConfigError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RenderConfigError(f"{field} must be finite")
    return result


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RenderConfigError(f"{field} must be a non-empty string")
    return value


def _load_render_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RenderConfigError(f"render config does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RenderConfigError(f"render config is not valid UTF-8 JSON: {error}") from error

    config = _mapping(value, "render config")
    _exact_keys(config, _TOP_LEVEL_KEYS, "render config")
    if config["schema_version"] != 1:
        raise RenderConfigError(
            f"schema_version must be 1, got {config['schema_version']!r}"
        )

    reference = _mapping(config["reference"], "reference")
    db = _mapping(config["db"], "db")
    image = _mapping(config["image"], "image")
    _exact_keys(reference, _REFERENCE_KEYS, "reference")
    _exact_keys(db, _DB_KEYS, "db")
    _exact_keys(image, _IMAGE_KEYS, "image")

    reference_power = _finite_number(
        reference["reference_power"], "reference.reference_power"
    )
    if reference_power <= 0.0:
        raise RenderConfigError("reference.reference_power must be greater than zero")
    for field in (
        "reference_id",
        "source_run_id",
        "source_artifact",
        "calibration",
    ):
        _non_empty_string(reference[field], f"reference.{field}")

    db_minimum = _finite_number(db["minimum"], "db.minimum")
    db_maximum = _finite_number(db["maximum"], "db.maximum")
    epsilon_ratio = _finite_number(db["epsilon_ratio"], "db.epsilon_ratio")
    if not db_minimum < db_maximum <= 0.0:
        raise RenderConfigError("db requires minimum < maximum <= 0")
    if epsilon_ratio != _EPSILON_RATIO:
        raise RenderConfigError(
            "db.epsilon_ratio must be 1e-12 for the v1 dB contract"
        )

    _non_empty_string(image["colormap"], "image.colormap")
    if image["colormap"] not in matplotlib.colormaps:
        raise RenderConfigError(f"unknown image.colormap: {image['colormap']!r}")
    dpi = image["dpi"]
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise RenderConfigError("image.dpi must be a positive integer")
    if not isinstance(image["show_colorbar"], bool):
        raise RenderConfigError("image.show_colorbar must be a boolean")
    if image["x_unit"] != "mm" or image["z_unit"] != "mm":
        raise RenderConfigError("Ticket 1 image units must both be 'mm'")
    return config


def _grid_coordinate(run: RunData, field: str) -> float:
    image_grid = run.manifest["image_grid"]
    value = image_grid.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"image_grid.{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"image_grid.{field} must be finite")
    return result


def _display_range(
    value: tuple[float, float] | None,
    full_range: tuple[float, float],
    field: str,
) -> tuple[float, float]:
    if value is None:
        return full_range
    minimum, maximum = value
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise RenderConfigError(f"{field} values must be finite")
    if not minimum < maximum:
        raise RenderConfigError(f"{field} requires MIN < MAX")
    full_minimum, full_maximum = full_range
    if minimum < full_minimum or maximum > full_maximum:
        raise RenderConfigError(
            f"{field} must remain inside manifest image grid "
            f"[{full_minimum}, {full_maximum}] mm"
        )
    return value


def _prepare_render_directory(run: RunData, render_id: str) -> Path:
    if not _SAFE_RENDER_ID.fullmatch(render_id):
        raise RenderConfigError(
            "render_id must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    render_directory = run.run_directory / "renders" / render_id
    if render_directory.exists():
        if not render_directory.is_dir() or any(render_directory.iterdir()):
            raise RenderConfigError(f"render directory must be new or empty: {render_directory}")
    else:
        render_directory.mkdir(parents=True)
    return render_directory


def _replace_file(partial: Path, target: Path) -> None:
    if target.exists():
        raise RenderConfigError(f"refusing to overwrite immutable output: {target}")
    os.replace(partial, target)


def _write_npy_atomic(path: Path, array: np.ndarray[Any, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as output:
        np.save(output, array, allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())
    _replace_file(partial, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    _replace_file(partial, path)


def _rollback_failed_render(render_directory: Path, outputs: Sequence[Path]) -> None:
    cleanup_errors: list[str] = []
    for output in outputs:
        for candidate in (output, output.with_name(output.name + ".partial")):
            try:
                candidate.unlink(missing_ok=True)
            except OSError as error:
                cleanup_errors.append(f"{candidate}: {error}")

    try:
        render_directory.rmdir()
    except FileNotFoundError:
        pass
    except OSError:
        # The directory is removed only when it is empty. An unexpected file is
        # never deleted by rollback.
        try:
            directory_is_empty = not any(render_directory.iterdir())
        except FileNotFoundError:
            directory_is_empty = False
        except OSError as error:
            cleanup_errors.append(f"failed to inspect {render_directory}: {error}")
            directory_is_empty = False
        if directory_is_empty:
            cleanup_errors.append(
                f"failed to remove empty render directory: {render_directory}"
            )

    if cleanup_errors:
        raise RenderConfigError(
            "failed to roll back incomplete render outputs: "
            + "; ".join(cleanup_errors)
        )


def render_run(
    run_directory: str | Path,
    render_config_path: str | Path,
    render_id: str,
    *,
    x_range_mm: tuple[float, float] | None = None,
    z_range_mm: tuple[float, float] | None = None,
) -> Path:
    """Render a validated CUDA run using one explicit shared power reference."""

    run = load_run(run_directory)
    config = _load_render_config(Path(render_config_path).resolve())
    reference = config["reference"]
    db_config = config["db"]
    image_config = config["image"]
    reference_power = float(reference["reference_power"])
    epsilon = reference_power * _EPSILON_RATIO

    power_db = (
        10.0 * np.log10(np.maximum(run.power, epsilon) / reference_power)
    ).astype("<f4", copy=False)
    if power_db.shape != run.power.shape or not np.all(np.isfinite(power_db)):
        raise ContractError("dB conversion must preserve shape and produce finite values")

    x_min_mm = _grid_coordinate(run, "x_min_m") * 1000.0
    x_max_mm = _grid_coordinate(run, "x_max_m") * 1000.0
    z_min_mm = _grid_coordinate(run, "z_min_m") * 1000.0
    z_max_mm = _grid_coordinate(run, "z_max_m") * 1000.0
    if not x_min_mm < x_max_mm or not z_min_mm < z_max_mm:
        raise ContractError("image grid requires increasing x and z coordinate bounds")
    display_x_range = _display_range(
        x_range_mm, (x_min_mm, x_max_mm), "x_range_mm"
    )
    display_z_range = _display_range(
        z_range_mm, (z_min_mm, z_max_mm), "z_range_mm"
    )

    render_directory = _prepare_render_directory(run, render_id)
    db_path = render_directory / "power_db_f32.npy"
    image_path = render_directory / "power_doppler.png"
    manifest_path = render_directory / "render_manifest.json"
    partial_image = image_path.with_name(image_path.name + ".partial")
    published_outputs = (db_path, image_path, manifest_path)
    try:
        _write_npy_atomic(db_path, power_db)

        figure, axes = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
        rendered = axes.imshow(
            power_db,
            cmap=image_config["colormap"],
            vmin=float(db_config["minimum"]),
            vmax=float(db_config["maximum"]),
            origin="upper",
            extent=[x_min_mm, x_max_mm, z_max_mm, z_min_mm],
            aspect="auto",
            interpolation="nearest",
        )
        axes.set_xlabel("Lateral position (mm)")
        axes.set_ylabel("Depth (mm)")
        axes.set_title("Power Doppler")
        axes.set_xlim(*display_x_range)
        axes.set_ylim(display_z_range[1], display_z_range[0])
        if image_config["show_colorbar"]:
            colorbar = figure.colorbar(rendered, ax=axes)
            colorbar.set_label("Power (dB re shared reference)")
        figure.savefig(
            partial_image,
            format="png",
            dpi=image_config["dpi"],
            metadata={"Software": "carotid-doppler Python renderer"},
        )
        plt.close(figure)
        _replace_file(partial_image, image_path)

        source_artifact = run.manifest["artifacts"]["power_linear"]["path"]
        render_manifest: dict[str, Any] = {
            "schema_version": 1,
            "render_id": render_id,
            "status": "complete",
            "source": {
                "run_id": run.run_id,
                "run_manifest": "../../run_manifest.json",
                "power_linear": source_artifact,
            },
            "reference": {
                "reference_id": reference["reference_id"],
                "reference_power": reference_power,
                "source_run_id": reference["source_run_id"],
                "source_artifact": reference["source_artifact"],
                "calibration": reference["calibration"],
            },
            "db": {
                "formula": "10*log10(max(power_linear, epsilon)/reference_power)",
                "resolved_formula": (
                    "10*log10(max(power_linear, "
                    "reference_power*1e-12)/reference_power)"
                ),
                "epsilon_formula": "reference_power*1e-12",
                "epsilon": epsilon,
                "epsilon_ratio": _EPSILON_RATIO,
                "minimum": float(db_config["minimum"]),
                "maximum": float(db_config["maximum"]),
                "array_clipped_to_display_range": False,
                "array_path": "power_db_f32.npy",
                "array_dtype": "float32_le",
                "array_shape": list(power_db.shape),
            },
            "image": {
                "path": "power_doppler.png",
                "colormap": image_config["colormap"],
                "dpi": image_config["dpi"],
                "show_colorbar": image_config["show_colorbar"],
                "array_layout": ["z", "x"],
                "pixel_index": "z*nx+x",
                "coordinate_source": "run_manifest.image_grid",
                "coordinate_unit": "mm",
                "x_range_mm": [x_min_mm, x_max_mm],
                "z_range_mm": [z_min_mm, z_max_mm],
                "display_x_range_mm": list(display_x_range),
                "display_z_range_mm": list(display_z_range),
                "display_crop_changes_power_db": False,
                "lateral_direction": "left_to_right_increasing_x",
                "depth_direction": "down",
                "x_display_order": "minimum_to_maximum_left_to_right",
                "z_display_order": "minimum_to_maximum_top_to_bottom",
                "transformations": {
                    "transpose": False,
                    "fliplr": False,
                    "flipud": False,
                },
            },
            "software": {
                "carotid_doppler": __version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
        }
        _write_json_atomic(manifest_path, render_manifest)
    except Exception as render_error:
        plt.close("all")
        try:
            _rollback_failed_render(render_directory, published_outputs)
        except RenderConfigError as cleanup_error:
            raise cleanup_error from render_error
        raise

    return render_directory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one complete CUDA linear-power run with a fixed reference."
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--render-config", required=True, type=Path)
    parser.add_argument("--render-id", required=True)
    parser.add_argument("--x-range-mm", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--z-range-mm", nargs=2, type=float, metavar=("MIN", "MAX"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        render_directory = render_run(
            arguments.run,
            arguments.render_config,
            arguments.render_id,
            x_range_mm=(tuple(arguments.x_range_mm) if arguments.x_range_mm else None),
            z_range_mm=(tuple(arguments.z_range_mm) if arguments.z_range_mm else None),
        )
    except (ContractError, RenderConfigError, OSError) as error:
        print(f"carotid_doppler.render: {error}", file=sys.stderr)
        return 2
    print(f"complete render directory: {render_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

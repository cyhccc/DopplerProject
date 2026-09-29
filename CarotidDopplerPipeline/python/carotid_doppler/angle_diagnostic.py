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
import numpy.typing as npt

from . import __version__
from .contract import ContractError, RunData, load_run, read_float32_artifact
from .render import RenderConfigError, _load_render_config


Float32Array = npt.NDArray[np.float32]


class AngleDiagnosticError(ValueError):
    """Raised when explicit five-angle diagnostic artifacts are invalid."""


_EXPECTED_ANGLES_DEG = (-6.0, -3.0, 0.0, 3.0, 6.0)
_POWER_COLORMAPS = {"hot", "afmhot", "gist_heat", "inferno"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EPSILON_RATIO = 1e-12


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AngleDiagnosticError(f"{field} must be a JSON object")
    return value


def _angles(run: RunData) -> list[float]:
    acquisition = _mapping(run.manifest["acquisition"], "acquisition")
    value = acquisition.get("transmit_angles_deg")
    if not isinstance(value, list) or len(value) != 5:
        raise AngleDiagnosticError(
            "acquisition.transmit_angles_deg must contain five angles"
        )
    angles: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AngleDiagnosticError(
                f"acquisition.transmit_angles_deg[{index}] must be finite"
            )
        angle = float(item)
        if not math.isfinite(angle):
            raise AngleDiagnosticError(
                f"acquisition.transmit_angles_deg[{index}] must be finite"
            )
        angles.append(angle)
    if not np.allclose(angles, _EXPECTED_ANGLES_DEG, rtol=0.0, atol=1e-9):
        raise AngleDiagnosticError(
            "transmit angle order must be exactly [-6, -3, 0, 3, 6] degrees"
        )
    return angles


def _validate_processing(run: RunData) -> tuple[int, int]:
    processing = _mapping(run.manifest["processing"], "processing")
    keep_start = processing.get("svd_keep_start")
    keep_end = processing.get("svd_keep_end")
    if (
        isinstance(keep_start, bool)
        or not isinstance(keep_start, int)
        or isinstance(keep_end, bool)
        or not isinstance(keep_end, int)
        or not 1 <= keep_start <= keep_end <= 381
    ):
        raise AngleDiagnosticError(
            "processing SVD range must satisfy "
            "1 <= svd_keep_start <= svd_keep_end <= 381"
        )
    if processing.get("svd_indexing") != "one_based_inclusive":
        raise AngleDiagnosticError(
            "processing.svd_indexing must be 'one_based_inclusive'"
        )
    expected_semantics = "coherent angle sum followed by one global temporal SVD"
    if processing.get("power_linear_semantics") != expected_semantics:
        raise AngleDiagnosticError(
            "processing.power_linear_semantics must declare the standard "
            "power_linear artifact as the coherent five-angle result"
        )
    for field in ("channel_repair", "motion_correction", "block_svd"):
        if processing.get(field) is not False:
            raise AngleDiagnosticError(f"processing.{field} must be false")
    return keep_start, keep_end


def _validate_real_diagnostic_contract(run: RunData) -> None:
    input_config = _mapping(run.manifest["input"], "input")
    if input_config.get("frame_count") != 381:
        raise AngleDiagnosticError("input.frame_count must be 381")
    if input_config.get("shape") != [381, 5, 2816, 128]:
        raise AngleDiagnosticError(
            "input.shape must be exactly [381, 5, 2816, 128]"
        )
    channel_mapping = _mapping(run.manifest["channel_mapping"], "channel_mapping")
    expected_mapping = {
        "source_order": "descending_x",
        "internal_order": "ascending_x",
        "reversed": True,
    }
    for field, expected in expected_mapping.items():
        if channel_mapping.get(field) != expected:
            raise AngleDiagnosticError(
                f"channel_mapping.{field} must be {expected!r}"
            )


def _load_power_maps(
    run: RunData,
) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array, str]:
    grid = _mapping(run.manifest["image_grid"], "image_grid")
    nz = grid.get("nz")
    nx = grid.get("nx")
    if isinstance(nz, bool) or not isinstance(nz, int) or nz <= 0:
        raise AngleDiagnosticError("image_grid.nz must be a positive integer")
    if isinstance(nx, bool) or not isinstance(nx, int) or nx <= 0:
        raise AngleDiagnosticError("image_grid.nx must be a positive integer")
    artifacts = _mapping(run.manifest["artifacts"], "artifacts")
    required = ("angle_power_linear", "incoherent_power_linear", "angle_coherence")
    missing = [key for key in required if artifacts.get(key) is None]
    if missing:
        raise AngleDiagnosticError(
            f"artifacts.{missing[0]} is required; pre-SVD angle_beamformed_iq "
            "must not be substituted for a Power Doppler artifact"
        )
    angle_power = read_float32_artifact(
        run.run_directory,
        artifacts["angle_power_linear"],
        field="artifacts.angle_power_linear",
        expected_shape=(5, nz, nx),
        expected_layout=["angle", "z", "x"],
        require_non_negative=True,
    )
    coherent_power = run.power
    incoherent_power = read_float32_artifact(
        run.run_directory,
        artifacts["incoherent_power_linear"],
        field="artifacts.incoherent_power_linear",
        expected_shape=(nz, nx),
        expected_layout=["z", "x"],
        require_non_negative=True,
    )

    coherence = read_float32_artifact(
        run.run_directory,
        artifacts["angle_coherence"],
        field="artifacts.angle_coherence",
        expected_shape=(nz, nx),
        expected_layout=["z", "x"],
        minimum=0.0,
        maximum=1.0,
    )
    coherence_source = "angle_coherence"
    if np.any(coherence > 1.00001):
        raise AngleDiagnosticError("angle coherence coefficient must be <= 1")
    coherence = np.clip(coherence, 0.0, 1.0).astype("<f4", copy=False)
    return angle_power, coherent_power, incoherent_power, coherence, coherence_source


def _region_metrics(
    mask: npt.NDArray[np.bool_],
    coherent_power: Float32Array,
    incoherent_power: Float32Array,
    coherence: Float32Array,
) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"pixel_count": 0, "pixel_fraction": 0.0}
    coherent_sum = float(coherent_power[mask].sum(dtype=np.float64))
    incoherent_sum = float(incoherent_power[mask].sum(dtype=np.float64))
    return {
        "pixel_count": count,
        "pixel_fraction": count / mask.size,
        "coherent_power_sum": coherent_sum,
        "incoherent_power_sum": incoherent_sum,
        "coherent_to_incoherent_power_ratio": (
            coherent_sum / incoherent_sum if incoherent_sum > 0.0 else None
        ),
        "normalized_coherent_power_ratio": (
            coherent_sum / (5.0 * incoherent_sum)
            if incoherent_sum > 0.0
            else None
        ),
        "coherence_mean": float(np.mean(coherence[mask], dtype=np.float64)),
        "coherence_median": float(np.median(coherence[mask])),
    }


def _metrics(
    angle_power: Float32Array,
    coherent_power: Float32Array,
    incoherent_power: Float32Array,
    coherence: Float32Array,
    angles: list[float],
    svd_keep_range: tuple[int, int],
) -> dict[str, Any]:
    positive = incoherent_power[incoherent_power > 0.0]
    support_threshold = (
        float(np.percentile(positive, 90.0)) if positive.size else None
    )
    masks = {
        "full_image": np.ones(coherence.shape, dtype=bool),
        "low_angle_coherence_lt_0p3": coherence < 0.3,
        "intermediate_angle_coherence_0p3_to_0p7": (
            (coherence >= 0.3) & (coherence < 0.7)
        ),
        "high_angle_coherence_ge_0p7": coherence >= 0.7,
        "strongest_incoherent_power_top_10_percent": (
            incoherent_power >= support_threshold
            if support_threshold is not None
            else np.zeros(coherence.shape, dtype=bool)
        ),
    }
    keep_start, keep_end = svd_keep_range
    svd_range_text = f"[{keep_start},{keep_end}]"
    return {
        "svd": {
            "keep_start": keep_start,
            "keep_end": keep_end,
            "indexing": "one_based_inclusive",
        },
        "definitions": {
            "angle_power": (
                f"CUDA: independently apply global temporal SVD {svd_range_text} to "
                "each angle, then sum_frame(abs(filtered_iq_angle)^2)"
            ),
            "coherent_power": (
                "CUDA: sum the five pre-SVD beamformed angle IQ arrays first, "
                f"apply global temporal SVD {svd_range_text} once, then "
                "sum_frame(abs(iq)^2)"
            ),
            "incoherent_power": (
                "CUDA: sum the five angle_power maps produced by independent "
                "per-angle SVD projections"
            ),
            "angle_coherence": (
                "CUDA pre-SVD normalized coherent power: "
                "sum_frame(abs(sum_angle(beamformed_iq))^2) / "
                "(5 * sum_frame_angle(abs(beamformed_iq)^2))"
            ),
            "normalized_coherent_power_ratio": (
                "coherent_power_sum / (5 * incoherent_power_sum)"
            ),
        },
        "angles": [
            {
                "angle_deg": angle,
                "power_sum": float(angle_power[index].sum(dtype=np.float64)),
                "power_mean": float(angle_power[index].mean(dtype=np.float64)),
            }
            for index, angle in enumerate(angles)
        ],
        "support_threshold_incoherent_power_p90": support_threshold,
        "regions": {
            name: _region_metrics(mask, coherent_power, incoherent_power, coherence)
            for name, mask in masks.items()
        },
    }


def _write_npy(path: Path, array: npt.NDArray[Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("xb") as output:
        np.save(output, array, allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, path)


def _extent(run: RunData) -> list[float]:
    grid = _mapping(run.manifest["image_grid"], "image_grid")
    coordinates: list[float] = []
    for field in ("x_min_m", "x_max_m", "z_min_m", "z_max_m"):
        value = grid.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AngleDiagnosticError(f"image_grid.{field} must be finite")
        coordinate = float(value) * 1000.0
        if not math.isfinite(coordinate):
            raise AngleDiagnosticError(f"image_grid.{field} must be finite")
        coordinates.append(coordinate)
    x_min, x_max, z_min, z_max = coordinates
    if not x_min < x_max or not z_min < z_max:
        raise AngleDiagnosticError("image grid coordinate bounds must increase")
    return [x_min, x_max, z_max, z_min]


def _display_range(
    value: tuple[float, float] | None,
    full_range: tuple[float, float],
    field: str,
) -> tuple[float, float]:
    if value is None:
        return full_range
    minimum, maximum = value
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise AngleDiagnosticError(f"{field} values must be finite")
    if not minimum < maximum:
        raise AngleDiagnosticError(f"{field} requires MIN < MAX")
    full_minimum, full_maximum = full_range
    if minimum < full_minimum or maximum > full_maximum:
        raise AngleDiagnosticError(
            f"{field} must remain inside manifest image grid "
            f"[{full_minimum}, {full_maximum}] mm"
        )
    return value


def _render(
    path: Path,
    power_db: Float32Array,
    coherence: Float32Array,
    angles: list[float],
    extent: list[float],
    *,
    display_x_range_mm: tuple[float, float],
    display_z_range_mm: tuple[float, float],
    colormap: str,
    db_minimum: float,
    db_maximum: float,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(14.4, 8.0), constrained_layout=True)
    titles = [f"Angle {angle:g} deg" for angle in angles] + [
        "Coherent five-angle power",
        "Incoherent five-angle power",
    ]
    power_images = []
    for index, title in enumerate(titles):
        axis = axes.flat[index]
        rendered = axis.imshow(
            power_db[index],
            cmap=colormap,
            vmin=db_minimum,
            vmax=db_maximum,
            origin="upper",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
        )
        power_images.append(rendered)
        axis.set_title(title)
        axis.set_xlabel("Lateral position (mm)")
        axis.set_ylabel("Depth (mm)")
        axis.set_xlim(*display_x_range_mm)
        axis.set_ylim(display_z_range_mm[1], display_z_range_mm[0])
    coherence_axis = axes.flat[7]
    coherence_image = coherence_axis.imshow(
        coherence,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
    )
    coherence_axis.set_title("DAS pre-SVD normalized power coherence")
    coherence_axis.set_xlabel("Lateral position (mm)")
    coherence_axis.set_ylabel("Depth (mm)")
    coherence_axis.set_xlim(*display_x_range_mm)
    coherence_axis.set_ylim(display_z_range_mm[1], display_z_range_mm[0])
    power_colorbar = figure.colorbar(
        power_images[0], ax=list(axes.flat[:7]), shrink=0.85
    )
    power_colorbar.set_label("Power (dB re shared absolute reference)")
    coherence_colorbar = figure.colorbar(
        coherence_image, ax=coherence_axis, shrink=0.85
    )
    coherence_colorbar.set_label("Angle coherence coefficient")
    figure.savefig(
        path,
        format="png",
        dpi=dpi,
        metadata={"Software": "carotid-doppler five-angle diagnostic"},
    )
    plt.close(figure)


def diagnose_angles(
    run_directory: str | Path,
    render_config_path: str | Path,
    diagnostic_id: str,
    *,
    x_range_mm: tuple[float, float] | None = None,
    z_range_mm: tuple[float, float] | None = None,
) -> Path:
    """Render explicit CUDA five-angle Power Doppler diagnostic maps."""

    if not _SAFE_ID.fullmatch(diagnostic_id):
        raise AngleDiagnosticError(
            "diagnostic_id must start with an alphanumeric character and contain "
            "only letters, digits, '.', '_' or '-'"
        )
    run = load_run(run_directory)
    angles = _angles(run)
    _validate_real_diagnostic_contract(run)
    svd_keep_range = _validate_processing(run)
    config = _load_render_config(Path(render_config_path).resolve())
    colormap = config["image"]["colormap"]
    if colormap not in _POWER_COLORMAPS:
        raise AngleDiagnosticError(
            "five-angle Power Doppler colormap must be one of: "
            + ", ".join(sorted(_POWER_COLORMAPS))
        )
    angle_power, coherent_power, incoherent_power, coherence, coherence_source = (
        _load_power_maps(run)
    )
    reference = config["reference"]
    reference_power = float(reference["reference_power"])
    epsilon = reference_power * _EPSILON_RATIO
    all_power = np.concatenate(
        (angle_power, coherent_power[None], incoherent_power[None]), axis=0
    )
    power_db = (
        10.0 * np.log10(np.maximum(all_power, epsilon) / reference_power)
    ).astype("<f4")
    metrics = _metrics(
        angle_power,
        coherent_power,
        incoherent_power,
        coherence,
        angles,
        svd_keep_range,
    )
    image_extent = _extent(run)
    display_x_range = _display_range(
        x_range_mm, (image_extent[0], image_extent[1]), "x_range_mm"
    )
    display_z_range = _display_range(
        z_range_mm, (image_extent[3], image_extent[2]), "z_range_mm"
    )

    output_directory = run.run_directory / "renders" / diagnostic_id
    if output_directory.exists():
        if not output_directory.is_dir() or any(output_directory.iterdir()):
            raise AngleDiagnosticError(
                f"diagnostic output directory must be new or empty: {output_directory}"
            )
    else:
        output_directory.mkdir(parents=True)
    outputs = [
        output_directory / "power_db_f32.npy",
        output_directory / "angle_coherence_f32.npy",
        output_directory / "five_angle_comparison.png",
        output_directory / "metrics.json",
        output_directory / "angle_diagnostic_manifest.json",
    ]
    try:
        _write_npy(outputs[0], power_db)
        _write_npy(outputs[1], coherence.astype("<f4", copy=False))
        partial_image = outputs[2].with_name(outputs[2].name + ".partial")
        _render(
            partial_image,
            power_db,
            coherence,
            angles,
            image_extent,
            display_x_range_mm=display_x_range,
            display_z_range_mm=display_z_range,
            colormap=colormap,
            db_minimum=float(config["db"]["minimum"]),
            db_maximum=float(config["db"]["maximum"]),
            dpi=int(config["image"]["dpi"]),
        )
        os.replace(partial_image, outputs[2])
        _write_json(outputs[3], metrics)
        _write_json(
            outputs[4],
            {
                "schema_version": 1,
                "diagnostic_id": diagnostic_id,
                "status": "complete",
                "source": {
                    "run_id": run.run_id,
                    "run_manifest": "../../run_manifest.json",
                    "angles_deg": angles,
                    "power_artifacts": [
                        "angle_power_linear",
                        "power_linear (coherent five-angle semantics)",
                        "incoherent_power_linear",
                    ],
                    "coherence_source": coherence_source,
                    "coherence_stage": "DAS pre-SVD",
                    "coherence_definition": (
                        "sum_frame(abs(sum_angle(beamformed_iq))^2) / "
                        "(5 * sum_frame_angle(abs(beamformed_iq)^2))"
                    ),
                    "pre_svd_angle_beamformed_iq_used_as_power": False,
                },
                "processing": {
                    "python_svd": False,
                    "svd_keep_start": svd_keep_range[0],
                    "svd_keep_end": svd_keep_range[1],
                    "svd_indexing": "one_based_inclusive",
                    "svd_range_changed": False,
                    "motion_correction_added": False,
                    "block_svd_added": False,
                    "channel_interpolation_added": False,
                },
                "reference": reference,
                "db": {
                    "formula": (
                        "10*log10(max(power, "
                        "reference_power*1e-12)/reference_power)"
                    ),
                    "reference_power": reference_power,
                    "epsilon": epsilon,
                    "minimum": float(config["db"]["minimum"]),
                    "maximum": float(config["db"]["maximum"]),
                    "shared_by_all_power_panels": True,
                    "stored_array_clipped_to_display_range": False,
                    "power_map_order": [
                        "angle_-6_deg",
                        "angle_-3_deg",
                        "angle_0_deg",
                        "angle_3_deg",
                        "angle_6_deg",
                        "coherent_five_angle",
                        "incoherent_five_angle",
                    ],
                },
                "image": {
                    "path": outputs[2].name,
                    "power_colormap": colormap,
                    "coherence_colormap": "gray",
                    "coherence_range": [0.0, 1.0],
                    "power_colorbar_shared": True,
                    "array_layout": ["z", "x"],
                    "depth_direction": "down",
                    "display_x_range_mm": list(display_x_range),
                    "display_z_range_mm": list(display_z_range),
                    "display_crop_changes_arrays_or_db": False,
                },
                "artifacts": {
                    "power_db": outputs[0].name,
                    "angle_coherence": outputs[1].name,
                    "metrics": outputs[3].name,
                },
                "software": {
                    "carotid_doppler": __version__,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "matplotlib": matplotlib.__version__,
                },
            },
        )
    except Exception:
        plt.close("all")
        for output in outputs:
            output.unlink(missing_ok=True)
            output.with_name(output.name + ".partial").unlink(missing_ok=True)
        try:
            output_directory.rmdir()
        except OSError:
            pass
        raise
    return output_directory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render CUDA-provided five-angle Power Doppler and coherence maps."
        )
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--render-config", required=True, type=Path)
    parser.add_argument("--diagnostic-id", required=True)
    parser.add_argument("--x-range-mm", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--z-range-mm", nargs=2, type=float, metavar=("MIN", "MAX"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        output = diagnose_angles(
            arguments.run,
            arguments.render_config,
            arguments.diagnostic_id,
            x_range_mm=(tuple(arguments.x_range_mm) if arguments.x_range_mm else None),
            z_range_mm=(tuple(arguments.z_range_mm) if arguments.z_range_mm else None),
        )
    except (
        AngleDiagnosticError,
        ContractError,
        RenderConfigError,
        OSError,
        ValueError,
    ) as error:
        print(f"carotid_doppler.angle_diagnostic: {error}", file=sys.stderr)
        return 2
    print(f"complete five-angle diagnostic directory: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

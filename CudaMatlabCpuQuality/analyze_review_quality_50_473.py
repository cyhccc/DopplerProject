#!/usr/bin/env python3
"""Reproducible CPU-versus-GPU power-Doppler quality analysis for reviewer 2.

The workflow is deliberately split in two:

1. ``--prepare-roi`` reads CPU reference images only, writes coordinate figures,
   and creates an unlocked ROI configuration template.
2. ``--analyze`` requires a locked CPU-only ROI configuration and then reads GPU
   images to calculate NRMSE, SSIM, figures, profiles, and provenance metadata.

Input files are float32 linear-power arrays in C order with shape 180 x 256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DEPS = Path(
    r"E:\Notes\Work\my\论文修改\基于连续帧并行优化的超快功率多普勒成像GPU加速方法_reading_files\review2_deps"
)
if DEFAULT_DEPS.is_dir():
    sys.path.insert(0, str(DEFAULT_DEPS))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402
from skimage import __version__ as skimage_version  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402


WIDTH = 256
HEIGHT = 180
SSIM_WIN_SIZE = 11
SSIM_RADIUS = SSIM_WIN_SIZE // 2
SSIM_SIGMA = 1.5
SSIM_DB_MIN = -30.0
SSIM_DB_MAX = 0.0
FRAME_LABELS = ("800", "60000")
ROI_NAMES = ("global", "weak_flow", "vessel_edge")
X0_MM = -6.35
DX_MM = 0.05
Z0_MM = 1.77408
DZ_MM = 0.04928
WEAK_FLOW_CANDIDATE = {"x0": 195, "y0": 29, "x1": 216, "y1": 49}
VESSEL_EDGE_CANDIDATE = {"x0": 89, "y0": 19, "x1": 108, "y1": 48}
DEFAULT_OUTPUT_DIR = Path(
    r"E:\Notes\Work\my\论文修改\基于连续帧并行优化的超快功率多普勒成像GPU加速方法_reading_files\review2_50_473"
)
PAPER_WIDTH_IN = 7.2
PAPER_LABEL_PT = 9.5
PAPER_TICK_PT = 9.0
PAPER_TITLE_PT = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CPU-only ROIs or analyze CPU/GPU Doppler quality."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-roi", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    parser.add_argument("--cpu-800", type=Path)
    parser.add_argument("--cpu-60000", type=Path)
    parser.add_argument("--gpu-800", type=Path)
    parser.add_argument("--gpu-60000", type=Path)
    parser.add_argument(
        "--frames",
        choices=("800", "60000", "both"),
        default="both",
        help="Frame-count subset for staged analysis; final analysis uses 'both'.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--roi-config",
        type=Path,
        help="ROI JSON. Defaults to OUTPUT_DIR/roi_config.json.",
    )
    parser.add_argument(
        "--overwrite-config",
        action="store_true",
        help="Allow --prepare-roi to replace an existing ROI configuration.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_power(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} file: {path}")
    data = np.fromfile(path, dtype=np.float32)
    expected = HEIGHT * WIDTH
    if data.size != expected:
        raise ValueError(
            f"Unexpected {label} size: expected {expected} float32 values, "
            f"got {data.size} ({path})."
        )
    image = data.reshape((HEIGHT, WIDTH), order="C")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{label} contains NaN or infinity: {path}")
    if np.any(image < 0.0):
        raise ValueError(f"{label} contains negative values but linear power is required: {path}")
    return image


def require_paths(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        switches = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(f"Required arguments: {switches}")


def doppler_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "doppler_black_red_yellow",
        [
            (0.00, "#000000"),
            (0.18, "#300000"),
            (0.40, "#8b0000"),
            (0.62, "#ff2a00"),
            (0.82, "#ffbf00"),
            (0.94, "#fff04a"),
            (1.00, "#fffef2"),
        ],
    )


def power_to_db(power: np.ndarray, cpu_peak: float) -> np.ndarray:
    if not np.isfinite(cpu_peak) or cpu_peak <= 0.0:
        raise ValueError(f"CPU reference peak must be positive, got {cpu_peak!r}.")
    result = np.full(power.shape, -np.inf, dtype=np.float64)
    positive = power > 0.0
    result[positive] = 10.0 * np.log10(power[positive].astype(np.float64) / cpu_peak)
    return result


def ssim_input(power: np.ndarray, cpu_peak: float) -> np.ndarray:
    db = power_to_db(power, cpu_peak)
    mapped = (np.clip(db, SSIM_DB_MIN, SSIM_DB_MAX) - SSIM_DB_MIN) / (
        SSIM_DB_MAX - SSIM_DB_MIN
    )
    return mapped.astype(np.float64, copy=False)


def difference_db(cpu: np.ndarray, gpu: np.ndarray, cpu_peak: float) -> tuple[np.ndarray, np.ndarray]:
    raw = gpu.astype(np.float64) - cpu.astype(np.float64)
    absolute = np.abs(raw)
    db = np.full(raw.shape, -np.inf, dtype=np.float64)
    nonzero = absolute > 0.0
    db[nonzero] = 10.0 * np.log10(absolute[nonzero] / cpu_peak)
    return raw, db


def base_config(cpu_paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "locked": False,
        "frame_locks": {"800": False, "60000": False},
        "selection_records": {"800": None, "60000": None},
        "selection_basis": "cpu_only",
        "selection_note": (
            "Choose weak-flow and edge coordinates from CPU reference coordinate figures only. "
            "Do not inspect GPU images or GPU error maps before locking this file."
        ),
        "image": {
            "width": WIDTH,
            "height": HEIGHT,
            "input_dtype": "float32",
            "input_order": "C",
            "coordinate_convention": "zero-based pixels; rectangles use exclusive x1/y1",
            "physical_axes_mm": {"x0": X0_MM, "dx": DX_MM, "z0": Z0_MM, "dz": DZ_MM},
        },
        "cpu_reference_files": {key: str(value.resolve()) for key, value in cpu_paths.items()},
        "rois": {
            "global": {
                "class": "global",
                "method": "interior",
                "border_px": SSIM_RADIUS,
                "selection_basis": "fixed_protocol",
            },
            "weak_flow": {
                "class": "weak_flow",
                "method": "rectangle",
                **WEAK_FLOW_CANDIDATE,
                "selection_basis": "cpu_only",
                "rationale": (
                    "Candidate from CPU only: recognizable weak fine vessel around "
                    "x=2.6..4.3 mm, z=3.2..4.9 mm. Review before locking."
                ),
            },
            "vessel_edge": {
                "class": "vessel_edge",
                "method": "cpu_threshold_contour_band",
                "threshold_db": -12.0,
                "band_px": 1,
                "box": dict(VESSEL_EDGE_CANDIDATE),
                "selection_basis": "cpu_only",
                "rationale": (
                    "Candidate from CPU only: strong vertical vessel boundary around "
                    "x=-1.9..-1.0 mm, z=2.7..4.1 mm; CPU-only review selected -12 dB."
                ),
            },
        },
        "profiles": {
            frame: {
                "axis": "horizontal",
                "index": 34,
                "start": 84,
                "stop": 114,
                "selection_basis": "cpu_only",
                "rationale": "Choose a CPU-reference cross-section before GPU inspection.",
            }
            for frame in FRAME_LABELS
        },
    }


def reference_figure(cpu_images: dict[str, np.ndarray], output_dir: Path) -> None:
    cmap = doppler_colormap()
    fig, axes = plt.subplots(
        len(cpu_images), 1, figsize=(10.2, 4.5 * len(cpu_images)), dpi=200, squeeze=False
    )
    image_artist = None
    for ax, (frame, power) in zip(axes[:, 0], cpu_images.items()):
        peak = float(np.max(power))
        db = power_to_db(power, peak)
        image_artist = ax.imshow(db, cmap=cmap, vmin=-30.0, vmax=0.0, origin="upper", aspect="equal")
        ax.set_title(f"CPU reference, {frame} frames")
        ax.set_xlabel("Lateral pixel x")
        ax.set_ylabel("Depth pixel y")
        ax.set_xticks(np.arange(0, WIDTH, 16), minor=True)
        ax.set_yticks(np.arange(0, HEIGHT, 10), minor=True)
        ax.set_xticks(np.arange(0, WIDTH + 1, 32))
        ax.set_yticks(np.arange(0, HEIGHT + 1, 20))
        ax.grid(which="major", color="white", linewidth=0.45, alpha=0.55)
        ax.grid(which="minor", color="white", linewidth=0.25, alpha=0.22)
        ax.set_xlim(-0.5, WIDTH - 0.5)
        ax.set_ylim(HEIGHT - 0.5, -0.5)
    assert image_artist is not None
    fig.colorbar(image_artist, ax=axes[:, 0].tolist(), label="Power relative to CPU peak (dB)", shrink=0.82)
    fig.suptitle("CPU-only ROI selection reference (zero-based pixel coordinates)", fontsize=13)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.92, bottom=0.07, hspace=0.32)
    save_figure(fig, output_dir / "cpu_reference_coordinates")
    if len(cpu_images) == 1:
        only_frame = next(iter(cpu_images))
        # Keep an immutable frame-specific reference when preparation is staged.
        reference_figure_single(cpu_images[only_frame], only_frame, output_dir)


def reference_figure_single(power: np.ndarray, frame: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.5), dpi=200, layout="constrained")
    peak = float(np.max(power))
    artist = ax.imshow(
        power_to_db(power, peak), cmap=doppler_colormap(), vmin=-30.0, vmax=0.0,
        origin="upper", aspect="equal"
    )
    ax.set_title(f"CPU reference, {frame} frames")
    ax.set_xlabel("Lateral pixel x")
    ax.set_ylabel("Depth pixel y")
    ax.set_xticks(np.arange(0, WIDTH, 16), minor=True)
    ax.set_yticks(np.arange(0, HEIGHT, 10), minor=True)
    ax.set_xticks(np.arange(0, WIDTH + 1, 32))
    ax.set_yticks(np.arange(0, HEIGHT + 1, 20))
    ax.grid(which="major", color="white", linewidth=0.45, alpha=0.55)
    ax.grid(which="minor", color="white", linewidth=0.25, alpha=0.22)
    ax.set_xlim(-0.5, WIDTH - 0.5)
    ax.set_ylim(HEIGHT - 0.5, -0.5)
    fig.colorbar(artist, ax=ax, label="Power relative to CPU peak (dB)", pad=0.035)
    save_figure(fig, output_dir / f"cpu_reference_coordinates_{frame}")


def candidate_roi_figure(cpu_images: dict[str, np.ndarray], output_dir: Path) -> None:
    """Show the predeclared candidate boxes using CPU images and nothing else."""
    cmap = doppler_colormap()
    for frame, power in cpu_images.items():
        peak = float(np.max(power))
        db = power_to_db(power, peak)
        fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), dpi=200, layout="constrained")
        boxes = (
            (None, "CPU reference"),
            (WEAK_FLOW_CANDIDATE, "Weak-flow candidate"),
            (VESSEL_EDGE_CANDIDATE, "Vessel-edge candidate"),
        )
        artist = None
        for ax, (box, title) in zip(axes, boxes):
            artist = ax.imshow(db, cmap=cmap, vmin=-30.0, vmax=0.0, origin="upper", aspect="equal")
            ax.set_title(title)
            ax.set_xlabel("Lateral pixel x")
            ax.set_ylabel("Depth pixel y")
            if box is None:
                for candidate, color in (
                    (WEAK_FLOW_CANDIDATE, "#00e5ff"),
                    (VESSEL_EDGE_CANDIDATE, "#66ff66"),
                ):
                    ax.add_patch(
                        Rectangle(
                            (candidate["x0"] - 0.5, candidate["y0"] - 0.5),
                            candidate["x1"] - candidate["x0"],
                            candidate["y1"] - candidate["y0"],
                            fill=False,
                            edgecolor=color,
                            linewidth=1.2,
                        )
                    )
            else:
                pad = 4
                ax.set_xlim(max(-0.5, box["x0"] - pad - 0.5), min(WIDTH - 0.5, box["x1"] + pad - 0.5))
                ax.set_ylim(min(HEIGHT - 0.5, box["y1"] + pad - 0.5), max(-0.5, box["y0"] - pad - 0.5))
                ax.add_patch(
                    Rectangle(
                        (box["x0"] - 0.5, box["y0"] - 0.5),
                        box["x1"] - box["x0"],
                        box["y1"] - box["y0"],
                        fill=False,
                        edgecolor="#00e5ff",
                        linewidth=1.0,
                    )
                )
                ax.set_xticks(np.arange(box["x0"], box["x1"] + 1, 5))
                ax.set_yticks(np.arange(box["y0"], box["y1"] + 1, 5))
                ax.grid(color="white", linewidth=0.35, alpha=0.35)
        assert artist is not None
        fig.colorbar(
            artist, ax=axes.tolist(), label="Power relative to CPU peak (dB)",
            fraction=0.025, pad=0.035, shrink=0.88
        )
        fig.suptitle(
            f"CPU-only candidate ROIs, {frame} frames | "
            f"x={X0_MM}+pixel*{DX_MM} mm, z={Z0_MM}+pixel*{DZ_MM} mm",
            fontsize=12,
        )
        save_figure(fig, output_dir / f"cpu_candidate_rois_{frame}")


def cpu_edge_mask_preview(
    cpu_images: dict[str, np.ndarray], config: dict[str, Any], output_dir: Path
) -> dict[str, int]:
    """Render the configured threshold contour from CPU references only."""
    counts: dict[str, int] = {}
    edge_spec = config["rois"]["vessel_edge"]
    for frame, cpu in cpu_images.items():
        cpu_peak = float(np.max(cpu))
        cpu_db = power_to_db(cpu, cpu_peak)
        nrmse_masks, _ = make_roi_masks(config, cpu, cpu_peak)
        edge_mask = nrmse_masks["vessel_edge"]
        counts[frame] = int(np.count_nonzero(edge_mask))
        x0, y0, x1, y1 = mask_bbox(rectangle_mask(edge_spec["box"], "rois.vessel_edge.box"), pad=4)
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), dpi=200, layout="constrained")
        for index, ax in enumerate(axes[:2]):
            artist = ax.imshow(
                cpu_db, cmap=doppler_colormap(), vmin=-30.0, vmax=0.0,
                origin="upper", aspect="equal"
            )
            ax.contour(edge_mask.astype(float), levels=[0.5], colors=["#66ff66"], linewidths=0.9)
            ax.set_title("CPU full view" if index == 0 else "CPU edge-box view")
            ax.set_xlabel("Lateral pixel x")
            ax.set_ylabel("Depth pixel y")
            if index == 1:
                ax.set_xlim(x0 - 0.5, x1 - 0.5)
                ax.set_ylim(y1 - 0.5, y0 - 0.5)
        axes[2].imshow(edge_mask, cmap="gray", vmin=0, vmax=1, origin="upper", aspect="equal")
        axes[2].set_xlim(x0 - 0.5, x1 - 0.5)
        axes[2].set_ylim(y1 - 0.5, y0 - 0.5)
        axes[2].set_title(f"CPU-only mask ({counts[frame]} pixels)")
        axes[2].set_xlabel("Lateral pixel x")
        axes[2].set_ylabel("Depth pixel y")
        fig.colorbar(artist, ax=axes[:2].tolist(), label="Power / CPU peak (dB)", pad=0.03, shrink=0.82)
        if edge_spec.get("method") == "fixed_cpu_reference_contour_band":
            figure_title = (
                f"Fixed 800-frame CPU contour applied to {frame} frames "
                f"({edge_spec['threshold_db']:g} dB, {edge_spec['band_px']}-pixel inner/outer band)"
            )
            output_stem = output_dir / f"cpu_edge_fixed800_mask_{frame}"
        else:
            figure_title = (
                f"CPU-only vessel-edge check, {frame} frames: "
                f"{edge_spec['threshold_db']:g} dB, {edge_spec['band_px']}-pixel inner/outer band"
            )
            output_stem = output_dir / f"cpu_edge_mask_preview_{frame}_native"
        fig.suptitle(figure_title, fontsize=11)
        save_figure(fig, output_stem)
    return counts


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def prepare_roi(args: argparse.Namespace, roi_config: Path) -> None:
    cpu_paths = {
        frame: path
        for frame, path in (("800", args.cpu_800), ("60000", args.cpu_60000))
        if path is not None
    }
    if not cpu_paths:
        raise ValueError("--prepare-roi requires --cpu-800 and/or --cpu-60000.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu_images = {frame: load_power(path, f"CPU {frame}") for frame, path in cpu_paths.items()}
    reference_figure(cpu_images, args.output_dir)
    candidate_roi_figure(cpu_images, args.output_dir)
    config_was_written = not roi_config.exists() or args.overwrite_config
    if config_was_written:
        roi_config.parent.mkdir(parents=True, exist_ok=True)
        roi_config.write_text(
            json.dumps(base_config(cpu_paths), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    preview_config = json.loads(roi_config.read_text(encoding="utf-8"))
    edge_mask_counts = cpu_edge_mask_preview(cpu_images, preview_config, args.output_dir)
    manifest = {
        "created_utc": utc_now(),
        "mode": "prepare_roi_cpu_only",
        "gpu_files_read": False,
        "cpu_inputs": {
            frame: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for frame, path in cpu_paths.items()
        },
        "outputs": {
            "coordinate_png": str((args.output_dir / "cpu_reference_coordinates.png").resolve()),
            "coordinate_pdf": str((args.output_dir / "cpu_reference_coordinates.pdf").resolve()),
            "candidate_roi_pngs": [
                str((args.output_dir / f"cpu_candidate_rois_{frame}.png").resolve()) for frame in cpu_images
            ],
            "roi_config": str(roi_config.resolve()),
            "roi_config_action": "written" if config_was_written else "preserved_existing",
            "cpu_only_edge_mask_pixel_counts": edge_mask_counts,
        },
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (args.output_dir / "roi_prepare_metadata.json").write_text(manifest_text, encoding="utf-8")
    frame_suffix = "_".join(cpu_images)
    (args.output_dir / f"roi_prepare_metadata_{frame_suffix}.json").write_text(
        manifest_text, encoding="utf-8"
    )
    print(f"[OK] CPU-only coordinate references: {args.output_dir}")
    if config_was_written:
        print(f"[ACTION] Fill CPU-selected coordinates, set frame lock(s), then analyze: {roi_config}")
    else:
        print(f"[OK] Preserved existing ROI configuration without modification: {roi_config}")


def as_int(value: Any, field: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"ROI config field {field} must be an integer, not {value!r}.")
    integer = int(value)
    if integer != value:
        raise ValueError(f"ROI config field {field} must be an integer, got {value!r}.")
    return integer


def rectangle_mask(spec: dict[str, Any], field: str) -> np.ndarray:
    x0 = as_int(spec.get("x0"), f"{field}.x0")
    y0 = as_int(spec.get("y0"), f"{field}.y0")
    x1 = as_int(spec.get("x1"), f"{field}.x1")
    y1 = as_int(spec.get("y1"), f"{field}.y1")
    if not (0 <= x0 < x1 <= WIDTH and 0 <= y0 < y1 <= HEIGHT):
        raise ValueError(f"Invalid exclusive rectangle in {field}: {(x0, y0, x1, y1)}.")
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def validate_config(config: dict[str, Any], frames: tuple[str, ...]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("ROI config schema_version must be 1.")
    frame_locks = config.get("frame_locks", {})
    unlocked = [frame for frame in frames if frame_locks.get(frame) is not True]
    if unlocked:
        raise ValueError(
            "CPU-only ROI configuration is not locked for frame count(s): " + ", ".join(unlocked)
        )
    if frames == FRAME_LABELS and config.get("locked") is not True:
        raise ValueError("Final two-frame analysis requires top-level locked=true.")
    if config.get("selection_basis") != "cpu_only":
        raise ValueError("ROI config selection_basis must be exactly 'cpu_only'.")
    image = config.get("image", {})
    if (image.get("width"), image.get("height"), image.get("input_order")) != (WIDTH, HEIGHT, "C"):
        raise ValueError("ROI config must describe 256 x 180 C-order input.")
    rois = config.get("rois", {})
    if set(rois) != set(ROI_NAMES):
        raise ValueError(f"ROI config must contain exactly: {', '.join(ROI_NAMES)}.")
    if rois["global"].get("method") != "interior":
        raise ValueError("Global ROI method must be 'interior'.")
    border = as_int(rois["global"].get("border_px"), "rois.global.border_px")
    if border < SSIM_RADIUS:
        raise ValueError(f"Global border_px must be at least {SSIM_RADIUS} for the SSIM window.")
    if rois["weak_flow"].get("method") != "rectangle":
        raise ValueError("Weak-flow ROI method must be 'rectangle'.")
    if rois["weak_flow"].get("selection_basis") != "cpu_only":
        raise ValueError("Weak-flow ROI selection_basis must be 'cpu_only'.")
    rectangle_mask(rois["weak_flow"], "rois.weak_flow")
    edge = rois["vessel_edge"]
    if edge.get("method") not in (
        "cpu_threshold_contour_band",
        "fixed_cpu_reference_contour_band",
    ):
        raise ValueError(
            "Vessel-edge method must be 'cpu_threshold_contour_band' or "
            "'fixed_cpu_reference_contour_band'."
        )
    if edge.get("selection_basis") != "cpu_only":
        raise ValueError("Vessel-edge ROI selection_basis must be 'cpu_only'.")
    threshold = edge.get("threshold_db")
    if threshold is None or not np.isfinite(float(threshold)) or not (-80.0 <= float(threshold) <= 0.0):
        raise ValueError("Vessel-edge threshold_db must be finite and between -80 and 0 dB.")
    band_px = as_int(edge.get("band_px"), "rois.vessel_edge.band_px")
    if band_px not in (1, 2):
        raise ValueError("Vessel-edge band_px must be 1 or 2.")
    rectangle_mask(edge.get("box", {}), "rois.vessel_edge.box")
    if edge.get("method") == "fixed_cpu_reference_contour_band":
        reference = edge.get("reference_cpu", {})
        reference_path = Path(str(reference.get("path", "")))
        expected_hash = str(reference.get("sha256", "")).lower()
        if reference.get("frame_count") != 800 or not reference_path.is_file():
            raise ValueError("Fixed vessel-edge ROI requires an existing 800-frame CPU reference path.")
        if len(expected_hash) != 64 or sha256_file(reference_path) != expected_hash:
            raise ValueError("Fixed vessel-edge CPU reference SHA256 does not match the ROI config.")
    profiles = config.get("profiles", {})
    for frame in FRAME_LABELS:
        profile = profiles.get(frame)
        if not isinstance(profile, dict):
            raise ValueError(f"Missing profiles.{frame} configuration.")
        if profile.get("selection_basis") != "cpu_only":
            raise ValueError(f"profiles.{frame}.selection_basis must be 'cpu_only'.")
        axis = profile.get("axis")
        if axis not in ("horizontal", "vertical"):
            raise ValueError(f"profiles.{frame}.axis must be horizontal or vertical.")
        index = as_int(profile.get("index"), f"profiles.{frame}.index")
        extent = HEIGHT if axis == "horizontal" else WIDTH
        length = WIDTH if axis == "horizontal" else HEIGHT
        if not 0 <= index < extent:
            raise ValueError(f"profiles.{frame}.index is outside the image.")
        start = as_int(profile.get("start"), f"profiles.{frame}.start")
        stop = as_int(profile.get("stop"), f"profiles.{frame}.stop")
        if not 0 <= start < stop <= length:
            raise ValueError(f"Invalid profiles.{frame} [start, stop) range.")


def valid_ssim_centers() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[SSIM_RADIUS : HEIGHT - SSIM_RADIUS, SSIM_RADIUS : WIDTH - SSIM_RADIUS] = True
    return mask


def make_roi_masks(
    config: dict[str, Any], cpu: np.ndarray, cpu_peak: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rois = config["rois"]
    global_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    border = int(rois["global"]["border_px"])
    global_mask[border : HEIGHT - border, border : WIDTH - border] = True
    weak_mask = rectangle_mask(rois["weak_flow"], "rois.weak_flow")
    edge_spec = rois["vessel_edge"]
    edge_box = rectangle_mask(edge_spec["box"], "rois.vessel_edge.box")
    edge_cpu = cpu
    edge_cpu_peak = cpu_peak
    if edge_spec.get("method") == "fixed_cpu_reference_contour_band":
        reference_path = Path(edge_spec["reference_cpu"]["path"])
        expected_hash = str(edge_spec["reference_cpu"]["sha256"]).lower()
        actual_hash = sha256_file(reference_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Fixed edge-mask CPU reference hash changed: expected {expected_hash}, got {actual_hash}."
            )
        edge_cpu = load_power(reference_path, "fixed edge-mask CPU reference")
        edge_cpu_peak = float(np.max(edge_cpu))
    edge_cpu_db = power_to_db(edge_cpu, edge_cpu_peak)
    foreground = edge_cpu_db >= float(edge_spec["threshold_db"])
    width = int(edge_spec["band_px"])
    structure = ndimage.generate_binary_structure(2, 1)
    outer = ndimage.binary_dilation(foreground, structure=structure, iterations=width)
    inner = ndimage.binary_erosion(foreground, structure=structure, iterations=width)
    edge_mask = np.logical_xor(outer, inner) & edge_box
    nrmse_masks = {
        "global": np.ones((HEIGHT, WIDTH), dtype=bool),
        "weak_flow": weak_mask,
        "vessel_edge": edge_mask,
    }
    ssim_masks = {"global": global_mask, "weak_flow": weak_mask, "vessel_edge": edge_mask}
    supported = valid_ssim_centers()
    for name, mask in ssim_masks.items():
        ssim_masks[name] = mask & supported
        if not np.any(nrmse_masks[name]):
            raise ValueError(f"ROI {name!r} has no NRMSE pixels.")
        if not np.any(ssim_masks[name]):
            raise ValueError(f"ROI {name!r} has no SSIM-window-supported pixels.")
    return nrmse_masks, ssim_masks


def calculate_pair(
    frame: str, cpu: np.ndarray, gpu: np.ndarray, config: dict[str, Any]
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    float,
]:
    cpu_peak = float(np.max(cpu))
    if cpu_peak <= 0.0:
        raise ValueError(f"CPU {frame} reference has no positive power.")
    cpu_mapped = ssim_input(cpu, cpu_peak)
    gpu_mapped = ssim_input(gpu, cpu_peak)
    _, ssim_map = structural_similarity(
        cpu_mapped,
        gpu_mapped,
        data_range=1.0,
        win_size=SSIM_WIN_SIZE,
        gaussian_weights=True,
        sigma=SSIM_SIGMA,
        use_sample_covariance=False,
        full=True,
    )
    nrmse_masks, ssim_masks = make_roi_masks(config, cpu, cpu_peak)
    raw_difference, raw_difference_db = difference_db(cpu, gpu, cpu_peak)
    rows: list[dict[str, Any]] = []
    cpu64 = cpu.astype(np.float64)
    gpu64 = gpu.astype(np.float64)
    for name in ROI_NAMES:
        nrmse_mask = nrmse_masks[name]
        ssim_mask = ssim_masks[name]
        denominator = float(np.linalg.norm(cpu64[nrmse_mask]))
        if denominator == 0.0:
            raise ValueError(f"CPU norm is zero in {frame}/{name}; NRMSE is undefined.")
        nrmse = float(np.linalg.norm((gpu64 - cpu64)[nrmse_mask]) / denominator * 100.0)
        rows.append(
            {
                "frame_count": int(frame),
                "roi_class": name,
                "nrmse_percent": nrmse,
                "ssim": float(np.mean(ssim_map[ssim_mask], dtype=np.float64)),
                "pixel_count": int(np.count_nonzero(ssim_mask)),
                "nrmse_pixel_count": int(np.count_nonzero(nrmse_mask)),
                "ssim_pixel_count": int(np.count_nonzero(ssim_mask)),
            }
        )
    return (
        rows,
        ssim_map,
        nrmse_masks,
        ssim_masks,
        raw_difference,
        raw_difference_db,
        cpu_peak,
    )


def mask_bbox(mask: np.ndarray, pad: int = 3) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(WIDTH, int(xs.max()) + 1 + pad),
        min(HEIGHT, int(ys.max()) + 1 + pad),
    )


def draw_roi_outlines(ax: plt.Axes, masks: dict[str, np.ndarray]) -> None:
    colors = {"weak_flow": "#00e5ff", "vessel_edge": "#66ff66"}
    for name in ("weak_flow", "vessel_edge"):
        mask = masks[name]
        if name == "weak_flow":
            x0, y0, x1, y1 = mask_bbox(mask, pad=0)
            ax.add_patch(
                Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False,
                          edgecolor=colors[name], linewidth=1.2)
            )
        else:
            ax.contour(mask.astype(float), levels=[0.5], colors=[colors[name]], linewidths=0.65)


def comparison_figure(
    pairs: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    frames = tuple(pairs)
    fig, axes = plt.subplots(
        len(frames), 3, figsize=(PAPER_WIDTH_IN, 2.35 * len(frames)), dpi=200, sharex=True,
        sharey=True, squeeze=False, layout="constrained"
    )
    cmap = doppler_colormap()
    artists: list[Any] = [None, None, None]
    for row, frame in enumerate(frames):
        item = pairs[frame]
        cpu_db = power_to_db(item["cpu"], item["cpu_peak"])
        gpu_db = power_to_db(item["gpu"], item["cpu_peak"])
        panels = (
            (cpu_db, "CPU", cmap, -30.0, 0.0),
            (gpu_db, "GPU", cmap, -30.0, 0.0),
            (item["difference_db"], "Raw absolute difference", "viridis", -80.0, 0.0),
        )
        for col, (array, title, panel_cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            artist = ax.imshow(array, origin="upper", aspect="equal", cmap=panel_cmap, vmin=vmin, vmax=vmax)
            artists[col] = artist
            draw_roi_outlines(ax, item["nrmse_masks"])
            if row == 0:
                short_title = ("CPU", "GPU", "|GPU - CPU|")[col]
                ax.set_title(short_title, fontsize=PAPER_TITLE_PT)
            panel_index = row * 3 + col
            ax.text(
                0.02, 0.97, f"({chr(ord('a') + panel_index)})", transform=ax.transAxes,
                ha="left", va="top", color="white", fontsize=PAPER_LABEL_PT, fontweight="bold",
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 1.5},
            )
            if row == len(frames) - 1:
                ax.set_xlabel("Lateral x (pixel)", fontsize=PAPER_LABEL_PT)
            if col == 0:
                ax.set_ylabel(f"{frame} frames\nDepth y (pixel)", fontsize=PAPER_LABEL_PT)
            ax.tick_params(labelsize=PAPER_TICK_PT)
    power_colorbar = fig.colorbar(
        artists[1], ax=axes[:, :2].ravel().tolist(), label="Power (dB)",
        fraction=0.032, pad=0.025, shrink=0.88
    )
    difference_colorbar = fig.colorbar(
        artists[2], ax=axes[:, 2].tolist(), label="Abs. difference (dB)",
        fraction=0.045, pad=0.025, shrink=0.88
    )
    for colorbar in (power_colorbar, difference_colorbar):
        colorbar.ax.tick_params(labelsize=PAPER_TICK_PT)
        colorbar.set_label(colorbar.ax.get_ylabel(), fontsize=PAPER_LABEL_PT)
    save_figure(fig, output_dir / "figure7_quality_comparison")


def roi_zoom_figure(frame: str, item: dict[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(
        2, 3, figsize=(PAPER_WIDTH_IN, 5.0), dpi=200, layout="constrained"
    )
    cmap = doppler_colormap()
    cpu_db = power_to_db(item["cpu"], item["cpu_peak"])
    gpu_db = power_to_db(item["gpu"], item["cpu_peak"])
    arrays = (cpu_db, gpu_db, item["difference_db"])
    titles = ("CPU", "GPU", "Raw absolute difference")
    artists: list[Any] = [None, None, None]
    roi_labels = (("weak_flow", "ROI A: weak flow"), ("vessel_edge", "ROI B: vessel edge"))
    for row, (name, display_name) in enumerate(roi_labels):
        mask = item["nrmse_masks"][name]
        x0, y0, x1, y1 = mask_bbox(mask, pad=4)
        for col, (array, title) in enumerate(zip(arrays, titles)):
            ax = axes[row, col]
            panel_cmap = cmap if col < 2 else "viridis"
            vmin, vmax = ((-30.0, 0.0) if col < 2 else (-80.0, 0.0))
            artists[col] = ax.imshow(
                array, origin="upper", aspect="equal", cmap=panel_cmap, vmin=vmin, vmax=vmax
            )
            outline_color = "#00e5ff" if name == "weak_flow" else "#66ff66"
            ax.contour(mask.astype(float), levels=[0.5], colors=[outline_color], linewidths=0.8)
            ax.set_xlim(x0 - 0.5, x1 - 0.5)
            ax.set_ylim(y1 - 0.5, y0 - 0.5)
            ax.set_title(title if row == 0 else "", fontsize=PAPER_TITLE_PT)
            panel_index = row * 3 + col
            ax.text(
                0.02, 0.97, f"({chr(ord('a') + panel_index)})", transform=ax.transAxes,
                ha="left", va="top", color="white", fontsize=PAPER_LABEL_PT, fontweight="bold",
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 1.2},
            )
            if row == 1:
                ax.set_xlabel("x (pixel)", fontsize=PAPER_LABEL_PT)
            if col == 0:
                ax.set_ylabel(f"{display_name}\ny (pixel)", fontsize=PAPER_LABEL_PT)
            ax.tick_params(labelsize=PAPER_TICK_PT)
    power_colorbar = fig.colorbar(
        artists[1], ax=axes[:, :2].ravel().tolist(), label="Power (dB)",
        fraction=0.038, pad=0.025, shrink=0.86
    )
    difference_colorbar = fig.colorbar(
        artists[2], ax=axes[:, 2].tolist(), label="Abs. difference (dB)",
        fraction=0.055, pad=0.025, shrink=0.86
    )
    for colorbar in (power_colorbar, difference_colorbar):
        colorbar.ax.tick_params(labelsize=PAPER_TICK_PT)
        colorbar.set_label(colorbar.ax.get_ylabel(), fontsize=PAPER_LABEL_PT)
    fig.suptitle(f"ROI comparison, {frame} frames", fontsize=10.5)
    save_figure(fig, output_dir / f"roi_zoom_{frame}")


def extract_profile(array: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    start, stop = int(spec["start"]), int(spec["stop"])
    if spec["axis"] == "horizontal":
        indices = np.arange(start, stop)
        values = array[int(spec["index"]), start:stop]
    else:
        indices = np.arange(start, stop)
        values = array[start:stop, int(spec["index"])]
    return indices, values


def profiles_output(
    pairs: dict[str, dict[str, Any]], config: dict[str, Any], output_dir: Path
) -> dict[str, dict[str, float]]:
    frames = tuple(pairs)
    fig, axes = plt.subplots(
        2,
        len(frames),
        figsize=(PAPER_WIDTH_IN, 4.4),
        dpi=200,
        squeeze=False,
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.0]},
        layout="constrained",
    )
    csv_rows: list[dict[str, Any]] = []
    residual_stats: dict[str, dict[str, float]] = {}
    for col, frame in enumerate(frames):
        item = pairs[frame]
        spec = config["profiles"][frame]
        indices, cpu_values = extract_profile(item["cpu"], spec)
        _, gpu_values = extract_profile(item["gpu"], spec)
        cpu_db = power_to_db(cpu_values, item["cpu_peak"])
        gpu_db = power_to_db(gpu_values, item["cpu_peak"])
        ax = axes[0, col]
        residual_ax = axes[1, col]
        ax.plot(indices, cpu_db, color="#111111", linewidth=1.5, label="CPU")
        ax.plot(indices, gpu_db, color="#d62728", linewidth=1.2, linestyle="--", label="GPU")
        ax.set_ylim(-30.0, 1.0)
        axis_label = "Lateral x (pixel)" if spec["axis"] == "horizontal" else "Depth y (pixel)"
        if col == 0:
            ax.set_ylabel("Power (dB)", fontsize=PAPER_LABEL_PT)
        ax.set_title(f"{frame} frames", fontsize=PAPER_TITLE_PT)
        ax.grid(True, color="#d8d8d8", linewidth=0.5)
        ax.legend(frameon=False, fontsize=PAPER_LABEL_PT, loc="best")
        ax.tick_params(labelsize=PAPER_TICK_PT)
        db_residual = gpu_db - cpu_db
        finite = np.isfinite(db_residual)
        if not np.any(finite):
            raise ValueError(f"No finite dB residual samples in the {frame}-frame profile.")
        max_abs = float(np.max(np.abs(db_residual[finite])))
        rms = float(np.sqrt(np.mean(np.square(db_residual[finite]), dtype=np.float64)))
        residual_stats[frame] = {"max_abs_db": max_abs, "rms_db": rms}
        residual_ax.axhline(0.0, color="#777777", linewidth=0.6)
        residual_ax.plot(indices, db_residual, color="#d62728", linewidth=1.0)
        residual_ax.set_ylim(-1.0, 1.0)
        residual_ax.set_xlabel(axis_label, fontsize=PAPER_LABEL_PT)
        if col == 0:
            residual_ax.set_ylabel("GPU - CPU\n(dB)", fontsize=PAPER_LABEL_PT)
        residual_ax.grid(True, color="#e2e2e2", linewidth=0.45)
        residual_ax.tick_params(labelsize=PAPER_TICK_PT)
        residual_ax.text(
            0.99,
            0.92,
            f"max|Δ|={max_abs:.4f} dB, RMS={rms:.4f} dB",
            transform=residual_ax.transAxes,
            ha="right",
            va="top",
            fontsize=PAPER_TICK_PT,
        )
        for index, cpu_raw, gpu_raw, cdb, gdb in zip(indices, cpu_values, gpu_values, cpu_db, gpu_db):
            csv_rows.append(
                {
                    "frame_count": int(frame),
                    "axis": spec["axis"],
                    "fixed_index": int(spec["index"]),
                    "sample_index": int(index),
                    "cpu_power": float(cpu_raw),
                    "gpu_power": float(gpu_raw),
                    "cpu_db_shared_peak": float(cdb),
                    "gpu_db_shared_peak": float(gdb),
                    "gpu_minus_cpu_db": float(gdb - cdb),
                }
            )
    save_figure(fig, output_dir / "profile_comparison")
    write_csv(output_dir / "profile_data.csv", csv_rows)
    return residual_stats


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace, roi_config: Path) -> None:
    frames = FRAME_LABELS if args.frames == "both" else (args.frames,)
    required: list[str] = []
    for frame in frames:
        required.extend((f"cpu_{frame}", f"gpu_{frame}"))
    require_paths(args, tuple(required))
    if not roi_config.is_file():
        raise FileNotFoundError(f"Missing locked ROI config: {roi_config}")
    config = json.loads(roi_config.read_text(encoding="utf-8"))
    validate_config(config, frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu_paths = {frame: getattr(args, f"cpu_{frame}") for frame in frames}
    gpu_paths = {frame: getattr(args, f"gpu_{frame}") for frame in frames}
    pairs: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    auxiliary: dict[str, np.ndarray] = {}
    for frame in frames:
        cpu = load_power(cpu_paths[frame], f"CPU {frame}")
        gpu = load_power(gpu_paths[frame], f"GPU {frame}")
        (
            rows,
            ssim_map,
            nrmse_masks,
            ssim_masks,
            raw_difference,
            raw_difference_db,
            cpu_peak,
        ) = calculate_pair(frame, cpu, gpu, config)
        metric_rows.extend(rows)
        pairs[frame] = {
            "cpu": cpu,
            "gpu": gpu,
            "ssim_map": ssim_map,
            "nrmse_masks": nrmse_masks,
            "ssim_masks": ssim_masks,
            "raw_difference": raw_difference,
            "difference_db": raw_difference_db,
            "cpu_peak": cpu_peak,
        }
        auxiliary[f"ssim_map_{frame}"] = ssim_map
        auxiliary[f"raw_gpu_minus_cpu_{frame}"] = raw_difference
        auxiliary[f"raw_abs_difference_db_{frame}"] = raw_difference_db
        for name in ROI_NAMES:
            auxiliary[f"nrmse_mask_{frame}_{name}"] = nrmse_masks[name]
            auxiliary[f"ssim_center_mask_{frame}_{name}"] = ssim_masks[name]
    expected_rows = 3 * len(frames)
    if len(metric_rows) != expected_rows:
        raise RuntimeError(f"Expected exactly {expected_rows} metric rows, got {len(metric_rows)}.")
    write_csv(args.output_dir / "quality_metrics.csv", metric_rows)
    (args.output_dir / "quality_metrics.json").write_text(
        json.dumps(metric_rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    comparison_figure(pairs, args.output_dir)
    for frame in frames:
        roi_zoom_figure(frame, pairs[frame], args.output_dir)
    profile_stats = profiles_output(pairs, config, args.output_dir)
    np.savez_compressed(args.output_dir / "quality_auxiliary.npz", **auxiliary)
    metadata = {
        "created_utc": utc_now(),
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "inputs": {
            frame: {
                "cpu": {"path": str(cpu_paths[frame].resolve()), "sha256": sha256_file(cpu_paths[frame])},
                "gpu": {"path": str(gpu_paths[frame].resolve()), "sha256": sha256_file(gpu_paths[frame])},
                "cpu_peak_linear_power": pairs[frame]["cpu_peak"],
            }
            for frame in frames
        },
        "roi_config": {
            "path": str(roi_config.resolve()),
            "sha256": sha256_file(roi_config),
            "definition": config,
            "effective_pixel_counts": {
                frame: {
                    name: {
                        "nrmse": int(np.count_nonzero(pairs[frame]["nrmse_masks"][name])),
                        "ssim_centers": int(np.count_nonzero(pairs[frame]["ssim_masks"][name])),
                    }
                    for name in ROI_NAMES
                }
                for frame in frames
            },
        },
        "array": {"dtype": "float32", "shape": [HEIGHT, WIDTH], "order": "C"},
        "nrmse": {
            "formula": "norm(GPU - CPU) / norm(CPU) * 100",
            "domain": "untruncated linear power",
            "error_capping": False,
            "individual_normalization": False,
            "global_scope": "all 46080 image pixels",
        },
        "ssim": {
            "normalization_reference": "shared CPU peak for each frame-count pair",
            "mapping": "10*log10(power/CPU_peak), clipped to [-30,0] dB, then mapped linearly to [0,1]",
            "win_size": SSIM_WIN_SIZE,
            "gaussian_weights": True,
            "sigma": SSIM_SIGMA,
            "use_sample_covariance": False,
            "covariance": "population",
            "data_range": 1.0,
            "aggregation": "mean of SSIM-map centers selected by each ROI mask",
            "global_border_excluded_px": int(config["rois"]["global"]["border_px"]),
            "minimum_image_border_for_all_roi_centers_px": SSIM_RADIUS,
            "individual_normalization": False,
        },
        "difference": {
            "stored_raw_formula": "GPU - CPU in untruncated linear power",
            "display_formula": "10*log10(abs(GPU - CPU)/CPU_peak)",
            "display_color_limits_db": [-80.0, 0.0],
            "numeric_error_capping": False,
        },
        "profiles": {
            "definitions": {frame: config["profiles"][frame] for frame in frames},
            "main_display_limits_db": [-30.0, 1.0],
            "residual_definition": "GPU_dB - CPU_dB with the same CPU peak",
            "residual_display_limits_db": [-1.0, 1.0],
            "residual_statistics": profile_stats,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_image": skimage_version,
            "matplotlib": matplotlib.__version__,
        },
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"[OK] Wrote {len(metric_rows)} metric rows and reproducibility artifacts to: "
        f"{args.output_dir}"
    )


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    roi_config = (args.roi_config or (args.output_dir / "roi_config.json")).resolve()
    if args.prepare_roi:
        prepare_roi(args, roi_config)
    else:
        analyze(args, roi_config)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)

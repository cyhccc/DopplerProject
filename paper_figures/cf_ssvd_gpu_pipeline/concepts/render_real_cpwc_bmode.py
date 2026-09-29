"""Render representative single-angle and five-angle CPWC B-mode images.

The input is the demodulated IQ stream produced by the 7883227 acquisition
pipeline.  All panels use one frame, one reconstruction grid, and one display
range so their contrast is directly comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


DEFAULT_IQ = Path(
    r"E:\Code\Cuda\LinearArrayCPWC_OPT_test\bins_7883227\RF_001_7883227_bs100bw_iq_float2.bin"
)
ANGLES_DEG = np.asarray([-6.0, -3.0, 0.0, 3.0, 6.0], dtype=np.float32)
NX, NZ = 256, 180
CHANNELS, SAMPLES, FRAMES = 128, 256, 800
WAVELENGTH = 0.09856e-3
FS = 15.625e6
F0 = 15.625e6
SOUND_SPEED = 1540.0
PITCH = 0.1e-3
F_NUMBER = 1.9
MIN_APERTURE = 1e-3
DISPLAY_RANGE_DB = 50.0


def _font() -> str:
    candidates = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial")
    installed = {item.name for item in font_manager.fontManager.ttflist}
    return next((name for name in candidates if name in installed), "DejaVu Sans")


def geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_axis = (
        -64.427759740259731 + np.arange(NX, dtype=np.float32) * 0.50730519480519476
    ) * WAVELENGTH
    z_axis = (18.0 + np.arange(NZ, dtype=np.float32) * 0.5) * WAVELENGTH
    grid_x, grid_z = np.meshgrid(x_axis, z_axis, indexing="xy")
    x = grid_x.ravel()
    z = grid_z.ravel()
    elem_x = (np.arange(CHANNELS, dtype=np.float32) - (CHANNELS - 1) / 2.0) * PITCH

    aperture_radius = np.maximum(z, F_NUMBER * MIN_APERTURE) / (2.0 * F_NUMBER)
    rx_delay = np.empty((CHANNELS, x.size), dtype=np.float32)
    rx_weight = np.empty((CHANNELS, x.size), dtype=bool)
    for channel, element_x in enumerate(elem_x):
        dx = x - element_x
        rx_delay[channel] = np.hypot(dx, z) * (FS / SOUND_SPEED)
        rx_weight[channel] = np.abs(dx) <= aperture_radius

    t0 = 2.0 * 10.0 * WAVELENGTH / SOUND_SPEED - 2.84375 / F0
    tx_offset_samples = t0 * FS
    tx_delay = np.stack(
        [
            (z * np.cos(np.deg2rad(angle)) + x * np.sin(np.deg2rad(angle)))
            * (FS / SOUND_SPEED)
            - tx_offset_samples
            for angle in ANGLES_DEG
        ],
        axis=0,
    ).astype(np.float32)
    return x_axis, z_axis, rx_delay, rx_weight, tx_delay


def beamform_one_frame(iq_path: Path, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = 5 * CHANNELS * SAMPLES * FRAMES * np.dtype(np.complex64).itemsize
    if iq_path.stat().st_size != expected:
        raise ValueError(f"Unexpected IQ size: expected {expected}, got {iq_path.stat().st_size}")
    iq = np.memmap(
        iq_path,
        mode="r",
        dtype=np.complex64,
        shape=(5, CHANNELS, SAMPLES, FRAMES),
    )
    x_axis, z_axis, rx_delay, rx_weight, tx_delay = geometry()
    outputs = np.zeros((5, NZ * NX), dtype=np.complex64)
    phase_scale = np.float32(2.0 * np.pi * F0 / FS)

    for angle_idx in range(5):
        out = outputs[angle_idx]
        for channel in range(CHANNELS):
            delay = tx_delay[angle_idx] + rx_delay[channel]
            valid = rx_weight[channel] & (delay >= 0.0) & (delay < SAMPLES - 1)
            ids = np.flatnonzero(valid)
            d = delay[ids]
            i0 = np.floor(d).astype(np.intp)
            frac = d - i0
            trace = iq[angle_idx, channel, :, frame]
            delayed = trace[i0] + (trace[i0 + 1] - trace[i0]) * frac
            phase = np.exp(1j * phase_scale * d).astype(np.complex64)
            out[ids] += delayed * phase
    return x_axis, z_axis, outputs.reshape(5, NZ, NX)


def db_image(image: np.ndarray, reference: float | None = None) -> np.ndarray:
    amplitude = np.abs(image)
    if reference is None:
        reference = float(amplitude.max())
    return 20.0 * np.log10(np.maximum(amplitude, reference * 1e-8) / reference)


def save_panel(path: Path, image_db: np.ndarray, x_mm: np.ndarray, z_mm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(2.25, 2.2), dpi=300)
    ax.imshow(
        image_db,
        cmap="gray",
        vmin=-DISPLAY_RANGE_DB,
        vmax=0.0,
        extent=(x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]),
        aspect="equal",
    )
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_concept(
    out_path: Path,
    representative: list[np.ndarray],
    compounded: np.ndarray,
    x_mm: np.ndarray,
    z_mm: np.ndarray,
) -> None:
    plt.rcParams.update({"font.family": _font(), "axes.unicode_minus": False})
    fig = plt.figure(figsize=(12.2, 2.9), dpi=300, facecolor="white")
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    canvas.add_patch(
        FancyBboxPatch(
            (0.012, 0.07), 0.976, 0.86,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            linewidth=1.4, edgecolor="#9BA8B4", facecolor="#F8FAFC",
        )
    )

    xs = [0.04, 0.185, 0.33]
    labels = ["−6°", "0°", "+6°"]
    for x, label, image_db in zip(xs, labels, representative):
        ax = fig.add_axes((x, 0.27, 0.115, 0.53))
        ax.imshow(
            image_db, cmap="gray", vmin=-DISPLAY_RANGE_DB, vmax=0,
            extent=(x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]), aspect="equal",
        )
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#334155"); spine.set_linewidth(0.8)
        canvas.text(x + 0.0575, 0.84, label, ha="center", va="center", fontsize=11, color="#172033")

    canvas.text(0.174, 0.535, "…", ha="center", va="center", fontsize=18, color="#526273")
    canvas.text(0.319, 0.535, "…", ha="center", va="center", fontsize=18, color="#526273")
    canvas.text(0.245, 0.15, "代表性单角度 B-mode（真实流程共5个角度）", ha="center", fontsize=10, color="#465568")

    canvas.add_patch(FancyArrowPatch((0.46, 0.53), (0.59, 0.53), arrowstyle="-|>", mutation_scale=15,
                                     linewidth=1.8, color="#2367A8"))
    canvas.add_patch(FancyBboxPatch((0.485, 0.59), 0.078, 0.17, boxstyle="round,pad=0.008",
                                    linewidth=1.2, edgecolor="#2367A8", facecolor="#E8F2FB"))
    canvas.text(0.524, 0.675, "Σ", ha="center", va="center", fontsize=23, color="#174E82")
    canvas.text(0.524, 0.43, "5角度\n相干复合", ha="center", va="center", fontsize=10, color="#174E82")

    ax = fig.add_axes((0.65, 0.22, 0.16, 0.62))
    ax.imshow(
        compounded, cmap="gray", vmin=-DISPLAY_RANGE_DB, vmax=0,
        extent=(x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]), aspect="equal",
    )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#174E82"); spine.set_linewidth(1.4)
    canvas.text(0.73, 0.14, "CPWC复合图像", ha="center", fontsize=11, fontweight="bold", color="#172033")
    canvas.text(0.895, 0.62, "CPWC-DAS", ha="center", fontsize=15, fontweight="bold", color="#174E82")
    canvas.text(0.895, 0.48, "波束形成", ha="center", fontsize=13, color="#334155")
    canvas.text(0.895, 0.34, "同一帧 · 同一坐标 · −50~0 dB", ha="center", fontsize=8.5, color="#66788A")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    # Store an opaque RGB image; some manuscript viewers mishandle RGBA PNGs.
    with Image.open(out_path) as rendered:
        rendered.convert("RGB").save(out_path, quality=95)


def make_concept_v2(
    out_path: Path,
    representative: list[np.ndarray],
    compounded: np.ndarray,
    x_mm: np.ndarray,
    z_mm: np.ndarray,
) -> None:
    """Render the narrow manuscript layout with vertical angle panels."""
    plt.rcParams.update({"font.family": _font(), "axes.unicode_minus": False})
    fig = plt.figure(figsize=(6.2, 4.35), dpi=300, facecolor="white")
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    canvas.text(
        0.5, 0.955, "CPWC-DAS波束形成",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#183B5B",
    )

    panel_x, panel_w, panel_h = 0.075, 0.225, 0.205
    panel_ys = [0.675, 0.395, 0.115]
    labels = ["−6°", "0°", "+6°"]
    centers: list[tuple[float, float]] = []
    for y, label, image_db in zip(panel_ys, labels, representative):
        ax = fig.add_axes((panel_x, y, panel_w, panel_h))
        ax.imshow(
            image_db, cmap="gray", vmin=-DISPLAY_RANGE_DB, vmax=0,
            extent=(x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]), aspect="auto",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#475569")
            spine.set_linewidth(0.75)
        cy = y + panel_h / 2
        centers.append((panel_x + panel_w, cy))
        canvas.text(panel_x - 0.018, cy, label, ha="right", va="center", fontsize=10.5, color="#1E293B")

    junction = (0.405, 0.5)
    for start_x, start_y in centers:
        canvas.add_patch(
            FancyArrowPatch(
                (start_x + 0.012, start_y), junction,
                arrowstyle="-|>", mutation_scale=9, linewidth=1.25,
                color="#2874AE", connectionstyle="arc3,rad=0",
            )
        )

    canvas.add_patch(
        FancyBboxPatch(
            (0.385, 0.43), 0.105, 0.14,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=1.25, edgecolor="#2874AE", facecolor="#EDF5FB",
        )
    )
    canvas.text(0.438, 0.507, "Σ", ha="center", va="center", fontsize=21, color="#165B8F")
    canvas.text(0.438, 0.375, "5角度\n相干复合", ha="center", va="center", fontsize=9.5, color="#165B8F")

    canvas.add_patch(
        FancyArrowPatch(
            (0.505, 0.5), (0.58, 0.5),
            arrowstyle="-|>", mutation_scale=12, linewidth=1.5, color="#2874AE",
        )
    )

    out_x, out_y, out_w, out_h = 0.605, 0.285, 0.315, 0.43
    ax = fig.add_axes((out_x, out_y, out_w, out_h))
    ax.imshow(
        compounded, cmap="gray", vmin=-DISPLAY_RANGE_DB, vmax=0,
        extent=(x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]), aspect="auto",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#165B8F")
        spine.set_linewidth(1.25)
    canvas.text(
        out_x + out_w / 2, 0.225, "CPWC复合图像",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1E293B",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.025, facecolor="white")
    plt.close(fig)
    with Image.open(out_path) as rendered:
        rendered.convert("RGB").save(out_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iq", type=Path, default=DEFAULT_IQ)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not 0 <= args.frame < FRAMES:
        raise ValueError(f"frame must be in [0, {FRAMES - 1}]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_axis, z_axis, single_angle = beamform_one_frame(args.iq, args.frame)
    compounded_complex = np.sum(single_angle, axis=0)
    # Normalize each B-mode panel to its own peak while applying the same
    # fixed −50…0 dB window.  This is the conventional B-mode comparison and
    # avoids making the single-angle panels artificially dark merely because
    # coherent compounding raises their absolute amplitude.
    references = np.asarray([np.abs(item).max() for item in single_angle], dtype=np.float64)
    compound_reference = float(np.abs(compounded_complex).max())
    single_db = np.stack([db_image(item, float(ref)) for item, ref in zip(single_angle, references)])
    compound_db = db_image(compounded_complex, compound_reference)
    x_mm, z_mm = x_axis * 1e3, z_axis * 1e3

    for idx, angle in enumerate(ANGLES_DEG):
        save_panel(args.output_dir / f"bmode_angle_{angle:+.0f}deg_frame{args.frame:03d}.png", single_db[idx], x_mm, z_mm)
    save_panel(args.output_dir / f"bmode_cpwc_5angle_frame{args.frame:03d}.png", compound_db, x_mm, z_mm)
    make_concept(
        args.output_dir / "cpwc_das_real_v1.png",
        [single_db[0], single_db[2], single_db[4]], compound_db, x_mm, z_mm,
    )
    make_concept_v2(
        args.output_dir / "cpwc_das_real_v2.png",
        [single_db[0], single_db[2], single_db[4]], compound_db, x_mm, z_mm,
    )
    np.savez_compressed(
        args.output_dir / f"cpwc_bmode_frame{args.frame:03d}.npz",
        x_mm=x_mm, z_mm=z_mm, angles_deg=ANGLES_DEG,
        single_angle_complex=single_angle, compounded_complex=compounded_complex,
    )
    print(
        f"frame={args.frame}; angle_references={references.tolist()}; "
        f"compound_reference={compound_reference:.7g}; output={args.output_dir}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Doppler image from either power_sum or power_dB float32 binaries."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--power-db", type=Path, help="Input float32 power_dB binary.")
    input_group.add_argument("--power-sum", type=Path, help="Input float32 linear power sum binary.")
    parser.add_argument("--width", type=int, required=True, help="Image width.")
    parser.add_argument("--height", type=int, required=True, help="Image height.")
    parser.add_argument(
        "--input-order",
        choices=("c", "fortran"),
        default="c",
        help="Binary layout order: 'c' for C/CUDA row-major binaries, 'fortran' for MATLAB column-major binaries.",
    )
    parser.add_argument("--output-png", type=Path, required=True, help="Output PNG path.")
    parser.add_argument(
        "--output-power-db",
        type=Path,
        default=None,
        help="Optional output float32 power_dB binary path after shared post-processing.",
    )
    parser.add_argument(
        "--peak-ref-value",
        type=float,
        default=None,
        help="Shared linear power reference used when converting power_sum to dB.",
    )
    parser.add_argument(
        "--peak-ref-power-sum",
        type=Path,
        default=None,
        help="Use the max value from this power_sum binary as the shared power reference.",
    )
    parser.add_argument("--display-min-db", type=float, default=-40.0, help="Lower display limit in dB.")
    parser.add_argument("--display-max-db", type=float, default=0.0, help="Upper display limit in dB.")
    parser.add_argument("--x-min-mm", type=float, default=None, help="Optional lateral-axis minimum in mm.")
    parser.add_argument("--x-max-mm", type=float, default=None, help="Optional lateral-axis maximum in mm.")
    parser.add_argument("--z-min-mm", type=float, default=None, help="Optional depth-axis minimum in mm.")
    parser.add_argument("--z-max-mm", type=float, default=None, help="Optional depth-axis maximum in mm.")
    parser.add_argument("--clean", action="store_true", help="Render without axes or colorbar.")
    parser.add_argument("--frameless", action="store_true", help="Hide title, axes, and ticks while retaining the colorbar.")
    parser.add_argument("--scale-bar-mm", type=float, default=0.0, help="White scale-bar length in mm.")
    return parser.parse_args()


def build_doppler_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "doppler_redyellow",
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


def load_float32_image(
    image_path: Path,
    width: int,
    height: int,
    label: str,
    input_order: str,
) -> np.ndarray:
    data = np.fromfile(image_path, dtype=np.float32)
    expected_count = width * height
    if data.size != expected_count:
        raise ValueError(
            f"Unexpected {label} size for {image_path}: "
            f"expected {expected_count} float32 values, got {data.size}."
        )
    order = "F" if input_order == "fortran" else "C"
    return data.reshape((height, width), order=order)


def resolve_peak_ref(
    power_sum: np.ndarray,
    peak_ref_value: float | None,
    peak_ref_power_sum: Path | None,
    width: int,
    height: int,
    input_order: str,
) -> float:
    if peak_ref_value is not None:
        peak_ref = float(peak_ref_value)
    elif peak_ref_power_sum is not None:
        reference_power = load_float32_image(
            peak_ref_power_sum,
            width,
            height,
            "peak_ref_power_sum",
            input_order,
        )
        peak_ref = float(np.max(reference_power))
    else:
        peak_ref = float(np.max(power_sum))

    if not np.isfinite(peak_ref) or peak_ref <= 0.0:
        raise ValueError(f"Invalid peak reference value: {peak_ref}")

    return peak_ref


def power_sum_to_db(
    power_sum: np.ndarray,
    peak_ref: float,
) -> np.ndarray:
    eps = np.finfo(np.float32).tiny
    power_db = 10.0 * np.log10(np.maximum(power_sum, eps) / peak_ref)
    return power_db.astype(np.float32, copy=False)


def save_png(
    power_db: np.ndarray,
    output_png: Path,
    display_min_db: float,
    display_max_db: float,
    x_min_mm: float | None,
    x_max_mm: float | None,
    z_min_mm: float | None,
    z_max_mm: float | None,
    clean: bool,
    frameless: bool,
    scale_bar_mm: float,
) -> None:
    cmap = build_doppler_colormap()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if clean:
        # Save the matrix directly so a 256x180 Doppler grid stays 256x180
        # instead of being stretched to a square matplotlib canvas.
        plt.imsave(
            output_png,
            power_db,
            cmap=cmap,
            vmin=display_min_db,
            vmax=display_max_db,
        )
        return

    height, width = power_db.shape
    use_physical_axes = None not in (x_min_mm, x_max_mm, z_min_mm, z_max_mm)
    if use_physical_axes:
        x_span = max(float(x_max_mm) - float(x_min_mm), 1.0e-6)
        z_span = max(float(z_max_mm) - float(z_min_mm), 1.0e-6)
        figure_width = 6.0
        figure_height = max(2.2, figure_width * z_span / x_span)
        extent = [float(x_min_mm), float(x_max_mm), float(z_max_mm), float(z_min_mm)]
        xlabel = "Lateral (mm)"
        ylabel = "Depth (mm)"
    else:
        figure_width = 6.0
        figure_height = max(2.0, figure_width * height / max(width, 1))
        extent = None
        xlabel = "Lateral index"
        ylabel = "Depth index"
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=180)
    image = ax.imshow(
        power_db,
        cmap=cmap,
        vmin=display_min_db,
        vmax=display_max_db,
        origin="upper",
        extent=extent,
        aspect="equal",
    )
    if frameless:
        ax.set_axis_off()
        if scale_bar_mm > 0.0 and use_physical_axes:
            x0, x1, z1, z0 = extent
            x_span, z_span = x1 - x0, z1 - z0
            x_start = x0 + 0.025 * x_span
            y = z1 - 0.09 * z_span
            ax.plot([x_start, x_start + scale_bar_mm], [y, y], color="white", linewidth=4, solid_capstyle="butt")
            ax.text(x_start + scale_bar_mm / 2, y - 0.025 * z_span, f"{scale_bar_mm:g} mm",
                    color="white", ha="center", va="bottom", fontsize=12, fontweight="bold")
    else:
        ax.set_title("Power Doppler")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=[display_min_db, (display_min_db + display_max_db) / 2, display_max_db],
        shrink=0.90,
    )
    colorbar.ax.set_xlabel("dB", fontsize=15, fontweight="bold", labelpad=8)
    colorbar.ax.tick_params(labelsize=14, width=1.5, length=6)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    fig.tight_layout()
    image_position = ax.get_position()
    colorbar_position = colorbar.ax.get_position()
    colorbar.ax.set_position([
        colorbar_position.x0,
        image_position.y0,
        colorbar_position.width,
        image_position.height,
    ])
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.power_sum is not None:
        power_sum = load_float32_image(
            args.power_sum,
            args.width,
            args.height,
            "power_sum",
            args.input_order,
        )
        peak_ref = resolve_peak_ref(
            power_sum,
            args.peak_ref_value,
            args.peak_ref_power_sum,
            args.width,
            args.height,
            args.input_order,
        )
        power_db = power_sum_to_db(power_sum, peak_ref)
        print(f"[Info] Converted power_sum to dB with shared peak_ref={peak_ref:.9g}")
    else:
        power_db = load_float32_image(
            args.power_db,
            args.width,
            args.height,
            "power_db",
            args.input_order,
        )

    if args.output_power_db is not None:
        args.output_power_db.parent.mkdir(parents=True, exist_ok=True)
        power_db.astype(np.float32, copy=False).tofile(args.output_power_db)
        print(f"[Info] Saved power_dB binary to {args.output_power_db}")

    save_png(
        power_db,
        args.output_png,
        args.display_min_db,
        args.display_max_db,
        args.x_min_mm,
        args.x_max_mm,
        args.z_min_mm,
        args.z_max_mm,
        args.clean,
        args.frameless,
        args.scale_bar_mm,
    )
    print(f"[Info] Saved Doppler PNG to {args.output_png}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MATLAB CPU and CUDA 7883227 Power Doppler outputs."
    )
    parser.add_argument("--cuda-power-sum", type=Path, required=True)
    parser.add_argument("--matlab-power-sum", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--display-min-db", type=float, default=-40.0)
    parser.add_argument("--display-max-db", type=float, default=0.0)
    parser.add_argument("--diff-min-db", type=float, default=-70.0)
    parser.add_argument("--diff-max-db", type=float, default=0.0)
    parser.add_argument(
        "--raw-diff-value-max-db",
        type=float,
        default=-30.0,
        help="Cap actual raw difference dB values above this value before plotting and raw-diff dB metrics.",
    )
    parser.add_argument(
        "--no-raw-diff-value-cap",
        action="store_true",
        help="Use uncapped raw difference dB values before plotting and raw-diff dB metrics.",
    )
    parser.add_argument(
        "--diff-reference",
        choices=["max-both", "cuda-peak"],
        default="max-both",
        help="Reference used for raw difference dB normalization.",
    )
    parser.add_argument("--cuda-label", default="CUDA")
    parser.add_argument("--matlab-label", default="MATLAB CPU")
    parser.add_argument("--scale-bar-mm", type=float, default=0.0)
    parser.add_argument("--scale-bar-label", default="")
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def load_float_image(path: Path, width: int, height: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    expected = width * height
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} float32 values, expected {expected}.")
    return data.reshape((height, width))


def imaging_extent_mm(width: int, height: int) -> list[float]:
    wavelength = 0.09856e-3
    origin_x_wl = -64.427759740259731
    delta_x_wl = 0.50730519480519476
    origin_z_wl = 18.0
    delta_z_wl = 0.5
    x0 = origin_x_wl * wavelength * 1e3
    x1 = (origin_x_wl + (width - 1) * delta_x_wl) * wavelength * 1e3
    z0 = origin_z_wl * wavelength * 1e3
    z1 = (origin_z_wl + (height - 1) * delta_z_wl) * wavelength * 1e3
    return [x0, x1, z1, z0]


def power_sum_to_db(power_sum: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.sqrt(np.maximum(power_sum, 0.0))))
    if peak <= 0.0 or not np.isfinite(peak):
        return np.full(power_sum.shape, -200.0, dtype=np.float32)
    magnitude = np.sqrt(np.maximum(power_sum, 0.0)) / peak
    return (20.0 * np.log10(np.maximum(magnitude, 1.0e-20))).astype(np.float32)


def normalize_display(power_db: np.ndarray, display_min_db: float, display_max_db: float) -> np.ndarray:
    clipped = np.clip(power_db, display_min_db, display_max_db)
    return ((clipped - display_min_db) / (display_max_db - display_min_db)).astype(np.float64)


def uniform_filter_reflect(image: np.ndarray, window: int) -> np.ndarray:
    if window % 2 != 1:
        raise ValueError("window must be odd.")
    radius = window // 2
    padded = np.pad(image, radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    return (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    ) / float(window * window)


def mean_ssim(image_a: np.ndarray, image_b: np.ndarray, window: int = 11, data_range: float = 1.0) -> float:
    image_a = image_a.astype(np.float64, copy=False)
    image_b = image_b.astype(np.float64, copy=False)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_a = uniform_filter_reflect(image_a, window)
    mu_b = uniform_filter_reflect(image_b, window)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = uniform_filter_reflect(image_a * image_a, window) - mu_a2
    sigma_b2 = uniform_filter_reflect(image_b * image_b, window) - mu_b2
    sigma_ab = uniform_filter_reflect(image_a * image_b, window) - mu_ab

    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    ssim_map = numerator / np.maximum(denominator, np.finfo(np.float64).eps)
    return float(np.mean(ssim_map))


def doppler_colormap():
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


def difference_colormap():
    return LinearSegmentedColormap.from_list(
        "raw_difference_black_purple_red_yellow_white",
        [
            (0.00, "#000000"),
            (0.18, "#14002e"),
            (0.36, "#4c0073"),
            (0.55, "#b1005a"),
            (0.72, "#f15a24"),
            (0.88, "#ffd447"),
            (1.00, "#fffef2"),
        ],
    )


def colorbar_ticks(display_min_db: float, display_max_db: float) -> np.ndarray:
    low = int(np.floor(display_min_db / 10.0) * 10)
    high = int(np.ceil(display_max_db / 10.0) * 10)
    return np.arange(high, low - 1, -10)


def add_scale_bar(ax, extent: list[float], length_mm: float, label: str) -> None:
    if length_mm <= 0.0:
        return
    x0, x1, z1, z0 = extent
    x_span = x1 - x0
    z_span = z1 - z0
    x_start = x0 + 0.09 * x_span
    x_end = x_start + length_mm
    y = z1 - 0.11 * z_span
    text_y = y - 0.05 * z_span
    label = label or f"{length_mm:g} mm"

    ax.plot(
        [x_start, x_end],
        [y, y],
        color="white",
        linewidth=3.0,
        solid_capstyle="butt",
        path_effects=[],
    )
    tick_height = 0.025 * z_span
    ax.plot([x_start, x_start], [y - tick_height, y + tick_height], color="white", linewidth=2.0)
    ax.plot([x_end, x_end], [y - tick_height, y + tick_height], color="white", linewidth=2.0)
    ax.text(
        (x_start + x_end) * 0.5,
        text_y,
        label,
        color="white",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def save_comparison_png(
    cuda_db: np.ndarray,
    matlab_db: np.ndarray,
    raw_diff_db: np.ndarray,
    output_png: Path,
    display_min_db: float,
    display_max_db: float,
    diff_min_db: float,
    diff_max_db: float,
    cuda_label: str,
    matlab_label: str,
    scale_bar_mm: float,
    scale_bar_label: str,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    doppler_cmap = doppler_colormap()
    diff_cmap = difference_colormap()
    extent = imaging_extent_mm(cuda_db.shape[1], cuda_db.shape[0])
    fig = plt.figure(figsize=(15.2, 3.9), dpi=180)
    grid = GridSpec(
        1,
        6,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.055, 0.22, 1.0, 0.055],
        wspace=0.36,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 4]),
    ]
    doppler_cax = fig.add_subplot(grid[0, 2])
    diff_cax = fig.add_subplot(grid[0, 5])

    for ax, img, title in [
        (axes[0], cuda_db, cuda_label),
        (axes[1], matlab_db, matlab_label),
    ]:
        im = ax.imshow(
            img,
            cmap=doppler_cmap,
            vmin=display_min_db,
            vmax=display_max_db,
            extent=extent,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Depth (mm)")
        add_scale_bar(ax, extent, scale_bar_mm, scale_bar_label)
    power_colorbar = fig.colorbar(
        im,
        cax=doppler_cax,
        ticks=colorbar_ticks(display_min_db, display_max_db),
    )
    power_colorbar.set_label("power [dB]")

    diff_im = axes[2].imshow(
        raw_diff_db,
        cmap=diff_cmap,
        vmin=diff_min_db,
        vmax=diff_max_db,
        extent=extent,
        aspect="equal",
    )
    axes[2].set_title(f"|{matlab_label} raw - {cuda_label} raw|")
    axes[2].set_xlabel("Lateral (mm)")
    axes[2].set_ylabel("Depth (mm)")
    diff_colorbar = fig.colorbar(
        diff_im,
        cax=diff_cax,
        ticks=colorbar_ticks(diff_min_db, diff_max_db),
    )
    diff_colorbar.set_label("raw difference [dB]")

    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cuda_power = load_float_image(args.cuda_power_sum, args.width, args.height)
    matlab_power = load_float_image(args.matlab_power_sum, args.width, args.height)

    cuda_db = power_sum_to_db(cuda_power)
    matlab_db = power_sum_to_db(matlab_power)
    cuda_display = normalize_display(cuda_db, args.display_min_db, args.display_max_db)
    matlab_display = normalize_display(matlab_db, args.display_min_db, args.display_max_db)

    raw_diff = matlab_power.astype(np.float64) - cuda_power.astype(np.float64)
    raw_diff_abs = np.abs(raw_diff)
    cuda_peak_power = max(float(np.max(np.maximum(cuda_power, 0.0))), np.finfo(np.float64).tiny)
    matlab_peak_power = max(float(np.max(np.maximum(matlab_power, 0.0))), np.finfo(np.float64).tiny)
    if args.diff_reference == "cuda-peak":
        reference_power = cuda_peak_power
    else:
        reference_power = max(cuda_peak_power, matlab_peak_power)
    raw_diff_db_uncapped = (10.0 * np.log10(np.maximum(raw_diff_abs, np.finfo(np.float64).tiny) / reference_power)).astype(np.float32)
    if args.no_raw_diff_value_cap:
        raw_diff_db = raw_diff_db_uncapped
        raw_diff_value_max_applied = None
    else:
        raw_diff_db = np.minimum(raw_diff_db_uncapped, args.raw_diff_value_max_db).astype(np.float32)
        raw_diff_value_max_applied = float(args.raw_diff_value_max_db)

    raw_rel_l2 = float(np.linalg.norm(raw_diff.ravel()) / max(np.linalg.norm(cuda_power.astype(np.float64).ravel()), np.finfo(np.float64).eps))
    raw_mae = float(np.mean(np.abs(raw_diff)))
    raw_max_abs = float(np.max(np.abs(raw_diff)))

    display_diff = matlab_display - cuda_display
    display_rel_l2 = float(np.linalg.norm(display_diff.ravel()) / max(np.linalg.norm(cuda_display.ravel()), np.finfo(np.float64).eps))

    clipped_cuda_db = np.clip(cuda_db, args.display_min_db, args.display_max_db)
    clipped_matlab_db = np.clip(matlab_db, args.display_min_db, args.display_max_db)
    clipped_diff_db = clipped_matlab_db - clipped_cuda_db

    metrics = {
        "raw_power_rel_l2": raw_rel_l2,
        "raw_power_mae": raw_mae,
        "raw_power_max_abs": raw_max_abs,
        "db_clipped_mae": float(np.mean(np.abs(clipped_diff_db))),
        "db_clipped_rmse": float(np.sqrt(np.mean(clipped_diff_db * clipped_diff_db))),
        "db_clipped_max_abs": float(np.max(np.abs(clipped_diff_db))),
        "display_rel_l2": display_rel_l2,
        "raw_diff_db_uncapped_max": float(np.max(raw_diff_db_uncapped)),
        "raw_diff_db_mean_clipped": float(np.mean(np.clip(raw_diff_db, args.diff_min_db, args.diff_max_db))),
        "raw_diff_db_max": float(np.max(raw_diff_db)),
        f"ssim_db{abs(int(round(args.display_min_db)))}_mean": mean_ssim(cuda_display, matlab_display),
    }

    txt_path = args.output_prefix.with_suffix(".txt")
    png_path = args.output_prefix.with_name(args.output_prefix.name + "_side_by_side.png")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"cuda_power_sum={args.cuda_power_sum}\n")
        f.write(f"matlab_power_sum={args.matlab_power_sum}\n")
        f.write(f"shape={args.height}x{args.width}\n")
        f.write(f"cuda_label={args.cuda_label}\n")
        f.write(f"matlab_label={args.matlab_label}\n")
        f.write(f"scale_bar_mm={args.scale_bar_mm:.9g}\n")
        f.write(f"scale_bar_label={args.scale_bar_label or f'{args.scale_bar_mm:g} mm'}\n")
        f.write(f"doppler_display_range_db={args.display_min_db},{args.display_max_db}\n")
        f.write(f"raw_difference_display_range_db={args.diff_min_db},{args.diff_max_db}\n")
        f.write(f"raw_difference_reference={args.diff_reference}\n")
        f.write(f"cuda_peak_power={cuda_peak_power:.9g}\n")
        f.write(f"matlab_peak_power={matlab_peak_power:.9g}\n")
        if args.diff_reference == "cuda-peak":
            f.write("raw_difference_uncapped_formula_db=10*log10(abs(matlab_power_sum-cuda_power_sum)/max(cuda_power_sum))\n")
        else:
            f.write("raw_difference_uncapped_formula_db=10*log10(abs(matlab_power_sum-cuda_power_sum)/max(max(cuda_power_sum),max(matlab_power_sum)))\n")
        if raw_diff_value_max_applied is None:
            f.write("raw_diff_db_value_max_applied=none\n")
            f.write("raw_difference_formula_db=raw_difference_uncapped_formula_db\n")
        else:
            f.write(f"raw_diff_db_value_max_applied={raw_diff_value_max_applied:.9g}\n")
            f.write("raw_difference_formula_db=min(raw_difference_uncapped_formula_db,raw_diff_db_value_max_applied)\n")
        for key, value in metrics.items():
            f.write(f"{key}={value:.9g}\n")
        f.write(f"side_by_side_png={png_path}\n")

    save_comparison_png(
        cuda_db,
        matlab_db,
        raw_diff_db,
        png_path,
        args.display_min_db,
        args.display_max_db,
        args.diff_min_db,
        args.diff_max_db,
        args.cuda_label,
        args.matlab_label,
        args.scale_bar_mm,
        args.scale_bar_label,
    )

    print(f"[Info] Saved comparison report to {txt_path}")
    print(f"[Info] Saved side-by-side PNG to {png_path}")
    for key, value in metrics.items():
        print(f"{key}={value:.9g}")


if __name__ == "__main__":
    main()

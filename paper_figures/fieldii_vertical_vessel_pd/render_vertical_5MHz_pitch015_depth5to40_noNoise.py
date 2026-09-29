from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(
    r"E:\Code\mat\y_branch_fieldii_gpu_pd\results_vertical_pw05_128f_tissue_bg_5MHz_pitch015mm_depth5to40mm_w4mm_blood3_tissue5_noNoise"
)
OUT_DIR = Path(r"D:\DopplerProject\paper_figures\fieldii_vertical_vessel_pd")

NX = 512
NZ = 512
X_MIN_MM = -10.0
X_MAX_MM = 10.0
Z_MIN_MM = 5.0
Z_MAX_MM = 40.0
DISPLAY_Z_MIN_MM = 15.0
DISPLAY_Z_MAX_MM = 40.0


def render(power_sum_path: Path, cmap: str, db_range: tuple[float, float], label: str, stem: str) -> None:
    power_sum = np.fromfile(power_sum_path, dtype=np.float32)
    if power_sum.size != NX * NZ:
        raise ValueError(f"Expected {NX * NZ} values, got {power_sum.size}: {power_sum_path}")

    power_sum = power_sum.reshape((NZ, NX), order="F")
    peak = float(np.nanmax(power_sum))
    if not np.isfinite(peak) or peak <= 0:
        peak = 1.0
    image_db = 10.0 * np.log10(np.maximum(power_sum, np.finfo(np.float32).tiny) / peak)

    mpl.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun"],
            "font.size": 7,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x_mm = np.linspace(X_MIN_MM, X_MAX_MM, NX)
    z_mm = np.linspace(Z_MIN_MM, Z_MAX_MM, NZ)
    z_mask = (z_mm >= DISPLAY_Z_MIN_MM) & (z_mm <= DISPLAY_Z_MAX_MM)
    image_db = image_db[z_mask, :]
    z_display_mm = z_mm[z_mask]
    fig = plt.figure(figsize=(7.0 / 2.54, 6.2 / 2.54), dpi=600)
    ax = fig.add_axes([0.16, 0.14, 0.64, 0.80])
    im = ax.imshow(
        image_db,
        cmap=cmap,
        vmin=db_range[0],
        vmax=db_range[1],
        extent=[x_mm[0], x_mm[-1], z_display_mm[-1], z_display_mm[0]],
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("宽度/mm", fontfamily="SimSun", fontsize=8)
    ax.set_ylabel("深度/mm", fontfamily="SimSun", fontsize=8)
    ax.tick_params(direction="out", length=3, width=0.5, labelsize=7)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")

    cax = fig.add_axes([0.84, 0.14, 0.055, 0.80])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontfamily="Times New Roman", fontsize=8)
    cb.ax.tick_params(direction="out", length=3, width=0.5, labelsize=7)
    for tick in cb.ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{stem}.png"
    svg_path = OUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=600)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")


def main() -> None:
    render(
        RESULTS_DIR / "power_doppler_power_sum.bin",
        "gray",
        (-60.0, 0.0),
        "B-mode/dB",
        "vertical_pw05_128f_gray_5MHz_pitch015mm_display15to40mm_w4mm_db60_70mm_noNoise",
    )
    render(
        RESULTS_DIR / "svd_rank3to124_cuda_matched" / "svd_power_doppler_power_sum.bin",
        "hot",
        (-30.0, 0.0),
        "Power Doppler/dB",
        "vertical_pw05_128f_svd_rank3to124_pd_5MHz_pitch015mm_display15to40mm_w4mm_db30_70mm_noNoise",
    )


if __name__ == "__main__":
    main()

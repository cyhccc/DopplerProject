from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(
    r"E:\Code\mat\y_branch_fieldii_gpu_pd\results_vertical_pw05_128f_tissue_bg_5to20mm_w5mm_blood20_tissue10"
)
POWER_SUM_PATH = RESULTS_DIR / "power_doppler_power_sum.bin"
OUT_DIR = Path(r"D:\DopplerProject\paper_figures\fieldii_vertical_vessel_pd")

NX = 512
NZ = 300
X_MIN_MM = -10.0
X_MAX_MM = 10.0
Z_MIN_MM = 1.0
Z_MAX_MM = 25.0
CROP_Z_MIN_MM = 5.0
CROP_Z_MAX_MM = 20.0
DISPLAY_MIN_DB = -60.0
DISPLAY_MAX_DB = 0.0


def render_image(image_db: np.ndarray, cmap: str, label: str, stem: str) -> tuple[Path, Path]:
    x_mm = np.linspace(X_MIN_MM, X_MAX_MM, NX)
    z_mm = np.linspace(Z_MIN_MM, Z_MAX_MM, NZ)
    z_mask = (z_mm >= CROP_Z_MIN_MM) & (z_mm <= CROP_Z_MAX_MM)
    img = image_db[z_mask, :]
    z_crop = z_mm[z_mask]

    fig = plt.figure(figsize=(7.0 / 2.54, 6.2 / 2.54), dpi=600)
    ax = fig.add_axes([0.16, 0.17, 0.64, 0.76])
    im = ax.imshow(
        img,
        cmap=cmap,
        vmin=DISPLAY_MIN_DB,
        vmax=DISPLAY_MAX_DB,
        extent=[x_mm[0], x_mm[-1], z_crop[-1], z_crop[0]],
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("宽度/mm", fontfamily="SimSun", fontsize=8)
    ax.set_ylabel("深度/mm", fontfamily="SimSun", fontsize=8)
    ax.tick_params(direction="out", length=3, width=0.5, labelsize=7)
    for label_obj in ax.get_xticklabels() + ax.get_yticklabels():
        label_obj.set_fontfamily("Times New Roman")

    cax = fig.add_axes([0.84, 0.17, 0.055, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontfamily="Times New Roman", fontsize=8)
    cb.ax.tick_params(direction="out", length=3, width=0.5, labelsize=7)
    for label_obj in cb.ax.get_yticklabels():
        label_obj.set_fontfamily("Times New Roman")

    png_path = OUT_DIR / f"{stem}.png"
    svg_path = OUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=600)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    power_sum = np.fromfile(POWER_SUM_PATH, dtype=np.float32)
    if power_sum.size != NX * NZ:
        raise ValueError(f"Expected {NX * NZ} values, got {power_sum.size}: {POWER_SUM_PATH}")

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

    gray_png, gray_svg = render_image(
        image_db,
        "gray",
        "B-mode/dB",
        "fieldii_vertical_vessel_tissue_bg_pw05_128f_gray_depth5to20mm_db60_70mm",
    )
    pd_png, pd_svg = render_image(
        image_db,
        "hot",
        "Power Doppler/dB",
        "fieldii_vertical_vessel_tissue_bg_pw05_128f_pd_depth5to20mm_db60_70mm",
    )

    print(f"Saved gray PNG: {gray_png}")
    print(f"Saved gray SVG: {gray_svg}")
    print(f"Saved PD PNG: {pd_png}")
    print(f"Saved PD SVG: {pd_svg}")


if __name__ == "__main__":
    main()

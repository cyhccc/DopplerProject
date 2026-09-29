from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path(
    r"E:\Code\mat\y_branch_fieldii_gpu_pd\results_vertical_pw05_128f_tissue_bg_40to60mm_w5mm_blood20_tissue10\preview_partial\power_doppler_power_sum.bin"
)
OUT_DIR = Path(r"D:\DopplerProject\paper_figures\fieldii_vertical_vessel_pd")

NX = 512
NZ = 300
X_MIN_MM = -6.0
X_MAX_MM = 6.0
Z_MIN_MM = 35.0
Z_MAX_MM = 65.0
CROP_Z_MIN_MM = 40.0
CROP_Z_MAX_MM = 60.0
DISPLAY_MIN_DB = -60.0
DISPLAY_MAX_DB = 0.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    power_sum = np.fromfile(DATA_PATH, dtype=np.float32)
    if power_sum.size != NX * NZ:
        raise ValueError(f"Expected {NX * NZ} values, got {power_sum.size}")
    power_sum = power_sum.reshape((NZ, NX), order="F")
    peak = np.nanmax(power_sum)
    if not np.isfinite(peak) or peak <= 0:
        peak = 1.0
    image_db = 10.0 * np.log10(np.maximum(power_sum, np.finfo(np.float32).tiny) / peak)

    x_mm = np.linspace(X_MIN_MM, X_MAX_MM, NX)
    z_mm = np.linspace(Z_MIN_MM, Z_MAX_MM, NZ)
    z_mask = (z_mm >= CROP_Z_MIN_MM) & (z_mm <= CROP_Z_MAX_MM)
    img = image_db[z_mask, :]
    z_crop = z_mm[z_mask]

    mpl.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun"],
            "font.size": 7,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(7.0 / 2.54, 6.0 / 2.54), dpi=600)
    ax = fig.add_axes([0.17, 0.17, 0.63, 0.76])
    im = ax.imshow(
        img,
        cmap="gray",
        vmin=DISPLAY_MIN_DB,
        vmax=DISPLAY_MAX_DB,
        extent=[x_mm[0], x_mm[-1], z_crop[-1], z_crop[0]],
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("宽度/mm", fontfamily="SimSun", fontsize=8)
    ax.set_ylabel("深度/mm", fontfamily="SimSun", fontsize=8)
    ax.tick_params(direction="out", length=3, width=0.5, labelsize=7)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")

    cax = fig.add_axes([0.84, 0.17, 0.055, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("B-mode/dB", fontfamily="Times New Roman", fontsize=8)
    cb.ax.tick_params(direction="out", length=3, width=0.5, labelsize=7)
    for label in cb.ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")

    base = OUT_DIR / "fieldii_vertical_vessel_tissue_bg_preview_2frames_depth40to60mm_db60_70mm"
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)
    print(base.with_suffix(".png"))
    print(base.with_suffix(".svg"))


if __name__ == "__main__":
    main()

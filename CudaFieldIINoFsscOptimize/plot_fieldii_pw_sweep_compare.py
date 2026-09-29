from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot our CUDA vs mach Field II PW sweep benchmark.")
    parser.add_argument("--our-csv", type=Path, required=True)
    parser.add_argument("--mach-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def read_by_pw(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {int(row["pw"]): row for row in csv.DictReader(f)}


def main() -> None:
    args = parse_args()
    our = read_by_pw(args.our_csv)
    mach = read_by_pw(args.mach_csv)
    pws = sorted(set(our) & set(mach))
    if not pws:
        raise RuntimeError("No common PW values found.")

    our_ms = np.array([float(our[pw]["beamform_median_ms"]) for pw in pws], dtype=np.float64)
    mach_ms = np.array([float(mach[pw]["beamform_median_ms"]) for pw in pws], dtype=np.float64)
    our_ms_per_frame = np.array(
        [float(our[pw]["beamform_median_ms_per_frame"]) for pw in pws],
        dtype=np.float64,
    )
    mach_ms_per_frame = np.array(
        [float(mach[pw]["beamform_median_ms_per_frame"]) for pw in pws],
        dtype=np.float64,
    )
    speedup = mach_ms / np.maximum(our_ms, np.finfo(np.float64).tiny)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_prefix.with_suffix(".csv")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pw",
                "frames",
                "input_gb",
                "phase_mode",
                "our_cuda_beamform_ms",
                "mach_beamform_ms",
                "our_cuda_ms_per_frame",
                "mach_ms_per_frame",
                "speedup_mach_over_our_cuda",
            ]
        )
        for i, pw in enumerate(pws):
            writer.writerow(
                [
                    pw,
                    our[pw]["frames"],
                    our[pw]["input_gb"],
                    our[pw]["phase_mode"],
                    f"{our_ms[i]:.9g}",
                    f"{mach_ms[i]:.9g}",
                    f"{our_ms_per_frame[i]:.9g}",
                    f"{mach_ms_per_frame[i]:.9g}",
                    f"{speedup[i]:.9g}",
                ]
            )

    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=180)
    ax.plot(pws, our_ms, marker="o", linewidth=2.2, markersize=5.2, label="Our CUDA", color="#1764ab")
    ax.plot(pws, mach_ms, marker="s", linewidth=2.2, markersize=5.2, label="mach", color="#c2410c")
    ax.set_xlabel("Plane waves")
    ax.set_ylabel("Beamforming time for 128 frames (ms)")
    ax.set_xticks(pws)
    ax.grid(axis="both", which="major", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(frameon=False, loc="upper left")

    annotation_pws = {6, 10, 12, 14, 16}
    y_offset = max(float(mach_ms.max() - mach_ms.min()) * 0.035, 1.0)
    ax.set_ylim(top=float(mach_ms.max() + 3.2 * y_offset))
    for i, pw in enumerate(pws):
        if pw in annotation_pws:
            ax.text(
                pw,
                mach_ms[i] + y_offset,
                f"{speedup[i]:.2f}x",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )

    ax.set_title("Field II vertical vessel PW sweep")
    fig.tight_layout()
    png = args.output_prefix.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    report = args.output_prefix.with_suffix(".txt")
    with report.open("w", encoding="utf-8") as f:
        f.write(f"our_csv={args.our_csv}\n")
        f.write(f"mach_csv={args.mach_csv}\n")
        f.write("Only beamforming calls are timed. File I/O and upload are reported but excluded from medians.\n")
        f.write("Dataset: Field II vertical vessel, pw=3..16, frames=128, samples=788, channels=128.\n")
        for i, pw in enumerate(pws):
            f.write(
                f"pw={pw}, our_cuda={our_ms[i]:.6f}ms, mach={mach_ms[i]:.6f}ms, "
                f"speedup={speedup[i]:.3f}x\n"
            )
        f.write(f"summary_csv={summary_csv}\n")
        f.write(f"plot_png={png}\n")

    print(f"[Info] Saved summary CSV to {summary_csv}")
    print(f"[Info] Saved plot to {png}")
    print(f"[Info] Saved report to {report}")


if __name__ == "__main__":
    main()

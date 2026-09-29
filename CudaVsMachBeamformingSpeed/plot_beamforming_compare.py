from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot our CUDA vs mach beamforming-only benchmark.")
    parser.add_argument("--our-csv", type=Path, required=True)
    parser.add_argument("--mach-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def read_by_frames(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {int(row["frames"]): row for row in csv.DictReader(f)}


def main() -> None:
    args = parse_args()
    our = read_by_frames(args.our_csv)
    mach = read_by_frames(args.mach_csv)
    frames = sorted(set(our) & set(mach))
    if not frames:
        raise RuntimeError("No common frame counts found.")

    our_ms = np.array([float(our[n]["beamform_median_ms"]) for n in frames], dtype=np.float64)
    mach_ms = np.array([float(mach[n]["beamform_median_ms"]) for n in frames], dtype=np.float64)
    speedup = mach_ms / np.maximum(our_ms, np.finfo(np.float64).tiny)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_prefix.with_suffix(".csv")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frames", "our_cuda_beamform_ms", "mach_beamform_ms", "speedup_mach_over_our_cuda"])
        for i, n in enumerate(frames):
            writer.writerow([n, f"{our_ms[i]:.9g}", f"{mach_ms[i]:.9g}", f"{speedup[i]:.9g}"])

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    ax.plot(frames, our_ms, marker="o", linewidth=2.2, markersize=5.5, label="Our CUDA", color="#1764ab")
    ax.plot(frames, mach_ms, marker="s", linewidth=2.2, markersize=5.5, label="mach", color="#d94801")
    ax.set_xlabel("Frames in RF001")
    ax.set_ylabel("Beamforming time (ms)")
    ax.set_xticks(frames)
    ax.grid(axis="both", which="major", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(frameon=False)
    for i, value in enumerate(speedup):
        ax.text(frames[i], max(our_ms[i], mach_ms[i]) * 1.04, f"{value:.1f}x", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    png = args.output_prefix.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    report = args.output_prefix.with_suffix(".txt")
    with report.open("w", encoding="utf-8") as f:
        f.write(f"our_csv={args.our_csv}\n")
        f.write(f"mach_csv={args.mach_csv}\n")
        f.write("Only beamforming is timed. RF001 is preloaded/uploaded before timed regions.\n")
        f.write("mach uses experimental.beamform over five transmits with preallocated GPU output.\n")
        for i, n in enumerate(frames):
            f.write(f"frames={n}, our_cuda={our_ms[i]:.6f}ms, mach={mach_ms[i]:.6f}ms, speedup={speedup[i]:.3f}x\n")
        f.write(f"summary_csv={summary_csv}\n")
        f.write(f"plot_png={png}\n")

    print(f"[Info] Saved summary CSV to {summary_csv}")
    print(f"[Info] Saved plot to {png}")
    print(f"[Info] Saved report to {report}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "fssc_ablation_rf001_seq5_repeat10.csv"
DEFAULT_OUTPUT_PREFIX = ROOT / "fssc_ablation_rf001_seq5_repeat10_line_7cm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RF001 seq=5 FSSC ablation within 7 cm width.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def read_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    frames = np.array([int(row["frames"]) for row in rows], dtype=np.int32)
    dynamic_ms = np.array([float(row["dynamic_generic_ms"]) for row in rows], dtype=np.float64)
    fssc_ms = np.array([float(row["fssc_specialized_ms"]) for row in rows], dtype=np.float64)
    speedup_percent = np.array([float(row["speedup_percent"]) for row in rows], dtype=np.float64)
    return frames, dynamic_ms, fssc_ms, speedup_percent


def main() -> None:
    args = parse_args()
    frames, dynamic_ms, fssc_ms, speedup_percent = read_rows(args.csv)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0 / 2.54, 4.7 / 2.54), dpi=600)
    ax2 = ax.twinx()

    ax.plot(frames, dynamic_ms, marker="o", markersize=2.8, linewidth=1.15, color="#264653", label="Dynamic")
    ax.plot(frames, fssc_ms, marker="s", markersize=2.6, linewidth=1.15, color="#d65f00", label="FSSC")
    ax2.plot(
        frames,
        speedup_percent,
        marker="^",
        markersize=2.5,
        linewidth=0.95,
        linestyle="--",
        color="#4c78a8",
        label="Speedup",
    )

    ax.set_xlabel("Frames")
    ax.set_ylabel("Beamforming time (ms)")
    ax2.set_ylabel("Speedup (%)")
    ax.set_xticks(frames)
    ax.grid(axis="y", linestyle=":", linewidth=0.45, alpha=0.55)

    speedup_min = float(np.floor(min(0.0, speedup_percent.min()) / 5.0) * 5.0)
    speedup_max = float(np.ceil(max(10.0, speedup_percent.max()) / 5.0) * 5.0)
    ax2.set_ylim(speedup_min, speedup_max)
    ax2.tick_params(width=0.55, length=2.4, pad=1.5, colors="#4c78a8")
    ax2.yaxis.label.set_color("#4c78a8")
    ax.tick_params(width=0.55, length=2.4, pad=1.5)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper center",
        bbox_to_anchor=(0.50, 1.16),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.9,
        borderaxespad=0.0,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.45)
    fig.savefig(args.output_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved {args.output_prefix.with_suffix('.png')}")
    print(f"Saved {args.output_prefix.with_suffix('.pdf')}")
    print(f"Saved {args.output_prefix.with_suffix('.svg')}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "fssc_ablation.csv"
OUTPUT_PREFIX = ROOT / "fssc_ablation_line_7cm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FSSC ablation line chart within 7 cm width.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    return parser.parse_args()


def read_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    seq = np.array([int(row["seq"]) for row in rows], dtype=np.int32)
    dynamic_ms = np.array([float(row["dynamic_generic_ms"]) for row in rows], dtype=np.float64)
    fssc_ms = np.array([float(row["fssc_specialized_ms"]) for row in rows], dtype=np.float64)
    speedup_percent = np.array([float(row["speedup_percent"]) for row in rows], dtype=np.float64)
    return seq, dynamic_ms, fssc_ms, speedup_percent


def main() -> None:
    args = parse_args()
    seq, dynamic_ms, fssc_ms, speedup_percent = read_rows(args.csv)

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

    width_in = 7.0 / 2.54
    height_in = 4.7 / 2.54
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=600)
    ax2 = ax.twinx()

    dynamic_color = "#264653"
    fssc_color = "#d65f00"
    speedup_color = "#4c78a8"

    ax.plot(
        seq,
        dynamic_ms,
        marker="o",
        markersize=2.7,
        linewidth=1.15,
        color=dynamic_color,
        label="Dynamic",
    )
    ax.plot(
        seq,
        fssc_ms,
        marker="s",
        markersize=2.5,
        linewidth=1.15,
        color=fssc_color,
        label="FSSC",
    )
    ax2.plot(
        seq,
        speedup_percent,
        marker="^",
        markersize=2.4,
        linewidth=0.95,
        color=speedup_color,
        linestyle="--",
        label="Speedup",
    )

    ax.set_xlabel("Plane-wave sequence length")
    ax.set_ylabel("Beamforming time (ms)")
    ax2.set_ylabel("Speedup (%)")
    ax.set_xlim(3.7, 16.3)
    ax.set_ylim(1.2, 6.0)
    speedup_min = float(np.floor(min(0.0, speedup_percent.min()) / 5.0) * 5.0)
    speedup_max = float(np.ceil(max(40.0, speedup_percent.max()) / 5.0) * 5.0)
    ax2.set_ylim(speedup_min, speedup_max)
    ax.set_xticks(seq)
    ax.set_yticks(np.arange(1.5, 6.1, 1.0))
    ax2.set_yticks(np.arange(speedup_min, speedup_max + 0.1, 10.0))
    ax.grid(axis="y", linestyle=":", linewidth=0.45, alpha=0.55)
    ax.tick_params(width=0.55, length=2.4, pad=1.5)
    ax2.tick_params(width=0.55, length=2.4, pad=1.5, colors=speedup_color)
    ax2.yaxis.label.set_color(speedup_color)

    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.16),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.9,
        borderaxespad=0.0,
    )

    fig.tight_layout(pad=0.45)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved {args.output_prefix.with_suffix('.png')}")
    print(f"Saved {args.output_prefix.with_suffix('.pdf')}")
    print(f"Saved {args.output_prefix.with_suffix('.svg')}")


if __name__ == "__main__":
    main()

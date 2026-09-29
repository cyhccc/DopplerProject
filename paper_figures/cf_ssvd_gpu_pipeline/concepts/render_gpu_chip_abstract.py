"""Render the paper-style abstract GPU icon as SVG and transparent PNG.

Both outputs are produced from the same deterministic geometry so the raster
preview stays visually identical to the editable vector artwork.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent
SVG_PATH = OUT_DIR / "gpu_chip_abstract_v1.svg"
PNG_PATH = OUT_DIR / "gpu_chip_abstract_v1.png"

DARK = "#173C5D"
BLUE = "#2F73B7"
PALE = "#EAF2F8"
PALE_ALT = "#DCEAF5"


def draw_icon(ax: plt.Axes) -> None:
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Six short, evenly spaced pins on every side.
    pin_positions = [152, 194, 236, 278, 320, 362]
    for p in pin_positions:
        ax.plot([80, 112], [p, p], color=DARK, linewidth=4.0,
                solid_capstyle="round")
        ax.plot([400, 432], [p, p], color=DARK, linewidth=4.0,
                solid_capstyle="round")
        ax.plot([p, p], [80, 112], color=DARK, linewidth=4.0,
                solid_capstyle="round")
        ax.plot([p, p], [400, 432], color=DARK, linewidth=4.0,
                solid_capstyle="round")

    # Flat, symmetric chip body and a restrained inner boundary.
    ax.add_patch(
        FancyBboxPatch(
            (112, 112), 288, 288,
            boxstyle="round,pad=0,rounding_size=9",
            facecolor="white", edgecolor=DARK, linewidth=5.0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (138, 138), 236, 236,
            boxstyle="round,pad=0,rounding_size=5",
            facecolor="none", edgecolor=BLUE, linewidth=2.6,
        )
    )

    # A visible 4 x 4 compute array. The central four cells remain implied
    # beneath the GPU core plate, keeping the icon legible at manuscript size.
    cell = 39
    gap = 10
    grid_x = 163
    grid_y = 163
    for row in range(4):
        for col in range(4):
            x = grid_x + col * (cell + gap)
            y = grid_y + row * (cell + gap)
            fill = PALE if (row + col) % 2 == 0 else PALE_ALT
            ax.add_patch(
                Rectangle(
                    (x, y), cell, cell,
                    facecolor=fill, edgecolor=BLUE, linewidth=1.8,
                )
            )

    # White core plate gives the label a calm focal point without shadows.
    ax.add_patch(
        FancyBboxPatch(
            (196, 217), 120, 78,
            boxstyle="round,pad=0,rounding_size=5",
            facecolor="white", edgecolor=DARK, linewidth=3.2,
        )
    )
    ax.text(
        256, 256, "GPU",
        ha="center", va="center",
        color=DARK, fontsize=31, fontweight="bold",
        family="DejaVu Sans",
    )


def main() -> None:
    # Keep the GPU label as editable SVG text instead of converting it to paths.
    rcParams["svg.fonttype"] = "none"
    fig = plt.figure(figsize=(6, 6), dpi=300)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.patch.set_alpha(0)
    draw_icon(ax)

    fig.savefig(
        SVG_PATH, format="svg", transparent=True,
        bbox_inches=None, pad_inches=0,
    )
    fig.savefig(
        PNG_PATH, format="png", transparent=True, dpi=300,
        bbox_inches=None, pad_inches=0,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()

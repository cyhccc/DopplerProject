"""Render a deterministic manuscript-ready RF data cube from the PALA HDF5 file.

The HDF5 array is exposed by h5py as (channel, sample, angle, frame).
The file's label_dim_* attributes describe the original MATLAB logical order,
which is reversed in the on-disk HDF5 view.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


SOURCE = Path(r"E:\Code\mat\dropper\RF_001_to_075\RF\RF_001.hdf5")
OUTPUT = Path(__file__).with_name("rf_data_cube_real_v1.png")

ANGLE_INDEX = 2  # 0 degrees, confirmed from angles_rad
SAMPLE_START = 32
SAMPLE_COUNT = 320
FRAME_COUNT = 64
FRONT_CHANNEL = 64


def signed_log_texture(values: np.ndarray, scale: float) -> np.ndarray:
    """Map bipolar RF samples to a robust, symmetric grayscale texture."""
    clipped = np.clip(values, -scale, scale)
    mapped = np.sign(clipped) * np.log1p(np.abs(clipped)) / np.log1p(scale)
    return cm.gray((mapped + 1.0) / 2.0)


def select_frame_start(rf: np.ndarray, count: int) -> int:
    """Pick the contiguous frame block with the highest mean log-RMS energy."""
    frame_rms = np.sqrt(np.mean(rf * rf, axis=(0, 1)))
    score = np.convolve(np.log1p(frame_rms), np.ones(count), mode="valid")
    return int(np.argmax(score))


def main() -> None:
    with h5py.File(SOURCE, "r") as h5:
        dataset = h5["/rf/rf"]
        shape = dataset.shape
        if shape != (128, 512, 5, 800):
            raise ValueError(f"Unexpected /rf/rf shape: {shape}")
        angles = np.asarray(dataset.attrs["angles_rad"], dtype=float)
        if not np.isclose(angles[ANGLE_INDEX], 0.0):
            raise ValueError("ANGLE_INDEX no longer points to the 0-degree acquisition")
        rf = dataset[:, :, ANGLE_INDEX, :].astype(np.float32)

    frame_start = select_frame_start(rf, FRAME_COUNT)
    frame_stop = frame_start + FRAME_COUNT
    sample_stop = SAMPLE_START + SAMPLE_COUNT
    volume = rf[:, SAMPLE_START:sample_stop, frame_start:frame_stop]

    # A single robust scale is shared by all faces so their contrast is comparable.
    scale = float(np.percentile(np.abs(volume), 99.6))

    # Scientifically consistent orthogonal views of channel x sample x frame.
    front = volume[FRONT_CHANNEL, :, :]          # sample x frame
    top = volume[:, -1, :]                       # channel x frame
    right = volume[:, :, -1].T                   # sample x channel

    front_rgba = signed_log_texture(front, scale)
    top_rgba = signed_log_texture(top, scale)
    right_rgba = signed_log_texture(right, scale)

    frames = np.linspace(0.0, 1.0, FRAME_COUNT)
    samples = np.linspace(0.0, 1.0, SAMPLE_COUNT)
    channels = np.linspace(0.0, 0.34, volume.shape[0])

    fig = plt.figure(figsize=(7.0, 5.25), dpi=300, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Front face: sample x frame matrix at a representative receive channel.
    xf, zf = np.meshgrid(frames, samples)
    yf = np.zeros_like(xf)
    ax.plot_surface(
        xf, yf, zf, facecolors=front_rgba, shade=False,
        rstride=1, cstride=1, linewidth=0, antialiased=False,
    )

    # Top exposed layer edges: channel x frame at the top sample of the crop.
    xt, yt = np.meshgrid(frames, channels)
    zt = np.ones_like(xt)
    ax.plot_surface(
        xt, yt, zt, facecolors=top_rgba, shade=False,
        rstride=1, cstride=1, linewidth=0, antialiased=False,
    )

    # Right face: sample x channel at the last selected frame.
    yr, zr = np.meshgrid(channels, samples)
    xr = np.ones_like(yr)
    ax.plot_surface(
        xr, yr, zr, facecolors=right_rgba, shade=False,
        rstride=1, cstride=1, linewidth=0, antialiased=False,
    )

    # Dense, fine lamination lines expose the tightly packed frame/channel layers.
    layer_color = (0.10, 0.10, 0.10, 0.34)
    for x in frames[::2]:
        ax.plot([x, x], [0, channels[-1]], [1.002, 1.002], color=layer_color, lw=0.28)
        ax.plot([x, x], [-0.002, -0.002], [0, 1], color=(0, 0, 0, 0.16), lw=0.22)
    for y in channels[::4]:
        ax.plot([0, 1], [y, y], [1.003, 1.003], color=(0, 0, 0, 0.20), lw=0.24)
        ax.plot([1.002, 1.002], [y, y], [0, 1], color=layer_color, lw=0.28)

    # Crisp cuboid silhouette.
    edge = (0.05, 0.05, 0.05, 0.95)
    for xs, ys, zs in [
        ([0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 0]),
        ([0, 1, 1, 0, 0], [channels[-1]] * 5, [1, 1, 0, 0, 1]),
        ([0, 0], [0, channels[-1]], [1, 1]),
        ([1, 1], [0, channels[-1]], [1, 1]),
        ([1, 1], [0, channels[-1]], [0, 0]),
    ]:
        ax.plot(xs, ys, zs, color=edge, lw=0.8)

    # Minimal double-headed dimension arrows and exact Chinese labels.
    arrow_color = (0.08, 0.08, 0.08, 1)
    # frame
    ax.quiver(0, -0.025, -0.065, 1, 0, 0, color=arrow_color, linewidth=0.8,
              arrow_length_ratio=0.025, normalize=False)
    ax.quiver(1, -0.025, -0.065, -1, 0, 0, color=arrow_color, linewidth=0.8,
              arrow_length_ratio=0.025, normalize=False)
    ax.text(0.50, -0.040, -0.105, "帧", ha="center", va="top", fontsize=10,
            fontfamily="Microsoft YaHei")
    # samples
    ax.quiver(-0.085, -0.005, 0, 0, 0, 1, color=arrow_color, linewidth=0.8,
              arrow_length_ratio=0.025, normalize=False)
    ax.quiver(-0.085, -0.005, 1, 0, 0, -1, color=arrow_color, linewidth=0.8,
              arrow_length_ratio=0.025, normalize=False)
    ax.text(-0.125, -0.010, 0.50, "采样点", ha="center", va="center", fontsize=10,
            fontfamily="Microsoft YaHei", rotation=90)
    # channels
    ax.quiver(1.045, 0, -0.035, 0, channels[-1], 0, color=arrow_color, linewidth=0.8,
              arrow_length_ratio=0.07, normalize=False)
    ax.quiver(1.045, channels[-1], -0.035, 0, -channels[-1], 0, color=arrow_color,
              linewidth=0.8, arrow_length_ratio=0.07, normalize=False)
    ax.text(1.055, channels[-1] * 0.64, -0.085, "通道", ha="left", va="top", fontsize=10,
            fontfamily="Microsoft YaHei")

    ax.view_init(elev=18, azim=-66)
    ax.set_xlim(-0.14, 1.13)
    ax.set_ylim(-0.08, 0.43)
    ax.set_zlim(-0.12, 1.08)
    ax.set_box_aspect((1.0, 0.42, 0.90))
    ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"source={SOURCE}")
    print(f"shape={shape}; h5py_order=channel,sample,angle,frame")
    print(f"angle_index={ANGLE_INDEX}; angle_deg={np.degrees(angles[ANGLE_INDEX]):.1f}")
    print(f"samples={SAMPLE_START}:{sample_stop}; frames={frame_start}:{frame_stop}")
    print(f"volume_min={int(volume.min())}; volume_max={int(volume.max())}; scale_p99.6={scale:.1f}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark mach beamforming only on RF001.")
    parser.add_argument(
        "--rf-bin",
        type=Path,
        default=Path(r"E:\Code\Cuda\LinearArrayCPWC_OPT_test\bins_7883227\RF_001_7883227_bs100bw_iq_float2.bin"),
    )
    parser.add_argument(
        "--mach-root",
        type=Path,
        default=Path(r"E:\Code\Python\mach-main\mach-main"),
        help="mach repository root that contains src/mach.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/benchmark_rf001_mach_beamforming.csv"))
    parser.add_argument("--frame-counts", default="64,128,256,512,800")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    return parser.parse_args()


def parse_frame_counts(csv_text: str) -> list[int]:
    values = [int(item) for item in csv_text.split(",") if item.strip()]
    if not values:
        raise ValueError("frame count list is empty")
    return values


def configure_geometry() -> dict[str, np.ndarray | float | int]:
    wavelength = 0.09856e-3
    origin_x_wavelength = -64.427759740259731
    delta_x_wavelength = 0.50730519480519476
    origin_z_wavelength = 18.0
    delta_z_wavelength = 0.5
    start_depth_wavelength = 10.0
    transmit_wave_peak_cycles = 2.84375

    seq = 5
    channels = 128
    samples = 256
    frames = 800
    f0 = 15.625e6
    fs = 15.625e6
    sound_speed = 1540.0
    pitch = 0.1e-3
    nx = 256
    nz = 180
    f_number = 1.9

    x_start = origin_x_wavelength * wavelength
    x_end = (origin_x_wavelength + (nx - 1) * delta_x_wavelength) * wavelength
    z_start = origin_z_wavelength * wavelength
    z_end = (origin_z_wavelength + (nz - 1) * delta_z_wavelength) * wavelength
    t0 = 2.0 * start_depth_wavelength * wavelength / sound_speed - transmit_wave_peak_cycles / f0

    xs = np.linspace(x_start, x_end, nx, dtype=np.float32)
    zs = np.linspace(z_start, z_end, nz, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="xy")
    scan_coords = np.stack(
        [grid_x.ravel(), np.zeros(nx * nz, dtype=np.float32), grid_z.ravel()],
        axis=1,
    ).astype(np.float32, copy=False)

    elem_x_origin = -0.5 * float(channels - 1) * pitch
    elem_x = elem_x_origin + np.arange(channels, dtype=np.float32) * pitch
    rx_coords = np.stack(
        [elem_x, np.zeros(channels, dtype=np.float32), np.zeros(channels, dtype=np.float32)],
        axis=1,
    ).astype(np.float32, copy=False)

    angles = np.linspace(-6.0, 6.0, seq, dtype=np.float32) * np.float32(np.pi / 180.0)
    tx_arrivals = []
    x_flat = scan_coords[:, 0]
    z_flat = scan_coords[:, 2]
    for angle in angles:
        tx_x = np.sin(angle, dtype=np.float32)
        tx_z = np.cos(angle, dtype=np.float32)
        tx_arrivals.append((z_flat * tx_z + x_flat * tx_x) / np.float32(sound_speed))

    return {
        "seq": seq,
        "channels": channels,
        "samples": samples,
        "total_frames": frames,
        "scan_coords": scan_coords,
        "rx_coords": rx_coords,
        "tx_arrivals": np.stack(tx_arrivals, axis=0).astype(np.float32, copy=False),
        "rx_start_s": float(t0),
        "sampling_freq_hz": float(fs),
        "sound_speed_m_s": float(sound_speed),
        "modulation_freq_hz": float(f0),
        "f_number": float(f_number),
    }


def stats(samples_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.mach_root / "src"))
    from mach import experimental  # noqa: PLC0415

    frame_counts = parse_frame_counts(args.frame_counts)
    geom = configure_geometry()
    total_frames = int(geom["total_frames"])
    seq = int(geom["seq"])
    channels = int(geom["channels"])
    samples = int(geom["samples"])

    if not args.rf_bin.exists():
        raise FileNotFoundError(args.rf_bin)

    rf_memmap = np.memmap(
        args.rf_bin,
        mode="r",
        dtype=np.complex64,
        shape=(seq, channels, samples, total_frames),
    )
    rx_coords = cp.asarray(geom["rx_coords"], dtype=cp.float32)
    scan_coords = cp.asarray(geom["scan_coords"], dtype=cp.float32)
    tx_arrivals_all = cp.asarray(geom["tx_arrivals"], dtype=cp.float32)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "frames",
                "warmup_repeats",
                "benchmark_repeats",
                "pack_upload_s",
                "beamform_avg_ms",
                "beamform_median_ms",
                "beamform_min_ms",
                "beamform_max_ms",
                "beamform_median_s",
            ]
        )

        print(f"[Info] mach beamforming benchmark on {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print(f"[Info] input={args.rf_bin}")
        for n_frames in frame_counts:
            if n_frames <= 0 or n_frames > total_frames:
                raise ValueError("frame count must stay within 1..800")

            pack_upload_start = time.perf_counter()
            host_subset = np.ascontiguousarray(rf_memmap[..., :n_frames], dtype=np.complex64)
            channel_data = cp.asarray(host_subset)
            out = cp.empty((scan_coords.shape[0], n_frames), dtype=cp.complex64)
            cp.cuda.Stream.null.synchronize()
            pack_upload_s = time.perf_counter() - pack_upload_start

            for _ in range(args.warmup):
                out.fill(0)
                cp.cuda.Stream.null.synchronize()
                experimental.beamform(
                    channel_data=channel_data,
                    rx_coords_m=rx_coords,
                    scan_coords_m=scan_coords,
                    tx_wave_arrivals_s=tx_arrivals_all,
                    out=out,
                    rx_start_s=float(geom["rx_start_s"]),
                    sampling_freq_hz=float(geom["sampling_freq_hz"]),
                    f_number=float(geom["f_number"]),
                    sound_speed_m_s=float(geom["sound_speed_m_s"]),
                    modulation_freq_hz=float(geom["modulation_freq_hz"]),
                    tukey_alpha=0.0,
                )
                cp.cuda.Stream.null.synchronize()

            samples_ms: list[float] = []
            for _ in range(args.repeat):
                out.fill(0)
                cp.cuda.Stream.null.synchronize()
                start = cp.cuda.Event()
                stop = cp.cuda.Event()
                start.record()
                experimental.beamform(
                    channel_data=channel_data,
                    rx_coords_m=rx_coords,
                    scan_coords_m=scan_coords,
                    tx_wave_arrivals_s=tx_arrivals_all,
                    out=out,
                    rx_start_s=float(geom["rx_start_s"]),
                    sampling_freq_hz=float(geom["sampling_freq_hz"]),
                    f_number=float(geom["f_number"]),
                    sound_speed_m_s=float(geom["sound_speed_m_s"]),
                    modulation_freq_hz=float(geom["modulation_freq_hz"]),
                    tukey_alpha=0.0,
                )
                stop.record()
                stop.synchronize()
                samples_ms.append(float(cp.cuda.get_elapsed_time(start, stop)))

            result = stats(samples_ms)
            writer.writerow(
                [
                    "mach",
                    n_frames,
                    args.warmup,
                    args.repeat,
                    f"{pack_upload_s:.9g}",
                    f"{result['avg_ms']:.9g}",
                    f"{result['median_ms']:.9g}",
                    f"{result['min_ms']:.9g}",
                    f"{result['max_ms']:.9g}",
                    f"{result['median_ms'] * 1.0e-3:.9g}",
                ]
            )
            print(
                f"[Bench] frames={n_frames}, mach_beamform_median={result['median_ms']:.3f} ms, "
                f"fps={1000.0 / max(result['median_ms'], 1.0e-6):.3f}"
            )

            del host_subset, channel_data, out
            cp.get_default_memory_pool().free_all_blocks()

    print(f"[Info] Saved mach beamforming CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

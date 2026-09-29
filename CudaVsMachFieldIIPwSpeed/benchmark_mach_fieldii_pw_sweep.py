from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np


FIELDII = {
    "frames": 128,
    "channels": 128,
    "samples": 788,
    "fs_hz": 30.0e6,
    "f0_hz": 5.0e6,
    "sound_speed_m_s": 1540.0,
    "pitch_m": 0.09e-3,
    "nx": 512,
    "nz": 180,
    "x_start_m": -3.0e-3,
    "x_end_m": 3.0e-3,
    "z_start_m": 2.0e-3,
    "z_end_m": 10.0e-3,
    "f_number": 1.75,
    "reference_tstart_samples": 57.58396810051636,
    "angle_start_deg": -9.0,
    "angle_end_deg": 9.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark mach beamforming on Field II PW 3..16 bins.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"E:\Code\mat\y_branch_fieldii_gpu_pd"),
        help="Root that contains results_vertical_pwXX_128f directories.",
    )
    parser.add_argument(
        "--mach-root",
        type=Path,
        default=Path(r"E:\Code\Python\mach-main\mach-main"),
        help="mach repository root that contains src/mach.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/benchmark_fieldii_pw_mach.csv"))
    parser.add_argument("--pw-values", default="3:16", help='PW list like "3,4,5" or range like "3:16".')
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--phase-mode", choices=["on", "off"], default="on")
    return parser.parse_args()


def parse_pw_values(text: str) -> list[int]:
    if ":" in text:
        start, stop = [int(part.strip()) for part in text.split(":", 1)]
        if start > stop:
            raise ValueError("PW range must be ascending")
        values = list(range(start, stop + 1))
    else:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("PW value list is empty")
    for pw in values:
        if pw < 1:
            raise ValueError("PW values must be positive")
    return values


def resolve_fieldii_bin(data_root: Path, pw: int) -> Path:
    pw_text = f"{pw:02d}"
    return (
        data_root
        / f"results_vertical_pw{pw_text}_128f"
        / "cuda_float2_bin"
        / f"vertical_pw{pw_text}_128f_iq_float2.bin"
    )


def configure_geometry(pw: int) -> dict[str, np.ndarray | float | int]:
    nx = int(FIELDII["nx"])
    nz = int(FIELDII["nz"])
    channels = int(FIELDII["channels"])
    pitch = float(FIELDII["pitch_m"])
    sound_speed = float(FIELDII["sound_speed_m_s"])

    xs = np.linspace(float(FIELDII["x_start_m"]), float(FIELDII["x_end_m"]), nx, dtype=np.float32)
    zs = np.linspace(float(FIELDII["z_start_m"]), float(FIELDII["z_end_m"]), nz, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="xy")
    scan_coords = np.stack(
        [grid_x.ravel(), np.zeros(nx * nz, dtype=np.float32), grid_z.ravel()],
        axis=1,
    ).astype(np.float32, copy=False)

    elem_x_origin = -0.5 * float(channels - 1) * pitch
    elem_x = elem_x_origin + np.arange(channels, dtype=np.float32) * np.float32(pitch)
    rx_coords = np.stack(
        [elem_x, np.zeros(channels, dtype=np.float32), np.zeros(channels, dtype=np.float32)],
        axis=1,
    ).astype(np.float32, copy=False)

    angles = np.linspace(
        float(FIELDII["angle_start_deg"]),
        float(FIELDII["angle_end_deg"]),
        pw,
        dtype=np.float32,
    ) * np.float32(np.pi / 180.0)
    tx_arrivals = []
    x_flat = scan_coords[:, 0]
    z_flat = scan_coords[:, 2]
    for angle in angles:
        tx_x = np.float32(np.sin(angle))
        tx_z = np.float32(np.cos(angle))
        tx_arrivals.append((z_flat * tx_z + x_flat * tx_x) / np.float32(sound_speed))

    return {
        "seq": pw,
        "channels": channels,
        "samples": int(FIELDII["samples"]),
        "frames": int(FIELDII["frames"]),
        "scan_coords": scan_coords,
        "rx_coords": rx_coords,
        "tx_arrivals": np.stack(tx_arrivals, axis=0).astype(np.float32, copy=False),
        "rx_start_s": float(FIELDII["reference_tstart_samples"]) / float(FIELDII["fs_hz"]),
        "sampling_freq_hz": float(FIELDII["fs_hz"]),
        "sound_speed_m_s": sound_speed,
        "modulation_freq_hz": float(FIELDII["f0_hz"]),
        "f_number": float(FIELDII["f_number"]),
        "nx": nx,
        "nz": nz,
    }


def stats(samples_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def expected_bytes(pw: int) -> int:
    return (
        pw
        * int(FIELDII["channels"])
        * int(FIELDII["samples"])
        * int(FIELDII["frames"])
        * np.dtype(np.complex64).itemsize
    )


def write_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "method",
            "pw",
            "frames",
            "channels",
            "samples",
            "nx",
            "nz",
            "phase_mode",
            "input_bytes",
            "input_gb",
            "warmup_repeats",
            "benchmark_repeats",
            "read_s",
            "upload_s",
            "beamform_avg_ms",
            "beamform_median_ms",
            "beamform_min_ms",
            "beamform_max_ms",
            "beamform_median_s",
            "beamform_median_ms_per_frame",
        ]
    )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.mach_root / "src"))
    from mach import experimental  # noqa: PLC0415

    pw_values = parse_pw_values(args.pw_values)
    phase_on = args.phase_mode == "on"
    modulation_freq_hz = float(FIELDII["f0_hz"]) if phase_on else 0.0

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        write_header(writer)

        device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        print(f"[Info] mach Field II PW sweep on {device_name}")
        print(f"[Info] data_root={args.data_root}")
        print(
            f"[Info] frames={FIELDII['frames']}, grid={FIELDII['nx']}x{FIELDII['nz']}, "
            f"samples={FIELDII['samples']}, phase_mode={args.phase_mode}"
        )

        for pw in pw_values:
            geom = configure_geometry(pw)
            bin_path = resolve_fieldii_bin(args.data_root, pw)
            if not bin_path.exists():
                raise FileNotFoundError(bin_path)
            byte_count = expected_bytes(pw)
            actual_bytes = bin_path.stat().st_size
            if actual_bytes != byte_count:
                raise ValueError(f"{bin_path} expected {byte_count} bytes, got {actual_bytes}")

            rf_memmap = np.memmap(
                bin_path,
                mode="r",
                dtype=np.complex64,
                shape=(
                    pw,
                    int(FIELDII["channels"]),
                    int(FIELDII["samples"]),
                    int(FIELDII["frames"]),
                ),
            )
            rx_coords = cp.asarray(geom["rx_coords"], dtype=cp.float32)
            scan_coords = cp.asarray(geom["scan_coords"], dtype=cp.float32)
            tx_arrivals = cp.asarray(geom["tx_arrivals"], dtype=cp.float32)

            upload_start = time.perf_counter()
            host_full = np.ascontiguousarray(rf_memmap, dtype=np.complex64)
            channel_data = cp.asarray(host_full)
            out = cp.empty((scan_coords.shape[0], int(FIELDII["frames"])), dtype=cp.complex64)
            cp.cuda.Stream.null.synchronize()
            upload_s = time.perf_counter() - upload_start

            for _ in range(args.warmup):
                out.fill(0)
                cp.cuda.Stream.null.synchronize()
                experimental.beamform(
                    channel_data=channel_data,
                    rx_coords_m=rx_coords,
                    scan_coords_m=scan_coords,
                    tx_wave_arrivals_s=tx_arrivals,
                    out=out,
                    rx_start_s=float(geom["rx_start_s"]),
                    sampling_freq_hz=float(geom["sampling_freq_hz"]),
                    f_number=float(geom["f_number"]),
                    sound_speed_m_s=float(geom["sound_speed_m_s"]),
                    modulation_freq_hz=modulation_freq_hz,
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
                    tx_wave_arrivals_s=tx_arrivals,
                    out=out,
                    rx_start_s=float(geom["rx_start_s"]),
                    sampling_freq_hz=float(geom["sampling_freq_hz"]),
                    f_number=float(geom["f_number"]),
                    sound_speed_m_s=float(geom["sound_speed_m_s"]),
                    modulation_freq_hz=modulation_freq_hz,
                    tukey_alpha=0.0,
                )
                stop.record()
                stop.synchronize()
                samples_ms.append(float(cp.cuda.get_elapsed_time(start, stop)))

            result = stats(samples_ms)
            median_s = result["median_ms"] * 1.0e-3
            median_ms_per_frame = result["median_ms"] / float(FIELDII["frames"])
            input_gb = actual_bytes / (1024.0**3)
            writer.writerow(
                [
                    "mach",
                    pw,
                    int(FIELDII["frames"]),
                    int(FIELDII["channels"]),
                    int(FIELDII["samples"]),
                    int(FIELDII["nx"]),
                    int(FIELDII["nz"]),
                    args.phase_mode,
                    actual_bytes,
                    f"{input_gb:.9g}",
                    args.warmup,
                    args.repeat,
                    "0",
                    f"{upload_s:.9g}",
                    f"{result['avg_ms']:.9g}",
                    f"{result['median_ms']:.9g}",
                    f"{result['min_ms']:.9g}",
                    f"{result['max_ms']:.9g}",
                    f"{median_s:.9g}",
                    f"{median_ms_per_frame:.9g}",
                ]
            )
            f.flush()
            print(
                f"[Bench] pw={pw}, input={input_gb:.3f} GiB, "
                f"mach_beamform_median={result['median_ms']:.3f} ms, "
                f"ms/frame={median_ms_per_frame:.3f}"
            )

            del rf_memmap, host_full, channel_data, out, tx_arrivals, rx_coords, scan_coords
            cp.get_default_memory_pool().free_all_blocks()

    print(f"[Info] Saved mach Field II PW CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot summary telemetry from split Baldr/SEIDR HDF5 output files.

Reads a selected range of part files, concatenates only small 1D/2D telemetry
arrays, and makes continuous time-series + histogram plots for:

  - Strehl before Baldr, estimated as exp(-var(phi))
  - Strehl after Baldr, estimated as exp(-var(phi))
  - OPD RMS before Baldr
  - OPD RMS after Baldr
  - reset/open-loop events

The script intentionally does NOT load residual_opd_nm image cubes by default.
It uses the scalar RMS telemetry already saved in the HDF5 files.

Example
-------
python plot_baldr_h5_batches.py \
  --input-dir baldr_json_im_then_ao_test \
  --stem test_baldr_closed_loop_1000_tt_watchdog \
  --part-start 0 \
  --part-stop 6 \
  --outdir batch_summary_0000_0006

Part range is inclusive: --part-start 0 --part-stop 6 reads part0000..part0006.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing split HDF5 part files.")
    p.add_argument("--stem", required=True, help="Filename stem before _partXXXX.h5, e.g. test_baldr_closed_loop_1000_tt_watchdog")
    p.add_argument("--part-start", type=int, required=True, help="First part index, inclusive.")
    p.add_argument("--part-stop", type=int, required=True, help="Last part index, inclusive.")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory for plots. Default: <input-dir>/<stem>_summary_<start>_<stop>")
    p.add_argument("--wavelength-m", type=float, default=None, help="Science/control wavelength for Strehl estimate. Default: HDF5 attr wvl0_m.")
    p.add_argument("--strehl-from", choices=["rms", "residual_cube"], default="rms", help="Use scalar RMS telemetry or load residual cube to compute piston-removed variance. Default avoids image-cube loading.")
    p.add_argument("--max-cube-frames", type=int, default=20000, help="Safety limit if --strehl-from residual_cube is used.")
    p.add_argument("--smooth-frames", type=int, default=1, help="Optional moving-average smoothing window for plotted time series only.")
    p.add_argument("--hist-bins", type=int, default=80)
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--show-invalid", action="store_true", help="Also shade invalid_residual_flag==0 frames when available.")
    return p.parse_args()


def part_path(input_dir: Path, stem: str, idx: int) -> Path:
    return input_dir / f"{stem}_part{idx:04d}.h5"


def read_1d_or_default(h5, name: str, n: int, default=0, dtype=float):
    if name in h5:
        return np.asarray(h5[name][...])
    return np.full(n, default, dtype=dtype)


def estimate_strehl_from_rms_nm(rms_nm: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Marechal approximation: Strehl ~ exp[-var(phi)], phi=2pi OPD/lambda.

    If RMS OPD is in nm and piston has been removed, var(phi)=(2pi*rms/lambda)^2.
    """
    rms_m = 1e-9 * np.asarray(rms_nm, dtype=float)
    var_phi = (2.0 * np.pi * rms_m / float(wavelength_m)) ** 2
    return np.exp(-var_phi)


def estimate_strehl_from_cube(h5, wavelength_m: float, max_frames: int):
    """Compute Strehl from residual_opd_nm cube without keeping the cube in memory."""
    if "residual_opd_nm" not in h5:
        raise KeyError("residual_opd_nm not found")
    d = h5["residual_opd_nm"]
    n = d.shape[0]
    if n > max_frames:
        raise RuntimeError(f"Refusing to read {n} residual frames; increase --max-cube-frames if intentional.")
    if "pupil_mask" in h5:
        pupil = np.asarray(h5["pupil_mask"][...], dtype=bool)
    else:
        pupil = np.isfinite(d[0])
    out = np.empty(n, dtype=float)
    for k in range(n):
        arr_nm = np.asarray(d[k], dtype=float)
        vals_nm = arr_nm[pupil]
        vals_nm = vals_nm[np.isfinite(vals_nm)]
        vals_nm = vals_nm - np.mean(vals_nm)
        rms_m = 1e-9 * np.std(vals_nm)
        out[k] = np.exp(-(2.0 * np.pi * rms_m / float(wavelength_m)) ** 2)
    return out


def moving_average(y: np.ndarray, w: int) -> np.ndarray:
    w = int(w)
    if w <= 1:
        return y
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(y, kernel, mode="same")


def contiguous_true_regions(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def shade_regions(ax, x: np.ndarray, mask: np.ndarray, color="0.85", alpha=0.55, label=None):
    first = True
    for a, b in contiguous_true_regions(mask):
        if a >= len(x):
            continue
        b = min(b, len(x) - 1)
        ax.axvspan(x[a], x[b], color=color, alpha=alpha, lw=0, label=label if first else None)
        first = False


def load_parts(args):
    arrays = {
        "frame": [],
        "rms_in": [],
        "rms_out": [],
        "rms_pre_naomi": [],
        "rms_post_naomi": [],
        "ctrl_in": [],
        "ctrl_out": [],
        "unctrl_in": [],
        "unctrl_out": [],
        "reset": [],
        "reset_reason": [],
        "valid": [],
        "strehl_out_cube": [],
    }
    manifest = []
    wavelength_m = args.wavelength_m
    fs_hz = None
    total_frames = 0

    for idx in range(args.part_start, args.part_stop + 1):
        path = part_path(args.input_dir, args.stem, idx)
        if not path.exists():
            raise FileNotFoundError(f"Missing part file: {path}")
        with h5py.File(path, "r") as h5:
            n = int(h5["rms_baldr_input_nm"].shape[0])
            g0 = int(h5.attrs.get("global_frame_start", total_frames))
            g1 = int(h5.attrs.get("global_frame_stop_exclusive", g0 + n))
            if wavelength_m is None:
                wavelength_m = float(h5.attrs.get("wvl0_m"))
            if fs_hz is None:
                fs_hz = float(h5.attrs.get("fs_hz", 1.0))

            frame = np.arange(g0, g0 + n, dtype=int)
            arrays["frame"].append(frame)
            arrays["rms_in"].append(np.asarray(h5["rms_baldr_input_nm"][...], dtype=float))
            arrays["rms_out"].append(np.asarray(h5["rms_after_baldr_nm"][...], dtype=float))
            arrays["rms_pre_naomi"].append(read_1d_or_default(h5, "rms_pre_naomi_nm", n, np.nan, float))
            arrays["rms_post_naomi"].append(read_1d_or_default(h5, "rms_post_naomi_nm", n, np.nan, float))
            arrays["ctrl_in"].append(read_1d_or_default(h5, "rms_control_component_baldr_input_nm", n, np.nan, float))
            arrays["ctrl_out"].append(read_1d_or_default(h5, "rms_control_component_after_baldr_nm", n, np.nan, float))
            arrays["unctrl_in"].append(read_1d_or_default(h5, "rms_uncontrolled_component_baldr_input_nm", n, np.nan, float))
            arrays["unctrl_out"].append(read_1d_or_default(h5, "rms_uncontrolled_component_after_baldr_nm", n, np.nan, float))
            arrays["reset"].append(read_1d_or_default(h5, "loop_reset_flag", n, 0, np.uint8).astype(bool))
            arrays["reset_reason"].append(read_1d_or_default(h5, "loop_reset_reason_code", n, 0, int))
            arrays["valid"].append(read_1d_or_default(h5, "valid_residual_flag", n, 1, np.uint8).astype(bool))
            if args.strehl_from == "residual_cube":
                arrays["strehl_out_cube"].append(estimate_strehl_from_cube(h5, wavelength_m, args.max_cube_frames))

            manifest.append({
                "part_index": idx,
                "path": str(path),
                "n_frames": n,
                "global_frame_start": g0,
                "global_frame_stop_exclusive": g1,
                "fs_hz": fs_hz,
                "wavelength_m": wavelength_m,
                "resets_in_part": int(np.sum(arrays["reset"][-1])),
                "invalid_frames_in_part": int(np.sum(~arrays["valid"][-1])),
            })
            total_frames += n

    cat = {k: np.concatenate(v) if v else np.array([]) for k, v in arrays.items()}
    return cat, manifest, float(wavelength_m), float(fs_hz)


def save_json(path: Path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_timeseries(outdir: Path, frame, time_s, rms_in, rms_out, strehl_in, strehl_out, reset, valid, args):
    rms_in_p = moving_average(rms_in, args.smooth_frames)
    rms_out_p = moving_average(rms_out, args.smooth_frames)
    strehl_in_p = moving_average(strehl_in, args.smooth_frames)
    strehl_out_p = moving_average(strehl_out, args.smooth_frames)

    fig, ax = plt.subplots(figsize=(13, 5))
    if args.show_invalid:
        shade_regions(ax, time_s, ~valid, color="0.82", alpha=0.65, label="invalid/replaced residual")
    shade_regions(ax, time_s, reset, color="tab:red", alpha=0.22, label="Baldr reset/opened")
    ax.plot(time_s, rms_in_p, label="Before Baldr", lw=1.1)
    ax.plot(time_s, rms_out_p, label="After Baldr", lw=1.1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pupil RMS OPD [nm]")
    ax.set_title("Baldr OPD RMS telemetry")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "timeseries_opd_rms_before_after_baldr.png", dpi=args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    if args.show_invalid:
        shade_regions(ax, time_s, ~valid, color="0.82", alpha=0.65, label="invalid/replaced residual")
    shade_regions(ax, time_s, reset, color="tab:red", alpha=0.22, label="Baldr reset/opened")
    ax.plot(time_s, strehl_in_p, label="Before Baldr", lw=1.1)
    ax.plot(time_s, strehl_out_p, label="After Baldr", lw=1.1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Approx. Strehl, exp[-var(phi)]")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Approximate Strehl telemetry")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "timeseries_strehl_before_after_baldr.png", dpi=args.dpi)
    plt.close(fig)


def plot_histograms(outdir: Path, rms_in, rms_out, strehl_in, strehl_out, valid, args):
    # Use only valid frames for after-Baldr histograms where replacement may have occurred.
    good = np.asarray(valid, dtype=bool)
    if not np.any(good):
        good = np.ones_like(good, dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rms_in[np.isfinite(rms_in)], bins=args.hist_bins, alpha=0.55, label="Before Baldr")
    ax.hist(rms_out[good & np.isfinite(rms_out)], bins=args.hist_bins, alpha=0.55, label="After Baldr, valid only")
    ax.set_xlabel("Pupil RMS OPD [nm]")
    ax.set_ylabel("Frames")
    ax.set_title("OPD RMS distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "histogram_opd_rms_before_after_baldr.png", dpi=args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(strehl_in[np.isfinite(strehl_in)], bins=args.hist_bins, alpha=0.55, label="Before Baldr")
    ax.hist(strehl_out[good & np.isfinite(strehl_out)], bins=args.hist_bins, alpha=0.55, label="After Baldr, valid only")
    ax.set_xlabel("Approx. Strehl")
    ax.set_ylabel("Frames")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("Approximate Strehl distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "histogram_strehl_before_after_baldr.png", dpi=args.dpi)
    plt.close(fig)


def plot_control_floor(outdir: Path, time_s, ctrl_in, ctrl_out, unctrl_in, unctrl_out, reset, valid, args):
    if np.all(~np.isfinite(ctrl_in)):
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    if args.show_invalid:
        shade_regions(ax, time_s, ~valid, color="0.82", alpha=0.65, label="invalid/replaced residual")
    shade_regions(ax, time_s, reset, color="tab:red", alpha=0.22, label="Baldr reset/opened")
    ax.plot(time_s, moving_average(ctrl_in, args.smooth_frames), label="Controllable before", lw=1.0)
    ax.plot(time_s, moving_average(ctrl_out, args.smooth_frames), label="Controllable after", lw=1.0)
    ax.plot(time_s, moving_average(unctrl_in, args.smooth_frames), label="Uncontrolled/floor before", lw=1.0, ls="--")
    ax.plot(time_s, moving_average(unctrl_out, args.smooth_frames), label="Uncontrolled/floor after", lw=1.0, ls="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("RMS OPD [nm]")
    ax.set_title("Projection onto Baldr control basis")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "timeseries_controlled_uncontrolled_opd.png", dpi=args.dpi)
    plt.close(fig)


def main():
    args = parse_args()
    if args.part_stop < args.part_start:
        raise ValueError("--part-stop must be >= --part-start")
    if args.outdir is None:
        args.outdir = args.input_dir / f"{args.stem}_summary_part{args.part_start:04d}_part{args.part_stop:04d}"
    args.outdir.mkdir(parents=True, exist_ok=True)

    data, manifest, wavelength_m, fs_hz = load_parts(args)
    frame = data["frame"]
    time_s = frame / fs_hz

    rms_in = data["rms_in"]
    rms_out = data["rms_out"]
    reset = data["reset"].astype(bool)
    valid = data["valid"].astype(bool)

    strehl_in = estimate_strehl_from_rms_nm(rms_in, wavelength_m)
    if args.strehl_from == "residual_cube" and data["strehl_out_cube"].size == rms_out.size:
        strehl_out = data["strehl_out_cube"]
    else:
        strehl_out = estimate_strehl_from_rms_nm(rms_out, wavelength_m)

    summary = {
        "stem": args.stem,
        "part_start": args.part_start,
        "part_stop": args.part_stop,
        "n_frames_loaded": int(frame.size),
        "frame_start": int(frame[0]) if frame.size else None,
        "frame_stop_inclusive": int(frame[-1]) if frame.size else None,
        "fs_hz": float(fs_hz),
        "wavelength_m": float(wavelength_m),
        "strehl_method": args.strehl_from,
        "n_reset_frames": int(np.sum(reset)),
        "n_invalid_residual_frames": int(np.sum(~valid)),
        "rms_in_nm_median": float(np.nanmedian(rms_in)),
        "rms_out_nm_median_valid": float(np.nanmedian(rms_out[valid])) if np.any(valid) else float(np.nanmedian(rms_out)),
        "strehl_in_median": float(np.nanmedian(strehl_in)),
        "strehl_out_median_valid": float(np.nanmedian(strehl_out[valid])) if np.any(valid) else float(np.nanmedian(strehl_out)),
        "part_manifest": manifest,
    }
    save_json(args.outdir / "batch_summary.json", summary)

    plot_timeseries(args.outdir, frame, time_s, rms_in, rms_out, strehl_in, strehl_out, reset, valid, args)
    plot_histograms(args.outdir, rms_in, rms_out, strehl_in, strehl_out, valid, args)
    plot_control_floor(args.outdir, time_s, data["ctrl_in"], data["ctrl_out"], data["unctrl_in"], data["unctrl_out"], reset, valid, args)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote plots and summary to: {args.outdir.resolve()}")
    print("Key files:")
    print("  timeseries_opd_rms_before_after_baldr.png")
    print("  timeseries_strehl_before_after_baldr.png")
    print("  histogram_opd_rms_before_after_baldr.png")
    print("  histogram_strehl_before_after_baldr.png")
    print("  timeseries_controlled_uncontrolled_opd.png")
    print("  batch_summary.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot Baldr/SEIDR HDF5 batch telemetry with post-Baldr performance measured
inside an eroded pupil mask.

This is the standard, non-comparative analysis script. You specify a single
boundary erosion width, e.g. --boundary-erosion-pixels 2, and the script uses
that mask for post-Baldr residual OPD/Strehl metrics computed directly from
residual_opd_nm.

Important convention
--------------------
The HDF5 files normally store only scalar pre-Baldr RMS telemetry
(rms_baldr_input_nm), not the full pre-Baldr OPD cube. Therefore:

  Before Baldr = rms_baldr_input_nm scalar telemetry from the run.
  After Baldr  = recomputed from residual_opd_nm using the eroded pupil mask.

If a future file contains an input cube named baldr_input_opd_nm, this script
will use it automatically when --before-source auto is selected.

Example
-------
python baldrapp/playground/onsky_sims/analysis_scripts/plot_baldr_h5_batches_inner_mask.py \
  --input-dir /Users/bencb/Downloads \
  --stem test_baldr_closed_loop_1000_tt_watchdog \
  --part-start 0 \
  --part-stop 9 \
  --outdir /Users/bencb/Downloads/batch_summary_inner_mask \
  --boundary-erosion-pixels 2 \
  --drop-invalid \
  --smooth-frames 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import binary_erosion
except Exception:
    binary_erosion = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing split HDF5 part files.")
    p.add_argument("--stem", required=True, help="Filename stem before _partXXXX.h5.")
    p.add_argument("--part-start", type=int, required=True, help="First part index, inclusive.")
    p.add_argument("--part-stop", type=int, required=True, help="Last part index, inclusive.")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory for plots.")
    p.add_argument("--wavelength-m", type=float, default=None, help="Wavelength for Strehl estimate. Default: HDF5 attr wvl0_m.")
    p.add_argument("--phase-key", default="residual_opd_nm", help="Post-Baldr residual phase cube dataset, in nm OPD.")
    p.add_argument("--pupil-key", default="pupil_mask", help="Cropped pupil mask dataset.")
    p.add_argument("--input-phase-key", default="baldr_input_opd_nm", help="Optional pre-Baldr input cube dataset, in nm OPD, if present.")
    p.add_argument("--before-source", choices=["auto", "scalar", "cube"], default="auto", help="Before-Baldr metric source. auto uses input cube if present, otherwise scalar telemetry.")
    p.add_argument("--boundary-erosion-pixels", type=int, default=2, help="Number of pupil-boundary pixels to remove for cube-based metrics. Applies to outer pupil and central obscuration boundaries.")
    p.add_argument("--drop-reset", action="store_true", help="Drop frames where loop_reset_flag != 0 from histograms/summary and grey them in timeseries.")
    p.add_argument("--drop-invalid", action="store_true", help="Drop frames where valid_residual_flag == 0 from histograms/summary and grey them in timeseries.")
    p.add_argument("--max-cube-frames", type=int, default=0, help="Safety limit for cube frames; 0 means no limit.")
    p.add_argument("--smooth-frames", type=int, default=1, help="Moving-average smoothing window for plotted time series only.")
    p.add_argument("--hist-bins", type=int, default=80)
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--show-invalid", action="store_true", help="Shade invalid/replaced frames on timeseries.")
    return p.parse_args()


def part_path(input_dir: Path, stem: str, idx: int) -> Path:
    return input_dir / f"{stem}_part{idx:04d}.h5"


def read_1d_or_default(h5, name: str, n: int, default=0, dtype=float):
    if name in h5:
        return np.asarray(h5[name][...])
    return np.full(n, default, dtype=dtype)


def estimate_strehl_from_rms_nm(rms_nm: np.ndarray, wavelength_m: float) -> np.ndarray:
    rms_m = 1e-9 * np.asarray(rms_nm, dtype=float)
    var_phi = (2.0 * np.pi * rms_m / float(wavelength_m)) ** 2
    return np.exp(-var_phi)


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


def make_eroded_pupil(pupil: np.ndarray, erosion_pixels: int) -> np.ndarray:
    pupil = np.asarray(pupil, dtype=bool)
    n = int(erosion_pixels)
    if n <= 0:
        return pupil.copy()
    if binary_erosion is None:
        raise RuntimeError("scipy.ndimage.binary_erosion is required for --boundary-erosion-pixels > 0. Install scipy or use --boundary-erosion-pixels 0.")
    eroded = pupil.copy()
    structure = np.ones((3, 3), dtype=bool)
    for _ in range(n):
        eroded = binary_erosion(eroded, structure=structure, border_value=0)
    if np.sum(eroded) == 0:
        raise RuntimeError(f"Boundary erosion of {n} pixels removed the whole pupil. Use a smaller value.")
    return eroded


def rms_from_frame_nm(frame_nm: np.ndarray, mask: np.ndarray) -> float:
    vals = np.asarray(frame_nm, dtype=float)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    vals = vals - np.nanmean(vals)  # piston remove inside chosen mask
    return float(np.nanstd(vals))


def compute_cube_rms_stream(h5, key: str, mask: np.ndarray, max_frames: int = 0) -> np.ndarray:
    if key not in h5:
        raise KeyError(f"Dataset '{key}' not found in {h5.filename}")
    d = h5[key]
    n = int(d.shape[0])
    if max_frames and n > max_frames:
        raise RuntimeError(f"Refusing to read {n} frames from {key}; increase --max-cube-frames or set 0 if intentional.")
    out = np.empty(n, dtype=float)
    for k in range(n):
        out[k] = rms_from_frame_nm(d[k], mask)
    return out


def load_parts(args):
    arrays = {
        "frame": [],
        "rms_in": [],
        "rms_out_inner": [],
        "rms_out_scalar": [],
        "rms_pre_naomi": [],
        "rms_post_naomi": [],
        "ctrl_in": [],
        "ctrl_out": [],
        "unctrl_in": [],
        "unctrl_out": [],
        "reset": [],
        "reset_reason": [],
        "valid": [],
    }
    manifest = []
    wavelength_m = args.wavelength_m
    fs_hz = None
    total_frames_seen = 0
    mask_summary = None
    before_source_used = None

    for idx in range(args.part_start, args.part_stop + 1):
        path = part_path(args.input_dir, args.stem, idx)
        if not path.exists():
            raise FileNotFoundError(f"Missing part file: {path}")
        with h5py.File(path, "r") as h5:
            if args.phase_key not in h5:
                raise KeyError(f"Could not find post-Baldr phase cube '{args.phase_key}' in {path}")
            if args.pupil_key not in h5:
                raise KeyError(f"Could not find pupil mask '{args.pupil_key}' in {path}")

            n = int(h5[args.phase_key].shape[0])
            g0 = int(h5.attrs.get("global_frame_start", total_frames_seen))
            g1 = int(h5.attrs.get("global_frame_stop_exclusive", g0 + n))
            if wavelength_m is None:
                wavelength_m = float(h5.attrs.get("wvl0_m"))
            if fs_hz is None:
                fs_hz = float(h5.attrs.get("fs_hz", 1.0))

            pupil = np.asarray(h5[args.pupil_key][...], dtype=bool)
            inner_mask = make_eroded_pupil(pupil, args.boundary_erosion_pixels)
            if mask_summary is None:
                mask_summary = {
                    "pupil_pixels_full": int(np.sum(pupil)),
                    "pupil_pixels_inner": int(np.sum(inner_mask)),
                    "boundary_erosion_pixels": int(args.boundary_erosion_pixels),
                    "fraction_pixels_kept": float(np.sum(inner_mask) / max(1, np.sum(pupil))),
                    "phase_shape": list(h5[args.phase_key].shape[1:]),
                }

            # Before-Baldr: use input cube if explicitly requested or available in auto mode.
            use_input_cube = False
            if args.before_source == "cube":
                if args.input_phase_key not in h5:
                    raise KeyError(f"--before-source cube requested but '{args.input_phase_key}' not found in {path}")
                use_input_cube = True
            elif args.before_source == "auto" and args.input_phase_key in h5:
                use_input_cube = True

            if use_input_cube:
                rms_in = compute_cube_rms_stream(h5, args.input_phase_key, inner_mask, args.max_cube_frames)
                before_source_used = f"cube:{args.input_phase_key} with eroded mask"
            else:
                rms_in = np.asarray(h5["rms_baldr_input_nm"][...], dtype=float)
                if before_source_used is None:
                    before_source_used = "scalar:rms_baldr_input_nm (not boundary-eroded; input cube unavailable)"

            rms_out_inner = compute_cube_rms_stream(h5, args.phase_key, inner_mask, args.max_cube_frames)

            frame = np.arange(g0, g0 + n, dtype=int)
            arrays["frame"].append(frame)
            arrays["rms_in"].append(rms_in)
            arrays["rms_out_inner"].append(rms_out_inner)
            arrays["rms_out_scalar"].append(np.asarray(h5["rms_after_baldr_nm"][...], dtype=float) if "rms_after_baldr_nm" in h5 else np.full(n, np.nan))
            arrays["rms_pre_naomi"].append(read_1d_or_default(h5, "rms_pre_naomi_nm", n, np.nan, float))
            arrays["rms_post_naomi"].append(read_1d_or_default(h5, "rms_post_naomi_nm", n, np.nan, float))
            arrays["ctrl_in"].append(read_1d_or_default(h5, "rms_control_component_baldr_input_nm", n, np.nan, float))
            arrays["ctrl_out"].append(read_1d_or_default(h5, "rms_control_component_after_baldr_nm", n, np.nan, float))
            arrays["unctrl_in"].append(read_1d_or_default(h5, "rms_uncontrolled_component_baldr_input_nm", n, np.nan, float))
            arrays["unctrl_out"].append(read_1d_or_default(h5, "rms_uncontrolled_component_after_baldr_nm", n, np.nan, float))
            arrays["reset"].append(read_1d_or_default(h5, "loop_reset_flag", n, 0, np.uint8).astype(bool))
            arrays["reset_reason"].append(read_1d_or_default(h5, "loop_reset_reason_code", n, 0, int))
            arrays["valid"].append(read_1d_or_default(h5, "valid_residual_flag", n, 1, np.uint8).astype(bool))

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
            total_frames_seen += n

    cat = {k: np.concatenate(v) if v else np.array([]) for k, v in arrays.items()}
    return cat, manifest, mask_summary, before_source_used, float(wavelength_m), float(fs_hz)


def plot_timeseries(outdir: Path, time_s, rms_in, rms_out, strehl_in, strehl_out, reset, valid, keep, args):
    rms_in_p = moving_average(rms_in, args.smooth_frames)
    rms_out_p = moving_average(rms_out, args.smooth_frames)
    strehl_in_p = moving_average(strehl_in, args.smooth_frames)
    strehl_out_p = moving_average(strehl_out, args.smooth_frames)

    fig, ax = plt.subplots(figsize=(13, 5))
    if args.show_invalid or args.drop_invalid:
        shade_regions(ax, time_s, ~valid, color="0.82", alpha=0.65, label="invalid/replaced residual")
    if args.drop_reset:
        shade_regions(ax, time_s, reset, color="tab:red", alpha=0.22, label="Baldr reset/opened")
    else:
        shade_regions(ax, time_s, reset, color="tab:red", alpha=0.22, label="Baldr reset/opened")
    ax.plot(time_s, rms_in_p, label="Before Baldr", lw=1.1)
    ax.plot(time_s, rms_out_p, label=f"After Baldr, inner mask {args.boundary_erosion_pixels}px", lw=1.1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pupil RMS OPD [nm]")
    ax.set_title("Baldr OPD RMS telemetry")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "timeseries_opd_rms_before_after_baldr.png", dpi=args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    if args.show_invalid or args.drop_invalid:
        shade_regions(ax, time_s, ~valid, color="0.82", alpha=0.65, label="invalid/replaced residual")
    shade_regions(ax, time_s, reset, color="tab:red", alpha=0.22, label="Baldr reset/opened")
    ax.plot(time_s, strehl_in_p, label="Before Baldr", lw=1.1)
    ax.plot(time_s, strehl_out_p, label=f"After Baldr, inner mask {args.boundary_erosion_pixels}px", lw=1.1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Approx. Strehl, exp[-var(phi)]")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Approximate Strehl telemetry")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "timeseries_strehl_before_after_baldr.png", dpi=args.dpi)
    plt.close(fig)

    improvement_nm = rms_in - rms_out
    fig, ax = plt.subplots(figsize=(13, 4.5))
    if args.show_invalid or args.drop_invalid:
        shade_regions(ax, time_s, ~valid, color="0.82", alpha=0.65, label="invalid/replaced residual")
    shade_regions(ax, time_s, reset, color="tab:red", alpha=0.22, label="Baldr reset/opened")
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(time_s, moving_average(improvement_nm, args.smooth_frames), lw=1.1, label="Before - after")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("OPD RMS improvement [nm]")
    ax.set_title("Baldr OPD improvement")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "timeseries_opd_improvement_nm.png", dpi=args.dpi)
    plt.close(fig)


def plot_histograms(outdir: Path, rms_in, rms_out, strehl_in, strehl_out, keep, args):
    good = np.asarray(keep, dtype=bool)
    if not np.any(good):
        good = np.ones_like(good, dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rms_in[good & np.isfinite(rms_in)], bins=args.hist_bins, alpha=0.55, label="Before Baldr")
    ax.hist(rms_out[good & np.isfinite(rms_out)], bins=args.hist_bins, alpha=0.55, label=f"After Baldr, inner {args.boundary_erosion_pixels}px")
    ax.set_xlabel("Pupil RMS OPD [nm]")
    ax.set_ylabel("Frames")
    ax.set_title("OPD RMS distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "histogram_opd_rms_before_after_baldr.png", dpi=args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(strehl_in[good & np.isfinite(strehl_in)], bins=args.hist_bins, alpha=0.55, label="Before Baldr")
    ax.hist(strehl_out[good & np.isfinite(strehl_out)], bins=args.hist_bins, alpha=0.55, label=f"After Baldr, inner {args.boundary_erosion_pixels}px")
    ax.set_xlabel("Approx. Strehl")
    ax.set_ylabel("Frames")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("Approximate Strehl distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "histogram_strehl_before_after_baldr.png", dpi=args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    imp = rms_in - rms_out
    ax.hist(imp[good & np.isfinite(imp)], bins=args.hist_bins, alpha=0.7)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("OPD RMS improvement [nm]")
    ax.set_ylabel("Frames")
    ax.set_title("Baldr OPD improvement distribution")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "histogram_opd_improvement_nm.png", dpi=args.dpi)
    plt.close(fig)


def plot_control_floor(outdir: Path, time_s, ctrl_in, ctrl_out, unctrl_in, unctrl_out, reset, valid, args):
    if np.all(~np.isfinite(ctrl_in)):
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    if args.show_invalid or args.drop_invalid:
        shade_regions(ax, time_s, ~valid, color="0.82", alpha=0.65, label="invalid/replaced residual")
    shade_regions(ax, time_s, reset, color="tab:red", alpha=0.22, label="Baldr reset/opened")
    ax.plot(time_s, moving_average(ctrl_in, args.smooth_frames), label="Controllable before", lw=1.0)
    ax.plot(time_s, moving_average(ctrl_out, args.smooth_frames), label="Controllable after", lw=1.0)
    ax.plot(time_s, moving_average(unctrl_in, args.smooth_frames), label="Uncontrolled/floor before", lw=1.0, ls="--")
    ax.plot(time_s, moving_average(unctrl_out, args.smooth_frames), label="Uncontrolled/floor after", lw=1.0, ls="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("RMS OPD [nm]")
    ax.set_title("Projection onto Baldr control basis, scalar telemetry")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "timeseries_controlled_uncontrolled_opd.png", dpi=args.dpi)
    plt.close(fig)


def save_mask_diagnostic(outdir: Path, input_dir: Path, stem: str, part_start: int, pupil_key: str, erosion: int, dpi: int):
    path = part_path(input_dir, stem, part_start)
    with h5py.File(path, "r") as h5:
        pupil = np.asarray(h5[pupil_key][...], dtype=bool)
    inner = make_eroded_pupil(pupil, erosion)
    img = np.zeros(pupil.shape, dtype=float)
    img[pupil] = 1.0
    img[inner] = 2.0
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(img, origin="lower", cmap="viridis", vmin=0, vmax=2)
    ax.set_title(f"Metric mask: eroded by {erosion} px")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["outside", "discarded boundary", "metric mask"])
    fig.tight_layout()
    fig.savefig(outdir / "metric_inner_pupil_mask.png", dpi=dpi)
    plt.close(fig)


def save_json(path: Path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    args = parse_args()
    if args.part_stop < args.part_start:
        raise ValueError("--part-stop must be >= --part-start")
    if args.outdir is None:
        args.outdir = args.input_dir / f"{args.stem}_inner{args.boundary_erosion_pixels}px_summary_part{args.part_start:04d}_part{args.part_stop:04d}"
    args.outdir.mkdir(parents=True, exist_ok=True)

    data, manifest, mask_summary, before_source_used, wavelength_m, fs_hz = load_parts(args)
    frame = data["frame"]
    time_s = frame / fs_hz

    rms_in = data["rms_in"]
    rms_out = data["rms_out_inner"]
    reset = data["reset"].astype(bool)
    valid = data["valid"].astype(bool)

    keep = np.ones_like(valid, dtype=bool)
    if args.drop_invalid:
        keep &= valid
    if args.drop_reset:
        keep &= ~reset

    strehl_in = estimate_strehl_from_rms_nm(rms_in, wavelength_m)
    strehl_out = estimate_strehl_from_rms_nm(rms_out, wavelength_m)

    improvement_nm = rms_in - rms_out
    strehl_gain = strehl_out - strehl_in

    summary = {
        "stem": args.stem,
        "part_start": args.part_start,
        "part_stop": args.part_stop,
        "n_frames_loaded": int(frame.size),
        "n_frames_used_for_hist_summary": int(np.sum(keep)),
        "frame_start": int(frame[0]) if frame.size else None,
        "frame_stop_inclusive": int(frame[-1]) if frame.size else None,
        "fs_hz": float(fs_hz),
        "wavelength_m": float(wavelength_m),
        "before_metric_source": before_source_used,
        "after_metric_source": f"cube:{args.phase_key} with eroded pupil mask",
        "mask_summary": mask_summary,
        "drop_invalid": bool(args.drop_invalid),
        "drop_reset": bool(args.drop_reset),
        "n_reset_frames": int(np.sum(reset)),
        "n_invalid_residual_frames": int(np.sum(~valid)),
        "rms_in_nm_median": float(np.nanmedian(rms_in[keep])) if np.any(keep) else float(np.nanmedian(rms_in)),
        "rms_out_nm_median": float(np.nanmedian(rms_out[keep])) if np.any(keep) else float(np.nanmedian(rms_out)),
        "rms_improvement_nm_median": float(np.nanmedian(improvement_nm[keep])) if np.any(keep) else float(np.nanmedian(improvement_nm)),
        "fraction_frames_improved": float(np.nanmean(improvement_nm[keep] > 0)) if np.any(keep) else float(np.nanmean(improvement_nm > 0)),
        "strehl_in_median": float(np.nanmedian(strehl_in[keep])) if np.any(keep) else float(np.nanmedian(strehl_in)),
        "strehl_out_median": float(np.nanmedian(strehl_out[keep])) if np.any(keep) else float(np.nanmedian(strehl_out)),
        "strehl_gain_median": float(np.nanmedian(strehl_gain[keep])) if np.any(keep) else float(np.nanmedian(strehl_gain)),
        "part_manifest": manifest,
    }
    save_json(args.outdir / "batch_summary.json", summary)
    save_mask_diagnostic(args.outdir, args.input_dir, args.stem, args.part_start, args.pupil_key, args.boundary_erosion_pixels, args.dpi)

    plot_timeseries(args.outdir, time_s, rms_in, rms_out, strehl_in, strehl_out, reset, valid, keep, args)
    plot_histograms(args.outdir, rms_in, rms_out, strehl_in, strehl_out, keep, args)
    plot_control_floor(args.outdir, time_s, data["ctrl_in"], data["ctrl_out"], data["unctrl_in"], data["unctrl_out"], reset, valid, args)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote plots and summary to: {args.outdir.resolve()}")
    print("Key files:")
    print("  metric_inner_pupil_mask.png")
    print("  timeseries_opd_rms_before_after_baldr.png")
    print("  timeseries_strehl_before_after_baldr.png")
    print("  timeseries_opd_improvement_nm.png")
    print("  histogram_opd_rms_before_after_baldr.png")
    print("  histogram_strehl_before_after_baldr.png")
    print("  histogram_opd_improvement_nm.png")
    print("  timeseries_controlled_uncontrolled_opd.png")
    print("  batch_summary.json")


if __name__ == "__main__":
    main()

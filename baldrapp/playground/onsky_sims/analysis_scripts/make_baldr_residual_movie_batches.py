#!/usr/bin/env python3
"""
Create a movie of Baldr residual OPD phase screens across several HDF5 batch files.

Input files are expected to be named

    <stem>_part0000.h5
    <stem>_part0001.h5
    ...

The script reads frames lazily, masks outside the pupil by default, and can
optionally skip reset/invalid frames.

Examples
--------
python baldrapp/playground/onsky_sims/make_baldr_residual_movie_batches.py \
  --input-dir ~/Downloads \
  --stem test_baldr_closed_loop_1000_tt_watchdog \
  --part-start 0 \
  --part-stop 5 \
  --output baldr_residual_phase_movie_parts0000_0007.mp4 \
  --frame-stride 20 \
  --movie-fps 25

Drop frames where Baldr opened/reset:

python baldrapp/playground/onsky_sims/analysis_scripts/make_baldr_residual_movie_batches.py \
  --input-dir ~/Downloads \
  --stem test_baldr_closed_loop_1000_tt_watchdog \
  --part-start 0 \
  --part-stop 9 \
  --output ~/Downloads/baldr_residual_phase_movie_parts0000_0009.mp4 \
  --drop-reset \
  --drop-invalid \
  --frame-stride 30 \
  --movie-fps 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing HDF5 part files.")
    p.add_argument("--stem", required=True, help="Filename stem before _partXXXX.h5.")
    p.add_argument("--part-start", type=int, required=True, help="First part index, inclusive.")
    p.add_argument("--part-stop", type=int, required=True, help="Last part index, inclusive.")
    p.add_argument("--part-digits", type=int, default=4, help="Number of digits in part index, default 4 for part0000.")
    p.add_argument("--output", type=Path, required=True, help="Output movie path, usually .mp4.")
    p.add_argument("--phase-key", default="residual_opd_nm", help="HDF5 phase cube dataset.")
    p.add_argument("--pupil-key", default="pupil_mask", help="HDF5 cropped pupil mask dataset.")
    p.add_argument("--frame-stride", type=int, default=2, help="Show every Nth selected simulation frame.")
    p.add_argument("--movie-fps", type=int, default=25, help="Output movie frame rate.")
    p.add_argument("--start-frame", type=int, default=0, help="Global selected-frame start index before stride. Usually 0.")
    p.add_argument("--stop-frame", type=int, default=None, help="Global selected-frame stop index before stride. Default: all.")
    p.add_argument("--scale-mode", default="global_percentile", choices=["global_percentile", "fixed", "per_frame"])
    p.add_argument("--percentile", type=float, default=99.0, help="Percentile for global/per-frame colour scaling.")
    p.add_argument("--fixed-vlim-nm", type=float, default=250.0, help="Fixed symmetric colour limit in nm.")
    p.add_argument("--n-scale-samples", type=int, default=200, help="Number of frames sampled for global colour scale.")
    p.add_argument("--drop-reset", action="store_true", help="Skip frames where loop_reset_flag != 0, if present.")
    p.add_argument("--drop-invalid", action="store_true", help="Skip frames where valid_residual_flag == 0, if present.")
    p.add_argument("--mask-outside-pupil", action="store_true", default=True, help="Mask outside pupil_mask with NaNs. Default: enabled.")
    p.add_argument("--no-mask-outside-pupil", dest="mask_outside_pupil", action="store_false", help="Do not mask outside pupil.")
    p.add_argument("--mask-fill", default="nan", choices=["nan", "zero"], help="Visual fill outside pupil if masking. Default: nan.")
    p.add_argument("--bitrate", type=int, default=3000, help="FFmpeg bitrate.")
    p.add_argument("--dpi", type=int, default=140, help="Movie DPI.")
    p.add_argument("--max-title-reset-markers", type=int, default=1, help="Show reset/invalid labels in title.")
    return p.parse_args()


def part_path(input_dir: Path, stem: str, idx: int, digits: int) -> Path:
    return input_dir / f"{stem}_part{idx:0{digits}d}.h5"


def pupil_rms_nm(frame_nm: np.ndarray, pupil: np.ndarray) -> float:
    vals = frame_nm[pupil]
    vals = vals[np.isfinite(vals)]
    return float(np.std(vals)) if vals.size else float("nan")


def keep_mask_for_file(h5: h5py.File, n: int, args) -> np.ndarray:
    keep = np.ones(n, dtype=bool)
    if args.drop_invalid:
        if "valid_residual_flag" in h5:
            keep &= np.asarray(h5["valid_residual_flag"][:], dtype=np.uint8).astype(bool)
        else:
            print("  NOTE: --drop-invalid requested but valid_residual_flag missing; not applied.")
    if args.drop_reset:
        if "loop_reset_flag" in h5:
            keep &= np.asarray(h5["loop_reset_flag"][:], dtype=np.uint8) == 0
        else:
            print("  NOTE: --drop-reset requested but loop_reset_flag missing; not applied.")
    return keep


def build_frame_index(paths: list[Path], args):
    records = []
    shapes = []
    fs_hz = None
    global_raw_offset = 0

    for part_i, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as h5:
            if args.phase_key not in h5:
                raise KeyError(f"{path} does not contain dataset {args.phase_key!r}")
            ds = h5[args.phase_key]
            if ds.ndim != 3:
                raise ValueError(f"{path}:{args.phase_key} has shape {ds.shape}; expected [frame,y,x]")
            n = int(ds.shape[0])
            shapes.append(tuple(ds.shape[1:]))
            if fs_hz is None:
                fs_hz = float(h5.attrs.get("fs_hz", 1000.0))
            keep = keep_mask_for_file(h5, n, args)
            local_idx = np.where(keep)[0]
            global_start = int(h5.attrs.get("global_frame_start", global_raw_offset))
            for k in local_idx:
                records.append({
                    "path": path,
                    "part": args.part_start + part_i,
                    "local": int(k),
                    "global_raw": int(global_start + k),
                })
            global_raw_offset += n

    if len(set(shapes)) != 1:
        raise ValueError(f"Spatial shape mismatch across files: {sorted(set(shapes))}")

    if not records:
        raise RuntimeError("No frames selected after drop-reset/drop-invalid filters.")

    # Apply global selected-frame window and stride after concatenating kept frames.
    start = max(0, int(args.start_frame))
    stop = len(records) if args.stop_frame is None else min(len(records), int(args.stop_frame))
    if stop <= start:
        raise ValueError("No frames selected: check --start-frame/--stop-frame.")
    records = records[start:stop:int(args.frame_stride)]
    if not records:
        raise ValueError("No frames selected after stride.")
    return records, shapes[0], fs_hz


def read_pupil(h5: h5py.File, args, shape) -> np.ndarray:
    if args.pupil_key not in h5:
        print(f"  NOTE: pupil dataset {args.pupil_key!r} missing; using all pixels as pupil.")
        return np.ones(shape, dtype=bool)
    pupil = np.asarray(h5[args.pupil_key][:], dtype=bool)
    if tuple(pupil.shape) != tuple(shape):
        raise ValueError(f"{args.pupil_key} shape {pupil.shape} does not match phase frame shape {shape}")
    return pupil


def mask_frame(frame: np.ndarray, pupil: np.ndarray, args) -> np.ndarray:
    frame = np.asarray(frame, dtype=float)
    if not args.mask_outside_pupil:
        return frame
    out = frame.copy()
    if args.mask_fill == "zero":
        out[~pupil] = 0.0
    else:
        out[~pupil] = np.nan
    return out


def get_frame_and_meta(record, args, shape):
    with h5py.File(record["path"], "r") as h5:
        ds = h5[args.phase_key]
        pupil = read_pupil(h5, args, shape)
        frame = np.asarray(ds[record["local"]], dtype=float)
        frame = mask_frame(frame, pupil, args)
        rms_stored = None
        if "rms_after_baldr_nm" in h5:
            rms_stored = float(np.asarray(h5["rms_after_baldr_nm"][record["local"]]))
        reset_flag = int(h5["loop_reset_flag"][record["local"]]) if "loop_reset_flag" in h5 else 0
        valid_flag = int(h5["valid_residual_flag"][record["local"]]) if "valid_residual_flag" in h5 else 1
    return frame, pupil, rms_stored, reset_flag, valid_flag


def estimate_vlim(records, args, shape):
    if args.scale_mode == "fixed":
        return float(args.fixed_vlim_nm)
    if args.scale_mode == "per_frame":
        return None

    sample_idx = np.linspace(0, len(records) - 1, min(args.n_scale_samples, len(records))).astype(int)
    vals = []
    for idx in sample_idx:
        frame, pupil, _, _, _ = get_frame_and_meta(records[idx], args, shape)
        vals.append(np.abs(frame[pupil]).ravel())
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(args.fixed_vlim_nm)
    v = float(np.nanpercentile(vals, args.percentile))
    if not np.isfinite(v) or v <= 0:
        v = float(args.fixed_vlim_nm)
    return v


def main():
    args = parse_args()
    if args.part_stop < args.part_start:
        raise ValueError("--part-stop must be >= --part-start")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    paths = [part_path(args.input_dir, args.stem, i, args.part_digits) for i in range(args.part_start, args.part_stop + 1)]
    records, shape, fs_hz = build_frame_index(paths, args)

    print("Input parts:")
    for p in paths:
        print(f"  {p}")
    print(f"Selected movie frames: {len(records)}")
    print(f"Frame shape: {shape}")
    print(f"Simulation sampling: {fs_hz:.1f} Hz")
    print(f"Movie FPS: {args.movie_fps}")
    print(f"Mask outside pupil: {args.mask_outside_pupil} ({args.mask_fill})")
    print(f"Output movie: {args.output}")

    print("Estimating colour scale...")
    vlim = estimate_vlim(records, args, shape)
    if vlim is None:
        print("Using per-frame colour scaling.")
    else:
        print(f"Using symmetric colour scale: ±{vlim:.2f} nm")

    first, pupil_first, rms_stored, reset_flag, valid_flag = get_frame_and_meta(records[0], args, shape)

    # Estimate RMS statistics from sampled/movie frames without loading all frames at once.
    rms_vals = []
    for rec in records:
        frame, pupil, rms_s, _, _ = get_frame_and_meta(rec, args, shape)
        rms_vals.append(rms_s if rms_s is not None else pupil_rms_nm(frame, pupil))
    rms_vals = np.asarray(rms_vals, dtype=float)
    mean_rms = float(np.nanmean(rms_vals))
    median_rms = float(np.nanmedian(rms_vals))
    print(f"Mean selected-frame residual RMS:   {mean_rms:.2f} nm")
    print(f"Median selected-frame residual RMS: {median_rms:.2f} nm")

    fig, ax = plt.subplots(figsize=(7, 6))
    if vlim is None:
        v0 = np.nanpercentile(np.abs(first[pupil_first]), args.percentile)
        if not np.isfinite(v0) or v0 <= 0:
            v0 = args.fixed_vlim_nm
        im = ax.imshow(first, origin="lower", cmap="RdBu_r", vmin=-v0, vmax=v0)
    else:
        im = ax.imshow(first, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Residual OPD [nm]")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    title = ax.set_title("")
    fig.tight_layout()

    writer = FFMpegWriter(
        fps=int(args.movie_fps),
        metadata={"title": "Baldr residual OPD phase screens", "artist": "BaldrApp"},
        bitrate=int(args.bitrate),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(args.output), dpi=int(args.dpi)):
        for i, rec in enumerate(records):
            frame, pupil, rms_s, reset_flag, valid_flag = get_frame_and_meta(rec, args, shape)
            rms_k = rms_s if rms_s is not None else pupil_rms_nm(frame, pupil)

            if vlim is None:
                vv = np.nanpercentile(np.abs(frame[pupil]), args.percentile)
                if not np.isfinite(vv) or vv <= 0:
                    vv = args.fixed_vlim_nm
                im.set_clim(-vv, vv)
            im.set_data(frame)

            t_k = rec["global_raw"] / fs_hz
            flags = []
            if reset_flag:
                flags.append("RESET")
            if not valid_flag:
                flags.append("INVALID")
            flag_txt = " | " + ",".join(flags) if flags else ""
            title.set_text(
                f"Baldr residual OPD | part {rec['part']:04d}"
            )
            # title.set_text(
            #     f"Baldr residual OPD | part {rec['part']:04d}, local {rec['local']} | global {rec['global_raw']} | t={t_k:.3f} s{flag_txt}\n"
            #     f"Frame RMS={rms_k:.1f} nm | mean shown RMS={mean_rms:.1f} nm | median shown RMS={median_rms:.1f} nm"
            # )
            writer.grab_frame()
            if (i + 1) % 50 == 0 or (i + 1) == len(records):
                print(f"  wrote movie frame {i+1}/{len(records)} from part {rec['part']:04d}, local frame {rec['local']}", flush=True)

    plt.close(fig)
    print(f"Done. Wrote movie: {args.output.resolve()}")


if __name__ == "__main__":
    main()

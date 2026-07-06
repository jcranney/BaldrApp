#!/usr/bin/env python3
"""
Merge Baldr HDF5 residual phase-screen batches into a FITS datacube only.

This lightweight version writes ONLY:

  - Primary FITS HDU containing the Baldr post-loop phase-screen cube
  - Primary FITS header metadata
  - Optional JSON sidecar summary

It does NOT append telemetry tables, pupil-mask extensions, modal tables, or
large static arrays.  The cube is copied in chunks, so the full HDF5 cube is
not loaded into RAM.

Input files are expected to be named:

  <stem>_part0000.h5
  <stem>_part0001.h5
  ...

Default dataset:

  residual_opd_nm

The FITS primary data shape is:

  [frame, y, x]

Examples
--------
Keep all frames:

python baldrapp/playground/onsky_sims/merge_baldr_h5_residuals_to_fits_cube_only_masked.py \
  --input-dir ~/Downloads/ \
  --stem test_baldr_closed_loop_1000_tt_watchdog \
  --part-start 0 \
  --part-stop 7 \
  --output post_baldr_residual_cube_parts0000_0007.fits \
  --overwrite

Drop frames where the watchdog reset/opened Baldr:

python baldrapp/playground/onsky_sims/merge_baldr_h5_residuals_to_fits_cube_only.py \
  --input-dir ~/Downloads/ \
  --stem test_baldr_closed_loop_1000_tt_watchdog \
  --part-start 0 \
  --part-stop 7 \
  --output post_baldr_residual_cube_parts0000_0007_noresets.fits \
  --drop-reset \
  --overwrite

Drop frames marked invalid/replaced by the patched writer:



python baldrapp/playground/onsky_sims/analysis_scripts/merge_baldr_h5_residuals_to_fits_cube_only_masked.py \
  --input-dir ~/Downloads/ \
  --stem test_baldr_closed_loop_1000_tt_watchdog \
  --part-start 0 \
  --part-stop 9 \
  --output ~/Downloads/post_baldr_residual_cube_parts0000_0009_noresets.fits \
  --drop-invalid \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import h5py
except Exception as exc:
    raise RuntimeError("This script requires h5py. Install with: python -m pip install h5py") from exc

try:
    from astropy.io import fits
except Exception as exc:
    raise RuntimeError("This script requires astropy. Install with: python -m pip install astropy") from exc


WHITELIST_ATTR_ALIASES = {
    "description": "DESC",
    "units_residual_opd": "BUNIT0",
    "fs_hz": "FS_HZ",
    "wvl0_m": "WVL0_M",
    "control_basis": "CBASIS",
    "control_modes": "NCMODES",
    "first_stage_modes": "NFSMODE",
    "gain": "GAIN",
    "leak": "LEAK",
    "baldr_lag_frames": "BLAGFR",
    "include_shotnoise": "SHOTNOI",
    "crop_padding_pixels": "CROPPAD",
    "residual_crop_y0": "CROPY0",
    "residual_crop_y1": "CROPY1",
    "residual_crop_x0": "CROPX0",
    "residual_crop_x1": "CROPX1",
    "tt_rms_nm": "TTRMSNM",
    "tt_rms_cmd": "TTRMSC",
    "tt_axis": "TTAXIS",
    "tt_frequencies_hz": "TTFREQS",
    "tt_cmd_rms_total_used": "TTCMDRMS",
    "reset_on_fail": "RESETON",
    "reset_rms_threshold_nm": "RSTTHR",
    "reset_hold_frames": "RSTHOLD",
    "max_loop_resets": "RSTMAX",
    "global_frame_start": "GFRM0",
    "global_frame_stop_exclusive": "GFRM1",
    "output_part_index": "OPART",
    "frames_per_output_file": "FRMPFILE",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing HDF5 part files.")
    p.add_argument("--stem", required=True, help="Filename stem before _partXXXX.h5.")
    p.add_argument("--part-start", type=int, required=True, help="First part index, inclusive.")
    p.add_argument("--part-stop", type=int, required=True, help="Last part index, inclusive.")
    p.add_argument("--part-digits", type=int, default=4, help="Number of digits in part index, default 4 for part0000.")
    p.add_argument("--dataset", default="residual_opd_nm", help="HDF5 dataset to merge into the primary FITS cube.")
    p.add_argument("--output", type=Path, required=True, help="Output FITS path.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--chunk-frames", type=int, default=256, help="Frames copied per I/O chunk.")
    p.add_argument("--drop-reset", action="store_true", help="Drop frames where loop_reset_flag != 0, when that dataset exists.")
    p.add_argument("--drop-invalid", action="store_true", help="Drop frames where valid_residual_flag == 0, when that dataset exists.")
    p.add_argument("--mask-outside-pupil", action="store_true", default=True, help="Set pixels outside pupil_mask to NaN before writing FITS cube. Default: enabled.")
    p.add_argument("--no-mask-outside-pupil", dest="mask_outside_pupil", action="store_false", help="Do not mask pixels outside the pupil before writing.")
    p.add_argument("--mask-fill", default="nan", choices=["nan", "zero"], help="Fill value outside pupil when masking. Default: nan.")
    p.add_argument("--summary-json", type=Path, default=None, help="Optional sidecar JSON summary path.")
    return p.parse_args()


def part_path(input_dir: Path, stem: str, idx: int, digits: int) -> Path:
    return input_dir / f"{stem}_part{idx:0{digits}d}.h5"


def as_python_scalar_or_short_string(value, max_string_len: int = 60):
    """Return a FITS-safe scalar value.

    Long strings are truncated for header compatibility. The full metadata is
    preserved in HISTORY chunks and in the JSON sidecar summary.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.reshape(-1)[0].item()
        else:
            value = json.dumps(value.tolist())
    if isinstance(value, (list, tuple, dict)):
        value = json.dumps(value)
    if isinstance(value, str) and len(value) > max_string_len:
        return value[: max_string_len - 3] + "..."
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)[:max_string_len]


def add_safe_card(header: fits.Header, key: str, value, comment: str = "") -> None:
    key = str(key).upper().replace(" ", "")[:8]
    if not key:
        return
    value = as_python_scalar_or_short_string(value)
    try:
        header[key] = (value, str(comment)[:47])
        # Force Astropy to format now; remove if not safe.
        _ = header.cards[key].image
    except Exception as exc:
        try:
            if key in header:
                del header[key]
        except Exception:
            pass
        msg = f"Dropped incompatible header card {key}: {value} ({exc})"
        add_history_chunks(header, msg)


def add_history_chunks(header: fits.Header, text: str, chunk: int = 68) -> None:
    text = str(text)
    for i in range(0, len(text), chunk):
        try:
            header.add_history(text[i : i + chunk])
        except Exception:
            # Last-resort: silently skip pathological history content.
            pass


def validate_header(header: fits.Header) -> fits.Header:
    safe = fits.Header()
    for card in list(header.cards):
        try:
            _ = card.image
            safe.append(card, end=True)
        except Exception as exc:
            add_history_chunks(safe, f"Dropped bad header card {card.keyword}: {exc}")
    return safe


def frame_keep_mask(h5: h5py.File, n: int, drop_invalid: bool, drop_reset: bool) -> np.ndarray:
    keep = np.ones(n, dtype=bool)
    if drop_invalid:
        if "valid_residual_flag" in h5:
            keep &= np.asarray(h5["valid_residual_flag"][:], dtype=np.uint8).astype(bool)
        else:
            print("  NOTE: --drop-invalid requested but valid_residual_flag missing; keeping all frames for this criterion.")
    if drop_reset:
        if "loop_reset_flag" in h5:
            keep &= np.asarray(h5["loop_reset_flag"][:], dtype=np.uint8) == 0
        else:
            print("  NOTE: --drop-reset requested but loop_reset_flag missing; keeping all frames for this criterion.")
    return keep


def read_file_info(paths: list[Path], dataset: str):
    infos = []
    first_shape = None
    first_dtype = None
    first_attrs = {}
    total_raw = 0
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        with h5py.File(p, "r") as h5:
            if dataset not in h5:
                raise KeyError(f"{p} does not contain dataset {dataset!r}")
            ds = h5[dataset]
            if ds.ndim != 3:
                raise ValueError(f"{p}:{dataset} has shape {ds.shape}; expected 3D [frame,y,x]")
            shape = tuple(ds.shape)
            if first_shape is None:
                first_shape = shape[1:]
                first_dtype = ds.dtype
                first_attrs = dict(h5.attrs)
            elif shape[1:] != first_shape:
                raise ValueError(f"Spatial shape mismatch: {p}:{dataset} has {shape[1:]}, expected {first_shape}")
            infos.append({"path": str(p), "n": int(shape[0]), "shape": list(shape), "dtype": str(ds.dtype)})
            total_raw += shape[0]
    return infos, first_shape, first_dtype, first_attrs, int(total_raw)


def collect_counts(paths: list[Path], dataset: str, drop_invalid: bool, drop_reset: bool):
    raw_counts = []
    kept_counts = []
    resets_dropped = []
    invalid_dropped = []
    for p in paths:
        with h5py.File(p, "r") as h5:
            n = int(h5[dataset].shape[0])
            raw_counts.append(n)
            keep = frame_keep_mask(h5, n, drop_invalid, drop_reset)
            kept_counts.append(int(np.sum(keep)))
            if "loop_reset_flag" in h5:
                resets_dropped.append(int(np.sum(np.asarray(h5["loop_reset_flag"][:], dtype=np.uint8) != 0)))
            else:
                resets_dropped.append(None)
            if "valid_residual_flag" in h5:
                invalid_dropped.append(int(np.sum(np.asarray(h5["valid_residual_flag"][:], dtype=np.uint8) == 0)))
            else:
                invalid_dropped.append(None)
    return raw_counts, kept_counts, resets_dropped, invalid_dropped


def make_primary_header(first_attrs: dict, args, total_kept: int, ny: int, nx: int, total_raw: int, paths: list[Path]) -> fits.Header:
    hdr = fits.Header()
    hdr["ORIGIN"] = "BaldrApp"
    hdr["CONTENT"] = "Baldr residual OPD cube"
    hdr["BUNIT"] = "nm OPD"
    hdr["DATASET"] = str(args.dataset)[:60]
    hdr["CUBEAXIS"] = "frame,y,x"
    hdr["NFRAMES"] = int(total_kept)
    hdr["RAWFRMS"] = int(total_raw)
    hdr["DROPRST"] = bool(args.drop_reset)
    hdr["DROPINV"] = bool(args.drop_invalid)
    hdr["MASKPUP"] = bool(args.mask_outside_pupil)
    add_safe_card(hdr, "MASKFILL", args.mask_fill, "Outside-pupil fill")
    hdr["PART0"] = int(args.part_start)
    hdr["PART1"] = int(args.part_stop)
    hdr["NPARTS"] = int(len(paths))
    hdr["NY"] = int(ny)
    hdr["NX"] = int(nx)
    add_safe_card(hdr, "INPSTEM", args.stem, "Input file stem")
    add_safe_card(hdr, "INDIR", str(args.input_dir), "Input directory")

    for attr_name, key in WHITELIST_ATTR_ALIASES.items():
        if attr_name in first_attrs:
            add_safe_card(hdr, key, first_attrs[attr_name], f"H5 attr {attr_name}")

    # Preserve complete HDF5 attributes as HISTORY chunks only. These are not
    # machine-ideal but are robust against FITS card-length failures.
    attr_dict = {}
    for k, v in first_attrs.items():
        if isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif not isinstance(v, (str, int, float, bool)):
            v = str(v)
        attr_dict[str(k)] = v
    add_history_chunks(hdr, "Full first-file HDF5 attrs JSON:")
    add_history_chunks(hdr, json.dumps(attr_dict, sort_keys=True))

    add_history_chunks(hdr, "Input HDF5 files:")
    for p in paths:
        add_history_chunks(hdr, p.name)

    return validate_header(hdr)



def get_pupil_mask_for_file(h5: h5py.File, spatial_shape) -> np.ndarray | None:
    """Return cropped pupil mask matching the residual cube shape, if available."""
    if "pupil_mask" not in h5:
        return None
    pupil = np.asarray(h5["pupil_mask"][:], dtype=bool)
    if tuple(pupil.shape) != tuple(spatial_shape):
        raise ValueError(
            f"pupil_mask shape {pupil.shape} does not match residual cube spatial shape {spatial_shape}."
        )
    return pupil


def apply_pupil_mask_block(block: np.ndarray, pupil: np.ndarray | None, args) -> np.ndarray:
    """Mask pixels outside the pupil in a frame block."""
    if not getattr(args, "mask_outside_pupil", True):
        return block
    if pupil is None:
        print("  NOTE: --mask-outside-pupil enabled but pupil_mask is missing; writing unmasked frames.")
        return block
    out = np.asarray(block, dtype=np.float32).copy()
    if args.mask_fill == "zero":
        out[:, ~pupil] = 0.0
    else:
        out[:, ~pupil] = np.nan
    return out

def write_cube(paths: list[Path], args, hdr: fits.Header, total_kept: int, spatial_shape):
    ny, nx = spatial_shape
    out = args.output
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; use --overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Allocate FITS file, then update via memmap. This avoids loading the full
    # cube into memory, although Astropy creates the primary data array object.
    data = np.empty((total_kept, ny, nx), dtype=np.float32)
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    fits.HDUList([hdu]).writeto(out, overwrite=args.overwrite, output_verify="fix")
    del data, hdu

    pos = 0
    with fits.open(out, mode="update", memmap=True) as hdul:
        for p in paths:
            with h5py.File(p, "r") as h5:
                ds = h5[args.dataset]
                n = int(ds.shape[0])
                keep = frame_keep_mask(h5, n, args.drop_invalid, args.drop_reset)
                idx = np.where(keep)[0]
                if len(idx) == 0:
                    continue
                for a in range(0, len(idx), int(args.chunk_frames)):
                    sub_idx = idx[a : a + int(args.chunk_frames)]
                    block = np.asarray(ds[sub_idx, :, :], dtype=np.float32)
                    pupil = get_pupil_mask_for_file(h5, spatial_shape)
                    block = apply_pupil_mask_block(block, pupil, args)
                    m = int(block.shape[0])
                    hdul[0].data[pos : pos + m, :, :] = block
                    print(f"  wrote output frames {pos}:{pos + m} from {p.name}", flush=True)
                    pos += m
        hdul.flush()

    if pos != total_kept:
        raise RuntimeError(f"Internal frame-count mismatch: wrote {pos}, expected {total_kept}")


def serialise_attrs(attrs: dict):
    out = {}
    for k, v in attrs.items():
        if isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif not isinstance(v, (str, int, float, bool)):
            v = str(v)
        out[str(k)] = v
    return out


def main():
    args = parse_args()
    if args.part_stop < args.part_start:
        raise ValueError("--part-stop must be >= --part-start")

    paths = [part_path(args.input_dir, args.stem, i, args.part_digits) for i in range(args.part_start, args.part_stop + 1)]
    infos, spatial_shape, dtype, first_attrs, total_raw = read_file_info(paths, args.dataset)
    raw_counts, kept_counts, reset_counts, invalid_counts = collect_counts(paths, args.dataset, args.drop_invalid, args.drop_reset)
    total_kept = int(sum(kept_counts))
    if total_kept <= 0:
        raise RuntimeError("No frames selected after filtering.")

    ny, nx = spatial_shape
    hdr = make_primary_header(first_attrs, args, total_kept, ny, nx, total_raw, paths)

    print("Input parts:")
    for p, raw, kept, nrst, ninv in zip(paths, raw_counts, kept_counts, reset_counts, invalid_counts):
        extra = []
        if nrst is not None:
            extra.append(f"reset-flagged={nrst}")
        if ninv is not None:
            extra.append(f"invalid={ninv}")
        extra_s = ", " + ", ".join(extra) if extra else ""
        print(f"  {p.name}: raw={raw}, kept={kept}{extra_s}")

    print(f"Writing FITS cube only: {args.output}")
    print(f"  shape = ({total_kept}, {ny}, {nx})")
    print(f"  dataset = {args.dataset}")
    print(f"  drop_reset = {args.drop_reset}")
    print(f"  drop_invalid = {args.drop_invalid}")
    print(f"  estimated primary cube size = {total_kept * ny * nx * 4 / 1024**3:.3f} GiB")

    write_cube(paths, args, hdr, total_kept, spatial_shape)

    summary = {
        "output": str(args.output),
        "input_dir": str(args.input_dir),
        "stem": args.stem,
        "part_start": int(args.part_start),
        "part_stop": int(args.part_stop),
        "dataset": args.dataset,
        "raw_frames": int(total_raw),
        "kept_frames": int(total_kept),
        "spatial_shape_yx": [int(ny), int(nx)],
        "drop_reset": bool(args.drop_reset),
        "drop_invalid": bool(args.drop_invalid),
        "mask_outside_pupil": bool(args.mask_outside_pupil),
        "mask_fill": str(args.mask_fill),
        "parts": [
            {
                "path": str(p),
                "raw_frames": int(raw),
                "kept_frames": int(kept),
                "reset_flagged_frames": None if nrst is None else int(nrst),
                "invalid_frames": None if ninv is None else int(ninv),
            }
            for p, raw, kept, nrst, ninv in zip(paths, raw_counts, kept_counts, reset_counts, invalid_counts)
        ],
        "hdf5_attrs_first_file": serialise_attrs(first_attrs),
    }
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()

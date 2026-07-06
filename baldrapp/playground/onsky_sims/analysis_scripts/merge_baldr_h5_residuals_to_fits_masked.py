#!/usr/bin/env python3
"""
Merge Baldr HDF5 residual phase-screen batches into one FITS datacube.

Reads files named like

    <stem>_part0000.h5
    <stem>_part0001.h5
    ...

and writes the post-Baldr/final residual OPD cube from dataset

    residual_opd_nm

to a FITS file with shape [NFRAME, NY, NX]. The values are normally nm OPD,
following the HDF5 dataset convention.

The script loads and writes in chunks so it does not need the whole cube in RAM.
It also writes useful scalar telemetry as FITS table extensions when available.

Example
-------
python merge_baldr_h5_residuals_to_fits.py \
  --input-dir baldr_json_im_then_ao_100k \
  --stem test_baldr_closed_loop_1000_tt_watchdog \
  --part-start 0 \
  --part-stop 6 \
  --output post_baldr_residual_cube_parts0000_0006.fits \
  --overwrite


python baldrapp/playground/onsky_sims/merge_baldr_h5_residuals_to_fits.py \
  --input-dir ~/Downloads \
  --stem test_baldr_closed_loop_1000_tt_watchdog \
  --part-start 0 \
  --part-stop 7 \
  --output ~/Downloads/post_baldr_residual_cube_parts0000_0006.fits \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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


SCALAR_DATASETS = [
    "rms_pre_naomi_nm",
    "rms_post_naomi_nm",
    "rms_tt_extra_opd_nm",
    "rms_baldr_input_nm",
    "rms_after_baldr_nm",
    "rms_control_component_baldr_input_nm",
    "rms_control_component_after_baldr_nm",
    "rms_uncontrolled_component_baldr_input_nm",
    "rms_uncontrolled_component_after_baldr_nm",
    "loop_reset_flag",
    "loop_reset_count",
    "loop_reset_reason_code",
    "valid_residual_flag",
]

MODAL_DATASETS = [
    "modal_reco_input_cmd",
    "modal_command_state_cmd",
    "modal_fit_coeff_baldr_input_cmd",
    "modal_fit_coeff_after_baldr_cmd",
    "modal_tt_extra_cmd",
]

STATIC_IMAGE_DATASETS = [
    "pupil_mask",
    "pupil_mask_full",
]

STATIC_ARRAY_DATASETS = [
    "control_M2C_command_basis",
    "control_opd_basis_nm_per_cmd",
    "interaction_matrix",
]


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
    p.add_argument("--compress", action="store_true", help="Write tiled compressed FITS image instead of ordinary PrimaryHDU. Slower, smaller.")
    p.add_argument("--compression-type", default="RICE_1", choices=["RICE_1", "GZIP_1", "GZIP_2", "PLIO_1", "HCOMPRESS_1"])
    p.add_argument("--drop-invalid", action="store_true", help="Only include frames with valid_residual_flag==1 when this dataset exists.")
    p.add_argument("--drop-reset", action="store_true", help="Only include frames with loop_reset_flag==0 when this dataset exists.")
    p.add_argument("--mask-outside-pupil", action="store_true", default=True, help="Set pixels outside pupil_mask to NaN before writing FITS cube. Default: enabled.")
    p.add_argument("--no-mask-outside-pupil", dest="mask_outside_pupil", action="store_false", help="Do not mask pixels outside the pupil before writing.")
    p.add_argument("--mask-fill", default="nan", choices=["nan", "zero"], help="Fill value outside pupil when masking. Default: nan.")
    p.add_argument("--write-modal-tables", action="store_true", help="Also write 2D modal telemetry table extensions. Can make FITS larger.")
    p.add_argument("--write-large-static", action="store_true", help="Also write larger static arrays such as interaction_matrix and control_opd_basis.")
    p.add_argument("--summary-json", type=Path, default=None, help="Optional JSON summary output path.")
    return p.parse_args()


def part_path(input_dir: Path, stem: str, idx: int, digits: int) -> Path:
    return input_dir / f"{stem}_part{idx:0{digits}d}.h5"


def safe_header_key(name: str, prefix: str = "H5") -> str:
    """Make an 8-character-ish FITS-safe key. Used only for selected common attrs."""
    clean = "".join(ch for ch in name.upper() if ch.isalnum())
    if not clean:
        clean = "ATTR"
    return (prefix + clean)[:8]


def add_header_card(header, key: str, value, comment: str = "") -> None:
    """Safely add a FITS header card.

    FITS headers are fragile for long string values, especially with HIERARCH
    keywords.  This routine writes a short, standards-safe value in the actual
    card and preserves the full text in HISTORY records.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple, np.ndarray, dict)):
        value = json.dumps(np.asarray(value).tolist() if not isinstance(value, dict) else value)

    # Astropy expects either a normal FITS key, or a single HIERARCH-prefixed
    # key.  Avoid accidentally creating "HIERARCH HIERARCH ...".
    key_to_write = str(key)
    if (len(key_to_write) > 8 or " " in key_to_write) and not key_to_write.upper().startswith("HIERARCH "):
        key_to_write = "HIERARCH " + key_to_write

    # Long strings in HIERARCH cards can fail verification.  Store a concise
    # card value and add the full value as HISTORY chunks.
    full_value = value
    if isinstance(value, str) and len(value) > 60:
        short_value = value[:57] + "..."
        short_comment = (comment + " (truncated; full value in HISTORY)").strip()
        try:
            header[key_to_write] = (short_value, short_comment[:47])
        except Exception:
            header.add_history(f"{key_to_write} = {short_value}")
        txt = f"FULL {key_to_write} = {full_value}"
        for i in range(0, len(txt), 68):
            header.add_history(txt[i:i+68])
        return

    try:
        header[key_to_write] = (value, comment)
    except Exception:
        sval = str(value)
        if len(sval) > 60:
            sval = sval[:57] + "..."
        try:
            header[key_to_write] = (sval, (comment + " (stringified)")[:47])
        except Exception:
            header.add_history(f"{key_to_write} = {sval}")



def validate_fits_header_compatible(header: fits.Header) -> fits.Header:
    """Return a header with only cards Astropy can format/verify.

    Any problematic metadata card is removed and represented as HISTORY.
    This is intentionally conservative: a FITS cube is more useful with safe
    metadata than with one over-long non-standard card that prevents writing.
    """
    safe = fits.Header()
    for card in list(header.cards):
        try:
            # Force Astropy to format the card now. This catches too-long strings.
            _ = card.image
            safe.append(card, end=True)
        except Exception as exc:
            key = str(card.keyword)
            val = str(card.value)
            msg = f"DROPPED BAD HEADER CARD {key}: {val} ({exc})"
            for i in range(0, len(msg), 68):
                safe.add_history(msg[i:i+68])
    try:
        safe.verify("exception")
    except Exception as exc:
        safe.add_history(f"Header verify warning after sanitise: {exc}"[:68])
    return safe

def read_file_info(paths: list[Path], dataset: str):
    infos = []
    first_shape = None
    first_dtype = None
    first_attrs = {}
    total_frames_raw = 0
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
            dtype = ds.dtype
            if first_shape is None:
                first_shape = shape[1:]
                first_dtype = dtype
                first_attrs = dict(h5.attrs)
            elif shape[1:] != first_shape:
                raise ValueError(f"Spatial shape mismatch: {p}:{dataset} has {shape[1:]}, expected {first_shape}")
            infos.append({"path": p, "n": shape[0], "shape": shape, "dtype": str(dtype)})
            total_frames_raw += shape[0]
    return infos, first_shape, first_dtype, first_attrs, total_frames_raw


def frame_keep_mask(h5, n: int, drop_invalid: bool, drop_reset: bool) -> np.ndarray:
    keep = np.ones(n, dtype=bool)
    if drop_invalid and "valid_residual_flag" in h5:
        keep &= np.asarray(h5["valid_residual_flag"][:], dtype=np.uint8).astype(bool)
    if drop_reset and "loop_reset_flag" in h5:
        keep &= np.asarray(h5["loop_reset_flag"][:], dtype=np.uint8) == 0
    return keep


def collect_lengths(paths: list[Path], dataset: str, drop_invalid: bool, drop_reset: bool):
    kept_counts = []
    raw_counts = []
    for p in paths:
        with h5py.File(p, "r") as h5:
            n = h5[dataset].shape[0]
            raw_counts.append(n)
            keep = frame_keep_mask(h5, n, drop_invalid, drop_reset)
            kept_counts.append(int(np.sum(keep)))
    return raw_counts, kept_counts


def make_primary_header(first_attrs: dict, args, total_frames: int, ny: int, nx: int, raw_frames: int, paths: list[Path]) -> fits.Header:
    hdr = fits.Header()
    hdr["ORIGIN"] = "BaldrApp"
    hdr["CONTENT"] = "Post-Baldr residual OPD datacube"
    hdr["BUNIT"] = "nm OPD"
    hdr["DATASET"] = args.dataset
    hdr["NFRAMES"] = int(total_frames)
    hdr["RAWFRAMS"] = int(raw_frames)
    hdr["DROPPINV"] = bool(args.drop_invalid)
    hdr["DROPRSET"] = bool(args.drop_reset)
    hdr["PART0"] = int(args.part_start)
    hdr["PART1"] = int(args.part_stop)
    hdr["NPARTS"] = int(len(paths))
    hdr["PARTDIG"] = int(args.part_digits)
    add_header_card(hdr, "INPSTEM", str(args.stem), "Input file stem")
    add_header_card(hdr, "INDIR", str(args.input_dir), "Input directory")
    hdr["CUBEAXIS"] = "frame,y,x"
    hdr["NAXIS1D"] = int(nx)
    hdr["NAXIS2D"] = int(ny)

    # Common HDF5 run attributes.
    # IMPORTANT: do not write arbitrary long HDF5 attrs as FITS header cards.
    # Long HIERARCH string cards are fragile in Astropy/FITS verification.
    # We only write short, whitelisted aliases as cards and preserve full attrs
    # in HISTORY records below.
    common = [
        "description", "units_residual_opd", "fs_hz", "wvl0_m", "control_basis",
        "control_modes", "first_stage_modes", "gain", "leak", "baldr_lag_frames",
        "include_shotnoise", "crop_padding_pixels", "residual_crop_y0",
        "residual_crop_y1", "residual_crop_x0", "residual_crop_x1", "tt_rms_nm",
        "tt_rms_cmd", "tt_axis", "tt_frequencies_hz", "tt_cmd_rms_total_used",
        "reset_on_fail", "reset_rms_threshold_nm", "reset_hold_frames",
        "max_loop_resets",
    ]
    aliases = {
        "fs_hz": "FS_HZ",
        "wvl0_m": "WVL0_M",
        "gain": "GAIN",
        "leak": "LEAK",
        "control_modes": "NCMODES",
        "first_stage_modes": "NFSMODE",
        "baldr_lag_frames": "BLAGFR",
        "include_shotnoise": "SHOTNOIS",
        "tt_rms_nm": "TTRMSNM",
        "tt_rms_cmd": "TTRMSC",
        "tt_axis": "TTAXIS",
        "reset_on_fail": "RESETON",
        "reset_rms_threshold_nm": "RSTTHR",
        "reset_hold_frames": "RSTHOLD",
    }
    for name in common:
        if name in first_attrs:
            key = aliases.get(name)
            if key is not None:
                add_header_card(hdr, key, first_attrs[name], f"HDF5 attr {name}")

    # Preserve all common HDF5 attrs in HISTORY using short chunks only.
    # This avoids non-standard or over-long HIERARCH string cards.
    hdr.add_history("Original HDF5 run attributes follow as JSON chunks.")
    attr_dict = {}
    for name in common:
        if name in first_attrs:
            v = first_attrs[name]
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            if isinstance(v, np.ndarray):
                v = v.tolist()
            attr_dict[name] = v
    attr_json = json.dumps(attr_dict, sort_keys=True)
    for i in range(0, len(attr_json), 68):
        hdr.add_history(attr_json[i:i+68])

    for i, p in enumerate(paths[:50]):
        add_header_card(hdr, f"HIERARCH INPUT PART {i:04d}", p.name, "Input HDF5 part file")
    if len(paths) > 50:
        add_header_card(hdr, "INPTRUNC", True, "Input part list truncated at 50 files")
    return validate_fits_header_compatible(hdr)


def make_scalar_table(paths: list[Path], dataset: str, args, total_frames: int) -> fits.BinTableHDU | None:
    present = []
    with h5py.File(paths[0], "r") as h5:
        for name in SCALAR_DATASETS:
            if name in h5 and h5[name].ndim == 1:
                present.append(name)
    if not present:
        return None

    cols_data = {name: np.empty(total_frames, dtype=np.float64) for name in present}
    global_frame = np.empty(total_frames, dtype=np.int64)
    part_index = np.empty(total_frames, dtype=np.int32)
    local_frame = np.empty(total_frames, dtype=np.int32)

    pos = 0
    global_offset_raw = 0
    for pidx, p in enumerate(paths):
        with h5py.File(p, "r") as h5:
            n = h5[dataset].shape[0]
            keep = frame_keep_mask(h5, n, args.drop_invalid, args.drop_reset)
            idx = np.where(keep)[0]
            m = len(idx)
            if m == 0:
                global_offset_raw += n
                continue
            global_frame[pos:pos+m] = global_offset_raw + idx
            part_index[pos:pos+m] = args.part_start + pidx
            local_frame[pos:pos+m] = idx
            for name in present:
                cols_data[name][pos:pos+m] = np.asarray(h5[name][idx], dtype=np.float64)
            pos += m
            global_offset_raw += n

    cols = [
        fits.Column(name="global_frame", format="K", array=global_frame),
        fits.Column(name="part_index", format="J", array=part_index),
        fits.Column(name="local_frame", format="J", array=local_frame),
    ]
    for name in present:
        arr = cols_data[name]
        fmt = "D"
        if name.startswith("loop_reset") or name == "valid_residual_flag":
            arr = arr.astype(np.int32)
            fmt = "J"
        cols.append(fits.Column(name=name[:68], format=fmt, array=arr))
    hdu = fits.BinTableHDU.from_columns(cols, name="TELEMETRY")
    hdu.header["COMMENT"] = "Scalar telemetry aligned with primary cube frame axis after filtering."
    return hdu


def make_modal_tables(paths: list[Path], dataset: str, args, total_frames: int) -> list[fits.BinTableHDU]:
    hdus = []
    with h5py.File(paths[0], "r") as h5:
        present = [name for name in MODAL_DATASETS if name in h5 and h5[name].ndim == 2]
    for name in present:
        # Determine mode count.
        with h5py.File(paths[0], "r") as h5:
            n_modes = h5[name].shape[1]
        data = np.empty((total_frames, n_modes), dtype=np.float32)
        pos = 0
        for p in paths:
            with h5py.File(p, "r") as h5:
                n = h5[dataset].shape[0]
                keep = frame_keep_mask(h5, n, args.drop_invalid, args.drop_reset)
                idx = np.where(keep)[0]
                m = len(idx)
                if m:
                    data[pos:pos+m, :] = np.asarray(h5[name][idx, :], dtype=np.float32)
                    pos += m
        col = fits.Column(name=name[:68], format=f"{n_modes}E", array=data)
        hdu = fits.BinTableHDU.from_columns([col], name=name[:68].upper())
        hdu.header["NMODES"] = int(n_modes)
        hdu.header["COMMENT"] = "Rows align with TELEMETRY/global_frame and primary cube frame axis."
        hdus.append(hdu)
    return hdus


def static_hdus(paths: list[Path], write_large_static: bool) -> list[fits.ImageHDU]:
    hdus = []
    with h5py.File(paths[0], "r") as h5:
        for name in STATIC_IMAGE_DATASETS:
            if name in h5:
                hdus.append(fits.ImageHDU(np.asarray(h5[name]), name=name.upper()[:68]))
        if write_large_static:
            for name in STATIC_ARRAY_DATASETS:
                if name in h5:
                    arr = np.asarray(h5[name])
                    # FITS image HDU supports nD numerical arrays.
                    hdus.append(fits.ImageHDU(arr, name=name.upper()[:68]))
    return hdus



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

def write_cube_uncompressed(paths, args, hdr, total_frames, spatial_shape, dtype):
    ny, nx = spatial_shape
    out = args.output
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; use --overwrite")

    # Create primary HDU with allocated data. Astropy will memory-map/update in place.
    data = np.empty((total_frames, ny, nx), dtype=np.float32)
    hdr = validate_fits_header_compatible(hdr)
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdul = fits.HDUList([hdu])
    hdul.writeto(out, overwrite=args.overwrite, output_verify="fix")
    del data, hdu, hdul

    with fits.open(out, mode="update", memmap=True) as hdul:
        pos = 0
        for p in paths:
            with h5py.File(p, "r") as h5:
                ds = h5[args.dataset]
                n = ds.shape[0]
                keep = frame_keep_mask(h5, n, args.drop_invalid, args.drop_reset)
                idx = np.where(keep)[0]
                if len(idx) == 0:
                    continue
                # Copy contiguous chunks when possible.
                for a in range(0, len(idx), args.chunk_frames):
                    sub_idx = idx[a:a+args.chunk_frames]
                    block = np.asarray(ds[sub_idx, :, :], dtype=np.float32)
                    pupil = get_pupil_mask_for_file(h5, spatial_shape)
                    block = apply_pupil_mask_block(block, pupil, args)
                    m = block.shape[0]
                    hdul[0].data[pos:pos+m, :, :] = block
                    pos += m
                    print(f"  wrote frames {pos-m}:{pos} from {p.name}", flush=True)
        hdul.flush()


def write_cube_compressed(paths, args, hdr, total_frames, spatial_shape):
    # Compressed image HDUs are usually written all-at-once by astropy; avoid this for huge cubes.
    # Here we still build in RAM, so warn in the doc/print. Use uncompressed for very large cubes.
    ny, nx = spatial_shape
    bytes_est = total_frames * ny * nx * 4
    print(
        "WARNING: --compress currently builds the cube in RAM before writing. "
        f"Estimated RAM for cube alone: {bytes_est/1024**3:.2f} GiB."
    )
    cube = np.empty((total_frames, ny, nx), dtype=np.float32)
    pos = 0
    for p in paths:
        with h5py.File(p, "r") as h5:
            ds = h5[args.dataset]
            n = ds.shape[0]
            keep = frame_keep_mask(h5, n, args.drop_invalid, args.drop_reset)
            idx = np.where(keep)[0]
            for a in range(0, len(idx), args.chunk_frames):
                sub_idx = idx[a:a+args.chunk_frames]
                block = np.asarray(ds[sub_idx, :, :], dtype=np.float32)
                pupil = get_pupil_mask_for_file(h5, spatial_shape)
                block = apply_pupil_mask_block(block, pupil, args)
                m = block.shape[0]
                cube[pos:pos+m] = block
                pos += m
    hdr = validate_fits_header_compatible(hdr)
    hdu = fits.CompImageHDU(data=cube, header=hdr, name="RESIDUAL_OPD", compression_type=args.compression_type)
    # Empty primary + compressed extension is standard for compressed images.
    hdr = validate_fits_header_compatible(hdr)
    primary = fits.PrimaryHDU(header=hdr)
    fits.HDUList([primary, hdu]).writeto(args.output, overwrite=args.overwrite, output_verify="fix")


def append_extensions(output: Path, hdus_to_append: list):
    if not hdus_to_append:
        return
    with fits.open(output, mode="append", memmap=False) as hdul:
        for hdu in hdus_to_append:
            hdul.append(hdu)
        hdul.flush()


def main():
    args = parse_args()
    if args.part_stop < args.part_start:
        raise ValueError("--part-stop must be >= --part-start")
    paths = [part_path(args.input_dir, args.stem, i, args.part_digits) for i in range(args.part_start, args.part_stop + 1)]

    infos, spatial_shape, dtype, first_attrs, total_raw = read_file_info(paths, args.dataset)
    raw_counts, kept_counts = collect_lengths(paths, args.dataset, args.drop_invalid, args.drop_reset)
    total_kept = int(sum(kept_counts))
    if total_kept <= 0:
        raise RuntimeError("No frames selected after filtering.")

    ny, nx = spatial_shape
    hdr = make_primary_header(first_attrs, args, total_kept, ny, nx, total_raw, paths)

    print("Input parts:")
    for p, raw, kept in zip(paths, raw_counts, kept_counts):
        print(f"  {p.name}: raw={raw}, kept={kept}")
    print(f"Writing FITS cube: {args.output}")
    print(f"  shape = ({total_kept}, {ny}, {nx})")
    print(f"  dataset = {args.dataset}")
    print(f"  estimated primary cube size = {total_kept * ny * nx * 4 / 1024**3:.3f} GiB")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.compress:
        write_cube_compressed(paths, args, hdr, total_kept, spatial_shape)
    else:
        write_cube_uncompressed(paths, args, hdr, total_kept, spatial_shape, dtype)

    # Append telemetry and metadata extensions.
    ext_hdus = []
    tel = make_scalar_table(paths, args.dataset, args, total_kept)
    if tel is not None:
        ext_hdus.append(tel)
    if args.write_modal_tables:
        ext_hdus.extend(make_modal_tables(paths, args.dataset, args, total_kept))
    ext_hdus.extend(static_hdus(paths, args.write_large_static))
    append_extensions(args.output, ext_hdus)

    summary = {
        "output": str(args.output),
        "input_dir": str(args.input_dir),
        "stem": args.stem,
        "part_start": args.part_start,
        "part_stop": args.part_stop,
        "dataset": args.dataset,
        "raw_frames": int(total_raw),
        "kept_frames": int(total_kept),
        "spatial_shape_yx": [int(ny), int(nx)],
        "drop_invalid": bool(args.drop_invalid),
        "mask_outside_pupil": bool(args.mask_outside_pupil),
        "mask_fill": str(args.mask_fill),
        "drop_reset": bool(args.drop_reset),
        "parts": [{"path": str(p), "raw_frames": int(r), "kept_frames": int(k)} for p, r, k in zip(paths, raw_counts, kept_counts)],
        "hdf5_attrs_first_file": {str(k): (v.item() if isinstance(v, np.generic) else str(v) if not isinstance(v, (int, float, bool, str)) else v) for k, v in first_attrs.items()},
    }
    summary_path = args.summary_json
    if summary_path is None:
        summary_path = args.output.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()

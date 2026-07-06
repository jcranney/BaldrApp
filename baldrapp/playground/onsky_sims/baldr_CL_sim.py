#!/usr/bin/env python3
"""
Baldr JSON calibration verification followed by optional closed-loop AO simulation.

This follows the older working BaldrApp calibration path:

  1. initialise ZWFS from a JSON config
  2. get N0/I0 reference intensities
  3. classify pupil regions
  4. build a zonal DM interaction matrix
  5. register the DM in detector/pixel space
  6. build a control interaction matrix in a DM command basis, e.g. Zernike_pinned_edges
  7. run real injection/reconstruction tests through the forward model
  8. ask for confirmation before running a rolling-atmosphere closed-loop AO simulation

The control IM is DM-command based, not direct OPD-modal based.

from BaldrApp in virtual environment (if you have it setup) run:
python baldrapp/playground/onsky_sims/baldr_CL_sim.py \
  --config baldrapp/playground/onsky_sims/baldr_naomi_fast_mono_config_v2.json \
  --outdir baldr_json_im_then_ao_test \
  --output test_baldr_closed_loop_1000_tt_watchdog.h5 \
  --control-modes 50 \
  --gain 0.2 \
  --leak 0.98 \
  --control-sign 1 \
  --n-frames 100000 \
  --target-pre-naomi-rms-nm 600 \
  --tt-rms-nm 80 \
  --tt-frequencies-hz 15 \
  --tt-axis tilt \
  --reset-on-fail \
  --reset-rms-threshold-nm 400 \
  --first-stage-lag-frames 2 \
  --baldr-lag-frames 1 \
  --include-shotnoise \
  --frames-per-output-file 10000


python baldrapp/playground/onsky_sims/baldr_CL_sim.py \
  --config baldrapp/playground/onsky_sims/baldr_naomi_fast_mono_config_v2.json \
  --outdir baldr_json_im_then_ao_test \
  --output test_baldr_closed_loop_1000_tt_watchdog.h5 \
  --control-modes 50 \ 
  --first-stage-gain 0.4 \
  --first-stage-leak 0.99 \
  --gain 0.2 \
  --leak 0.98 \ 
  --control-sign 1 \
  --n-frames 1000 \
  --target-pre-naomi-rms-nm 600 \
  --tt-rms-nm 80 \
  --tt-frequencies-hz 15 \
  --tt-axis tilt \
  --reset-on-fail \
  --reset-rms-threshold-nm 400 \
  --first-stage-lag-frames 2 \
  --baldr-lag-frames 1 \
  --include-shotnoise \
  --frames-per-output-file 10000

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _prepend_local_repo_to_syspath() -> None:
    here = Path(__file__).resolve()
    candidates = []
    candidates.extend(list(here.parents))
    candidates.append(Path.cwd().resolve())
    candidates.extend(list(Path.cwd().resolve().parents))
    for c in candidates:
        if (c / "baldrapp" / "common" / "baldr_core.py").exists():
            sys.path.insert(0, str(c))
            return


_prepend_local_repo_to_syspath()

from baldrapp.common import baldr_core as bldr  # noqa: E402
from baldrapp.common import DM_basis            # noqa: E402
from baldrapp.common import DM_registration     # noqa: E402
from baldrapp.common import utilities as util    # noqa: E402
from baldrapp.common import phasescreens as ps   # noqa: E402

try:
    import pyzelda.ztools as ztools              # noqa: E402
except Exception as exc:  # pragma: no cover
    ztools = None
    _ztools_import_error = exc

try:
    import h5py                                  # noqa: E402
except Exception:
    h5py = None


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_image(path: Path, image, title="", cbar_label="", cmap="viridis", log=False, vlim=None):
    arr = np.asarray(image, dtype=float)
    if log:
        floor = np.nanmax(arr) * 1e-8 if np.nanmax(arr) > 0 else 1e-30
        arr = np.log10(np.maximum(arr, floor))
    fig, ax = plt.subplots(figsize=(7, 6))
    if vlim is None:
        im = ax.imshow(arr, origin="lower", cmap=cmap)
    else:
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    ax.set_title(title)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    cb = fig.colorbar(im, ax=ax)
    if cbar_label:
        cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_im_rows(outdir: Path, IM: np.ndarray, image_shape, prefix: str, n_plot: int):
    n = min(n_plot, IM.shape[0])
    for i in range(n):
        save_image(
            outdir / f"{prefix}_mode_{i+1:03d}.png",
            IM[i].reshape(image_shape),
            title=f"{prefix} mode {i+1}: dS/dcmd",
            cbar_label="signal / command unit",
        )


def make_detector_from_config(zwfs, cfg: dict, args):
    if args.no_detector:
        return None
    det_cfg = cfg.get("detector", {})
    if args.detector_binning is not None:
        return bldr.detector(
            binning=int(args.detector_binning),
            dit=float(args.detector_dit),
            ron=float(args.detector_ron),
            qe=float(args.detector_qe),
        )
    if det_cfg and bool(det_cfg.get("enabled", True)):
        return bldr.detector(
            binning=int(det_cfg.get("binning", 1)),
            dit=float(det_cfg.get("dit", 1.0)),
            ron=float(det_cfg.get("ron", 0.0)),
            qe=float(det_cfg.get("qe", 1.0)),
        )
    return None


def make_calibration_amp(zwfs, cfg: dict) -> np.ndarray:
    source = cfg.get("source", {})
    prof_internal = cfg.get("source_profiles", {}).get("internal", {})
    flux = None
    for d in (source, prof_internal):
        if "photons_per_second_per_pixel_per_nm" in d:
            flux = float(d["photons_per_second_per_pixel_per_nm"])
            break
    if flux is None:
        flux = 1000.0
    return np.sqrt(flux) * zwfs.grid.pupil_mask.astype(float)


def normalise_signal(I: np.ndarray, zwfs, method: str) -> np.ndarray:
    """Return the pixel signal compatible with bldr.build_IM normalisation."""
    I = np.asarray(I, dtype=float)
    I0 = np.asarray(zwfs.reco.I0, dtype=float)
    method_l = method.strip().lower()
    eps = 1e-30
    if method_l == "subframe mean":
        return (I / (np.mean(I) + eps) - I0 / (np.mean(I0) + eps)).reshape(-1)
    if method_l == "clear pupil mean":
        filt = np.asarray(zwfs.reco.interior_pup_filt, dtype=bool)
        denom = np.mean(np.asarray(zwfs.reco.N0, dtype=float)[filt]) + eps
        return (I / denom - I0 / denom).reshape(-1)
    raise ValueError(f"Unknown normalization method: {method}")


def build_reconstructor_from_im(IM: np.ndarray, svd_rcond: float, tikhonov: float, n_keep: int | None):
    U, S, Vt = np.linalg.svd(IM, full_matrices=False)
    if n_keep is None:
        n_keep = int(np.sum(S > svd_rcond * np.max(S)))
    n_keep = max(1, min(int(n_keep), len(S)))
    W = np.zeros_like(S)
    if tikhonov > 0:
        W[:n_keep] = S[:n_keep] / (S[:n_keep] ** 2 + tikhonov ** 2)
    else:
        W[:n_keep] = 1.0 / S[:n_keep]
    I2M = U @ (W[:, None] * Vt)
    response = I2M @ IM.T
    return I2M, response, S, n_keep


def save_response_diagnostics(outdir: Path, IM: np.ndarray, args):
    I2M, response, S, n_keep = build_reconstructor_from_im(
        IM, args.svd_rcond, args.tikhonov, args.svd_n_keep
    )
    save_image(
        outdir / "modal_response_I2M_IMT.png",
        response,
        title="Modal response: I2M @ IM.T",
        cbar_label="recovered / injected",
        cmap="RdBu_r",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(S) + 1), S, marker=".")
    ax.axvline(n_keep, color="k", linestyle="--", linewidth=1, label=f"kept={n_keep}")
    ax.set_title("Control IM singular values")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "modal_IM_singular_values.png", dpi=160)
    plt.close(fig)
    diag = np.diag(response)
    off = response - np.diag(diag)
    summary = {
        "n_modes": int(IM.shape[0]),
        "n_pixels": int(IM.shape[1]),
        "n_svd_kept": int(n_keep),
        "singular_values_max": float(np.max(S)),
        "singular_values_min": float(np.min(S)),
        "response_diag_median": float(np.median(diag)),
        "response_diag_min": float(np.min(diag)),
        "response_diag_max": float(np.max(diag)),
        "response_offdiag_abs_median": float(np.median(np.abs(off))),
        "response_offdiag_abs_max": float(np.max(np.abs(off))),
    }
    with open(outdir / "reconstructor_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return I2M, summary


def save_registration_overlay(outdir: Path, image, zwfs):
    if not hasattr(zwfs, "dm2pix_registration"):
        return
    coords = np.asarray(zwfs.dm2pix_registration.actuator_coord_list_pixel_space)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(image, origin="lower", cmap="viridis")
    ax.scatter(coords[:, 0], coords[:, 1], s=16, facecolors="none", edgecolors="r", linewidths=0.8)
    ax.set_title("DM actuator registration over reference image")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    fig.tight_layout()
    fig.savefig(outdir / "dm_registration_overlay.png", dpi=180)
    plt.close(fig)


def get_dm_opd(zwfs, command_vector):
    return bldr.get_dm_displacement(
        command_vector=command_vector,
        gain=zwfs.dm.opd_per_cmd,
        sigma=zwfs.grid.dm_coord.act_sigma_wavesp,
        X=zwfs.grid.wave_coord.X,
        Y=zwfs.grid.wave_coord.Y,
        x0=zwfs.grid.dm_coord.act_x0_list_wavesp,
        y0=zwfs.grid.dm_coord.act_y0_list_wavesp,
    )


def run_real_injection_tests(outdir, zwfs, amp, opd0, opd_internal, detector, use_pyzelda, args, I2M):
    """Inject real DM modal commands, propagate, reconstruct from pixels."""
    M2C = np.asarray(zwfs.reco.M2C_0, dtype=float)  # modes x command-vector length
    n_modes = M2C.shape[0]
    tests = []

    print("\nRunning real single-mode injection/reconstruction tests through get_frame...")
    for j in range(min(args.n_injection_tests, n_modes)):
        for a in args.injection_amps:
            zwfs.dm.current_cmd = zwfs.dm.dm_flat + float(a) * M2C[j]
            I = bldr.get_frame(opd0, amp, opd_internal, zwfs, detector=detector, include_shotnoise=False, use_pyZelda=use_pyzelda)
            s = normalise_signal(I, zwfs, args.normalization_method)
            c = I2M @ s
            others = np.delete(c, j) if n_modes > 1 else np.array([0.0])
            tests.append({
                "kind": "single_mode",
                "mode_1based": j + 1,
                "injected_cmd": float(a),
                "recovered_same_mode": float(c[j]),
                "recovered_over_injected": float(c[j] / a) if a != 0 else np.nan,
                "max_abs_other_modes": float(np.max(np.abs(others))),
                "all_recovered_coefficients_json": json.dumps(c.tolist()),
            })

    rng = np.random.default_rng(args.random_seed)
    random_rows = []
    print("Running random multimode injection/reconstruction tests...")
    for k in range(args.n_random_injection_tests):
        coeff = rng.normal(size=n_modes)
        coeff *= args.random_injection_rms_cmd / (np.std(coeff) + 1e-30)
        zwfs.dm.current_cmd = zwfs.dm.dm_flat + coeff @ M2C
        I = bldr.get_frame(opd0, amp, opd_internal, zwfs, detector=detector, include_shotnoise=False, use_pyZelda=use_pyzelda)
        s = normalise_signal(I, zwfs, args.normalization_method)
        rec = I2M @ s
        err = rec - coeff
        random_rows.append({
            "test_index": k,
            "injected_rms_cmd": float(np.std(coeff)),
            "recovered_rms_cmd": float(np.std(rec)),
            "error_rms_cmd": float(np.std(err)),
            "corrcoef": float(np.corrcoef(coeff, rec)[0, 1]) if np.std(rec) > 0 else np.nan,
            "injected_json": json.dumps(coeff.tolist()),
            "recovered_json": json.dumps(rec.tolist()),
        })

    zwfs.dm.current_cmd = zwfs.dm.dm_flat.copy()

    with open(outdir / "real_single_mode_injection_tests.csv", "w", newline="") as f:
        if tests:
            writer = csv.DictWriter(f, fieldnames=list(tests[0].keys()))
            writer.writeheader()
            writer.writerows(tests)
    with open(outdir / "real_random_multimode_injection_tests.csv", "w", newline="") as f:
        if random_rows:
            writer = csv.DictWriter(f, fieldnames=list(random_rows[0].keys()))
            writer.writeheader()
            writer.writerows(random_rows)

    # Plot single-mode recovered/injected for first amplitude in list.
    if tests:
        amp0 = float(args.injection_amps[0])
        xs, ys = [], []
        for r in tests:
            if abs(r["injected_cmd"] - amp0) < 1e-15:
                xs.append(r["mode_1based"])
                ys.append(r["recovered_over_injected"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, marker="o")
        ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Mode index")
        ax.set_ylabel("Recovered / injected")
        ax.set_title(f"Real single-mode reconstruction, injected={amp0:g} cmd")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "real_single_mode_recovered_over_injected.png", dpi=160)
        plt.close(fig)

    if random_rows:
        inj = np.array([r["injected_rms_cmd"] for r in random_rows])
        rec = np.array([r["recovered_rms_cmd"] for r in random_rows])
        err = np.array([r["error_rms_cmd"] for r in random_rows])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(len(inj)), inj, marker="o", label="injected RMS")
        ax.plot(np.arange(len(rec)), rec, marker="o", label="recovered RMS")
        ax.plot(np.arange(len(err)), err, marker="o", label="error RMS")
        ax.set_xlabel("Random test")
        ax.set_ylabel("Command RMS")
        ax.set_title("Real random multimode reconstruction")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "real_random_multimode_reconstruction_rms.png", dpi=160)
        plt.close(fig)

    return tests, random_rows



def save_reconstruction_bar(path: Path, injected: np.ndarray, recovered: np.ndarray, title: str, n_show: int = 40):
    n = min(n_show, len(recovered))
    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.18, injected[:n], width=0.36, label="injected")
    ax.bar(x + 0.18, recovered[:n], width=0.36, label="recovered")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Command coefficient")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def command_to_opd_delta(zwfs, M2C: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """Convert modal command coefficients into DM OPD relative to the DM flat."""
    flat = np.asarray(zwfs.dm.dm_flat, dtype=float)
    return get_dm_opd(zwfs, flat + coeff @ M2C) - get_dm_opd(zwfs, flat)


def run_visual_injection_and_sign_tests(outdir, zwfs, amp, opd_internal, detector, use_pyzelda, args, I2M):
    """Save input OPD, ZWFS intensity, signal, and reconstructed coefficients for real injections.

    These tests inject an upstream OPD with exactly the same shape as the DM command
    basis mode. This is closer to the AO use case than setting the DM itself, because
    the DM should then apply the opposite OPD to cancel it.
    """
    M2C = np.asarray(zwfs.reco.M2C_0, dtype=float)
    n_modes = M2C.shape[0]
    pupil = np.asarray(zwfs.grid.pupil_mask, dtype=bool)
    print("Building OPD projection onto the Baldr/DM control basis for diagnostics...")
    B_control_nm, P_opd_to_cmd = build_control_opd_projection(
        zwfs, M2C, pupil, rcond=args.opd_projection_rcond
    )
    flat_opd = get_dm_opd(zwfs, zwfs.dm.dm_flat)
    zwfs.dm.current_cmd = zwfs.dm.dm_flat.copy()

    print("\nRunning visual upstream-aberration injection/reconstruction diagnostics...")
    rows = []
    n_vis = min(int(args.n_visual_injection_tests), n_modes)
    a = float(args.visual_injection_amp_cmd)

    for j in range(n_vis):
        coeff = np.zeros(n_modes)
        coeff[j] = a
        opd_ab = command_to_opd_delta(zwfs, M2C, coeff)

        zwfs.dm.current_cmd = zwfs.dm.dm_flat.copy()
        I = bldr.get_frame(opd_ab, amp, opd_internal, zwfs, detector=detector, include_shotnoise=False, use_pyZelda=use_pyzelda)
        sig = normalise_signal(I, zwfs, args.normalization_method)
        rec = I2M @ sig

        # Candidate correction: if rec is the DM-equivalent aberration, the physical correction is -rec.
        opd_corr_minus = command_to_opd_delta(zwfs, M2C, -rec)
        opd_corr_plus = command_to_opd_delta(zwfs, M2C, +rec)
        resid_minus = opd_ab + opd_corr_minus
        resid_plus = opd_ab + opd_corr_plus

        rms_in = pupil_rms_nm(opd_ab, pupil)
        rms_minus = pupil_rms_nm(resid_minus, pupil)
        rms_plus = pupil_rms_nm(resid_plus, pupil)
        rows.append({
            "mode_1based": j + 1,
            "injected_cmd": a,
            "recovered_same_mode_cmd": float(rec[j]),
            "recovered_over_injected": float(rec[j] / a) if a else np.nan,
            "input_opd_rms_nm": float(rms_in),
            "residual_rms_if_apply_minus_rec_nm": float(rms_minus),
            "residual_rms_if_apply_plus_rec_nm": float(rms_plus),
            "recommended_physical_dm_sign": "minus_rec" if rms_minus < rms_plus else "plus_rec",
            "all_recovered_coefficients_json": json.dumps(rec.tolist()),
        })

        stem = f"visual_mode_{j+1:03d}_{a:g}cmd"
        save_image(outdir / f"{stem}_input_opd_nm.png", 1e9 * opd_ab, f"Input aberration mode {j+1}: OPD", "nm OPD", cmap="RdBu_r")
        save_image(outdir / f"{stem}_zwfs_intensity.png", I, f"ZWFS intensity for injected mode {j+1}", "Intensity")
        save_image(outdir / f"{stem}_zwfs_signal.png", sig.reshape(I.shape), f"ZWFS signal for injected mode {j+1}", "Signal", cmap="RdBu_r")
        save_reconstruction_bar(outdir / f"{stem}_reconstruction_coefficients.png", coeff, rec, f"Reconstruction for injected mode {j+1}", n_show=n_modes)
        save_image(outdir / f"{stem}_residual_if_minus_rec_nm.png", 1e9 * resid_minus, f"Residual if DM applies -reconstructed command", "nm OPD", cmap="RdBu_r")
        save_image(outdir / f"{stem}_residual_if_plus_rec_nm.png", 1e9 * resid_plus, f"Residual if DM applies +reconstructed command", "nm OPD", cmap="RdBu_r")

    with open(outdir / "visual_upstream_injection_sign_tests.csv", "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # A random multimode visual test.
    rng = np.random.default_rng(args.random_seed + 100)
    coeff = rng.normal(size=n_modes)
    coeff *= float(args.random_injection_rms_cmd) / (np.std(coeff) + 1e-30)
    opd_ab = command_to_opd_delta(zwfs, M2C, coeff)
    zwfs.dm.current_cmd = zwfs.dm.dm_flat.copy()
    I = bldr.get_frame(opd_ab, amp, opd_internal, zwfs, detector=detector, include_shotnoise=False, use_pyZelda=use_pyzelda)
    sig = normalise_signal(I, zwfs, args.normalization_method)
    rec = I2M @ sig
    save_image(outdir / "visual_random_multimode_input_opd_nm.png", 1e9 * opd_ab, "Random multimode input aberration: OPD", "nm OPD", cmap="RdBu_r")
    save_image(outdir / "visual_random_multimode_zwfs_intensity.png", I, "ZWFS intensity for random multimode injection", "Intensity")
    save_image(outdir / "visual_random_multimode_zwfs_signal.png", sig.reshape(I.shape), "ZWFS signal for random multimode injection", "Signal", cmap="RdBu_r")
    save_reconstruction_bar(outdir / "visual_random_multimode_reconstruction_coefficients.png", coeff, rec, "Random multimode reconstruction", n_show=n_modes)

    if rows:
        med_minus = float(np.median([r["residual_rms_if_apply_minus_rec_nm"] for r in rows]))
        med_plus = float(np.median([r["residual_rms_if_apply_plus_rec_nm"] for r in rows]))
        print("\nDM correction sign sanity check from upstream OPD injections:")
        print(f"  median residual if applying -reconstructed command: {med_minus:.3f} nm")
        print(f"  median residual if applying +reconstructed command: {med_plus:.3f} nm")
        if med_minus < med_plus:
            print("  Recommended physical DM correction: current_cmd = flat - reconstructed @ M2C")
            print("  With this script's convention, use --control-sign 1")
        else:
            print("  Recommended physical DM correction: current_cmd = flat + reconstructed @ M2C")
            print("  With this script's convention, use --control-sign -1")

    zwfs.dm.current_cmd = zwfs.dm.dm_flat.copy()
    return rows


def make_first_stage_basis(zwfs, nterms: int):
    if ztools is None:
        raise RuntimeError(f"pyzelda.ztools is required for the first-stage Zernike basis: {_ztools_import_error}")
    basis_cropped = ztools.zernike.zernike_basis(nterms=max(nterms, 5), npix=zwfs.grid.N)
    template = np.zeros(zwfs.grid.pupil_mask.shape)
    return np.array([util.insert_concentric(np.nan_to_num(b, 0), template) for b in basis_cropped])


def pupil_rms_nm(opd_m: np.ndarray, pupil: np.ndarray) -> float:
    m = np.asarray(pupil).astype(bool)
    return float(1e9 * np.std(np.asarray(opd_m)[m]))


def remove_piston_over_pupil(opd_m: np.ndarray, pupil: np.ndarray) -> np.ndarray:
    """Return OPD with the active-pupil mean piston removed.

    The simulation RMS diagnostics already use np.std, so they are piston-insensitive.
    This helper is for the saved phase-screen cube, so movies and downstream users
    do not see arbitrary pupil-mean offsets.
    """
    out = np.asarray(opd_m, dtype=float).copy()
    m = np.asarray(pupil).astype(bool)
    if np.any(m):
        out[m] -= np.mean(out[m])
    return out


def pupil_crop_slices(pupil: np.ndarray, padding: int = 2):
    """Return y/x slices enclosing the active pupil, with optional padding."""
    p = np.asarray(pupil).astype(bool)
    yy, xx = np.where(p)
    if len(yy) == 0:
        raise ValueError("Cannot crop pupil: pupil mask contains no active pixels.")
    pad = max(0, int(padding))
    y0 = max(0, int(yy.min()) - pad)
    y1 = min(p.shape[0], int(yy.max()) + pad + 1)
    x0 = max(0, int(xx.min()) - pad)
    x1 = min(p.shape[1], int(xx.max()) + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def calibrate_atmospheric_scale(zwfs, basis, cfg, args):
    if args.target_pre_naomi_rms_nm <= 0:
        return args.phase_scaling_factor
    wvl0 = float(zwfs.optics.wvl0)
    dx = float(zwfs.grid.D) / float(zwfs.grid.N)
    r0 = float(args.r0_500_m) * (wvl0 / float(args.atm_reference_wavelength_m)) ** (6.0 / 5.0)
    scrn = ps.PhaseScreenKolmogorov(
        nx_size=int(zwfs.grid.dim), pixel_scale=dx, r0=r0, L0=float(args.L0_m), random_seed=int(args.random_seed)
    )
    vals = []
    for _ in range(max(1, args.atm_calibration_frames)):
        scrn.add_row()
        pre_phase = args.phase_scaling_factor * scrn.scrn
        pre_opd = (wvl0 / (2 * np.pi)) * pre_phase * basis[0]
        vals.append(pupil_rms_nm(pre_opd, zwfs.grid.pupil_mask))
    measured = float(np.median(vals)) if vals else 0.0
    scale = float(args.phase_scaling_factor)
    if measured > 0:
        scale *= float(args.target_pre_naomi_rms_nm) / measured
    print(
        f"Atmosphere calibration: median pre-Naomi RMS at scale={args.phase_scaling_factor:g} "
        f"is {measured:.1f} nm; using phase_scaling_factor={scale:.4g} "
        f"for target {args.target_pre_naomi_rms_nm:.1f} nm."
    )
    return scale


def parse_frequency_list(freqs: str):
    if freqs is None or str(freqs).strip() == "":
        return []
    return [float(x.strip()) for x in str(freqs).split(",") if x.strip()]


def tt_modal_extra_command(k: int, fs_hz: float, args, n_modes: int, tt_cmd_rms_total: float = None) -> np.ndarray:
    """Optional TT vibration injected as an upstream disturbance in the control basis.

    --tt-axis controls where the injected vibration goes:
      both : split total RMS over modes 0 and 1 and all frequency lines
      tip  : inject only into mode 0, split total RMS over frequency lines
      tilt : inject only into mode 1, split total RMS over frequency lines

    Preferred input is --tt-rms-nm, interpreted as total OPD RMS over the selected
    axis/axes and all requested sinusoidal lines. It is converted to command units
    using the measured OPD RMS per unit command for the selected control mode(s).

    Legacy input --tt-rms-cmd remains supported and is interpreted directly as total
    command RMS over the selected axis/axes and all requested lines.
    """
    extra = np.zeros(n_modes)
    if n_modes < 2:
        return extra

    freqs = parse_frequency_list(args.tt_frequencies_hz)
    if not freqs:
        return extra

    if tt_cmd_rms_total is None:
        tt_cmd_rms_total = float(args.tt_rms_cmd)
    if tt_cmd_rms_total <= 0:
        return extra

    axis = str(getattr(args, "tt_axis", "both")).strip().lower()
    if axis == "both":
        axes = [0, 1]
    elif axis == "tip":
        axes = [0]
    elif axis == "tilt":
        axes = [1]
    else:
        raise ValueError(f"Unknown --tt-axis: {axis!r}. Use 'both', 'tip', or 'tilt'.")

    t = k / fs_hz
    n_components = len(axes) * len(freqs)
    comp_rms_cmd = tt_cmd_rms_total / np.sqrt(float(n_components))
    amp_cmd = np.sqrt(2.0) * comp_rms_cmd

    # Fixed phases for deterministic reproducibility.
    phases = np.linspace(0.13, 4.27, n_components, endpoint=True)
    q = 0
    for f in freqs:
        for ax in axes:
            extra[ax] += amp_cmd * np.sin(2 * np.pi * f * t + phases[q])
            q += 1
    return extra


def compute_tt_command_rms_from_nm(args, B_control_nm: np.ndarray, pupil: np.ndarray) -> float:
    """Convert requested total TT OPD RMS in nm into total RMS in command units.

    The first two control modes are assumed to be tip/tilt-like. The conversion uses
    the selected axis/axes from --tt-axis:
      both : average calibration of modes 0 and 1
      tip  : calibration of mode 0 only
      tilt : calibration of mode 1 only
    """
    if float(args.tt_rms_nm) <= 0:
        return float(args.tt_rms_cmd)
    if B_control_nm.shape[0] < 2:
        return 0.0

    axis = str(getattr(args, "tt_axis", "both")).strip().lower()
    rms0 = float(np.std(B_control_nm[0][pupil]))
    rms1 = float(np.std(B_control_nm[1][pupil]))

    if axis == "both":
        nm_per_cmd = 0.5 * (abs(rms0) + abs(rms1))
        axis_desc = "modes 0/1 (tip+tilt)"
    elif axis == "tip":
        nm_per_cmd = abs(rms0)
        axis_desc = "mode 0 only (tip-like)"
    elif axis == "tilt":
        nm_per_cmd = abs(rms1)
        axis_desc = "mode 1 only (tilt-like)"
    else:
        raise ValueError(f"Unknown --tt-axis: {axis!r}. Use 'both', 'tip', or 'tilt'.")

    if nm_per_cmd <= 0 or not np.isfinite(nm_per_cmd):
        raise RuntimeError("Could not convert TT nm RMS to command units: selected TT control mode(s) have zero OPD response.")
    tt_cmd = float(args.tt_rms_nm) / nm_per_cmd
    print(
        f"TT vibration: requested total RMS={args.tt_rms_nm:.2f} nm over {axis_desc} and lines "
        f"{parse_frequency_list(args.tt_frequencies_hz)} Hz; "
        f"TT calibration={nm_per_cmd:.3f} nm RMS per cmd; total RMS={tt_cmd:.5g} cmd."
    )
    return tt_cmd


def build_control_opd_projection(zwfs, M2C: np.ndarray, pupil: np.ndarray, rcond: float = 1e-6):
    """Build a least-squares projector from OPD screens to the DM control basis.

    Returns
    -------
    B_nm : ndarray, shape (n_modes, ny, nx)
        OPD response of each control mode, in nm OPD per unit command.
    P : ndarray, shape (n_modes, n_pupil_pixels)
        Matrix mapping pupil OPD samples in nm to least-squares command coefficients.
    """
    n_modes = M2C.shape[0]
    B = []
    z = np.zeros(n_modes)
    for j in range(n_modes):
        c = z.copy()
        c[j] = 1.0
        B.append(1e9 * command_to_opd_delta(zwfs, M2C, c))
    B_nm = np.asarray(B, dtype=float)
    A = B_nm[:, pupil].T  # n_pix x n_modes
    P = np.linalg.pinv(A, rcond=float(rcond))  # n_modes x n_pix
    return B_nm, P


def project_opd_nm_to_control_basis(opd_m: np.ndarray, pupil: np.ndarray, B_nm: np.ndarray, P: np.ndarray):
    """Project an OPD screen onto the DM control OPD basis.

    Returns command coefficients, fitted controllable OPD in nm, controllable RMS,
    and residual/uncontrollable RMS, all evaluated over the pupil.
    """
    y_nm = 1e9 * np.asarray(opd_m, dtype=float)[pupil]
    coeff = P @ y_nm
    fit_nm = coeff @ B_nm[:, pupil]
    rem_nm = y_nm - fit_nm
    return coeff, fit_nm, float(np.std(fit_nm)), float(np.std(rem_nm))


def run_ao_simulation(outdir, zwfs, cfg, amp, opd_internal, detector, use_pyzelda, args, I2M):
    if h5py is None:
        raise RuntimeError("h5py is required for AO output. Install h5py or run diagnostics only.")

    print("\nStarting AO simulation...")
    M2C = np.asarray(zwfs.reco.M2C_0, dtype=float)
    n_modes = M2C.shape[0]
    pupil = np.asarray(zwfs.grid.pupil_mask, dtype=bool)
    crop_y, crop_x = pupil_crop_slices(pupil, padding=args.crop_padding_pixels)
    pupil_crop = pupil[crop_y, crop_x]
    crop_shape = pupil_crop.shape
    print(
        f"Saving cropped residual OPD screens with shape={crop_shape} "
        f"from full grid shape={pupil.shape}; "
        f"y=[{crop_y.start}:{crop_y.stop}], x=[{crop_x.start}:{crop_x.stop}]."
    )

    print("Building OPD projection onto the Baldr/DM control basis for diagnostics...")
    B_control_nm, P_opd_to_cmd = build_control_opd_projection(
        zwfs, M2C, pupil, rcond=args.opd_projection_rcond
    )
    tt_cmd_rms_total = compute_tt_command_rms_from_nm(args, B_control_nm, pupil)
    wvl0 = float(zwfs.optics.wvl0)
    dx = float(zwfs.grid.D) / float(zwfs.grid.N)
    r0 = float(args.r0_500_m) * (wvl0 / float(args.atm_reference_wavelength_m)) ** (6.0 / 5.0)
    basis = make_first_stage_basis(zwfs, args.first_stage_modes)
    phase_scaling_factor = calibrate_atmospheric_scale(zwfs, basis, cfg, args)

    scrn = ps.PhaseScreenKolmogorov(
        nx_size=int(zwfs.grid.dim), pixel_scale=dx, r0=r0, L0=float(args.L0_m), random_seed=int(args.random_seed)
    )

    reco_lag = []
    for _ in range(max(0, args.first_stage_lag_frames)):
        scrn.add_row()
        _, reco = bldr.first_stage_ao(
            scrn,
            Nmodes_removed=int(args.first_stage_modes),
            basis=basis,
            phase_scaling_factor=phase_scaling_factor,
            return_reconstructor=True,
        )
        reco_lag.append(reco)


    # Initialize the first-stage applied correction (2D phase map)
    first_stage_U = np.zeros_like(zwfs.grid.pupil_mask, dtype=float)
    
    # Initialize the lag queue for the first stage measurements. 
    # If lag is N frames, we need N empty measurements in the queue to start.
    fs_lag_frames = max(1, args.first_stage_lag_frames)
    fs_error_queue = [np.zeros_like(first_stage_U) for _ in range(fs_lag_frames)]

    modal_state = np.zeros(n_modes)
    baldr_lag_frames = max(0, int(args.baldr_lag_frames))
    baldr_error_lag = [np.zeros(n_modes) for _ in range(baldr_lag_frames)]
    zwfs.dm.current_cmd = zwfs.dm.dm_flat.copy()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = outdir / output_path

    # ------------------------------------------------------------------
    # Output splitting policy.
    #
    # Recommended mode: --frames-per-output-file N, because the number of
    # frames per HDF5 file is deterministic and independent of gzip behaviour.
    # If only --max-output-file-gb is provided, we estimate a conservative
    # frames-per-file from the raw per-frame payload. Compression usually makes
    # the actual file smaller than this estimate.
    # ------------------------------------------------------------------
    explicit_frames_per_file = int(getattr(args, "frames_per_output_file", 0))
    max_output_file_gb = float(getattr(args, "max_output_file_gb", 0.0))

    bytes_per_frame = int(np.prod(crop_shape)) * np.dtype("f4").itemsize
    bytes_per_frame += 5 * n_modes * np.dtype("f4").itemsize  # modal arrays
    bytes_per_frame += 16 * np.dtype("f4").itemsize            # scalar telemetry margin
    bytes_per_frame += 32                                      # reset/valid small arrays margin

    static_overhead_bytes = 0
    static_overhead_bytes += int(np.prod(pupil_crop.shape)) * np.dtype("u1").itemsize
    static_overhead_bytes += int(np.prod(pupil.shape)) * np.dtype("u1").itemsize
    static_overhead_bytes += int(np.prod(M2C.shape)) * np.dtype("f4").itemsize
    static_overhead_bytes += int(np.prod(np.asarray(zwfs.reco.IM).shape)) * np.dtype("f4").itemsize
    static_overhead_bytes += int(np.prod(B_control_nm[:, crop_y, crop_x].shape)) * np.dtype("f4").itemsize

    if explicit_frames_per_file > 0:
        frames_per_file = explicit_frames_per_file
        split_reason = f"explicit --frames-per-output-file={frames_per_file}"
    elif max_output_file_gb > 0:
        max_bytes = max_output_file_gb * (1024 ** 3)
        # Leave margin for HDF5 metadata/compression chunks and static datasets.
        usable_bytes = max(1, int(0.80 * max_bytes - static_overhead_bytes))
        frames_per_file = max(1, int(usable_bytes // max(1, bytes_per_frame)))
        split_reason = (
            f"estimated from --max-output-file-gb={max_output_file_gb:g}; "
            f"raw bytes/frame≈{bytes_per_frame}, static≈{static_overhead_bytes}"
        )
    else:
        frames_per_file = int(args.n_frames)
        split_reason = "no splitting requested"

    frames_per_file = max(1, min(int(frames_per_file), int(args.n_frames)))
    split_outputs = frames_per_file < int(args.n_frames)
    n_parts_est = int(np.ceil(int(args.n_frames) / frames_per_file))

    print(
        f"HDF5 output policy: frames_per_file={frames_per_file}, "
        f"n_parts={n_parts_est}, split_outputs={split_outputs} ({split_reason})."
    )

    output_parts = []

    def part_path_for(part_index: int) -> Path:
        if not split_outputs:
            return output_path
        digits = max(3, int(getattr(args, "output_part_digits", 4)))
        return output_path.with_name(f"{output_path.stem}_part{part_index:0{digits}d}{output_path.suffix}")

    def open_output_part(part_index: int, global_start: int, n_local_frames: int):
        part_path = part_path_for(part_index)
        print(
            f"\nOpening HDF5 output part {part_index+1}/{n_parts_est}: {part_path} "
            f"for global frames [{global_start}, {global_start+n_local_frames}).",
            flush=True,
        )
        h5 = h5py.File(part_path, "w")
        h5.attrs["description"] = "Baldr closed-loop residual OPD screens from JSON/DM-calibrated control IM"
        h5.attrs["units_residual_opd"] = "nm OPD"
        h5.attrs["fs_hz"] = float(args.fs_hz)
        h5.attrs["wvl0_m"] = wvl0
        h5.attrs["control_basis"] = args.control_basis
        h5.attrs["control_modes"] = int(n_modes)
        h5.attrs["first_stage_modes"] = int(args.first_stage_modes)
        h5.attrs["gain"] = float(args.gain)
        h5.attrs["leak"] = float(args.leak)
        h5.attrs["baldr_lag_frames"] = int(baldr_lag_frames)
        h5.attrs["include_shotnoise"] = bool(args.include_shotnoise)
        h5.attrs["crop_padding_pixels"] = int(args.crop_padding_pixels)
        h5.attrs["residual_crop_y0"] = int(crop_y.start)
        h5.attrs["residual_crop_y1"] = int(crop_y.stop)
        h5.attrs["residual_crop_x0"] = int(crop_x.start)
        h5.attrs["residual_crop_x1"] = int(crop_x.stop)
        h5.attrs["tt_rms_nm"] = float(args.tt_rms_nm)
        h5.attrs["tt_rms_cmd"] = float(args.tt_rms_cmd)
        h5.attrs["tt_axis"] = str(args.tt_axis)
        h5.attrs["tt_frequencies_hz"] = str(args.tt_frequencies_hz)
        h5.attrs["tt_cmd_rms_total_used"] = float(tt_cmd_rms_total)
        h5.attrs["reset_on_fail"] = bool(args.reset_on_fail)
        h5.attrs["reset_rms_threshold_nm"] = float(args.reset_rms_threshold_nm)
        h5.attrs["reset_hold_frames"] = int(args.reset_hold_frames)
        h5.attrs["max_loop_resets"] = int(args.max_loop_resets)
        h5.attrs["global_frame_start"] = int(global_start)
        h5.attrs["global_frame_stop_exclusive"] = int(global_start + n_local_frames)
        h5.attrs["n_global_frames_requested"] = int(args.n_frames)
        h5.attrs["output_part_index"] = int(part_index)
        h5.attrs["output_part_count_estimate"] = int(n_parts_est)
        h5.attrs["frames_per_output_file"] = int(frames_per_file)
        h5.attrs["max_output_file_gb"] = float(max_output_file_gb)

        h5.create_dataset("pupil_mask", data=pupil_crop.astype(np.uint8), compression="gzip")
        h5.create_dataset("pupil_mask_full", data=pupil.astype(np.uint8), compression="gzip")
        h5.create_dataset("control_M2C_command_basis", data=M2C.astype(np.float32), compression="gzip")
        h5.create_dataset("interaction_matrix", data=np.asarray(zwfs.reco.IM, dtype=np.float32), compression="gzip")
        h5.create_dataset("control_opd_basis_nm_per_cmd", data=B_control_nm[:, crop_y, crop_x].astype(np.float32), compression="gzip")

        chunk_frames = min(int(args.chunk_frames), int(n_local_frames))
        chunks_img = (max(1, chunk_frames),) + tuple(crop_shape)
        chunks_modes = (max(1, chunk_frames), n_modes)

        d = {}
        d["residual_opd_nm"] = h5.create_dataset(
            "residual_opd_nm", shape=(n_local_frames,) + tuple(crop_shape), dtype="f4", chunks=chunks_img, compression="gzip"
        )
        for key in [
            "rms_pre_naomi_nm",
            "rms_post_naomi_nm",
            "rms_baldr_input_nm",
            "rms_after_baldr_nm",
            "rms_control_component_baldr_input_nm",
            "rms_control_component_after_baldr_nm",
            "rms_uncontrolled_component_baldr_input_nm",
            "rms_uncontrolled_component_after_baldr_nm",
            "rms_tt_extra_opd_nm",
        ]:
            d[key] = h5.create_dataset(key, shape=(n_local_frames,), dtype="f4")

        for key in [
            "modal_reco_input_cmd",
            "modal_command_state_cmd",
            "modal_fit_coeff_baldr_input_cmd",
            "modal_fit_coeff_after_baldr_cmd",
            "modal_tt_extra_cmd",
        ]:
            d[key] = h5.create_dataset(key, shape=(n_local_frames, n_modes), dtype="f4", chunks=chunks_modes, compression="gzip")

        d["loop_reset_flag"] = h5.create_dataset("loop_reset_flag", shape=(n_local_frames,), dtype="u1")
        d["loop_reset_count"] = h5.create_dataset("loop_reset_count", shape=(n_local_frames,), dtype="i4")
        d["loop_reset_reason_code"] = h5.create_dataset("loop_reset_reason_code", shape=(n_local_frames,), dtype="i4")
        d["valid_residual_flag"] = h5.create_dataset("valid_residual_flag", shape=(n_local_frames,), dtype="u1")

        output_parts.append(
            {
                "part_index": int(part_index),
                "path": str(part_path),
                "global_frame_start": int(global_start),
                "global_frame_stop_exclusive": int(global_start + n_local_frames),
                "n_frames": int(n_local_frames),
            }
        )
        return h5, d, part_path

    n_loop_resets = 0
    reset_hold_counter = 0
    previous_good_residual = None

    h5 = None
    d = None
    current_part_index = -1
    local_k = 0
    current_part_path = None

    try:
        for k in range(args.n_frames):
            # Open the next file part if needed.
            if h5 is None or local_k >= int(h5.attrs["global_frame_stop_exclusive"]) - int(h5.attrs["global_frame_start"]):
                if h5 is not None:
                    h5.flush()
                    h5.close()
                    print(f"Closed HDF5 output part: {current_part_path}", flush=True)
                current_part_index += 1
                global_start = k
                n_local_frames = min(frames_per_file, int(args.n_frames) - k)
                h5, d, current_part_path = open_output_part(current_part_index, global_start, n_local_frames)
                local_k = 0

            for _ in range(max(1, args.rows_per_frame)):
                scrn.add_row()

            # new 
            _, reco_now = bldr.first_stage_ao(
                scrn,
                Nmodes_removed=int(args.first_stage_modes),
                basis=basis,
                phase_scaling_factor=phase_scaling_factor,
                return_reconstructor=True,
            )

            # 2. The WFS measurement is the open-loop projection minus the currently applied DM shape
            current_measurement = reco_now - first_stage_U
            
            # 3. Push measurement to the delay queue and pop the oldest one
            fs_error_queue.append(current_measurement)
            delayed_measurement = fs_error_queue.pop(0)

            # 4. Update the leaky integrator state
            first_stage_U = (args.first_stage_leak * first_stage_U) + (args.first_stage_gain * delayed_measurement)

            # 5. Calculate the phases entering the second stage
            pre_phase = phase_scaling_factor * scrn.scrn * basis[0]
            post_phase = basis[0] * (pre_phase - first_stage_U)

            # _, reco_now = bldr.first_stage_ao(
            #     scrn,
            #     Nmodes_removed=int(args.first_stage_modes),
            #     basis=basis,
            #     phase_scaling_factor=phase_scaling_factor,
            #     return_reconstructor=True,
            # )
            # if args.first_stage_lag_frames > 0:
            #     reco_lag.append(reco_now)
            #     reco_use = reco_lag.pop(0)
            # else:
            #     reco_use = reco_now

            # pre_phase = phase_scaling_factor * scrn.scrn * basis[0]
            # post_phase = basis[0] * (phase_scaling_factor * scrn.scrn - reco_use)

            opd_pre = (wvl0 / (2 * np.pi)) * pre_phase
            opd_post = (wvl0 / (2 * np.pi)) * post_phase

            # Optional extra TT vibration in the same DM command basis, applied upstream.
            extra_cmd = tt_modal_extra_command(k, args.fs_hz, args, n_modes, tt_cmd_rms_total)
            extra_opd = get_dm_opd(zwfs, zwfs.dm.dm_flat + extra_cmd @ M2C) - get_dm_opd(zwfs, zwfs.dm.dm_flat)
            opd_in = opd_post + extra_opd

            # ------------------------------------------------------------
            # Measurement and control update, with an optional watchdog reset.
            # Reset frames are marked invalid and the saved phase cube uses
            # the previous good residual rather than the open-loop reset frame.
            # ------------------------------------------------------------
            reset_flag = 0
            reset_reason = 0
            opd_flat = get_dm_opd(zwfs, zwfs.dm.dm_flat)

            if reset_hold_counter > 0:
                modal_state[:] = 0.0
                baldr_error_lag = [np.zeros(n_modes) for _ in range(baldr_lag_frames)]
                zwfs.dm.current_cmd = zwfs.dm.dm_flat.copy()
                e = np.zeros(n_modes)
                if previous_good_residual is not None:
                    residual = previous_good_residual.copy()
                else:
                    residual = remove_piston_over_pupil(opd_in, pupil)
                reset_flag = 1
                reset_reason = 2
                reset_hold_counter -= 1

            else:
                # Measurement with the DM state currently applied. get_frame adds this DM OPD internally.
                dm_cmd_before = np.asarray(zwfs.dm.current_cmd, dtype=float).copy()
                opd_dm_before = get_dm_opd(zwfs, dm_cmd_before)
                I = bldr.get_frame(
                    opd_in,
                    amp,
                    opd_internal,
                    zwfs,
                    detector=detector,
                    include_shotnoise=args.include_shotnoise,
                    use_pyZelda=use_pyzelda,
                )
                s = normalise_signal(I, zwfs, args.normalization_method)
                e = I2M @ s

                if baldr_lag_frames > 0:
                    # Physical delayed loop: this frame's residual is set by the command
                    # already on the DM. The newly measured error is queued for later.
                    residual = opd_in + (opd_dm_before - opd_flat)
                    baldr_error_lag.append(e.copy())
                    e_apply = baldr_error_lag.pop(0)
                    modal_state = args.leak * modal_state + args.control_sign * args.gain * e_apply
                    zwfs.dm.current_cmd = zwfs.dm.dm_flat - modal_state @ M2C
                else:
                    # Legacy zero-latency / same-frame correction used by previous versions.
                    modal_state = args.leak * modal_state + args.control_sign * args.gain * e
                    zwfs.dm.current_cmd = zwfs.dm.dm_flat - modal_state @ M2C
                    opd_dm_after = get_dm_opd(zwfs, zwfs.dm.current_cmd)
                    residual = opd_in + (opd_dm_after - opd_flat)

                residual_rms_nm = pupil_rms_nm(residual, pupil)
                if (
                    args.reset_on_fail
                    and np.isfinite(residual_rms_nm)
                    and residual_rms_nm > float(args.reset_rms_threshold_nm)
                ):
                    n_loop_resets += 1
                    if n_loop_resets > int(args.max_loop_resets):
                        raise RuntimeError(
                            f"Maximum loop resets exceeded: {n_loop_resets} > {args.max_loop_resets}. "
                            f"Last residual RMS was {residual_rms_nm:.2f} nm."
                        )

                    print(
                        f"\n*** LOOP WATCHDOG RESET at frame {k+1}/{args.n_frames}: "
                        f"after-Baldr RMS={residual_rms_nm:.1f} nm exceeds "
                        f"{args.reset_rms_threshold_nm:.1f} nm. "
                        f"Resetting DM and integrator. Total resets={n_loop_resets}. ***\n",
                        flush=True,
                    )

                    modal_state[:] = 0.0
                    baldr_error_lag = [np.zeros(n_modes) for _ in range(baldr_lag_frames)]
                    zwfs.dm.current_cmd = zwfs.dm.dm_flat.copy()

                    # Do not save the open-loop reset frame as a valid corrected residual.
                    if previous_good_residual is not None:
                        residual = previous_good_residual.copy()
                    else:
                        residual = remove_piston_over_pupil(opd_in, pupil)
                    e = np.zeros(n_modes)

                    reset_flag = 1
                    reset_reason = 1
                    reset_hold_counter = max(0, int(args.reset_hold_frames))

            cin_fit, _, ctrl_in_rms, unctrl_in_rms = project_opd_nm_to_control_basis(
                opd_in, pupil, B_control_nm, P_opd_to_cmd
            )
            cout_fit, _, ctrl_out_rms, unctrl_out_rms = project_opd_nm_to_control_basis(
                residual, pupil, B_control_nm, P_opd_to_cmd
            )

            valid_residual = 1 if reset_flag == 0 else 0

            # Always save piston-removed OPD maps.
            residual_save = remove_piston_over_pupil(residual, pupil)

            # Only update previous-good memory from genuine non-reset frames.
            if valid_residual:
                previous_good_residual = residual_save.copy()

            d["residual_opd_nm"][local_k] = (1e9 * residual_save[crop_y, crop_x]).astype(np.float32)
            d["rms_pre_naomi_nm"][local_k] = pupil_rms_nm(opd_pre, pupil)
            d["rms_post_naomi_nm"][local_k] = pupil_rms_nm(opd_post, pupil)
            d["rms_baldr_input_nm"][local_k] = pupil_rms_nm(opd_in, pupil)
            d["rms_after_baldr_nm"][local_k] = pupil_rms_nm(residual, pupil)
            d["modal_reco_input_cmd"][local_k] = e.astype(np.float32)
            d["modal_command_state_cmd"][local_k] = modal_state.astype(np.float32)
            d["modal_fit_coeff_baldr_input_cmd"][local_k] = cin_fit.astype(np.float32)
            d["modal_fit_coeff_after_baldr_cmd"][local_k] = cout_fit.astype(np.float32)
            d["rms_control_component_baldr_input_nm"][local_k] = ctrl_in_rms
            d["rms_control_component_after_baldr_nm"][local_k] = ctrl_out_rms
            d["rms_uncontrolled_component_baldr_input_nm"][local_k] = unctrl_in_rms
            d["rms_uncontrolled_component_after_baldr_nm"][local_k] = unctrl_out_rms
            d["rms_tt_extra_opd_nm"][local_k] = pupil_rms_nm(extra_opd, pupil)
            d["modal_tt_extra_cmd"][local_k] = extra_cmd.astype(np.float32)
            d["loop_reset_flag"][local_k] = np.uint8(reset_flag)
            d["loop_reset_count"][local_k] = int(n_loop_resets)
            d["loop_reset_reason_code"][local_k] = int(reset_reason)
            d["valid_residual_flag"][local_k] = np.uint8(valid_residual)

            if (k + 1) % args.progress_every == 0 or (k + 1) == args.n_frames:
                print(
                    f"[{k+1:8d}/{args.n_frames}] "
                    f"pre-Naomi={d['rms_pre_naomi_nm'][local_k]:8.1f} nm, post-Naomi={d['rms_post_naomi_nm'][local_k]:8.1f} nm, "
                    f"TT={d['rms_tt_extra_opd_nm'][local_k]:6.1f} nm, Baldr in={d['rms_baldr_input_nm'][local_k]:8.1f} nm, "
                    f"after Baldr={d['rms_after_baldr_nm'][local_k]:8.1f} nm, "
                    f"ctrl={ctrl_in_rms:7.1f}->{ctrl_out_rms:7.1f} nm, "
                    f"floor={unctrl_in_rms:7.1f}->{unctrl_out_rms:7.1f} nm, "
                    f"|e|rms={np.std(e):8.4f} cmd, |u|rms={np.std(modal_state):8.4f} cmd, "
                    f"resets={n_loop_resets:d}, part={current_part_index}",
                    flush=True,
                )

            local_k += 1

    finally:
        if h5 is not None:
            h5.flush()
            h5.close()
            print(f"Closed HDF5 output part: {current_part_path}", flush=True)

    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json") if split_outputs else output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest = {
        "requested_output": str(output_path),
        "split_outputs": bool(split_outputs),
        "frames_per_file": int(frames_per_file),
        "n_global_frames_requested": int(args.n_frames),
        "n_parts": int(len(output_parts)),
        "max_output_file_gb": float(max_output_file_gb),
        "bytes_per_frame_estimate_raw": int(bytes_per_frame),
        "static_overhead_bytes_estimate_raw": int(static_overhead_bytes),
        "parts": output_parts,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if split_outputs:
        print("AO simulation wrote split HDF5 outputs:")
        for part in output_parts:
            print(f"  {Path(part['path']).resolve()}")
        print(f"Manifest wrote: {manifest_path.resolve()}")
    else:
        print(f"AO simulation wrote: {output_path.resolve()}")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--outdir", default=Path("baldr_json_im_verify_then_ao"), type=Path)
    p.add_argument("--output", default="baldr_closed_loop_residuals.h5", type=Path)

    p.add_argument("--use-pyzelda", action="store_true")
    p.add_argument("--no-detector", action="store_true")
    p.add_argument("--detector-binning", type=int, default=None)
    p.add_argument("--detector-dit", type=float, default=1.0)
    p.add_argument("--detector-ron", type=float, default=0.0)
    p.add_argument("--detector-qe", type=float, default=1.0)

    p.add_argument("--imgs-to-mean", type=int, default=1)
    p.add_argument("--normalization-method", default="subframe mean", choices=["subframe mean", "clear pupil mean"])
    p.add_argument("--zonal-basis", default="Zonal")
    p.add_argument("--zonal-modes", type=int, default=140)
    p.add_argument("--zonal-poke", type=float, default=0.05)
    p.add_argument("--control-basis", default="Zernike_pinned_edges")
    p.add_argument("--control-modes", type=int, default=40)
    p.add_argument("--control-poke", type=float, default=0.05)
    p.add_argument("--n-plot-modes", type=int, default=8)
    p.add_argument("--plot-registration-debug", action="store_true")

    p.add_argument("--svd-rcond", type=float, default=1e-3)
    p.add_argument("--svd-n-keep", type=int, default=None)
    p.add_argument("--tikhonov", type=float, default=0.0)
    p.add_argument("--opd-projection-rcond", type=float, default=1e-6, help="Rcond for projecting OPD screens onto the DM control OPD basis")

    p.add_argument("--n-injection-tests", type=int, default=8)
    p.add_argument("--injection-amps", type=float, nargs="+", default=[0.01, 0.03, 0.05])
    p.add_argument("--n-random-injection-tests", type=int, default=5)
    p.add_argument("--random-injection-rms-cmd", type=float, default=0.02)
    p.add_argument("--n-visual-injection-tests", type=int, default=4)
    p.add_argument("--visual-injection-amp-cmd", type=float, default=0.03)
    p.add_argument("--random-seed", type=int, default=2)

    p.add_argument("--diagnostic-only", action="store_true", help="Stop after calibration/injection tests")
    p.add_argument("--yes", action="store_true", help="Do not ask before starting AO simulation")

    p.add_argument("--n-frames", type=int, default=1000)
    p.add_argument("--fs-hz", type=float, default=1000.0)
    p.add_argument("--gain", type=float, default=0.03)
    p.add_argument("--leak", type=float, default=0.995)
    p.add_argument("--control-sign", type=float, default=1.0, choices=[-1.0, 1.0])
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--chunk-frames", type=int, default=128)
    p.add_argument("--frames-per-output-file", type=int, default=0, help="Split AO HDF5 output into fixed-size parts with this many frames per file. 0 disables unless --max-output-file-gb is set.")
    p.add_argument("--max-output-file-gb", type=float, default=0.0, help="Approximate maximum HDF5 part size in GiB. Uses a conservative raw-size estimate to choose frames per part. 0 disables size-based splitting.")
    p.add_argument("--output-part-digits", type=int, default=4, help="Number of digits in split output filenames, e.g. part0000.")

    p.add_argument("--r0-500-m", type=float, default=0.126)
    p.add_argument("--L0-m", type=float, default=25.0)
    p.add_argument("--atm-reference-wavelength-m", type=float, default=5.0e-7)
    p.add_argument("--rows-per-frame", type=int, default=1)
    p.add_argument("--phase-scaling-factor", type=float, default=1.0)
    p.add_argument("--target-pre-naomi-rms-nm", type=float, default=1000.0)
    p.add_argument("--atm-calibration-frames", type=int, default=100)
    p.add_argument("--first-stage-modes", type=int, default=14)
    p.add_argument("--first-stage-lag-frames", type=int, default=0)
    p.add_argument("--tt-rms-nm", type=float, default=0.0, help="Optional total TT vibration RMS in nm OPD, split over selected --tt-axis/axes and all TT frequency lines")
    p.add_argument("--tt-frequencies-hz", default="15,50", help="Comma-separated TT vibration frequencies in Hz, e.g. '15,50'")
    p.add_argument("--tt-axis", default="both", choices=["both", "tip", "tilt"], help="Which TT axis to inject vibration into: both, tip/mode-0 only, or tilt/mode-1 only")
    p.add_argument("--tt-rms-cmd", type=float, default=0.0, help="Legacy optional total TT vibration RMS in DM command basis units; ignored if --tt-rms-nm > 0")
    p.add_argument("--baldr-lag-frames", type=int, default=0, help="Integer-frame Baldr control latency. 0 preserves the legacy same-frame correction; >0 applies delayed corrections from previous measurements.")
    p.add_argument("--include-shotnoise", action="store_true", help="Include photon shot noise in the live Baldr AO frames. Calibration/reference/IM frames remain noiseless.")
    p.add_argument("--crop-padding-pixels", type=int, default=2, help="Padding around active pupil when saving residual_opd_nm cropped phase screens.")
    p.add_argument("--reset-on-fail", action="store_true", help="Reset DM and modal integrator if the after-Baldr residual exceeds --reset-rms-threshold-nm.")
    p.add_argument("--reset-rms-threshold-nm", type=float, default=250.0, help="Failure threshold for after-Baldr residual RMS in nm. If exceeded, reset DM/control loop.")
    p.add_argument("--reset-hold-frames", type=int, default=0, help="Optional number of frames to hold DM flat after a reset before reclosing.")
    p.add_argument("--max-loop-resets", type=int, default=1000000, help="Maximum allowed loop resets before raising an error.")

    p.add_argument("--first-stage-gain", type=float, default=0.4, help="Loop gain for the first stage AO leaky integrator.")
    p.add_argument("--first-stage-leak", type=float, default=0.99, help="Integrator leak for the first stage AO.")

    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_json(args.config)

    print("Initialising ZWFS from JSON...")
    print(f"  baldr_core imported from: {getattr(bldr, '__file__', '<unknown>')}")
    zwfs = bldr.init_zwfs_from_json(
        args.config,
        derive_fresnel=False,
        derive_spectrum=False,
        instantiate_detector=False,
    )
    if hasattr(zwfs, "fresnel_relay"):
        zwfs.fresnel_relay.enabled = False
    if hasattr(zwfs, "spectrum"):
        zwfs.spectrum.enabled = False

    detector = make_detector_from_config(zwfs, cfg, args)
    if detector is not None:
        zwfs.detector = detector
        print(f"Using detector: binning={detector.binning}, dit={detector.dit}, ron={detector.ron}, qe={detector.qe}")
    else:
        if hasattr(zwfs, "detector"):
            delattr(zwfs, "detector")
        print("Using detector=None: intensities remain in wave-space grid.")

    opd0 = np.zeros_like(zwfs.grid.pupil_mask, dtype=float)
    opd_internal = np.zeros_like(zwfs.grid.pupil_mask, dtype=float)
    amp = make_calibration_amp(zwfs, cfg)
    zwfs.dm.current_cmd = zwfs.dm.dm_flat.copy()

    print("Getting reference intensities N0/I0...")
    I0 = bldr.get_I0(opd0, amp, opd_internal, zwfs, detector=detector, include_shotnoise=False, use_pyZelda=args.use_pyzelda).astype(float)
    N0 = bldr.get_N0(opd0, amp, opd_internal, zwfs, detector=detector, include_shotnoise=False, use_pyZelda=args.use_pyzelda).astype(float)
    np.save(args.outdir / "I0.npy", I0)
    np.save(args.outdir / "N0.npy", N0)
    save_image(args.outdir / "reference_intensity_I0.png", I0, "Baldr reference intensity I0", "Intensity")
    save_image(args.outdir / "reference_intensity_I0_log.png", I0, "Baldr reference intensity I0 log10", "log10 intensity", log=True)
    save_image(args.outdir / "clear_pupil_N0.png", N0, "Clear pupil N0", "Intensity")
    save_image(args.outdir / "clear_pupil_mask_model.png", zwfs.grid.pupil_mask, "Model pupil mask", "mask")

    print("Classifying pupil regions...")
    zwfs = bldr.classify_pupil_regions(opd0, amp, opd_internal, zwfs, detector=detector, use_pyZelda=args.use_pyzelda, mode="bright")
    plt.close("all")
    save_image(args.outdir / "classified_pupil_filt.png", zwfs.pupil_regions.pupil_filt, "Detected pupil filter", "mask")
    save_image(args.outdir / "classified_secondary_filt.png", zwfs.pupil_regions.secondary_strehl_filt, "Detected secondary filter", "mask")
    save_image(args.outdir / "classified_outer_strehl_filt.png", zwfs.pupil_regions.outer_strehl_filt, "Detected outer Strehl filter", "mask")

    print("\nBuilding zonal IM for DM registration...")
    zwfs = bldr.build_IM(
        zwfs,
        calibration_opd_input=opd0,
        calibration_amp_input=amp,
        opd_internal=opd_internal,
        basis=args.zonal_basis,
        Nmodes=args.zonal_modes,
        poke_amp=args.zonal_poke,
        poke_method="double_sided_poke",
        normalization_method=args.normalization_method,
        imgs_to_mean=args.imgs_to_mean,
        detector=detector,
        use_pyZelda=args.use_pyzelda,
    )
    zonal_IM = np.asarray(zwfs.reco.IM)
    np.save(args.outdir / "zonal_IM.npy", zonal_IM)
    save_im_rows(args.outdir, zonal_IM, I0.shape, "zonal_IM_signal", args.n_plot_modes)

    print("\nRegistering DM in pixel space from zonal IM...")
    zwfs = bldr.register_DM_in_pixelspace_from_IM(zwfs, plot_intermediate_results=args.plot_registration_debug)
    plt.close("all")
    save_registration_overlay(args.outdir, I0, zwfs)
    if hasattr(zwfs, "dm2pix_registration"):
        reg = zwfs.dm2pix_registration
        with open(args.outdir / "dm_registration_summary.json", "w") as f:
            json.dump({
                "DM_center_pixel_space": np.asarray(reg.DM_center_pixel_space).tolist(),
                "n_actuator_pixel_coords": int(len(reg.actuator_coord_list_pixel_space)),
                "first_five_actuator_pixel_coords": np.asarray(reg.actuator_coord_list_pixel_space[:5]).tolist(),
            }, f, indent=2)

    print("\nBuilding control/modal IM with the registered DM model...")
    zwfs = bldr.build_IM(
        zwfs,
        calibration_opd_input=opd0,
        calibration_amp_input=amp,
        opd_internal=opd_internal,
        basis=args.control_basis,
        Nmodes=args.control_modes,
        poke_amp=args.control_poke,
        poke_method="double_sided_poke",
        normalization_method=args.normalization_method,
        imgs_to_mean=args.imgs_to_mean,
        detector=detector,
        use_pyZelda=args.use_pyzelda,
    )
    control_IM = np.asarray(zwfs.reco.IM)
    np.save(args.outdir / "control_IM.npy", control_IM)
    np.save(args.outdir / "control_M2C_command_basis.npy", np.asarray(zwfs.reco.M2C_0))
    save_im_rows(args.outdir, control_IM, I0.shape, "control_IM_signal", args.n_plot_modes)

    print("\nBuilding reconstructor diagnostics from control IM...")
    I2M, reco_summary = save_response_diagnostics(args.outdir, control_IM, args)
    print(json.dumps(reco_summary, indent=2))

    run_real_injection_tests(args.outdir, zwfs, amp, opd0, opd_internal, detector, args.use_pyzelda, args, I2M)
    run_visual_injection_and_sign_tests(args.outdir, zwfs, amp, opd_internal, detector, args.use_pyzelda, args, I2M)

    with open(args.outdir / "run_summary.json", "w") as f:
        json.dump({
            "config": str(args.config),
            "outdir": str(args.outdir),
            "image_shape": list(I0.shape),
            "zonal_basis": args.zonal_basis,
            "zonal_modes": int(args.zonal_modes),
            "control_basis": args.control_basis,
            "control_modes": int(args.control_modes),
            "normalization_method": args.normalization_method,
            "reconstructor_summary": reco_summary,
        }, f, indent=2)

    print("\nCalibration and injection diagnostics complete.")
    print(f"Diagnostics written to: {args.outdir.resolve()}")
    print("Key new files:")
    print("  real_single_mode_injection_tests.csv")
    print("  real_single_mode_recovered_over_injected.png")
    print("  real_random_multimode_injection_tests.csv")
    print("  real_random_multimode_reconstruction_rms.png")
    print("  visual_mode_XXX_input_opd_nm.png / zwfs_intensity.png / zwfs_signal.png")
    print("  visual_upstream_injection_sign_tests.csv")

    if args.diagnostic_only:
        print("--diagnostic-only set: stopping before AO simulation.")
        return
    if not args.yes:
        reply = input("\nBegin rolling-atmosphere Baldr AO simulation and write residual OPD screens? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Stopping before AO simulation.")
            return

    run_ao_simulation(args.outdir, zwfs, cfg, amp, opd_internal, detector, args.use_pyzelda, args, I2M)


if __name__ == "__main__":
    main()

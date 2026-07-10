## Run ./shm_creator_sim first to create the shm in the right format.
# Do not pass data here otherwise it will overwrite/recreate with the Python
# shmlib.py wrapper format, which may differ from the RTC-required format.


##############################################################################################################
# reverted back to stable version, i tried to implement adding dynamic phasemask changing throug baldrapp/apps/paranal_simulator/fake_configs/phasemask_properties_physica_NOTWORKING.json
# which relies on an updated (still backwards compatible) spectrum.py script which is here
# These changes have not been commited or implemented


import numpy as np
import json
import zmq
import time
from xaosim.shmlib import shm  # type: ignore
import subprocess
from pathlib import Path

from baldrapp.common import baldr_core as bldr
from baldrapp.common import utilities as util
from baldrapp.common import phasescreens as ps
from baldrapp.common import spectrum as spec
import pyzelda.ztools as ztools  # type: ignore
from types import SimpleNamespace
from typing import Dict, Optional

############### ADDING IN TO TRY FIX PHASEMASK CHANGE ISSUES


def load_phasemask_properties(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Phasemask properties file not found: {path}")

    with open(path, "r") as f:
        cfg = json.load(f)

    if "phasemask" in cfg:
        cfg = cfg["phasemask"]

    if "masks" not in cfg:
        raise ValueError("phasemask_properties.json must contain phasemask.masks")

    return cfg


def restore_optics_state(optics, state):
    """
    Restore optics fields to a previous baseline state and remove runtime
    phase-mask fields that were not present originally.
    """
    runtime_keys = [
        "active_phasemask",
        "current_mask",
        "theta_rad_wvl0",
        "mask_diam_lambdaD_wvl0",
    ]

    for key in runtime_keys:
        if key not in state and hasattr(optics, key):
            delattr(optics, key)

    for key, value in state.items():
        setattr(optics, key, value)


def precompute_active_phasemask_spectrum_cache(zwfs):
    """
    Precompute theta(lambda) and mask_diam(lambda) for the current active mask
    on the current zwfs.spectrum wavelength grid.

    This avoids repeated material interpolation during the frame loop.
    """
    if not hasattr(zwfs.optics, "active_phasemask"):
        return None

    pm = zwfs.optics.active_phasemask

    # Prevent stale cache being used while rebuilding.
    for key in [
        "cached_wavelengths_m",
        "cached_theta_rad",
        "cached_mask_diam_lambdaD",
    ]:
        if hasattr(pm, key):
            delattr(pm, key)

    wavelengths_m = np.asarray(zwfs.spectrum.wavelengths, dtype=float)

    theta_grid = np.array(
        [spec.theta_at_wavelength(zwfs.optics, w) for w in wavelengths_m], dtype=float
    )

    mask_diam_grid = np.array(
        [
            spec.phasemask_diameter_at_wavelength(
                zwfs.optics,
                wavelength_m=w,
                default_wvl0=zwfs.optics.wvl0,
            )
            for w in wavelengths_m
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(theta_grid)):
        raise ValueError(
            f"active phasemask theta grid contains non-finite values "
            f"for mask {getattr(pm, 'name', '<unknown>')!r}"
        )

    if not np.all(np.isfinite(mask_diam_grid)):
        raise ValueError(
            f"active phasemask diameter grid contains non-finite values "
            f"for mask {getattr(pm, 'name', '<unknown>')!r}"
        )

    if np.any(mask_diam_grid <= 0):
        raise ValueError(
            f"active phasemask diameter grid contains non-positive values "
            f"for mask {getattr(pm, 'name', '<unknown>')!r}"
        )

    pm.cached_wavelengths_m = wavelengths_m.copy()
    pm.cached_theta_rad = theta_grid
    pm.cached_mask_diam_lambdaD = mask_diam_grid

    return pm


# ============================================================
# Helpers
# ============================================================


def get_git_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    )


def convert_12x12_to_140(arr):
    """
    Convert a 12x12 DM command image with four unused corners into the
    140-element BMC multi-3.5 command vector.
    """
    arr = np.asarray(arr)

    if arr.shape != (12, 12):
        raise ValueError("Input must be a 12x12 array.")

    flat = arr.flatten()
    corner_indices = [0, 11, 132, 143]
    return np.delete(flat, corner_indices)


# def get_cfg_value(ns, name, default=None):
#     return getattr(ns, name, default) if ns is not None else default
def get_cfg_value(ns, name, default=None):
    if ns is None:
        return default

    if isinstance(ns, dict):
        return ns.get(name, default)

    return getattr(ns, name, default)


def get_photon_flux_density_from_config(zwfs):
    """
    Return photons / second / wave-space-pixel / nm for the simulator source.

    Preferred new JSON config field:
        zwfs.source.photons_per_second_per_pixel_per_nm

    Fallback:
        1.0e5 photons / s / pixel / nm

    This keeps the simulator independent of the old pyZELDA/throughput/magnitude
    calculation path.
    """
    source_cfg = getattr(zwfs, "source", None)

    return float(
        get_cfg_value(
            source_cfg,
            "photons_per_second_per_pixel_per_nm",
            1.0e5,
        )
    )


def apply_source_profile_fixed_grid(zwfs, profile_name, source_profiles):
    """
    Apply source flux and blackbody temperature while keeping the existing
    wavelength grid fixed.

    This updates:
        zwfs.source.photons_per_second_per_pixel_per_nm
        zwfs.source.active_profile
        zwfs.spectrum.weights
        zwfs.spectrum.weights_normalized
        zwfs.spectrum.weights_nm
        zwfs.stellar.spectrum.temperature_K

    It does NOT change:
        zwfs.spectrum.wavelengths

    Therefore it does NOT invalidate the active_phasemask wavelength cache.
    """
    if source_profiles is None:
        raise ValueError("source_profiles is None")

    if isinstance(source_profiles, dict):
        profiles = source_profiles
    else:
        profiles = vars(source_profiles)

    if profile_name not in profiles:
        raise ValueError(
            f"Unknown source profile {profile_name!r}. "
            f"Available profiles: {sorted(profiles.keys())}"
        )

    profile = profiles[profile_name]

    flux_model = str(get_cfg_value(profile, "flux_model", "")).strip()
    if flux_model != "photon_density":
        raise ValueError(
            f"Unsupported flux_model={flux_model!r} for source profile "
            f"{profile_name!r}. Only 'photon_density' is supported."
        )

    flux_density = float(
        get_cfg_value(
            profile,
            "photons_per_second_per_pixel_per_nm",
            np.nan,
        )
    )

    if not np.isfinite(flux_density) or flux_density < 0:
        raise ValueError(
            f"Invalid photons_per_second_per_pixel_per_nm for profile "
            f"{profile_name!r}: {flux_density!r}"
        )

    temperature_K = float(get_cfg_value(profile, "temperature_K", np.nan))

    if not np.isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError(
            f"Invalid temperature_K for profile {profile_name!r}: " f"{temperature_K!r}"
        )

    # Update flux source.
    if not hasattr(zwfs, "source") or zwfs.source is None:
        from types import SimpleNamespace

        zwfs.source = SimpleNamespace()

    zwfs.source.photons_per_second_per_pixel_per_nm = flux_density
    zwfs.source.active_profile = profile_name
    zwfs.source.flux_model = flux_model

    # Keep existing wavelength grid fixed.
    wavelengths_m = np.asarray(zwfs.spectrum.wavelengths, dtype=float)

    # Get weighting mode from current stellar.spectrum if present.
    stellar_spectrum = getattr(getattr(zwfs, "stellar", None), "spectrum", None)
    weighting = get_cfg_value(stellar_spectrum, "weighting", "photon")
    normalize = get_cfg_value(stellar_spectrum, "normalize", "sum")

    # Recompute blackbody weights on the fixed wavelength grid.
    weights = spec.blackbody_weights(
        wavelengths_m,
        temperature_K=temperature_K,
        weighting=weighting,
    )

    weights_normalized = spec.normalize_weights(
        weights,
        normalize=normalize,
    )

    zwfs.spectrum.weights = weights
    zwfs.spectrum.weights_normalized = weights_normalized

    # Preserve current effective bandwidth convention.
    bandwidth_nm = float(getattr(zwfs.stellar, "bandwidth", 1.0))
    zwfs.spectrum.weights_nm = weights_normalized * bandwidth_nm

    # Keep stellar metadata informative.
    if stellar_spectrum is not None:
        if isinstance(stellar_spectrum, dict):
            stellar_spectrum["temperature_K"] = temperature_K
        else:
            stellar_spectrum.temperature_K = temperature_K

    zwfs.stellar.active_source_profile = profile_name

    return zwfs


def get_internal_opd_from_config(zwfs):
    """
    Build static/internal OPD from the new JSON config.

    Currently supports:
        internal_aberrations.parabolic_scratches

    If no internal-aberrations section is present, returns zeros.
    """
    opd_internal = np.zeros_like(zwfs.grid.pupil_mask, dtype=float)
    dx = zwfs.grid.D / zwfs.grid.N

    ia = getattr(zwfs, "internal_aberrations", None)
    if ia is None or not bool(get_cfg_value(ia, "enabled", False)):
        return opd_internal

    scratches = getattr(ia, "parabolic_scratches", None)
    if scratches is not None and bool(get_cfg_value(scratches, "enabled", False)):
        width_list = None

        if hasattr(scratches, "width_list"):
            width_list = list(scratches.width_list)
        elif hasattr(scratches, "width_list_dx_factor"):
            width_list = [float(v) * dx for v in scratches.width_list_dx_factor]

        if width_list is None:
            width_list = [2.0 * dx]

        opd_internal = util.apply_parabolic_scratches(
            opd_internal,
            dx=dx,
            dy=dx,
            list_a=list(get_cfg_value(scratches, "list_a", [0.1])),
            list_b=list(get_cfg_value(scratches, "list_b", [0.0])),
            list_c=list(get_cfg_value(scratches, "list_c", [-2.0])),
            width_list=width_list,
            depth_list=list(get_cfg_value(scratches, "depth_list_m", [100e-9])),
        )

    return zwfs.grid.pupil_mask * opd_internal


# def set_phase_mask_state(zwfs, mask_inserted, original_optics_state):
#     """
#     Apply phase-mask in/out state to the configured analytic propagation path.

#     For mask out, force theta=0 and theta_mode='constant'. This is robust even
#     when the configured mask normally uses theta_mode='physical_depth'.
#     """
#     if mask_inserted:
#         zwfs.optics.theta = original_optics_state["theta"]
#         zwfs.optics.theta_mode = original_optics_state["theta_mode"]
#     else:
#         zwfs.optics.theta = 0.0
#         zwfs.optics.theta_mode = "constant"


def set_phase_mask_state(
    zwfs,
    mask_name,
    original_optics_state,
    phasemask_properties,
    verbose=False,
):
    """
    Apply current MDS phase-mask state to zwfs.optics.

    MDS convention:
        ""   -> mask out
        "J3" -> apply named physical mask from phasemask_properties.json

    New convention:
        zwfs.optics.active_phasemask stores the physical/chromatic model.
        zwfs.optics.theta and zwfs.optics.mask_diam are synced to wvl0 values
        for backwards compatibility.
    """
    mask_name = "" if mask_name is None else str(mask_name).strip()

    restore_optics_state(zwfs.optics, original_optics_state)

    # ----------------------------
    # Mask out
    # ----------------------------
    if mask_name == "":
        zwfs.optics.theta = 0.0
        zwfs.optics.theta_mode = "constant"
        zwfs.optics.current_mask = ""

        # Do not leave a physical active mask in place when mask is out.
        if hasattr(zwfs.optics, "active_phasemask"):
            delattr(zwfs.optics, "active_phasemask")

        if verbose:
            print("[PHASEMASK] mask out: theta=0", flush=True)

        return False

    # ----------------------------
    # Named mask
    # ----------------------------
    masks = phasemask_properties.get("masks", {})
    if mask_name not in masks:
        # Backwards-compatible fallback: use whatever was in baldr_config.json.
        zwfs.optics.current_mask = mask_name

        if verbose:
            print(
                f"[PHASEMASK] No entry for {mask_name!r}; "
                "using baseline baldr_config optics.",
                flush=True,
            )

        return True

    pm = spec.normalise_phasemask_entry(
        mask_name=mask_name,
        mask_entry=masks[mask_name],
        optics=zwfs.optics,
    )

    zwfs.optics.active_phasemask = pm
    precompute_active_phasemask_spectrum_cache(zwfs)
    zwfs.optics.current_mask = mask_name

    # if verbose and hasattr(pm, "cached_theta_rad"):

    #     print(
    #         f"[PHASEMASK CACHE] {mask_name}: "
    #         f"{len(pm.cached_wavelengths_m)} wavelengths cached",
    #         flush=True,
    #     )

    # Legacy compatibility fields at wvl0.
    zwfs.optics.theta = pm.theta_rad_wvl0
    zwfs.optics.theta_mode = "constant"

    zwfs.optics.mask_diam = pm.mask_diam_lambdaD_wvl0
    zwfs.optics.mask_diam_mode = "lambda_over_D"

    # Useful explicit diagnostics.
    zwfs.optics.theta_rad_wvl0 = pm.theta_rad_wvl0
    zwfs.optics.mask_diam_lambdaD_wvl0 = pm.mask_diam_lambdaD_wvl0

    if verbose:
        print(
            f"[PHASEMASK] active={mask_name} "
            f"theta_wvl0={pm.theta_rad_wvl0:.6g} "
            f"mask_diam_wvl0={pm.mask_diam_lambdaD_wvl0:.6g} "
            f"theta_model={pm.theta_model} "
            f"diameter_model={pm.diameter_model}",
            flush=True,
        )

    return True


def read_dm_command(dm_shm):
    """
    Read the combined 12x12 simulated DM shared-memory image and convert it to
    the Baldr 140-vector command.

    The simulator currently treats this shared-memory command as the full DM
    command used by the optical model, preserving the behaviour of the previous
    script.
    """
    dmcmd_2d = dm_shm.get_data()
    return convert_12x12_to_140(dmcmd_2d)


# ============================================================
# Dynamic atmosphere / AO / scintillation helpers
# ============================================================


def scale_r0_to_wavelength(r0_ref_m, wavelength_m, reference_wavelength_m):
    """Scale Fried parameter from reference_wavelength_m to wavelength_m."""
    return float(r0_ref_m) * (float(wavelength_m) / float(reference_wavelength_m)) ** (
        6.0 / 5.0
    )


def make_first_stage_ao_basis(zwfs, n_modes_removed):
    """
    Build a Zernike basis on the BaldrApp wave-space grid for first-stage AO.

    basis[0] is the filled disk support used by bldr.first_stage_ao(...) for
    modal projection/removal.
    """
    nterms = max(50, int(n_modes_removed) + 5)

    basis_cropped = ztools.zernike.zernike_basis(
        nterms=nterms,
        npix=int(zwfs.grid.N),
    )

    basis_template = np.zeros_like(zwfs.grid.pupil_mask, dtype=float)

    basis = np.array(
        [
            util.insert_concentric(np.nan_to_num(b, nan=0.0), basis_template)
            for b in basis_cropped
        ]
    )

    return basis


def upsample_by_factor(arr, factor):
    """Nearest-neighbour upsample of a square 2D array by an integer factor."""
    arr = np.asarray(arr)
    return np.repeat(np.repeat(arr, int(factor), axis=0), int(factor), axis=1)


def upsample_to_size(arr, target_size):
    """Nearest-neighbour upsample/pad/crop a square 2D array to target_size."""
    arr = np.asarray(arr)

    if arr.shape[0] == target_size:
        return arr

    factor = max(1, int(target_size) // int(arr.shape[0]))
    out = upsample_by_factor(arr, factor)

    if out.shape[0] < target_size:
        pad = target_size - out.shape[0]
        out = np.pad(out, ((0, pad), (0, pad)), mode="edge")

    if out.shape[0] > target_size:
        out = out[:target_size, :target_size]

    return out


def update_scintillation_amplitude(
    scint_screen,
    zwfs,
    pxl_scale,
    wavelength_m,
    rows_per_frame=1,
    propagation_distance_m=10000.0,
    renormalize_mean_intensity=True,
    clip_negative_intensity=True,
):
    """
    Evolve a high-altitude phase screen and return scintillation amplitude.

    This uses baldrapp.common.phasescreens.angularSpectrum, not aotools.
    The returned quantity is field amplitude, so amp_input is multiplied by it.
    """
    for _ in range(int(rows_per_frame)):
        scint_screen.add_row()

    wavefront = np.exp(1j * scint_screen.scrn)

    propagated = ps.angularSpectrum(
        inputComplexAmp=wavefront,
        z=float(propagation_distance_m),
        wvl=float(wavelength_m),
        inputSpacing=float(pxl_scale),
        outputSpacing=float(pxl_scale),
    )

    intensity = np.abs(propagated) ** 2

    if clip_negative_intensity:
        intensity = np.clip(intensity, 0.0, np.inf)

    amp = np.sqrt(intensity)
    amp = upsample_to_size(amp, zwfs.grid.pupil_mask.shape[0])

    if renormalize_mean_intensity:
        pupil = zwfs.grid.pupil_mask.astype(bool)
        mean_intensity = np.mean(amp[pupil] ** 2)
        if mean_intensity > 0:
            amp = amp / np.sqrt(mean_intensity)

    return amp


def init_dynamic_atmosphere_for_beam(zwfs):
    """
    Initialise Kolmogorov phase, first-stage AO basis, and scintillation state
    from the JSON-derived zwfs namespace.
    """
    wvl0 = float(zwfs.optics.wvl0)
    dx = float(zwfs.grid.D) / float(zwfs.grid.N)

    atm_cfg = getattr(zwfs, "atmosphere", None)
    phase_cfg = getattr(atm_cfg, "phase", None)
    scint_cfg = getattr(atm_cfg, "scintillation", None)
    ao_cfg = getattr(zwfs, "first_stage_ao", None)

    phase_enabled = bool(get_cfg_value(phase_cfg, "enabled", False))
    ao_enabled = bool(get_cfg_value(ao_cfg, "enabled", False))
    scint_enabled = bool(get_cfg_value(scint_cfg, "enabled", False))

    phase_screen = None
    if phase_enabled:
        r0_500_m = float(get_cfg_value(phase_cfg, "r0_500_m", 0.126))
        reference_wavelength_m = float(
            get_cfg_value(phase_cfg, "reference_wavelength_m", 500e-9)
        )
        r0_wvl_m = scale_r0_to_wavelength(
            r0_500_m,
            wvl0,
            reference_wavelength_m,
        )

        phase_screen = ps.PhaseScreenKolmogorov(
            nx_size=int(zwfs.grid.dim),
            pixel_scale=dx,
            r0=r0_wvl_m,
            L0=float(get_cfg_value(phase_cfg, "L0_m", 25.0)),
            random_seed=get_cfg_value(phase_cfg, "random_seed", None),
        )

    n_modes_removed = int(get_cfg_value(ao_cfg, "Nmodes_removed", 0))
    ao_basis = None
    if phase_enabled and ao_enabled and n_modes_removed > 0:
        ao_basis = make_first_stage_ao_basis(zwfs, n_modes_removed)

    scint_screen = None
    if scint_enabled:
        scint_r0_500_m = float(get_cfg_value(scint_cfg, "r0_500_m", 0.126))
        scint_ref_wvl_m = float(
            get_cfg_value(scint_cfg, "reference_wavelength_m", 500e-9)
        )
        scint_r0_wvl_m = scale_r0_to_wavelength(
            scint_r0_500_m,
            wvl0,
            scint_ref_wvl_m,
        )

        scint_screen = ps.PhaseScreenVonKarman(
            nx_size=int(zwfs.grid.dim),
            pixel_scale=dx,
            r0=scint_r0_wvl_m,
            L0=float(get_cfg_value(scint_cfg, "L0_m", 25.0)),
            random_seed=get_cfg_value(scint_cfg, "random_seed", None),
        )

    return {
        "dx": dx,
        "phase_enabled": phase_enabled,
        "phase_screen": phase_screen,
        "phase_rows_per_frame": int(get_cfg_value(phase_cfg, "rows_per_frame", 1)),
        "phase_scaling_factor": float(
            get_cfg_value(phase_cfg, "phase_scaling_factor", 1.0)
        ),
        "ao_enabled": ao_enabled,
        "ao_basis": ao_basis,
        "n_modes_removed": n_modes_removed,
        "ao_phase_scaling_factor": float(
            get_cfg_value(ao_cfg, "phase_scaling_factor", 1.0)
        ),
        "scint_enabled": scint_enabled,
        "scint_screen": scint_screen,
        "scint_rows_per_frame": int(get_cfg_value(scint_cfg, "rows_per_frame", 1)),
        "scint_propagation_distance_m": float(
            get_cfg_value(scint_cfg, "propagation_distance_m", 10000.0)
        ),
        "scint_renormalize_mean_intensity": bool(
            get_cfg_value(scint_cfg, "renormalize_mean_intensity", True)
        ),
        "scint_clip_negative_intensity": bool(
            get_cfg_value(scint_cfg, "clip_negative_intensity", True)
        ),
    }


def generate_atmosphere_and_source_for_beam(zwfs, amp_input_0, dynamic_state):
    """
    Evolve configured atmosphere/scintillation and return:
        opd_input : OPD in metres after optional first-stage AO
        amp_input : source amplitude after optional scintillation

    The Baldr DM is not included here. bldr.get_frame_configured(...) adds the
    current DM command internally.
    """
    pupil = zwfs.grid.pupil_mask
    wvl0 = float(zwfs.optics.wvl0)

    if dynamic_state["phase_enabled"]:
        scrn = dynamic_state["phase_screen"]
        for _ in range(int(dynamic_state["phase_rows_per_frame"])):
            scrn.add_row()

        phase_scaling = float(dynamic_state["phase_scaling_factor"])

        if (
            dynamic_state["ao_enabled"]
            and dynamic_state["ao_basis"] is not None
            and dynamic_state["n_modes_removed"] > 0
        ):
            phase_after_ao = bldr.first_stage_ao(
                atm_scrn=scrn,
                Nmodes_removed=dynamic_state["n_modes_removed"],
                basis=dynamic_state["ao_basis"],
                phase_scaling_factor=(
                    phase_scaling * float(dynamic_state["ao_phase_scaling_factor"])
                ),
                return_reconstructor=False,
            )
        else:
            phase_after_ao = pupil * phase_scaling * scrn.scrn

        opd_input = pupil * (wvl0 / (2.0 * np.pi)) * phase_after_ao
    else:
        opd_input = np.zeros_like(pupil, dtype=float)

    if dynamic_state["scint_enabled"]:
        amp_scint = update_scintillation_amplitude(
            scint_screen=dynamic_state["scint_screen"],
            zwfs=zwfs,
            pxl_scale=dynamic_state["dx"],
            wavelength_m=wvl0,
            rows_per_frame=dynamic_state["scint_rows_per_frame"],
            propagation_distance_m=dynamic_state["scint_propagation_distance_m"],
            renormalize_mean_intensity=dynamic_state[
                "scint_renormalize_mean_intensity"
            ],
            clip_negative_intensity=dynamic_state["scint_clip_negative_intensity"],
        )
        amp_input = amp_input_0 * amp_scint
    else:
        amp_input = amp_input_0

    return opd_input, amp_input


def center_crop_or_pad_to_shape(arr, target_shape, fill_value=0.0):
    """
    Center-crop or center-pad a 2D array to target_shape = (ny, nx).
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("center_crop_or_pad_to_shape expects a 2D array.")

    ty, tx = map(int, target_shape)
    sy, sx = arr.shape
    out = np.full((ty, tx), fill_value, dtype=arr.dtype)

    src_y0 = max(0, (sy - ty) // 2)
    src_x0 = max(0, (sx - tx) // 2)
    src_y1 = min(sy, src_y0 + ty)
    src_x1 = min(sx, src_x0 + tx)

    dst_y0 = max(0, (ty - sy) // 2)
    dst_x0 = max(0, (tx - sx) // 2)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


# ============================================================
# Runtime simulator-control ZMQ helpers
# ============================================================


def make_initial_sim_control():
    """
    Live runtime-control state.

    The JSON config remains the startup/default configuration. These fields are
    intentionally runtime overrides for a future GUI/control client.
    """
    return {
        "mode": "onsky",
        "phase_enabled": True,
        "ao_enabled": True,
        "scint_enabled": True,
        "edge_offset_m": None,
        "coldstop_x_offset_m": None,
        "coldstop_y_offset_m": None,
        "pupil_misconjugation_m": None,
        "sleep_time_s": None,
    }
    # return {
    #     "mode": "onsky",              # "onsky" or "internal"
    #     "phase_enabled": True,         # used if atmosphere generation is present
    #     "ao_enabled": True,            # used if atmosphere generation is present
    #     "scint_enabled": True,         # used if atmosphere generation is present
    #     "edge_offset_m": None,         # None => JSON/default Fresnel value
    #     "coldstop_x_offset_m": None,   # None => JSON/default Fresnel value
    #     "coldstop_y_offset_m": None,   # None => JSON/default Fresnel value
    #     "sleep_time_s": None,          # None => JSON/default value
    # }


def desired_source_profile_from_mode(sim_control):
    mode = str(sim_control["mode"]).lower()

    if mode == "internal":
        return "internal"

    if mode == "onsky":
        return "onsky"

    raise ValueError(f"Unknown simulator mode: {mode!r}")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def make_control_reply(ok, message, sim_control, runtime_status=None):
    reply = {
        "ok": bool(ok),
        "message": str(message),
        "control": sim_control,
    }
    if runtime_status is not None:
        reply["runtime_status"] = runtime_status
    return json.dumps(json_safe(reply))


def bool_from_token(token):
    token = str(token).strip().lower()
    if token in ["1", "true", "t", "yes", "y", "on", "enable", "enabled"]:
        return True
    if token in ["0", "false", "f", "no", "n", "off", "disable", "disabled"]:
        return False
    raise ValueError(f"Cannot parse boolean token: {token!r}")


def apply_fresnel_control_to_all_beams(zwfs_ns, original_fresnel_state, sim_control):
    """Apply live Fresnel alignment overrides to every beam namespace."""
    for beam, zwfs in zwfs_ns.items():
        if not hasattr(zwfs, "fresnel_relay"):
            continue

        fr = zwfs.fresnel_relay
        original = original_fresnel_state[beam]

        fr.edge_offset = (
            original["edge_offset"]
            if sim_control["edge_offset_m"] is None
            else float(sim_control["edge_offset_m"])
        )
        fr.coldstop_x_offset = (
            original["coldstop_x_offset"]
            if sim_control["coldstop_x_offset_m"] is None
            else float(sim_control["coldstop_x_offset_m"])
        )
        fr.coldstop_y_offset = (
            original["coldstop_y_offset"]
            if sim_control["coldstop_y_offset_m"] is None
            else float(sim_control["coldstop_y_offset_m"])
        )
        fr.pupil_misconjugation = (
            original["pupil_misconjugation"]
            if sim_control["pupil_misconjugation_m"] is None
            else float(sim_control["pupil_misconjugation_m"])
        )

        if hasattr(fr, "z_focus_to_detector_nominal"):
            fr.z_focus_to_detector = float(fr.z_focus_to_detector_nominal) + float(
                fr.pupil_misconjugation
            )


def runtime_dynamic_state_from_control(dynamic_state, sim_control):
    """
    Return a shallow runtime view of a dynamic-atmosphere state dictionary.
    This is used only if the active script defines generate_atmosphere... and
    dynamic_atmosphere_state. It is harmless otherwise.
    """
    state = dynamic_state.copy()

    if sim_control["mode"] == "internal":
        state["phase_enabled"] = False
        state["ao_enabled"] = False
        state["scint_enabled"] = False
        return state

    state["phase_enabled"] = bool(
        dynamic_state.get("phase_enabled", False) and sim_control["phase_enabled"]
    )
    state["ao_enabled"] = bool(
        dynamic_state.get("ao_enabled", False)
        and sim_control["ao_enabled"]
        and state["phase_enabled"]
    )
    state["scint_enabled"] = bool(
        dynamic_state.get("scint_enabled", False) and sim_control["scint_enabled"]
    )

    return state


def handle_control_command(command, sim_control, runtime_status=None):
    """
    Supported commands:
      status
      help
      mode onsky|internal
      phase on|off
      ao on|off
      scint on|off
      preset onsky|internal|phase_only|scint_only
      set edge_offset_mm <x>
      set edge_offset_m <x>
      set coldstop_x_um <x>
      set coldstop_y_um <y>
      set coldstop_offset_um <x> <y>
      set sleep_time_s <s>
      reset_offsets
    """
    command = str(command).strip()
    tokens = command.split()

    if not tokens:
        return make_control_reply(False, "Empty command", sim_control, runtime_status)

    cmd = tokens[0].lower()

    try:
        if cmd in ["help", "?"]:
            return make_control_reply(
                True,
                "Commands: status; mode onsky|internal; phase/ao/scint on|off; "
                "preset onsky|internal|phase_only|scint_only; "
                "set edge_offset_mm <x>; set coldstop_x_um <x>; "
                "set coldstop_y_um <y>; set coldstop_offset_um <x> <y>; "
                "set pupil_misconjugation_mm <z>; "
                "set sleep_time_s <s>; reset_offsets",
                sim_control,
                runtime_status,
            )

        if cmd == "status":
            return make_control_reply(True, "status", sim_control, runtime_status)

        if cmd == "mode":
            if len(tokens) != 2 or tokens[1].lower() not in ["onsky", "internal"]:
                return make_control_reply(
                    False, "Usage: mode onsky|internal", sim_control, runtime_status
                )
            sim_control["mode"] = tokens[1].lower()
            return make_control_reply(
                True, f"mode set to {sim_control['mode']}", sim_control, runtime_status
            )

        if cmd in ["phase", "ao", "scint"]:
            if len(tokens) != 2:
                return make_control_reply(
                    False, f"Usage: {cmd} on|off", sim_control, runtime_status
                )
            key = f"{cmd}_enabled"
            sim_control[key] = bool_from_token(tokens[1])
            return make_control_reply(
                True, f"{key} set to {sim_control[key]}", sim_control, runtime_status
            )

        if cmd == "preset":
            if len(tokens) != 2:
                return make_control_reply(
                    False,
                    "Usage: preset onsky|internal|phase_only|scint_only",
                    sim_control,
                    runtime_status,
                )
            preset = tokens[1].lower()
            if preset == "onsky":
                sim_control["mode"] = "onsky"
                sim_control["phase_enabled"] = True
                sim_control["ao_enabled"] = True
                sim_control["scint_enabled"] = True
            elif preset == "internal":
                sim_control["mode"] = "internal"
            elif preset == "phase_only":
                sim_control["mode"] = "onsky"
                sim_control["phase_enabled"] = True
                sim_control["ao_enabled"] = True
                sim_control["scint_enabled"] = False
            elif preset == "scint_only":
                sim_control["mode"] = "onsky"
                sim_control["phase_enabled"] = False
                sim_control["ao_enabled"] = False
                sim_control["scint_enabled"] = True
            else:
                return make_control_reply(
                    False, f"Unknown preset: {preset}", sim_control, runtime_status
                )
            return make_control_reply(
                True, f"preset set to {preset}", sim_control, runtime_status
            )

        if cmd == "reset_offsets":
            sim_control["edge_offset_m"] = None
            sim_control["coldstop_x_offset_m"] = None
            sim_control["coldstop_y_offset_m"] = None
            sim_control["pupil_misconjugation_m"] = None
            return make_control_reply(
                True,
                "Fresnel offsets reset to JSON defaults",
                sim_control,
                runtime_status,
            )

        # if cmd == "reset_offsets":
        #     sim_control["edge_offset_m"] = None
        #     sim_control["coldstop_x_offset_m"] = None
        #     sim_control["coldstop_y_offset_m"] = None
        #     return make_control_reply(True, "Fresnel offsets reset to JSON defaults", sim_control, runtime_status)

        if cmd == "set":
            if len(tokens) < 3:
                return make_control_reply(
                    False, "Usage: set <parameter> <value>", sim_control, runtime_status
                )

            key = tokens[1].lower()

            if key == "edge_offset_mm":
                sim_control["edge_offset_m"] = float(tokens[2]) * 1e-3
            elif key == "edge_offset_m":
                sim_control["edge_offset_m"] = float(tokens[2])
            elif key == "coldstop_x_um":
                sim_control["coldstop_x_offset_m"] = float(tokens[2]) * 1e-6
            elif key == "coldstop_y_um":
                sim_control["coldstop_y_offset_m"] = float(tokens[2]) * 1e-6
            elif key == "coldstop_x_m":
                sim_control["coldstop_x_offset_m"] = float(tokens[2])
            elif key == "coldstop_y_m":
                sim_control["coldstop_y_offset_m"] = float(tokens[2])
            elif key == "coldstop_offset_um":
                if len(tokens) != 4:
                    return make_control_reply(
                        False,
                        "Usage: set coldstop_offset_um <x_um> <y_um>",
                        sim_control,
                        runtime_status,
                    )
                sim_control["coldstop_x_offset_m"] = float(tokens[2]) * 1e-6
                sim_control["coldstop_y_offset_m"] = float(tokens[3]) * 1e-6

            elif key == "pupil_misconjugation_mm":
                sim_control["pupil_misconjugation_m"] = float(tokens[2]) * 1e-3
            elif key == "pupil_misconjugation_um":
                sim_control["pupil_misconjugation_m"] = float(tokens[2]) * 1e-6
            elif key == "pupil_misconjugation_m":
                sim_control["pupil_misconjugation_m"] = float(tokens[2])

            elif key == "sleep_time_s":
                sim_control["sleep_time_s"] = float(tokens[2])
            else:
                return make_control_reply(
                    False, f"Unknown set parameter: {key}", sim_control, runtime_status
                )

            return make_control_reply(True, f"set {key}", sim_control, runtime_status)

        return make_control_reply(
            False, f"Unknown command: {command}", sim_control, runtime_status
        )

    except Exception as exc:
        return make_control_reply(
            False,
            f"Error handling command {command!r}: {exc}",
            sim_control,
            runtime_status,
        )


def poll_control_socket(control_socket, sim_control, runtime_status=None):
    """Drain all pending simulator-control commands without blocking."""

    while True:
        try:
            command = control_socket.recv_string(flags=zmq.NOBLOCK)
        except zmq.Again:
            break

        reply = handle_control_command(command, sim_control, runtime_status)
        control_socket.send_string(reply)
        print(f"[SIM_CONTROL] {command} -> {reply}", flush=True)


# ============================================================
# Configuration / initialisation
# ============================================================

root_path = get_git_root()

# New generic BaldrApp JSON config. This should include:
#   grid, optics, dm, stellar.spectrum, fresnel_relay, detector, source, etc.
config_path = (
    root_path / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"
)

# Initialise one independent ZWFS namespace per beam.
# The configured frame dispatcher will automatically use:
#   - polychromatic propagation if zwfs.spectrum.enabled and n_wvl > 1
#   - Fresnel relay propagation if zwfs.fresnel_relay.enabled is True
zwfs_ns: Dict[int, SimpleNamespace] = {}
amp_input = {}
opd_internal = {}
original_optics_state = {}
original_fresnel_state = {}
dynamic_atmosphere_state = {}

for beam in [1, 2, 3, 4]:
    zwfs = bldr.init_zwfs_from_json(config_path)

    # Optional per-simulator override retained from the old script.
    zwfs.dm.actuator_coupling_factor = 0.7

    # Recompute DM registration after changing actuator coupling factor.
    # This keeps act_sigma_wavesp consistent with the updated coupling.
    zwfs = bldr.update_dm_registration_wavespace(
        zwfs.dm2wavespace_registration.dm_to_wavesp_transform_matrix,
        zwfs,
    )

    flux_density = get_photon_flux_density_from_config(zwfs)
    amp_input[beam] = np.sqrt(flux_density) * zwfs.grid.pupil_mask

    opd_internal[beam] = get_internal_opd_from_config(zwfs)

    dynamic_atmosphere_state[beam] = init_dynamic_atmosphere_for_beam(zwfs)

    # original_optics_state[beam] = {
    #     "theta": float(zwfs.optics.theta),
    #     "theta_mode": str(getattr(zwfs.optics, "theta_mode", "constant")),
    # }

    original_optics_state[beam] = dict(vars(zwfs.optics))

    if hasattr(zwfs, "fresnel_relay"):
        original_fresnel_state[beam] = {
            "edge_offset": float(getattr(zwfs.fresnel_relay, "edge_offset", 0.0)),
            "coldstop_x_offset": float(
                getattr(zwfs.fresnel_relay, "coldstop_x_offset", 0.0)
            ),
            "coldstop_y_offset": float(
                getattr(zwfs.fresnel_relay, "coldstop_y_offset", 0.0)
            ),
            "pupil_misconjugation": float(
                getattr(zwfs.fresnel_relay, "pupil_misconjugation", 0.0)
            ),
        }
        # original_fresnel_state[beam] = {
        #     "edge_offset": float(getattr(zwfs.fresnel_relay, "edge_offset", 0.0)),
        #     "coldstop_x_offset": float(getattr(zwfs.fresnel_relay, "coldstop_x_offset", 0.0)),
        #     "coldstop_y_offset": float(getattr(zwfs.fresnel_relay, "coldstop_y_offset", 0.0)),
        # }
    else:
        original_fresnel_state[beam] = {
            "edge_offset": 0.0,
            "coldstop_x_offset": 0.0,
            "coldstop_y_offset": 0.0,
            "pupil_misconjugation": 0.0,
        }
        # original_fresnel_state[beam] = {
        #     "edge_offset": 0.0,
        #     "coldstop_x_offset": 0.0,
        #     "coldstop_y_offset": 0.0,
        # }

    zwfs_ns[beam] = zwfs


default_tel = 1
use_pyZelda = False
default_zwfs = zwfs_ns[default_tel]
if default_zwfs is None:
    raise RuntimeError(f"zwfs_ns[{default_tel}] not initialised")

print("\n=== Baldr simulator optical configuration ===")
print("config_path:", config_path)
print("grid N:", default_zwfs.grid.N)
print("grid dim:", default_zwfs.grid.dim)
print("detector binning:", default_zwfs.detector.binning)
print(
    "stellar.bandwidth [nm]:", getattr(default_zwfs.stellar, "bandwidth", None)
)
print("spectrum wavelengths [um]:", default_zwfs.spectrum.wavelengths * 1e6)
print("sum spectrum weights_nm:", np.sum(default_zwfs.spectrum.weights_nm))
print("fresnel enabled:", getattr(default_zwfs.fresnel_relay, "enabled", None))
print(
    "atmosphere phase enabled:", dynamic_atmosphere_state[default_tel]["phase_enabled"]
)
print("first-stage AO enabled:", dynamic_atmosphere_state[default_tel]["ao_enabled"])
print(
    "first-stage AO modes removed:",
    dynamic_atmosphere_state[default_tel]["n_modes_removed"],
)
print("scintillation enabled:", dynamic_atmosphere_state[default_tel]["scint_enabled"])


# ============================================================
# Shared-memory layout
# ============================================================

# Prefer split JSON from simulator_runtime config if present.
runtime_cfg = getattr(default_zwfs, "simulator_runtime", None)
shared_memory_cfg = getattr(runtime_cfg, "shared_memory", None)

source_profiles = getattr(default_zwfs, "source_profiles", None)

if source_profiles is None:
    print(
        "[SOURCE] No source_profiles block found; using existing source/stellar config.",
        flush=True,
    )
else:
    if not isinstance(source_profiles, dict):
        source_profiles = vars(source_profiles)

    print(
        f"[SOURCE] Available source profiles: {sorted(source_profiles.keys())}",
        flush=True,
    )

split_filename = get_cfg_value(
    shared_memory_cfg,
    "split_json",
    str(root_path / "baldrapp/apps/paranal_simulator/fake_configs/cred1_split.json"),
)
split_filename = Path(split_filename)
if not split_filename.is_absolute():
    split_filename = root_path / split_filename

with open(split_filename, "r") as file:
    split_dict = json.load(file)

# nrs should ideally be read from the camera server/config.
nrs = int(get_cfg_value(runtime_cfg, "nrs", 5))
global_frame_size = [256, 320, nrs]

baldr_frame_sizes = []
baldr_frame_corners = []

for beam in [1, 2, 3, 4]:
    nx = split_dict[f"baldr{beam}"]["xsz"]
    ny = split_dict[f"baldr{beam}"]["ysz"]
    # NumPy image shape is (ny, nx), while split JSON uses xsz/ysz.
    baldr_frame_sizes.append([ny, nx])

    x0 = split_dict[f"baldr{beam}"]["x0"]
    y0 = split_dict[f"baldr{beam}"]["y0"]
    baldr_frame_corners.append([x0, y0])


# ============================================================
# SHM initialisation
# ============================================================

baldr_sub_shms = {}
dm_shms = {}

f_cred1_global = get_cfg_value(
    shared_memory_cfg,
    "cred1_global",
    "/dev/shm/cred1.im.shm",
)

global_frame_shm = shm(f_cred1_global, nosem=False)
global_frame_shm.set_data(np.zeros(global_frame_size).astype(dtype=np.uint16))

for ct, beam in enumerate([1, 2, 3, 4]):
    f_baldr = get_cfg_value(
        shared_memory_cfg,
        "baldr_template",
        "/dev/shm/baldr{beam}.im.shm",
    ).format(beam=beam)

    f_dm = get_cfg_value(
        shared_memory_cfg,
        "dm_template",
        "/dev/shm/dm{beam}.im.shm",
    ).format(beam=beam)

    ss = shm(f_baldr, nosem=False)
    ss.set_data(np.zeros(baldr_frame_sizes[ct]))
    baldr_sub_shms[beam] = ss

    # Should be running sim_mdm_server.
    # If not, initialise elsewhere as:
    #   shm(f_dm, data=np.zeros([12, 12]), nosem=False)
    dm_shms[beam] = shm(f_dm, nosem=False)


# ============================================================
# Camera / MDS server state
# ============================================================

ctx = zmq.Context()

# Simulator runtime-control socket. A GUI/client can connect here to toggle
# on-sky/internal mode and adjust Fresnel alignment parameters without editing
# the JSON config or restarting the simulator.

sim_control = make_initial_sim_control()

runtime_status = {}
#     "frame": None,
#     "cnt0": None,
#     "cnt1": None,
#     "last_beam1_intensity_sum": None,
#     "last_beam1_subim_sum": None,
#     "source_profile": None,
#     "source_flux_density": None,
#     "source_temperature_K": None,
# }

control_zmq = get_cfg_value(
    getattr(runtime_cfg, "servers", None),
    "sim_control_zmq",
    "tcp://127.0.0.1:6670",
)
control_socket = ctx.socket(zmq.REP)
control_socket.bind(control_zmq)
print("Simulator control ZMQ:", control_zmq, flush=True)

cam_socket = ctx.socket(zmq.REQ)
camera_zmq = get_cfg_value(
    getattr(runtime_cfg, "servers", None),
    "camera_zmq",
    "tcp://127.0.0.1:6667",
)
cam_socket.connect(camera_zmq)
cam_socket.send_string('cli "gain"')
print("Camera reply:", cam_socket.recv_string())

# TODO: query camera server for offset/noise when available.
det_cfg = getattr(default_zwfs, "detector_config", None)
adu_offset = float(get_cfg_value(det_cfg, "adu_offset", 1000.0))
noise_std = float(get_cfg_value(det_cfg, "noise_std_adu", 100.0))
include_shotnoise = bool(get_cfg_value(det_cfg, "include_shotnoise", True))

mds_socket = ctx.socket(zmq.REQ)
mds_zmq = get_cfg_value(
    getattr(runtime_cfg, "servers", None),
    "mds_zmq",
    "tcp://127.0.0.1:5555",
)
mds_socket.connect(mds_zmq)

mds_socket.send_string("on SBB")
print("MDS reply (on):", mds_socket.recv_string())
mds_socket.send_string("off SBB")
print("MDS reply (off):", mds_socket.recv_string())

assert len(dm_shms) == len(baldr_sub_shms)

# Move to the configured phasemask in MDS state.
default_mask = get_cfg_value(
    getattr(runtime_cfg, "phasemask", None),
    "default_mask",
    "J3",
)

for beam in [1, 2, 3, 4]:
    mds_socket.send_string(f"fpm_move {beam} {default_mask}")
    print(mds_socket.recv_string())


# ============================================================
# Phasemask
# ============================================================
phasemask_runtime_cfg = getattr(runtime_cfg, "phasemask", None)

phasemask_properties_path = get_cfg_value(
    phasemask_runtime_cfg,
    "properties_file",
    "baldrapp/apps/paranal_simulator/fake_configs/phasemask_properties.json",
)

phasemask_properties_path = Path(phasemask_properties_path)
if not phasemask_properties_path.is_absolute():
    phasemask_properties_path = root_path / phasemask_properties_path

with open(phasemask_properties_path, "r") as f:
    phasemask_properties = json.load(f)

if "phasemask" in phasemask_properties:
    phasemask_properties = phasemask_properties["phasemask"]

if "masks" not in phasemask_properties:
    raise ValueError("phasemask_properties.json must contain phasemask.masks")

print(
    f"[PHASEMASK] Loaded {len(phasemask_properties['masks'])} masks from "
    f"{phasemask_properties_path}",
    flush=True,
)

# phasemask_properties_path = Path(phasemask_properties_path)
# if not phasemask_properties_path.is_absolute():
#     phasemask_properties_path = root_path / phasemask_properties_path

# phasemask_properties = load_phasemask_properties(phasemask_properties_path)

# print(
#     f"[PHASEMASK] Loaded {len(phasemask_properties['masks'])} mask definitions "
#     f"from {phasemask_properties_path}",
#     flush=True,
# )


# ============================================================
# Main simulator loop
# ============================================================

liveindex = global_frame_shm.mtdata["cnt0"]

zero_opd = {
    beam: np.zeros_like(zwfs_ns[beam].grid.pupil_mask, dtype=float)
    for beam in [1, 2, 3, 4]
}

sleep_time_s = float(get_cfg_value(runtime_cfg, "sleep_time_s", 0.01))

beams_shown = [1]  # [1, 2, 3, 4]

last_mask_name = {}


# ============================================================
# Initial source profile
# ============================================================

last_source_profile = {}

if source_profiles is not None:
    desired_profile = desired_source_profile_from_mode(sim_control)

    for beam in [1, 2, 3, 4]:
        zwfs_ns[beam] = apply_source_profile_fixed_grid(
            zwfs_ns[beam],
            desired_profile,
            source_profiles,
        )

        amp_input[beam] = (
            np.sqrt(get_photon_flux_density_from_config(zwfs_ns[beam]))
            * zwfs_ns[beam].grid.pupil_mask
        )

        last_source_profile[beam] = desired_profile

    profile = source_profiles[desired_profile]

    runtime_status["source_profile"] = desired_profile
    runtime_status["source_flux_density"] = get_photon_flux_density_from_config(
        default_zwfs
    )
    runtime_status["source_temperature_K"] = get_cfg_value(
        profile,
        "temperature_K",
        None,
    )

    print(
        f"[SOURCE] Initial source profile {desired_profile!r}: "
        f"flux_density={runtime_status['source_flux_density']:.6g} "
        f"phot/s/pix/nm, "
        f"T={runtime_status['source_temperature_K']} K",
        flush=True,
    )

while True:
    time_at_start = time.time()
    poll_control_socket(
        control_socket=control_socket,
        sim_control=sim_control,
        runtime_status=runtime_status,
    )

    apply_fresnel_control_to_all_beams(
        zwfs_ns=zwfs_ns,
        original_fresnel_state=original_fresnel_state,
        sim_control=sim_control,
    )

    # ------------------------------------------------------------
    # Check source change
    # ------------------------------------------------------------
    if source_profiles is not None:
        desired_profile = desired_source_profile_from_mode(sim_control)

        for beam in [1, 2, 3, 4]:
            if desired_profile != last_source_profile[beam]:
                old_wavelengths = np.asarray(
                    zwfs_ns[beam].spectrum.wavelengths,
                    dtype=float,
                ).copy()

                zwfs_ns[beam] = apply_source_profile_fixed_grid(
                    zwfs_ns[beam],
                    desired_profile,
                    source_profiles,
                )

                new_wavelengths = np.asarray(
                    zwfs_ns[beam].spectrum.wavelengths,
                    dtype=float,
                )

                # This should not happen under the fixed-grid convention.
                # If it ever does, force the phasemask state/cache to rebuild.
                if not np.array_equal(old_wavelengths, new_wavelengths):
                    last_mask_name[beam] = ""
                    print(
                        f"[SOURCE] wavelength grid changed for beam {beam}; "
                        "forcing phasemask cache rebuild",
                        flush=True,
                    )

                amp_input[beam] = (
                    np.sqrt(get_photon_flux_density_from_config(zwfs_ns[beam]))
                    * zwfs_ns[beam].grid.pupil_mask
                )

                last_source_profile[beam] = desired_profile

                if beam == default_tel:
                    profile = source_profiles[desired_profile]

                    runtime_status["source_profile"] = desired_profile
                    runtime_status["source_flux_density"] = (
                        get_photon_flux_density_from_config(zwfs_ns[beam])
                    )
                    runtime_status["source_temperature_K"] = get_cfg_value(
                        profile,
                        "temperature_K",
                        None,
                    )

                    print(
                        f"[SOURCE] switched to {desired_profile!r}: "
                        f"flux_density="
                        f"{runtime_status['source_flux_density']:.6g} "
                        f"phot/s/pix/nm, "
                        f"T={runtime_status['source_temperature_K']} K",
                        flush=True,
                    )

    # ------------------------------------------------------------
    # Read MDS phase-mask state and update each beam's optical config.
    # Empty fpm_whereami response is interpreted as mask out.
    # ------------------------------------------------------------

    for beam in [1, 2, 3, 4]:
        mds_socket.send_string(f"fpm_whereami {beam}")
        mask_name = mds_socket.recv_string().strip()

        if mask_name != last_mask_name.get(beam, None):
            set_phase_mask_state(
                zwfs=zwfs_ns[beam],
                mask_name=mask_name,
                original_optics_state=original_optics_state[beam],
                phasemask_properties=phasemask_properties,
                verbose=True,
            )

            last_mask_name[beam] = mask_name

    # for beam in [1, 2, 3, 4]:
    #     mds_socket.send_string(f"fpm_whereami {beam}")
    #     mask_name = mds_socket.recv_string().strip()

    #     set_phase_mask_state(
    #         zwfs=zwfs_ns[beam],
    #         mask_name=mask_name,
    #         original_optics_state=original_optics_state[beam],
    #         phasemask_properties=phasemask_properties,
    #         verbose=(beam == 1 and liveindex % 100 == 0),
    #     )
    # # for beam in [1, 2, 3, 4]:
    # #     mds_socket.send_string(f"fpm_whereami {beam}")
    # #     mask = mds_socket.recv_string()

    # #     mask_inserted = (mask != "")
    # #     set_phase_mask_state(
    # #         zwfs_ns[beam],
    # #         mask_inserted=mask_inserted,
    # #         original_optics_state=original_optics_state[beam],
    # #     )

    # ------------------------------------------------------------
    # Camera shared-memory counters.
    # ------------------------------------------------------------
    cnt0 = liveindex
    cnt1 = liveindex % global_frame_size[2]

    # ------------------------------------------------------------
    # Per-beam propagation and subframe SHM update.
    # ------------------------------------------------------------
    for ct, beam in enumerate(beams_shown):

        dmcmd = read_dm_command(dm_shms[beam])

        # Current BaldrApp get_frame_configured/get_frame_fresnel internally
        # adds the OPD from zwfs_ns.dm.current_cmd, so do not also pass the DM
        # OPD through opd_input. This avoids double-counting the DM.
        zwfs_ns[beam].dm.current_cmd = dmcmd.copy()

        # If this script has dynamic atmosphere functions/state defined, use them
        # in on-sky mode. Otherwise fall back to the current internal/static path.
        if (
            sim_control["mode"] == "onsky"
            and "generate_atmosphere_and_source_for_beam" in globals()
            and "dynamic_atmosphere_state" in globals()
        ):
            runtime_dynamic_state = runtime_dynamic_state_from_control(
                dynamic_atmosphere_state[beam],
                sim_control,
            )
            opd_input_runtime, amp_input_runtime = (
                generate_atmosphere_and_source_for_beam(
                    zwfs=zwfs_ns[beam],
                    amp_input_0=amp_input[beam],
                    dynamic_state=runtime_dynamic_state,
                )
            )
        else:
            opd_input_runtime = zero_opd[beam]
            amp_input_runtime = amp_input[beam]

        intensity = bldr.get_frame_configured(
            opd_input=opd_input_runtime,
            amp_input=amp_input_runtime,
            opd_internal=opd_internal[beam],
            zwfs_ns=zwfs_ns[beam],
            detector=zwfs_ns[beam].detector,
            include_shotnoise=include_shotnoise,
            use_pyZelda=use_pyZelda,
        )

        det_cfg_beam = getattr(zwfs_ns[beam], "detector_config", None)
        crop_to_pixels = get_cfg_value(det_cfg_beam, "crop_to_pixels", None)
        crop_after_detection = bool(
            get_cfg_value(det_cfg_beam, "crop_after_detection", False)
        )

        if crop_after_detection and crop_to_pixels is not None:
            intensity = center_crop_or_pad_to_shape(
                intensity,
                tuple(crop_to_pixels),
            )

        # Always force the optical image to match the SHM subframe shape.
        # This keeps the simulator robust if detector crop or cred1_split changes.
        intensity = center_crop_or_pad_to_shape(
            intensity,
            tuple(baldr_frame_sizes[ct]),
        )

        if beam == 1:
            runtime_status["last_beam1_intensity_sum"] = float(np.nansum(intensity))
            # print(
            #     "beam 1 intensity:",
            #     "shape", intensity.shape,
            #     "min", np.nanmin(intensity),
            #     "max", np.nanmax(intensity),
            #     "sum", np.nansum(intensity),
            #     "mode", sim_control["mode"],
            #     "dtype", intensity.dtype,
            #     flush=True,
            # )

        # Detector/background model in ADU-like units. The optical image is
        # inserted concentrically into the configured Baldr subframe.
        subim_tmp = adu_offset + noise_std * np.random.randn(*baldr_frame_sizes[ct])

        subim_tmp += intensity

        # if beam == 1:
        #     runtime_status["last_beam1_subim_sum"] = float(np.nansum(subim_tmp))
        #     print(
        #         "beam 1 subim:",
        #         "shape", subim_tmp.shape,
        #         "min", np.nanmin(subim_tmp),
        #         "max", np.nanmax(subim_tmp),
        #         "sum", np.nansum(subim_tmp),
        #         flush=True,
        #     )

        baldr_sub_shms[beam].set_data(subim_tmp.astype(dtype=np.int32))
        baldr_sub_shms[beam].mtdata["cnt0"] = cnt0
        baldr_sub_shms[beam].mtdata["cnt1"] = cnt1
        baldr_sub_shms[beam].post_sems(1)

    # ------------------------------------------------------------
    # Global CRED1 frame SHM update for display/debugging.
    # ------------------------------------------------------------
    global_frame_shm.mtdata["cnt0"] = cnt0
    global_frame_shm.mtdata["cnt1"] = cnt1

    global_im_tmp = adu_offset + noise_std * np.random.randn(*global_frame_size[:-1])

    for ct, beam in enumerate(beams_shown):
        x0, y0 = baldr_frame_corners[ct]
        ysz, xsz = baldr_frame_sizes[ct]

        global_im_tmp[y0 : y0 + ysz, x0 : x0 + xsz] = (
            baldr_sub_shms[beam].get_data().copy()
        )

    gframe_now = global_frame_shm.get_data().copy()

    runtime_status["frame"] = int(liveindex)
    runtime_status["cnt0"] = int(cnt0)
    runtime_status["cnt1"] = int(cnt1)

    print(f"frame {cnt1}", flush=True)

    gframe_now[cnt1, :, :] = global_im_tmp
    global_frame_shm.set_data(gframe_now.astype(np.uint16))

    sleep_this_frame_s = (
        sleep_time_s
        if sim_control["sleep_time_s"] is None
        else float(sim_control["sleep_time_s"])
    )
    # we don't want to sleep, but we want to limit the loop period
    # time.sleep(sleep_this_frame_s)
    duration_so_far = time.time() - time_at_start
    if duration_so_far < sleep_this_frame_s:
        time.sleep(sleep_this_frame_s - duration_so_far)
    global_frame_shm.post_sems(1)

    liveindex += 1

# ============================================================
# Usage notes
# ============================================================
#
# python3 -m venv venv
# git clone BaldrApp
# cd /to/cloned/directory
#
# git clone pyZelda fork
# cd /to/cloned/directory
# pip install -e .
#
# cd dcs/simulation
# run bash script to start servers:
#
# ./shm_creator_sim
# ./sim_mdm_server
# source venv/bin/activate
# python3 -i simulation/baldr_sim.py
#
# View:
#   shmview /dev/shm/cred1.im.shm
#
# DM GUI:
#   lab-MDM-control &

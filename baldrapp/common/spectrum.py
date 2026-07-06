"""
Spectral sampling utilities for BaldrApp.

This module provides lightweight machinery for polychromatic simulations.
It intentionally does not perform ZWFS propagation, Fresnel propagation,
detector modelling, or DM modelling. Those remain in baldr_core.py and
fresnel.py.

Main responsibilities:

    1. Build a wavelength grid.
    2. Build relative spectral weights.
    3. Normalize spectral weights.
    4. Provide theta(lambda) helpers for chromatic phase-mask behaviour.

The preferred JSON config layout is:

    "stellar": {
        "bandwidth": 300.0,
        "spectrum": {
            "enabled": true,
            "mode": "blackbody",
            "temperature_K": 3500.0,
            "weighting": "photon",
            "wvl_min": 1.5e-6,
            "wvl_max": 1.8e-6,
            "n_wvl": 7
        }
    }

This module only consumes the nested stellar.spectrum dictionary. The scalar
stellar.bandwidth and weights_nm convention are handled in config_helper.py.

The expected use is:

    spectrum_cfg = cfghelp.get_spectrum_config_from_cfg(cfg)

    zwfs_ns.spectrum = spec.derive_spectrum(
        spectrum_cfg,
        default_wvl0=zwfs_ns.optics.wvl0,
    )

    zwfs_ns.spectrum = cfghelp.complete_spectrum_integration_fields(
        zwfs_ns.spectrum
    )

Then Baldr frame-generation functions can loop over:

    zwfs_ns.spectrum.wavelengths
    zwfs_ns.spectrum.weights_normalized
    zwfs_ns.spectrum.weights_nm
"""

from types import SimpleNamespace

import numpy as np

from baldrapp.common import utilities as util


# ============================================================
# Small conversion helpers
# ============================================================

def _as_namespace(obj):
    """
    Convert dictionaries to SimpleNamespace recursively.

    If obj is already a SimpleNamespace, it is returned unchanged.
    Lists are preserved, with any dictionaries inside converted.
    """
    if isinstance(obj, SimpleNamespace):
        return obj

    if isinstance(obj, dict):
        return SimpleNamespace(
            **{k: _as_namespace(v) for k, v in obj.items()}
        )

    if isinstance(obj, list):
        return [_as_namespace(v) for v in obj]

    return obj


def default_monochromatic_spectrum(default_wvl0):
    """
    Return the backwards-compatible monochromatic spectrum.

    This is the default when no spectrum section is provided in the config.
    """
    return SimpleNamespace(
        enabled=False,
        mode="monochromatic",
        weighting="none",
        normalize="sum",
        wavelengths=np.array([float(default_wvl0)], dtype=float),
        weights=np.array([1.0], dtype=float),
        weights_normalized=np.array([1.0], dtype=float),
    )


# ============================================================
# Spectral weighting
# ============================================================

def normalize_weights(weights, normalize="sum"):
    """
    Normalize spectral weights.

    Parameters
    ----------
    weights : array-like
        Non-negative spectral weights.
    normalize : str
        Supported options:

        "sum"
            Normalize so sum(weights) = 1. This is the recommended default
            for polychromatic intensity summation.

        "peak"
            Normalize by peak first, then normalize to unit sum. This gives
            the same final result as "sum" for positive weights, but is kept
            for clarity and future use.

        "none" or "raw"
            Return raw weights unchanged.

    Returns
    -------
    weights_normalized : ndarray
    """
    weights = np.asarray(weights, dtype=float)

    if np.any(weights < 0):
        raise ValueError("Spectrum weights must be non-negative.")

    if np.sum(weights) <= 0:
        raise ValueError("Spectrum weights must have positive sum.")

    normalize = str(normalize).lower()

    if normalize == "sum":
        return weights / np.sum(weights)

    if normalize == "peak":
        w = weights / np.max(weights)
        return w / np.sum(w)

    if normalize in ["none", "raw"]:
        return weights.copy()

    raise ValueError(f"Unsupported spectrum normalization mode: {normalize}")


def blackbody_weights(wavelengths_m, temperature_K, weighting="photon"):
    """
    Compute relative blackbody spectral weights.

    This uses util.planck_law if available in BaldrApp utilities. The returned
    values are relative weights only; absolute units are not required because
    the weights are normally normalized afterwards.

    Parameters
    ----------
    wavelengths_m : array-like
        Wavelengths in metres.
    temperature_K : float
        Blackbody temperature in Kelvin.
    weighting : str
        "energy"
            Use B_lambda.

        "photon"
            Use photon-counting weights proportional to B_lambda * lambda.
            This is usually the better choice for detector photon counts.

    Returns
    -------
    weights : ndarray
        Relative spectral weights.
    """
    wavelengths_m = np.asarray(wavelengths_m, dtype=float)

    if np.any(wavelengths_m <= 0):
        raise ValueError("All wavelengths must be positive.")

    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive.")

    # utilities.planck_law in BaldrApp has historically been used with SI-like
    # wavelength values. If your local implementation expects different units,
    # adjust here rather than in the frame-generation code.
    weights_energy = np.asarray(
        util.planck_law(wavelengths_m, temperature_K),
        dtype=float,
    )

    weighting = str(weighting).lower()

    if weighting == "energy":
        return weights_energy

    if weighting == "photon":
        # Photon number is proportional to energy flux divided by photon energy,
        # so relative photon weighting gains a factor lambda.
        return weights_energy * wavelengths_m

    raise ValueError(f"Unsupported blackbody weighting mode: {weighting}")


def derive_spectrum(spectrum_config, default_wvl0):
    """
    Derive wavelength and weight arrays from a spectrum config section.

    Parameters
    ----------
    spectrum_config : dict or SimpleNamespace or None
        Spectrum configuration. If None, a backwards-compatible monochromatic
        spectrum is returned.
    default_wvl0 : float
        Default central wavelength in metres, usually zwfs_ns.optics.wvl0.

    Supported modes
    ---------------
    monochromatic / single
        One wavelength. Uses spectrum.wvl0 if supplied, otherwise default_wvl0.

    flat
        Uniform weights over a wavelength grid.

    blackbody
        Blackbody spectral weights over a wavelength grid.

    table
        User-supplied wavelengths and weights.

    Example JSON
    ------------
    "spectrum": {
      "enabled": true,
      "mode": "blackbody",
      "temperature_K": 3500,
      "weighting": "photon",
      "wvl_min": 1.50e-6,
      "wvl_max": 1.80e-6,
      "n_wvl": 7,
      "normalize": "sum"
    }

    Returns
    -------
    spectrum_ns : SimpleNamespace
        Contains:

            enabled
            mode
            wavelengths
            weights
            weights_normalized
    """
    if spectrum_config is None:
        return default_monochromatic_spectrum(default_wvl0)

    spectrum_ns = _as_namespace(spectrum_config)

    if not hasattr(spectrum_ns, "enabled"):
        spectrum_ns.enabled = True

    if not spectrum_ns.enabled:
        return default_monochromatic_spectrum(default_wvl0)

    if not hasattr(spectrum_ns, "mode"):
        spectrum_ns.mode = "flat"

    if not hasattr(spectrum_ns, "normalize"):
        spectrum_ns.normalize = "sum"

    mode = str(spectrum_ns.mode).lower()

    if mode in ["monochromatic", "single"]:
        wvl = float(getattr(spectrum_ns, "wvl0", default_wvl0))
        wavelengths = np.array([wvl], dtype=float)
        weights = np.array([1.0], dtype=float)

    elif mode == "flat":
        wvl_min = float(getattr(spectrum_ns, "wvl_min", default_wvl0))
        wvl_max = float(getattr(spectrum_ns, "wvl_max", default_wvl0))
        n_wvl = int(getattr(spectrum_ns, "n_wvl", 1))

        if n_wvl < 1:
            raise ValueError("spectrum.n_wvl must be >= 1.")

        wavelengths = np.linspace(wvl_min, wvl_max, n_wvl)
        weights = np.ones_like(wavelengths)

    elif mode == "blackbody":
        if not hasattr(spectrum_ns, "temperature_K"):
            raise ValueError(
                "spectrum.temperature_K is required for blackbody mode."
            )

        wvl_min = float(getattr(spectrum_ns, "wvl_min", default_wvl0))
        wvl_max = float(getattr(spectrum_ns, "wvl_max", default_wvl0))
        n_wvl = int(getattr(spectrum_ns, "n_wvl", 5))
        weighting = str(getattr(spectrum_ns, "weighting", "photon")).lower()

        if n_wvl < 1:
            raise ValueError("spectrum.n_wvl must be >= 1.")

        wavelengths = np.linspace(wvl_min, wvl_max, n_wvl)
        weights = blackbody_weights(
            wavelengths,
            temperature_K=float(spectrum_ns.temperature_K),
            weighting=weighting,
        )

    elif mode == "table":
        if not hasattr(spectrum_ns, "wavelengths_m"):
            raise ValueError(
                "spectrum.wavelengths_m is required for table mode."
            )
        if not hasattr(spectrum_ns, "weights"):
            raise ValueError(
                "spectrum.weights is required for table mode."
            )

        wavelengths = np.asarray(spectrum_ns.wavelengths_m, dtype=float)
        weights = np.asarray(spectrum_ns.weights, dtype=float)

        if wavelengths.shape != weights.shape:
            raise ValueError(
                "spectrum.wavelengths_m and spectrum.weights must have "
                "the same shape."
            )

    else:
        raise ValueError(f"Unsupported spectrum.mode: {spectrum_ns.mode}")

    if np.any(wavelengths <= 0):
        raise ValueError("All spectrum wavelengths must be positive.")

    if np.any(weights < 0):
        raise ValueError("All spectrum weights must be non-negative.")

    if np.sum(weights) <= 0:
        raise ValueError("Spectrum weights must have positive sum.")

    weights_normalized = normalize_weights(
        weights,
        normalize=spectrum_ns.normalize,
    )

    spectrum_ns.wavelengths = wavelengths
    spectrum_ns.weights = weights
    spectrum_ns.weights_normalized = weights_normalized

    return spectrum_ns


# ============================================================
# Phase-mask chromaticity helpers
# ============================================================
def theta_at_wavelength(optics, wavelength_m):
    """
    Return the ZWFS phase-mask phase shift at a given wavelength.

    New optional behaviour
    ----------------------
    If optics.active_phasemask exists, use it as the canonical phasemask model.

    Backwards-compatible behaviour
    ------------------------------
    If optics.active_phasemask does not exist, fall back to the existing
    optics.theta / optics.theta_mode convention.
    """
    wavelength_m = float(wavelength_m)

    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError(f"wavelength_m must be finite and positive; got {wavelength_m!r}")

    # ============================================================
    # New active physical/named phasemask path
    # ============================================================
    if hasattr(optics, "active_phasemask"):
        pm = optics.active_phasemask


        if not bool(getattr(pm, "inserted", True)):
            return 0.0


        if hasattr(pm, "cached_wavelengths_m") and hasattr(pm, "cached_theta_rad"):
            #idx = np.where(pm.cached_wavelengths_m == wavelength_m)[0]
            idx = np.where(
                np.isclose(
                    pm.cached_wavelengths_m,
                    wavelength_m,
                    rtol=0.0,
                    atol=1e-15))[0]
            if len(idx) == 1:
                return float(pm.cached_theta_rad[idx[0]])
            
        if not hasattr(pm, "theta_model"):
            raise ValueError("active_phasemask must contain theta_model.")


        theta_model = str(pm.theta_model)

        if theta_model == "constant":
            if not hasattr(pm, "theta_rad"):
                raise ValueError(
                    f"active_phasemask {getattr(pm, 'name', '<unknown>')!r} "
                    "has theta_model='constant' but no theta_rad."
                )

            theta = float(pm.theta_rad)

            if not np.isfinite(theta):
                raise ValueError(
                    f"active_phasemask {getattr(pm, 'name', '<unknown>')!r} "
                    f"has non-finite theta_rad={theta!r}."
                )

            return theta

        if theta_model == "physical_depth":
            if not hasattr(pm, "mask_depth_um") or pm.mask_depth_um is None:
                raise ValueError(
                    f"active_phasemask {getattr(pm, 'name', '<unknown>')!r} "
                    "has theta_model='physical_depth' but no mask_depth_um."
                )

            if not hasattr(pm, "dot_material") or pm.dot_material is None:
                raise ValueError(
                    f"active_phasemask {getattr(pm, 'name', '<unknown>')!r} "
                    "has theta_model='physical_depth' but no dot_material."
                )

            wavelength_um = wavelength_m * 1e6
            depth_um = float(pm.mask_depth_um)
            dot_material = str(pm.dot_material)

            theta = float(
                util.get_phasemask_phaseshift(
                    wvl=wavelength_um,
                    depth=depth_um,
                    dot_material=dot_material,
                )
            )

            if not np.isfinite(theta):
                raise ValueError(
                    f"Non-finite theta for active_phasemask "
                    f"{getattr(pm, 'name', '<unknown>')!r}: "
                    f"wavelength_um={wavelength_um:.6g}, "
                    f"mask_depth_um={depth_um:.6g}, "
                    f"dot_material={dot_material!r}. "
                    "This usually means the material optical-constants table "
                    "does not cover this wavelength, or the material name is wrong."
                )

            return theta

        raise ValueError(
            f"Unsupported active_phasemask theta_model={theta_model!r}. "
            "Allowed values are 'constant' and 'physical_depth'."
        )

    # ============================================================
    # Existing legacy optics path
    # ============================================================
    theta_mode = str(getattr(optics, "theta_mode", "constant")).lower()

    if theta_mode == "constant":
        if not hasattr(optics, "theta"):
            raise ValueError(
                "optics.theta is required when theta_mode='constant'."
            )

        theta = float(optics.theta)

        if not np.isfinite(theta):
            raise ValueError(f"optics.theta must be finite; got {theta!r}")

        return theta

    if theta_mode == "physical_depth":
        if not hasattr(optics, "mask_depth"):
            raise ValueError(
                "optics.mask_depth is required when theta_mode='physical_depth'. "
                "Legacy optics.mask_depth is interpreted as metres."
            )

        dot_material = getattr(optics, "dot_material", "N_1405")

        wavelength_um = wavelength_m * 1e6
        depth_um = float(optics.mask_depth) * 1e6

        theta = float(
            util.get_phasemask_phaseshift(
                wvl=wavelength_um,
                depth=depth_um,
                dot_material=dot_material,
            )
        )

        if not np.isfinite(theta):
            raise ValueError(
                f"Non-finite theta from legacy optics physical_depth mode: "
                f"wavelength_um={wavelength_um:.6g}, "
                f"depth_um={depth_um:.6g}, "
                f"dot_material={dot_material!r}."
            )

        return theta

    raise ValueError(f"Unsupported optics.theta_mode: {theta_mode}")

# def theta_at_wavelength(optics, wavelength_m):
#     """
#     Return the ZWFS phase-mask phase shift at a given wavelength.

#     Backwards-compatible default:

#         theta_mode = "constant"

#     so the function simply returns optics.theta.

#     Optional physical-depth mode:

#         theta_mode = "physical_depth"

#     then the phase shift is calculated using util.get_phasemask_phaseshift.
#     This is useful when the mask depth and material are known and the phase
#     shift should vary with wavelength.

#     Expected optics fields for physical_depth mode
#     ----------------------------------------------
#     optics.theta_mode = "physical_depth"
#     optics.mask_depth = mask depth in metres
#     optics.dot_material = material name accepted by util.get_phasemask_phaseshift

#     Returns
#     -------
#     theta : float
#         Phase shift in radians.
#     """
#     theta_mode = str(getattr(optics, "theta_mode", "constant")).lower()

#     if theta_mode == "constant":
#         return float(optics.theta)

#     if theta_mode == "physical_depth":
#         if not hasattr(optics, "mask_depth"):
#             raise ValueError(
#                 "optics.mask_depth is required when theta_mode is "
#                 "'physical_depth'."
#             )

#         dot_material = getattr(optics, "dot_material", "N_1405")

#         # util.get_phasemask_phaseshift expects wavelength and depth in um
#         # in the current BaldrApp utilities implementation.
#         wavelength_um = float(wavelength_m) * 1e6
#         depth_um = float(optics.mask_depth) * 1e6

#         return float(
#             util.get_phasemask_phaseshift(
#                 wvl=wavelength_um,
#                 depth=depth_um,
#                 dot_material=dot_material,
#             )
#         )

#     raise ValueError(f"Unsupported optics.theta_mode: {theta_mode}")


def phasemask_diameter_at_wavelength(
    optics,
    wavelength_m,
    default_wvl0=None,
):
    """
    Return phase-mask diameter in the legacy dimensionless convention expected
    by the ZWFS propagation functions.

    New optional behaviour
    ----------------------
    If optics.active_phasemask exists, use it as the canonical phasemask model.

    Backwards-compatible behaviour
    ------------------------------
    If optics.active_phasemask does not exist, fall back to the existing
    optics.mask_diam / optics.mask_diam_mode convention.
    """
    del default_wvl0  # currently unused, kept for API compatibility

    wavelength_m = float(wavelength_m)

    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError(f"wavelength_m must be finite and positive; got {wavelength_m!r}")

    # ============================================================
    # New active physical/named phasemask path
    # ============================================================
    if hasattr(optics, "active_phasemask"):
        pm = optics.active_phasemask

        if not bool(getattr(pm, "inserted", True)):
            # Diameter is irrelevant when theta=0 / mask out.
            # Return the legacy value if available so callers do not crash.
            if hasattr(optics, "mask_diam"):
                return float(optics.mask_diam)
            return 0.0

        if hasattr(pm, "cached_wavelengths_m") and hasattr(pm, "cached_mask_diam_lambdaD"):
            idx = np.where(
                np.isclose(
                    pm.cached_wavelengths_m,
                    wavelength_m,
                    rtol=0.0,
                    atol=1e-15,
                )
            )[0]
            if len(idx) == 1:
                return float(pm.cached_mask_diam_lambdaD[idx[0]])


        if not hasattr(pm, "diameter_model"):
            raise ValueError("active_phasemask must contain diameter_model.")

        diameter_model = str(pm.diameter_model)


        if diameter_model == "lambda_over_D":
            if not hasattr(pm, "mask_diam_lambdaD") or pm.mask_diam_lambdaD is None:
                raise ValueError(
                    f"active_phasemask {getattr(pm, 'name', '<unknown>')!r} "
                    "has diameter_model='lambda_over_D' but no mask_diam_lambdaD."
                )

            mask_diam = float(pm.mask_diam_lambdaD)

        elif diameter_model == "physical":
            if not hasattr(pm, "mask_diam_m") or pm.mask_diam_m is None:
                raise ValueError(
                    f"active_phasemask {getattr(pm, 'name', '<unknown>')!r} "
                    "has diameter_model='physical' but no mask_diam_m."
                )

            if not hasattr(optics, "F_number"):
                raise ValueError(
                    "optics.F_number is required to convert physical "
                    "phase-mask diameter to lambda/D units."
                )

            F_number = float(optics.F_number)

            if not np.isfinite(F_number) or F_number <= 0:
                raise ValueError(
                    f"optics.F_number must be finite and positive; got {F_number!r}"
                )

            mask_diam = float(pm.mask_diam_m) / (
                1.22 * F_number * wavelength_m
            )

        else:
            raise ValueError(
                f"Unsupported active_phasemask diameter_model={diameter_model!r}. "
                "Allowed values are 'lambda_over_D' and 'physical'."
            )

        if not np.isfinite(mask_diam) or mask_diam <= 0:
            raise ValueError(
                f"Invalid phase-mask diameter for active_phasemask "
                f"{getattr(pm, 'name', '<unknown>')!r}: {mask_diam!r}"
            )

        return mask_diam

    # ============================================================
    # Existing legacy optics path
    # ============================================================
    mode = str(
        getattr(optics, "mask_diam_mode", "lambda_over_D")
    ).lower()

    if mode in ["lambda_over_d", "lambda/d", "dimensionless"]:
        if not hasattr(optics, "mask_diam"):
            raise ValueError(
                "optics.mask_diam is required when mask_diam_mode='lambda_over_D'."
            )

        mask_diam = float(optics.mask_diam)

        if not np.isfinite(mask_diam) or mask_diam <= 0:
            raise ValueError(
                f"optics.mask_diam must be finite and positive; got {mask_diam!r}"
            )

        return mask_diam

    if mode in ["physical_um", "um", "micron", "microns"]:
        if not hasattr(optics, "mask_diam_um"):
            raise ValueError(
                "optics.mask_diam_um is required when "
                "mask_diam_mode='physical_um'."
            )

        if not hasattr(optics, "F_number"):
            raise ValueError(
                "optics.F_number is required when mask_diam_mode='physical_um'."
            )

        mask_diam_m = float(optics.mask_diam_um) * 1e-6
        mask_diam = mask_diam_m / (
            1.22 * float(optics.F_number) * wavelength_m
        )

        if not np.isfinite(mask_diam) or mask_diam <= 0:
            raise ValueError(
                f"Converted physical_um mask diameter is invalid: {mask_diam!r}"
            )

        return mask_diam

    if mode in ["physical_m", "m", "metre", "meter", "meters", "metres"]:
        if not hasattr(optics, "mask_diam_m"):
            raise ValueError(
                "optics.mask_diam_m is required when "
                "mask_diam_mode='physical_m'."
            )

        if not hasattr(optics, "F_number"):
            raise ValueError(
                "optics.F_number is required when mask_diam_mode='physical_m'."
            )

        mask_diam = float(optics.mask_diam_m) / (
            1.22 * float(optics.F_number) * wavelength_m
        )

        if not np.isfinite(mask_diam) or mask_diam <= 0:
            raise ValueError(
                f"Converted physical_m mask diameter is invalid: {mask_diam!r}"
            )

        return mask_diam

    raise ValueError(f"Unsupported optics.mask_diam_mode: {mode}")

# def phasemask_diameter_at_wavelength(
#     optics,
#     wavelength_m,
#     default_wvl0=None,
# ):
#     """
#     Return phase-mask diameter in the legacy dimensionless convention expected
#     by the ZWFS propagation functions.

#     Supported modes
#     ---------------
#     mask_diam_mode = "lambda_over_D"
#         optics.mask_diam is already dimensionless and is returned unchanged.

#     mask_diam_mode = "physical_um"
#         optics.mask_diam_um is a physical focal-plane diameter in microns.
#         It is converted to the legacy dimensionless convention at wavelength_m:

#             mask_diam = mask_diam_m / (1.22 * F_number * wavelength_m)

#     mask_diam_mode = "physical_m"
#         optics.mask_diam_m is a physical focal-plane diameter in metres.
#     """
#     mode = str(
#         getattr(optics, "mask_diam_mode", "lambda_over_D")
#     ).lower()

#     wavelength_m = float(wavelength_m)

#     if wavelength_m <= 0:
#         raise ValueError("wavelength_m must be positive.")

#     if mode in ["lambda_over_d", "lambda/d", "lambda_over_D".lower(), "dimensionless"]:
#         return float(optics.mask_diam)

#     if mode in ["physical_um", "um", "micron", "microns"]:
#         if not hasattr(optics, "mask_diam_um"):
#             raise ValueError(
#                 "optics.mask_diam_um is required when "
#                 "mask_diam_mode='physical_um'."
#             )

#         mask_diam_m = float(optics.mask_diam_um) * 1e-6

#         return mask_diam_m / (
#             1.22 * float(optics.F_number) * wavelength_m
#         )

#     if mode in ["physical_m", "m", "metre", "meter", "meters", "metres"]:
#         if not hasattr(optics, "mask_diam_m"):
#             raise ValueError(
#                 "optics.mask_diam_m is required when "
#                 "mask_diam_mode='physical_m'."
#             )

#         return float(optics.mask_diam_m) / (
#             1.22 * float(optics.F_number) * wavelength_m
#         )

#     raise ValueError(f"Unsupported optics.mask_diam_mode: {mode}")

# # def phasemask_diameter_at_wavelength(optics, wavelength_m, default_wvl0):
# #     """
# #     Return the phase-mask diameter parameter at a given wavelength.

# #     Backwards-compatible default:

# #         mask_diam_mode = "lambda_over_D"

# #     In this mode, optics.mask_diam is interpreted exactly as current BaldrApp
# #     code interprets it: a diameter in lambda/D-like units. Therefore the same
# #     numerical value is returned at every wavelength.

# #     Optional physical mode:

# #         mask_diam_mode = "physical"

# #     In this mode, optics.mask_diam_phys_m is interpreted as a physical mask
# #     diameter. The returned lambda/D-like diameter is scaled relative to wvl0:

# #         mask_diam(lambda) = mask_diam(wvl0) * default_wvl0 / lambda

# #     This assumes the current optics.mask_diam corresponds to the physical
# #     mask size expressed in lambda/D units at default_wvl0.

# #     Returns
# #     -------
# #     mask_diam : float
# #         Phase-mask diameter in the convention expected by
# #         get_zwfs_output_field.
# #     """
# #     mode = str(getattr(optics, "mask_diam_mode", "lambda_over_D")).lower()

# #     if mode in ["lambda_over_d", "lambda/d", "lambda_over_D".lower()]:
# #         return float(optics.mask_diam)

# #     if mode == "physical":
# #         # Keep this deliberately simple and backwards compatible. We do not
# #         # require a physical mask diameter unless later code needs it. The
# #         # existing optics.mask_diam is treated as the value at default_wvl0.
# #         return float(optics.mask_diam) * float(default_wvl0) / float(wavelength_m)

# #     raise ValueError(f"Unsupported optics.mask_diam_mode: {mode}")


# ============================================================
# Convenience iteration
# ============================================================

def iter_spectrum(spectrum_ns):
    """
    Iterate over normalized spectral samples.

    Yields
    ------
    wavelength_m : float
    weight : float
    """
    for wavelength_m, weight in zip(
        spectrum_ns.wavelengths,
        spectrum_ns.weights_normalized,
    ):
        yield float(wavelength_m), float(weight)
        
        
        
        
        
        
        
        
def normalise_phasemask_entry(mask_name, mask_entry, optics):
    """
    Convert one strict phasemask_properties.json mask entry into a canonical
    active_phasemask namespace.

    This function does not mutate optics. The simulator will later attach the
    returned object as:

        zwfs.optics.active_phasemask = normalise_phasemask_entry(...)

    Strict accepted schemas
    -----------------------

    Physical phase depth + physical focal-plane diameter:

        {
          "theta_model": "physical_depth",
          "mask_depth_um": 0.654,
          "dot_material": "N_1405",
          "diameter_model": "physical",
          "mask_diam_um": 44.0
        }

    Constant phase + lambda/D diameter:

        {
          "theta_model": "constant",
          "theta_rad": 1.5707963267948966,
          "diameter_model": "lambda_over_D",
          "mask_diam_lambdaD": 1.06
        }

    Notes
    -----
    - mask_depth_um is always microns.
    - mask_diam_um is always microns.
    - mask_diam_lambdaD is always the legacy BaldrApp dimensionless convention.
    - theta_rad is always radians.
    - optics.wvl0 and optics.F_number are used only to compute diagnostic /
      legacy central-wavelength values.
    """
    if not isinstance(mask_entry, dict):
        raise TypeError(
            f"Mask entry for {mask_name!r} must be a dictionary, "
            f"got {type(mask_entry).__name__}."
        )

    if optics is None:
        raise ValueError("optics must be provided.")

    if not hasattr(optics, "wvl0"):
        raise ValueError("optics.wvl0 is required.")

    if not hasattr(optics, "F_number"):
        raise ValueError("optics.F_number is required.")

    name = str(mask_name)
    entry = dict(mask_entry)

    allowed_keys = {
        "theta_model",
        "theta_rad",
        "mask_depth_um",
        "dot_material",
        "diameter_model",
        "mask_diam_um",
        "mask_diam_lambdaD",
    }

    unknown_keys = sorted(set(entry.keys()) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Mask {name!r} contains unsupported keys: {unknown_keys}. "
            f"Allowed keys are: {sorted(allowed_keys)}"
        )

    if "theta_model" not in entry:
        raise ValueError(
            f"Mask {name!r} must define 'theta_model'. "
            "Allowed values: 'constant', 'physical_depth'."
        )

    if "diameter_model" not in entry:
        raise ValueError(
            f"Mask {name!r} must define 'diameter_model'. "
            "Allowed values: 'lambda_over_D', 'physical'."
        )

    theta_model = str(entry["theta_model"]).strip()
    diameter_model = str(entry["diameter_model"]).strip()

    if theta_model not in {"constant", "physical_depth"}:
        raise ValueError(
            f"Mask {name!r} has invalid theta_model={theta_model!r}. "
            "Allowed values: 'constant', 'physical_depth'."
        )

    if diameter_model not in {"lambda_over_D", "physical"}:
        raise ValueError(
            f"Mask {name!r} has invalid diameter_model={diameter_model!r}. "
            "Allowed values: 'lambda_over_D', 'physical'."
        )

    wvl0_m = float(optics.wvl0)
    F_number = float(optics.F_number)

    if not np.isfinite(wvl0_m) or wvl0_m <= 0:
        raise ValueError(f"optics.wvl0 must be finite and positive; got {wvl0_m!r}")

    if not np.isfinite(F_number) or F_number <= 0:
        raise ValueError(
            f"optics.F_number must be finite and positive; got {F_number!r}"
        )

    wvl0_um = wvl0_m * 1e6

    # ------------------------------------------------------------------
    # Theta model
    # ------------------------------------------------------------------
    theta_rad = None
    theta_rad_wvl0 = None
    mask_depth_um = None
    dot_material = None

    if theta_model == "constant":
        required = {"theta_model", "theta_rad", "diameter_model"}
        allowed_for_theta = {"theta_model", "theta_rad"}

        if "theta_rad" not in entry:
            raise ValueError(
                f"Mask {name!r} has theta_model='constant' but is missing "
                "'theta_rad'."
            )

        forbidden = {"mask_depth_um", "dot_material"} & set(entry.keys())
        if forbidden:
            raise ValueError(
                f"Mask {name!r} has theta_model='constant' but also defines "
                f"{sorted(forbidden)}. Remove physical-depth fields."
            )

        theta_rad = float(entry["theta_rad"])
        if not np.isfinite(theta_rad):
            raise ValueError(
                f"Mask {name!r} theta_rad must be finite; got {theta_rad!r}."
            )

        theta_rad_wvl0 = theta_rad

    elif theta_model == "physical_depth":
        if "mask_depth_um" not in entry:
            raise ValueError(
                f"Mask {name!r} has theta_model='physical_depth' but is "
                "missing 'mask_depth_um'."
            )

        if "dot_material" not in entry:
            raise ValueError(
                f"Mask {name!r} has theta_model='physical_depth' but is "
                "missing 'dot_material'."
            )

        forbidden = {"theta_rad"} & set(entry.keys())
        if forbidden:
            raise ValueError(
                f"Mask {name!r} has theta_model='physical_depth' but also "
                f"defines {sorted(forbidden)}. Remove constant-theta fields."
            )

        mask_depth_um = float(entry["mask_depth_um"])
        if not np.isfinite(mask_depth_um) or mask_depth_um <= 0:
            raise ValueError(
                f"Mask {name!r} mask_depth_um must be finite and positive; "
                f"got {mask_depth_um!r}."
            )

        dot_material = str(entry["dot_material"]).strip()
        if not dot_material:
            raise ValueError(f"Mask {name!r} dot_material must be non-empty.")

        theta_rad_wvl0 = float(
            util.get_phasemask_phaseshift(
                wvl=wvl0_um,
                depth=mask_depth_um,
                dot_material=dot_material,
            )
        )

        if not np.isfinite(theta_rad_wvl0):
            raise ValueError(
                f"Mask {name!r} produced non-finite theta at wvl0. "
                f"wvl0_um={wvl0_um}, mask_depth_um={mask_depth_um}, "
                f"dot_material={dot_material!r}. Check material data coverage."
            )

    # ------------------------------------------------------------------
    # Diameter model
    # ------------------------------------------------------------------
    mask_diam_um = None
    mask_diam_m = None
    mask_diam_lambdaD = None
    mask_diam_lambdaD_wvl0 = None

    if diameter_model == "lambda_over_D":
        if "mask_diam_lambdaD" not in entry:
            raise ValueError(
                f"Mask {name!r} has diameter_model='lambda_over_D' but is "
                "missing 'mask_diam_lambdaD'."
            )

        forbidden = {"mask_diam_um"} & set(entry.keys())
        if forbidden:
            raise ValueError(
                f"Mask {name!r} has diameter_model='lambda_over_D' but also "
                f"defines {sorted(forbidden)}. Remove physical-diameter fields."
            )

        mask_diam_lambdaD = float(entry["mask_diam_lambdaD"])
        if not np.isfinite(mask_diam_lambdaD) or mask_diam_lambdaD <= 0:
            raise ValueError(
                f"Mask {name!r} mask_diam_lambdaD must be finite and positive; "
                f"got {mask_diam_lambdaD!r}."
            )

        mask_diam_lambdaD_wvl0 = mask_diam_lambdaD

    elif diameter_model == "physical":
        if "mask_diam_um" not in entry:
            raise ValueError(
                f"Mask {name!r} has diameter_model='physical' but is missing "
                "'mask_diam_um'."
            )

        forbidden = {"mask_diam_lambdaD"} & set(entry.keys())
        if forbidden:
            raise ValueError(
                f"Mask {name!r} has diameter_model='physical' but also defines "
                f"{sorted(forbidden)}. Remove lambda/D diameter fields."
            )

        mask_diam_um = float(entry["mask_diam_um"])
        if not np.isfinite(mask_diam_um) or mask_diam_um <= 0:
            raise ValueError(
                f"Mask {name!r} mask_diam_um must be finite and positive; "
                f"got {mask_diam_um!r}."
            )

        mask_diam_m = mask_diam_um * 1e-6

        mask_diam_lambdaD_wvl0 = mask_diam_m / (
            1.22 * F_number * wvl0_m
        )

        if (
            not np.isfinite(mask_diam_lambdaD_wvl0)
            or mask_diam_lambdaD_wvl0 <= 0
        ):
            raise ValueError(
                f"Mask {name!r} produced invalid lambda/D diameter at wvl0: "
                f"{mask_diam_lambdaD_wvl0!r}."
            )

    pm = SimpleNamespace(
        name=name,
        inserted=True,

        # Strict canonical model names
        theta_model=theta_model,
        diameter_model=diameter_model,

        # Constant-theta model
        theta_rad=theta_rad,

        # Physical-depth model
        mask_depth_um=mask_depth_um,
        dot_material=dot_material,

        # Lambda/D diameter model
        mask_diam_lambdaD=mask_diam_lambdaD,

        # Physical-diameter model
        mask_diam_um=mask_diam_um,
        mask_diam_m=mask_diam_m,

        # Central-wavelength compatibility/diagnostic values
        wvl0_m=wvl0_m,
        wvl0_um=wvl0_um,
        F_number=F_number,
        theta_rad_wvl0=theta_rad_wvl0,
        mask_diam_lambdaD_wvl0=mask_diam_lambdaD_wvl0,

        # Raw strict entry for debugging
        raw_entry=entry,
    )

    return pm




### Tests 
# from types import SimpleNamespace
# import numpy as np
# from baldrapp.common import spectrum as spec

# optics = SimpleNamespace(
#     wvl0=1.65e-6,
#     F_number=21.2,
#     theta=1.5707963267948966,
#     theta_mode="constant",
#     mask_diam=1.06,
#     mask_diam_mode="lambda_over_D",
# )

# entry = {
#     "theta_model": "physical_depth",
#     "mask_depth_um": 0.654,
#     "dot_material": "N_1405",
#     "diameter_model": "physical",
#     "mask_diam_um": 44.0,
# }

# pm = spec.normalise_phasemask_entry(
#     mask_name="J3",
#     mask_entry=entry,
#     optics=optics,
# )

# optics.active_phasemask = pm

# for w in np.linspace(1.5e-6, 1.8e-6, 7):
#     print(
#         f"{w*1e6:.3f} um",
#         "theta =", spec.theta_at_wavelength(optics, w),
#         "diam =", spec.phasemask_diameter_at_wavelength(optics, w),
#     )
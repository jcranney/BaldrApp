"""
fresnel.py

Scalar diffraction and Fresnel propagation utilities.

This module is deliberately independent of beam_trace.py. It only depends on
sampled complex fields, pixel scales, wavelength, and propagation distance.

Conventions
-----------
- Fields are 2D complex NumPy arrays.
- dx, dy are physical pixel scales in metres / pixel.
- wavelength is in metres.
- z is propagation distance in metres.
- Positive z propagates forward.
- FFTs use NumPy's native unnormalised convention.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from functools import lru_cache

ArrayLike = np.ndarray
NormalizationMode = Optional[Literal["peak", "sum"]]


# ============================================================
# Coordinate / frequency grids
# ============================================================

@lru_cache(maxsize=None)
def make_coordinate_grid(
    shape: Tuple[int, int],
    dx: float,
    dy: Optional[float] = None,
    indexing: Literal["xy", "ij"] = "xy",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return centred physical coordinate grids.

    Parameters
    ----------
    shape:
        Array shape as (ny, nx).
    dx, dy:
        Pixel scales [m / pixel]. If dy is None, dy = dx.
    indexing:
        Passed to np.meshgrid. Default "xy" returns X, Y arrays with shape
        (ny, nx).

    Returns
    -------
    X, Y:
        Coordinate grids in metres.
    """
    if dy is None:
        dy = dx

    ny, nx = shape
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy

    return np.meshgrid(x, y, indexing=indexing)

@lru_cache(maxsize=None)
def make_frequency_grid(
    shape: Tuple[int, int],
    dx: float,
    dy: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return spatial-frequency grids compatible with np.fft.fft2.

    Frequencies are in cycles / metre and are intentionally not fftshifted,
    because they are used directly with unshifted np.fft.fft2 outputs.
    """
    if dy is None:
        dy = dx

    ny, nx = shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)

    return np.meshgrid(fx, fy)


# ============================================================
# Transfer functions
# ============================================================

@lru_cache(maxsize=None)
def angular_spectrum_transfer_function(
    shape: Tuple[int, int],
    wavelength: float,
    dx: float,
    z: float,
    dy: Optional[float] = None,
    bandlimit: bool = True,
    include_global_phase: bool = True,
) -> np.ndarray:
    """
    Angular-spectrum transfer function.

    This is the scalar free-space propagator in homogeneous media, apart from
    optional removal of evanescent frequencies.

    Parameters
    ----------
    shape:
        Field shape as (ny, nx).
    wavelength:
        Wavelength [m].
    dx, dy:
        Pixel scales [m / pixel].
    z:
        Propagation distance [m].
    bandlimit:
        If True, evanescent spatial frequencies are suppressed.
        If False, evanescent terms are included using complex kz.
    include_global_phase:
        If False, removes the carrier exp(i k z), leaving only the relative
        spatial-frequency-dependent phase.

    Returns
    -------
    H:
        Complex transfer function with the same shape as the input field.
    """
    if dy is None:
        dy = dx

    FX, FY = make_frequency_grid(shape, dx, dy)

    k = 2.0 * np.pi / wavelength
    lambda_fx = wavelength * FX
    lambda_fy = wavelength * FY

    arg = 1.0 - lambda_fx**2 - lambda_fy**2

    if bandlimit:
        valid = arg >= 0.0
        H = np.zeros(shape, dtype=complex)

        if include_global_phase:
            phase = k * z * np.sqrt(np.maximum(arg, 0.0))
        else:
            phase = k * z * (np.sqrt(np.maximum(arg, 0.0)) - 1.0)

        H[valid] = np.exp(1j * phase[valid])
        return H

    kz = k * np.sqrt(arg.astype(complex))

    if include_global_phase:
        return np.exp(1j * kz * z)

    return np.exp(1j * (kz - k) * z)

@lru_cache(maxsize=None)
def fresnel_transfer_function(
    shape: Tuple[int, int],
    wavelength: float,
    dx: float,
    z: float,
    dy: Optional[float] = None,
    include_global_phase: bool = True,
) -> np.ndarray:
    """
    Paraxial Fresnel transfer function.

    This keeps the same input/output sampling.

    H(fx, fy) = exp(i k z) exp[-i pi lambda z (fx^2 + fy^2)]
    """
    if dy is None:
        dy = dx

    FX, FY = make_frequency_grid(shape, dx, dy)

    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))

    if include_global_phase:
        k = 2.0 * np.pi / wavelength
        H = np.exp(1j * k * z) * H

    return H


# ============================================================
# Propagation
# ============================================================

def angular_spectrum_propagate(
    field: ArrayLike,
    wavelength: float,
    dx: float,
    z: float,
    dy: Optional[float] = None,
    bandlimit: bool = True,
    include_global_phase: bool = True,
) -> np.ndarray:
    """
    Propagate a 2D scalar field using the angular-spectrum method.
    """
    if dy is None:
        dy = dx

    field = np.asarray(field, dtype=complex)

    H = angular_spectrum_transfer_function(
        field.shape,
        wavelength=wavelength,
        dx=dx,
        dy=dy,
        z=z,
        bandlimit=bandlimit,
        include_global_phase=include_global_phase,
    )

    return np.fft.ifft2(np.fft.fft2(field) * H)


def fresnel_transfer_function_propagate(
    field: ArrayLike,
    wavelength: float,
    dx: float,
    z: float,
    dy: Optional[float] = None,
    include_global_phase: bool = True,
) -> np.ndarray:
    """
    Propagate a 2D scalar field using the Fresnel transfer-function method.
    """
    if dy is None:
        dy = dx

    field = np.asarray(field, dtype=complex)

    H = fresnel_transfer_function(
        field.shape,
        wavelength=wavelength,
        dx=dx,
        dy=dy,
        z=z,
        include_global_phase=include_global_phase,
    )

    return np.fft.ifft2(np.fft.fft2(field) * H)


def propagate(
    field: ArrayLike,
    wavelength: float,
    dx: float,
    z: float,
    dy: Optional[float] = None,
    method: Literal["angular_spectrum", "fresnel"] = "angular_spectrum",
    include_global_phase: bool = True,
    bandlimit: bool = True,
) -> np.ndarray:
    """
    Convenience propagation wrapper.

    Parameters
    ----------
    method:
        "angular_spectrum" or "fresnel".
    """
    if method == "angular_spectrum":
        return angular_spectrum_propagate(
            field,
            wavelength=wavelength,
            dx=dx,
            dy=dy,
            z=z,
            bandlimit=bandlimit,
            include_global_phase=include_global_phase,
        )

    if method == "fresnel":
        return fresnel_transfer_function_propagate(
            field,
            wavelength=wavelength,
            dx=dx,
            dy=dy,
            z=z,
            include_global_phase=include_global_phase,
        )

    raise ValueError("method must be 'angular_spectrum' or 'fresnel'.")


# ============================================================
# Simple optical elements
# ============================================================
@lru_cache(maxsize=None)
def thin_lens_phase(
    shape: Tuple[int, int],
    wavelength: float,
    dx: float,
    focal_length: float,
    dy: Optional[float] = None,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Complex phase factor for an ideal thin lens.

    phase = exp[-i pi (x^2 + y^2) / (lambda f)]
    """
    if dy is None:
        dy = dx

    X, Y = make_coordinate_grid(shape, dx, dy)

    if center is not None:
        X = X - center[0]
        Y = Y - center[1]

    return np.exp(-1j * np.pi * (X**2 + Y**2) / (wavelength * focal_length))


def apply_thin_lens(
    field: ArrayLike,
    wavelength: float,
    dx: float,
    focal_length: float,
    dy: Optional[float] = None,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Apply an ideal thin-lens phase to a sampled field.
    """
    field = np.asarray(field, dtype=complex)

    return field * thin_lens_phase(
        field.shape,
        wavelength=wavelength,
        dx=dx,
        dy=dy,
        focal_length=focal_length,
        center=center,
    )

@lru_cache(maxsize=None)
def circular_aperture(
    shape: Tuple[int, int],
    dx: float,
    radius: float,
    dy: Optional[float] = None,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Boolean circular aperture mask.

    Parameters
    ----------
    radius:
        Aperture radius [m].
    center:
        Optional aperture centre (x0, y0) [m].
    """
    if dy is None:
        dy = dx

    X, Y = make_coordinate_grid(shape, dx, dy)

    if center is not None:
        X = X - center[0]
        Y = Y - center[1]

    return (X**2 + Y**2) <= radius**2


def rectangular_aperture(
    shape: Tuple[int, int],
    dx: float,
    width: float,
    height: Optional[float] = None,
    dy: Optional[float] = None,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Boolean rectangular aperture mask.

    Parameters
    ----------
    width, height:
        Full aperture widths [m]. If height is None, height = width.
    """
    if dy is None:
        dy = dx
    if height is None:
        height = width

    X, Y = make_coordinate_grid(shape, dx, dy)

    if center is not None:
        X = X - center[0]
        Y = Y - center[1]

    return (np.abs(X) <= 0.5 * width) & (np.abs(Y) <= 0.5 * height)


def apply_aperture(field: ArrayLike, aperture: ArrayLike) -> np.ndarray:
    """
    Apply an arbitrary aperture mask to a field.
    """
    field = np.asarray(field, dtype=complex)
    aperture = np.asarray(aperture)

    if aperture.shape != field.shape:
        raise ValueError("aperture must have the same shape as field.")

    return field * aperture


def apply_opd(
    field: ArrayLike,
    opd_m: ArrayLike,
    wavelength: float,
    mask: Optional[ArrayLike] = None,
) -> np.ndarray:
    """
    Apply an optical path difference map to a field.

    OPD is in metres.
    """
    field = np.asarray(field, dtype=complex)
    opd_m = np.asarray(opd_m, dtype=float)

    if opd_m.shape != field.shape:
        raise ValueError("opd_m must have the same shape as field.")

    phasor = np.exp(1j * 2.0 * np.pi * opd_m / wavelength)

    if mask is None:
        return field * phasor

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != field.shape:
        raise ValueError("mask must have the same shape as field.")

    return np.where(mask, field * phasor, field)


# ============================================================
# PSF / Fourier optics helpers
# ============================================================

def fft_psf(
    field: ArrayLike,
    pad_to: Optional[int] = None,
    normalize: NormalizationMode = None,
) -> np.ndarray:
    """
    Fraunhofer-style FFT PSF helper.

    Parameters
    ----------
    field:
        Input complex pupil field.
    pad_to:
        Optional square padded size. Must be >= both field dimensions.
    normalize:
        None:
            raw intensity.
        "peak":
            divide by peak intensity.
        "sum":
            divide by total intensity.

    Returns
    -------
    psf:
        Shifted intensity PSF.
    """
    field = np.asarray(field, dtype=complex)

    if pad_to is not None:
        ny, nx = field.shape
        if pad_to < max(ny, nx):
            raise ValueError("pad_to must be >= field dimensions.")

        py = pad_to - ny
        px = pad_to - nx

        field = np.pad(
            field,
            ((py // 2, py - py // 2), (px // 2, px - px // 2)),
            mode="constant",
        )

    ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    psf = np.abs(ef) ** 2

    if normalize == "peak":
        peak = np.max(psf)
        if peak > 0:
            psf = psf / peak
    elif normalize == "sum":
        total = np.sum(psf)
        if total > 0:
            psf = psf / total
    elif normalize is not None:
        raise ValueError("normalize must be None, 'peak', or 'sum'.")

    return psf


def focal_plane_coordinates(
    shape: Tuple[int, int],
    dx_pupil: float,
    wavelength: float,
    focal_length: float,
    pad_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return physical focal-plane coordinates for an FFT of a pupil field.

    This is useful when interpreting Fraunhofer PSFs.

    Parameters
    ----------
    shape:
        FFT array shape after any padding.
    dx_pupil:
        Pupil-plane pixel scale [m/pixel].
    wavelength:
        Wavelength [m].
    focal_length:
        Focal length [m].
    pad_factor:
        Included for explicit bookkeeping. Normally leave at 1.0 if shape
        already includes padding.
    """
    del pad_factor

    ny, nx = shape
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_pupil))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_pupil))

    x_foc = wavelength * focal_length * fx
    y_foc = wavelength * focal_length * fy

    return np.meshgrid(x_foc, y_foc)


def lambda_over_d_coordinates(
    shape: Tuple[int, int],
    dx_pupil: float,
    pupil_diameter: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return FFT-plane coordinates in lambda/D units.

    For Fraunhofer coordinates, angular coordinate divided by lambda/D is
    simply spatial frequency times pupil diameter.
    """
    ny, nx = shape

    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_pupil))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_pupil))

    x_ld = fx * pupil_diameter
    y_ld = fy * pupil_diameter

    return np.meshgrid(x_ld, y_ld)


# ============================================================
# Diagnostics / comparison helpers
# ============================================================

def energy(field: ArrayLike, dx: float, dy: Optional[float] = None) -> float:
    """
    Return integrated field intensity.

    E = sum(|field|^2) dx dy
    """
    if dy is None:
        dy = dx

    field = np.asarray(field, dtype=complex)
    return float(np.sum(np.abs(field) ** 2) * dx * dy)


def best_fit_complex_scale(reference: ArrayLike, test: ArrayLike) -> complex:
    """
    Return alpha such that alpha * test best matches reference in least squares.
    """
    reference = np.asarray(reference, dtype=complex)
    test = np.asarray(test, dtype=complex)

    denom = np.vdot(test, test)
    if np.abs(denom) == 0:
        return 0.0 + 0.0j

    return np.vdot(test, reference) / denom


def remove_global_complex_scale(reference: ArrayLike, test: ArrayLike) -> np.ndarray:
    """
    Remove the best-fit complex scalar from test relative to reference.

    Useful for comparing propagated fields where global phase is arbitrary.
    """
    test = np.asarray(test, dtype=complex)
    alpha = best_fit_complex_scale(reference, test)
    return alpha * test


def relative_l2_error(
    reference: ArrayLike,
    test: ArrayLike,
    remove_scale: bool = True,
) -> float:
    """
    Relative L2 error between two complex fields.

    If remove_scale=True, a best-fit global complex scale is removed first.
    """
    reference = np.asarray(reference, dtype=complex)
    test = np.asarray(test, dtype=complex)

    if remove_scale:
        test = remove_global_complex_scale(reference, test)

    denom = np.linalg.norm(reference.ravel())
    if denom == 0:
        return float(np.linalg.norm(test.ravel()))

    return float(np.linalg.norm((reference - test).ravel()) / denom)


def assert_energy_conserved(
    field0: ArrayLike,
    field1: ArrayLike,
    dx: float,
    dy: Optional[float] = None,
    rtol: float = 1e-10,
) -> None:
    """
    Raise AssertionError if integrated intensity is not conserved.
    """
    e0 = energy(field0, dx, dy)
    e1 = energy(field1, dx, dy)

    if not np.isclose(e0, e1, rtol=rtol, atol=0.0):
        raise AssertionError(f"Energy not conserved: E0={e0:.16e}, E1={e1:.16e}")
    


# ============================================================
# Scaled Fresnel / focal-plane transforms
# ============================================================
@lru_cache(maxsize=None)
def fresnel_prepare(
    field_shape: Tuple[int, int],
    wavelength: float,
    dx: float,
    z: float,
    dy: Optional[float] = None,
    include_global_phase: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    One-step Fresnel propagation with scaled output sampling.

    Unlike fresnel_transfer_function_propagate, this method does not keep the
    same input/output pixel scale. The output sampling is:

        dx_out = wavelength * z / (N_x * dx)
        dy_out = wavelength * z / (N_y * dy)

    This is useful for propagating to a focal plane, where the natural sampling
    is Fourier-transform sampling.

    Parameters
    ----------
    field:
        Input complex field.
    wavelength:
        Wavelength [m].
    dx, dy:
        Input-plane pixel scales [m/pixel].
    z:
        Propagation distance [m].
    include_global_phase:
        If True, include exp(i k z) / (i lambda z). Usually irrelevant for
        intensity-only calculations.

    Returns
    -------
    field_out:
        Propagated complex field.
    dx_out, dy_out:
        Output-plane pixel scales [m/pixel].
    """
    if dy is None:
        dy = dx

    ny, nx = field_shape

    k = 2.0 * np.pi / wavelength

    # Input coordinates
    X1, Y1 = make_coordinate_grid(field_shape, dx, dy)

    # Output sampling
    dx_out = wavelength * abs(z) / (nx * dx)
    dy_out = wavelength * abs(z) / (ny * dy)

    X2, Y2 = make_coordinate_grid(field_shape, dx_out, dy_out)

    # Fresnel one-step form:
    #
    # U2(x2,y2) = exp(ikz)/(i lambda z)
    #             exp[i k/(2z) (x2^2+y2^2)]
    #             FFT{ U1(x1,y1) exp[i k/(2z)(x1^2+y1^2)] } dx dy
    #
    # With centred FFT bookkeeping.
    quad_in = np.exp(1j * k * (X1**2 + Y1**2) / (2.0 * z))
    quad_out = np.exp(1j * k * (X2**2 + Y2**2) / (2.0 * z))
    
    prefactor = (dx * dy) / (1j * wavelength * z)

    if include_global_phase:
        prefactor *= np.exp(1j * k * z)

    quad_out = prefactor * quad_out

    return quad_in, quad_out, k, dx_out, dy_out


def fresnel_one_step_propagate(
    field: ArrayLike,
    wavelength: float,
    dx: float,
    z: float,
    dy: Optional[float] = None,
    include_global_phase: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """
    One-step Fresnel propagation with scaled output sampling.

    Unlike fresnel_transfer_function_propagate, this method does not keep the
    same input/output pixel scale. The output sampling is:

        dx_out = wavelength * z / (N_x * dx)
        dy_out = wavelength * z / (N_y * dy)

    This is useful for propagating to a focal plane, where the natural sampling
    is Fourier-transform sampling.

    Parameters
    ----------
    field:
        Input complex field.
    wavelength:
        Wavelength [m].
    dx, dy:
        Input-plane pixel scales [m/pixel].
    z:
        Propagation distance [m].
    include_global_phase:
        If True, include exp(i k z) / (i lambda z). Usually irrelevant for
        intensity-only calculations.

    Returns
    -------
    field_out:
        Propagated complex field.
    dx_out, dy_out:
        Output-plane pixel scales [m/pixel].
    """
    field = np.asarray(field, dtype=complex)
    if dy is None:
        dy = dx
    quad_in, quad_out, k, dx_out, dy_out = fresnel_prepare(field.shape, wavelength, dx, z, dy, include_global_phase)
    
    spectrum = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(field * quad_in)
        )
    )

    field_out = quad_out * spectrum

    return field_out, dx_out, dy_out


def lens_focal_plane_field(
    field: ArrayLike,
    wavelength: float,
    dx: float,
    focal_length: float,
    dy: Optional[float] = None,
    include_global_phase: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute the sampled focal-plane field after an ideal thin lens.

    This uses the standard Fourier-transform scaling of a lens. The output
    sampling is:

        dx_focal = wavelength * focal_length / (N_x * dx)
        dy_focal = wavelength * focal_length / (N_y * dy)

    This is usually the right way to compute a properly sampled focal-plane PSF
    from a pupil field.

    Parameters
    ----------
    field:
        Input pupil-plane complex field before the lens.
    wavelength:
        Wavelength [m].
    dx, dy:
        Pupil-plane pixel scales [m/pixel].
    focal_length:
        Lens focal length [m].
    include_global_phase:
        Whether to include the constant prefactor phase.

    Returns
    -------
    field_focal:
        Complex field in the focal plane.
    dx_focal, dy_focal:
        Focal-plane pixel scales [m/pixel].
    """
    if dy is None:
        dy = dx

    field = np.asarray(field, dtype=complex)
    ny, nx = field.shape

    k = 2.0 * np.pi / wavelength

    dx_focal = wavelength * focal_length / (nx * dx)
    dy_focal = wavelength * focal_length / (ny * dy)

    Xf, Yf = make_coordinate_grid(field.shape, dx_focal, dy_focal)

    spectrum = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(field)
        )
    )

    prefactor = (dx * dy) / (1j * wavelength * focal_length)

    if include_global_phase:
        prefactor *= np.exp(1j * k * focal_length)

    # Paraxial quadratic phase in focal plane. This has no effect on intensity.
    quad_focal = np.exp(1j * k * (Xf**2 + Yf**2) / (2.0 * focal_length))

    field_focal = prefactor * quad_focal * spectrum

    return field_focal, dx_focal, dy_focal


def intensity_and_normalized(image_field: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience helper returning raw and peak-normalized intensity.
    """
    intensity = np.abs(np.asarray(image_field, dtype=complex)) ** 2
    peak = np.nanmax(intensity)

    if peak > 0:
        return intensity, intensity / peak

    return intensity, intensity



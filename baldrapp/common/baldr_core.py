import numpy as np  ##(version 2.1.1 works but incompatiple with numba)

# from numba import njit
import os

# import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from types import SimpleNamespace
from scipy.stats import pearsonr
from scipy.signal import TransferFunction, bode
from scipy.ndimage import binary_erosion
import datetime
from astropy.io import fits  # type: ignore
import time
import pickle
import pyzelda.zelda as zelda  # type: ignore
import pyzelda.ztools as ztools  # type: ignore
import pyzelda.utils.aperture as aperture  # type: ignore
from . import utilities as util
from . import DM_basis
from . import phasescreens
from . import DM_registration

from . import config_helper as cfghelp
from . import spectrum as spec
from . import fresnel
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, List, Any
from numpy.typing import NDArray

"""
1/11/25 - updated get_pupil_intensity to use fft and not mft for speed, added coldstop diam and coldstop_offset as input argunments removes and remove precomupted mask 

changed get_frame with use_pyZelda = False to use updated 'get_pupil_intensity()' method which include zwfs.optics.coldstop_diam , and remove precomupted mask since we use fft and not mft now for speed 

25/5/26 - updating with fresnel propagation and baldr specific optics, updating polychromatic (gneeralised spectrum) configuration

"""


_AUTO_SPECTRAL_BANDWIDTH = "auto"


def _resolve_detector_spectral_bandwidth(
    zwfs_ns,
    spectral_bandwidth: Optional[float | str] = _AUTO_SPECTRAL_BANDWIDTH,
):
    """
    Resolve the bandwidth passed to detect(...).

    spectral_bandwidth == "auto"
        Backwards-compatible default. Use zwfs_ns.stellar.bandwidth if present.

    spectral_bandwidth is a float
        Use this explicit bandwidth in nm.

    spectral_bandwidth is None
        Treat the input image as already spectrally integrated.
    """
    if spectral_bandwidth != _AUTO_SPECTRAL_BANDWIDTH:
        if type(spectral_bandwidth) is float:
            return spectral_bandwidth
        else:
            raise ValueError('spectral bandwidth must be either float, "auto", or None')

    if hasattr(zwfs_ns, "stellar") and hasattr(zwfs_ns.stellar, "bandwidth"):
        return zwfs_ns.stellar.bandwidth

    return None


# PID and leaky integrator copied from /Users/bencb/Documents/asgard-alignment/playground/open_loop_tests_HO.py
class PIDController:
    def __init__(
        self,
        kp=None,
        ki=None,
        kd=None,
        upper_limit=None,
        lower_limit=None,
        setpoint=None,
    ):
        if kp is None:
            kp = np.zeros(1)
        if ki is None:
            ki = np.zeros(1)
        if kd is None:
            kd = np.zeros(1)
        if lower_limit is None:
            lower_limit = np.zeros(1)
        if upper_limit is None:
            upper_limit = np.ones(1)
        if setpoint is None:
            setpoint = np.zeros(1)

        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.lower_limit = np.array(lower_limit)
        self.upper_limit = np.array(upper_limit)
        self.setpoint = np.array(setpoint)
        self.ctrl_type = "PID"

        size = len(self.kp)
        self.output = np.zeros(size)
        self.integrals = np.zeros(size)
        self.prev_errors = np.zeros(size)

    def process(self, measured):
        measured = np.array(measured)
        size = len(self.setpoint)

        if len(measured) != size:
            raise ValueError(f"Input vector size must match setpoint size: {size}")

        # Check all vectors have the same size
        error_message = []
        for attr_name in ["kp", "ki", "kd", "lower_limit", "upper_limit"]:
            if len(getattr(self, attr_name)) != size:
                error_message.append(attr_name)

        if error_message:
            raise ValueError(
                f"Input vectors of incorrect size: {' '.join(error_message)}"
            )

        if len(self.integrals) != size:
            print(
                "Reinitializing integrals, prev_errors, and output to zero with correct size."
            )
            self.integrals = np.zeros(size)
            self.prev_errors = np.zeros(size)
            self.output = np.zeros(size)

        for i in range(size):
            error = measured[i] - self.setpoint[i]  # same as rtc

            if self.ki[i] != 0:  # ONLY INTEGRATE IF KI IS NONZERO!!
                self.integrals[i] += error
                self.integrals[i] = np.clip(
                    self.integrals[i], self.lower_limit[i], self.upper_limit[i]
                )

            derivative = error - self.prev_errors[i]
            self.output[i] = (
                self.kp[i] * error
                + self.ki[i] * self.integrals[i]
                + self.kd[i] * derivative
            )
            self.prev_errors[i] = error

        return self.output

    def set_all_gains_to_zero(self):
        self.kp = np.zeros(len(self.kp))
        self.ki = np.zeros(len(self.ki))
        self.kd = np.zeros(len(self.kd))

    def reset(self):
        self.integrals.fill(0.0)
        self.prev_errors.fill(0.0)
        self.output.fill(0.0)

    def get_transfer_function(self, mode_index=0):
        """
        Returns the transfer function for the specified mode index.

        Parameters:
        - mode_index: Index of the mode for which to get the transfer function (default is 0).

        Returns:
        - scipy.signal.TransferFunction: Transfer function object.
        """
        if mode_index >= len(self.kp):
            raise IndexError("Mode index out of range.")

        # Extract gains for the selected mode
        kp = self.kp[mode_index]
        ki = self.ki[mode_index]
        kd = self.kd[mode_index]

        # Numerator and denominator for the PID transfer function: G(s) = kp + ki/s + kd*s
        # Which can be expressed as G(s) = (kd*s^2 + kp*s + ki) / s
        num = [kd, kp, ki]  # coefficients of s^2, s, and constant term
        den = [1, 0]  # s term in the denominator for integral action

        return TransferFunction(num, den)

    def plot_bode(self, mode_index=0):
        """
        Plots the Bode plot for the transfer function of a specified mode.

        Parameters:
        - mode_index: Index of the mode for which to plot the Bode plot (default is 0).
        """
        # Get transfer function
        tf = self.get_transfer_function(mode_index)

        # Generate Bode plot data
        w, mag, phase = bode(tf)

        # Plot magnitude and phase
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # Magnitude plot
        ax1.semilogx(w, mag)  # Bode magnitude plot
        ax1.set_title(f"Bode Plot for Mode {mode_index}")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Phase plot
        ax2.semilogx(w, phase)  # Bode phase plot
        ax2.set_xlabel("Frequency (rad/s)")
        ax2.set_ylabel("Phase (degrees)")
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plt.show()


class LeakyIntegrator:
    def __init__(self, ki=None, lower_limit=None, upper_limit=None, kp=None):
        # If no arguments are passed, initialize with default values
        if ki is None or lower_limit is None or upper_limit is None or kp is None:
            self.ki = []
            self.lower_limit = []
            self.upper_limit = []
            self.kp = []
        else:
            if len(ki) == 0:
                raise ValueError("ki vector cannot be empty.")
            if len(lower_limit) != len(ki) or len(upper_limit) != len(ki):
                raise ValueError(
                    "Lower and upper limit vectors must match ki vector size."
                )
            if kp is None or len(kp) != len(ki):
                raise ValueError("kp vector must be the same size as ki vector.")

            self.ki = np.array(ki)
            self.output = np.zeros(len(ki))
            self.lower_limit = np.array(lower_limit)
            self.upper_limit = np.array(upper_limit)
            self.kp = np.array(kp)  # kp is a vector now
        self.ctrl_type = "Leaky"

    def process(self, input_vector):
        input_vector = np.array(input_vector)

        # Error checks
        if len(input_vector) != len(self.ki):
            raise ValueError("Input vector size must match ki vector size.")

        size = len(self.ki)
        error_message = ""

        if len(self.ki) != size:
            error_message += "ki "
        if len(self.lower_limit) != size:
            error_message += "lower_limit "
        if len(self.upper_limit) != size:
            error_message += "upper_limit "
        if len(self.kp) != size:
            error_message += "kp "

        if error_message:
            raise ValueError("Input vectors of incorrect size: " + error_message)

        if len(self.output) != size:
            print(
                f"output.size() != size.. reinitializing output to zero with correct size"
            )
            self.output = np.zeros(size)

        # Process with the kp vector
        self.output = self.ki * self.output + self.kp * input_vector
        self.output = np.clip(self.output, self.lower_limit, self.upper_limit)

        return self.output

    def set_all_gains_to_zero(self):
        self.ki = np.zeros(len(self.ki))
        self.kp = np.zeros(len(self.kp))

    def reset(self):
        self.output = np.zeros(len(self.ki))


class detector:
    def __init__(self, dit, ron, qe, binning):
        """_summary_

        Args:
            binning (tuple): _description_ binning factor (rows to sum, columns to sum)
            qe (scalar): _description_ quantum efficiency of detector
            dit (scalar): _description_ integration time of detector
            ron (int): _description_. readout noise in electrons per pixel
        """
        self.dit = dit
        self.ron = ron
        self.qe = qe
        self.binning = binning

    def detect(self, i, include_shotnoise=True, spectral_bandwidth=None):
        """
        Convert an input photon-rate image into a detected detector frame.

        Unit convention
        ---------------
        This detector method supports two closely related input conventions,
        controlled by the `spectral_bandwidth` argument.

        1. Monochromatic or spectral-density input

            If `spectral_bandwidth` is not None, the input image `i` is assumed
            to have units:

                photons / second / wave-space-pixel / nm

            The detector first spatially bins the image, then multiplies by:

                spectral_bandwidth * dit * qe

            to obtain detected photoelectrons.

            This is the historical BaldrApp convention and corresponds to a flat
            spectrum across the supplied spectral bandwidth.

        2. Already spectrally integrated input

            If `spectral_bandwidth` is None, the input image `i` is assumed to
            already have units:

                photons / second / wave-space-pixel

            In this case no spectral-bandwidth multiplication is applied. The
            detector only applies spatial binning, detector integration time, QE,
            optional shot noise, and read noise.

        Polychromatic simulations
        -------------------------
        For non-flat or explicitly sampled spectra, spectral integration should be
        performed before calling this detector method.

        If each monochromatic image `I_density_k` has units:

            photons / second / wave-space-pixel / nm

        then the spectral integration must include wavelength-bin widths. For a
        discrete wavelength grid this is generally:

            I_rate = sum_k I_density_k * spectral_shape_k * delta_lambda_k_nm

        where `delta_lambda_k_nm` is the wavelength-bin width associated with
        wavelength sample k, and `spectral_shape_k` is the relative spectral shape
        evaluated at that sample.

        Equivalently, define integration weights:

            weights_nm_k = spectral_shape_k * delta_lambda_k_nm

        with units of nm. Then:

            I_rate = sum_k weights_nm_k * I_density_k

        The resulting `I_rate` has units:

            photons / second / wave-space-pixel

        and should be passed to this detector method with:

            spectral_bandwidth=None

        If using fractional weights normalized such that:

            sum_k weight_k = 1

        then the full-band integration can also be written as:

            I_rate = bandwidth_nm * sum_k weight_k * I_density_k

        but this is only equivalent when:

            bandwidth_nm * weight_k

        correctly represents the wavelength-bin integration weight for sample k.
        To avoid ambiguity, higher-level polychromatic frame-generation code should
        prefer explicit `weights_nm` or `delta_lambda_k_nm` integration weights.

        Design note
        -----------
        The detector intentionally does not inspect `zwfs_ns.spectrum` or any
        instrument-level spectral configuration. Spectral weighting, chromatic ZWFS
        propagation, Fresnel propagation, and bandwidth integration should be
        handled by higher-level frame-generation functions before this detector
        method is called.

        In other words:

            - Use `spectral_bandwidth` only for the legacy flat-spectrum density
            convention.
            - Use `spectral_bandwidth=None` when the image has already been
            integrated over wavelength.

        Parameters
        ----------
        i : ndarray
            Input photon image. Units depend on `spectral_bandwidth`, as described
            above.

        include_shotnoise : bool
            If True, apply Poisson shot noise after conversion to expected detected
            photoelectrons.

        spectral_bandwidth :  float, None, or "auto"
            Spectral bandwidth in nm for the legacy flat-spectrum convention.
            If None, `i` is treated as already spectrally integrated.

        Returns
        -------
        detected_image : ndarray
            Detector image after binning, detector integration time, QE, optional
            shot noise, and read noise.
        """

        """_summary_
        
        copy of the detect function generalized for this class
        
        assumes input intensity is in photons per second per pixel per nm, 
        if spectral_bandwidth is None than returns photons per pixel per nm of input light,
        otherwise returns photons per pixel
        
        Args:
            i (2D array like): _description_ input intensity (abs(field)**2!!!) before being detected on a detector (generally higher spatial resolution than detector)
            include_shotnoise (bool, optional): _description_. Defaults to True. Sample poisson distribution for each pixel (input intensity is the expected value)
            spectral_bandwidth (_type_, optional): _description_. Defaults to None. if spectral_bandwidth is None than returns photons per pixel per nm of input light,
        """

        i = sum_subarrays(array=i, block_size=(self.binning, self.binning))

        if spectral_bandwidth is None:
            i *= self.qe * self.dit
        else:
            i *= self.qe * self.dit * spectral_bandwidth

        if include_shotnoise:
            noisy_intensity = np.random.poisson(lam=i)
        else:  # no noise
            noisy_intensity = i

        if self.ron > 0:
            noisy_intensity += np.random.normal(
                0, self.ron, noisy_intensity.shape
            ).astype(int)

        return np.clip(noisy_intensity, 0, np.inf)


class StrehlModelGeneric(ABC):
    name: str
    detector_pupilmask: SimpleNamespace
    N0: NDArray


class StrehlModel(StrehlModelGeneric):
    def __init__(
        self,
        model_description="Linear regression model fitting intensities to Strehl ratio.",
    ):
        """
        Initialize the StrehlModel.

        Args:
            model_description (str): A string description of the model.
        """
        self.coefficients = None
        self.intercept = None
        self.pixel_indices = None
        self.model_description = model_description

    def fit(self, X, y, pixel_filter):
        """
        Fits the linear model of the form y = sum(alpha_i * x_i) + intercept using the normal equation.

        Args:
            X (NDArray): A 3D matrix of shape (M, N, K) where M is the number of data points,
                            and N x K is the grid of pixel intensities (best if they are normalized! ).
            y (NDArray): A vector of shape (M,) corresponding to the measured Strehl ratio.
            pixel_filter (NDArray): A boolean array of shape (N, K) that defines which pixels to use in the model.
        """
        # Ensure the pixel_filter has the correct shape
        assert (
            pixel_filter.shape == X[0].shape
        ), "pixel_filter must have the same shape as a single grid (N, K)"

        # X is shape (M, N, K). Flatten the N x K grid for each data point into a 1D array of length N * K.
        M, N, K = X.shape
        X_flattened = X.reshape(M, N * K)

        self.pixel_filter = pixel_filter

        # Select the pixel indices based on the boolean pixel_filter
        self.pixel_indices = np.where(pixel_filter)
        self.pixel_indices = np.ravel_multi_index(
            self.pixel_indices, (N, K)
        )  # Convert 2D indices to 1D

        # Select the relevant pixel indices (features subset) for fitting
        X_subset = X_flattened[:, self.pixel_indices]

        # Add a column of ones to X for the intercept term
        X_bias = np.hstack(
            [np.ones((M, 1)), X_subset]
        )  # Add a column of ones to X_subset for the intercept term

        # Solve for theta using the normal equation: theta = (X^T X)^-1 X^T y
        theta = np.linalg.inv(X_bias.T @ X_bias) @ (X_bias.T @ y)

        # Extract the intercept and coefficients
        self.intercept = theta[0]  # First element is the intercept
        self.coefficients = theta[1:]  # Remaining elements are the coefficients

    def apply_model(self, X):
        """
        Applies the fitted linear model to new 3D data.

        Args:
            X (NDArray): A 3D matrix of shape (M_new, N, K) where M_new is the number of new data points,
                            and N x K is the grid of pixel intensities.

        Returns:
            NDArray: The predicted Strehl ratio for each new data point.
        """
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.array(X)  # ensure it is an numpy array

        # X is shape (M_new, N, K). Flatten the N x K grid for each data point into a 1D array of length N * K.
        M_new, N, K = X.shape
        X_flattened = X.reshape(M_new, N * K)

        # Select only the relevant pixels (pixel indices from the fitting process)
        X_subset = X_flattened[:, self.pixel_indices]

        # Apply the model: y_pred = X_subset @ coefficients + intercept
        y_pred = X_subset @ self.coefficients + self.intercept

        return y_pred

    def describe(self):
        """
        Prints a description of the model.
        """
        print(f"Model Description: {self.model_description}")
        if self.coefficients is not None and self.intercept is not None:
            print(f"Coefficients: {self.coefficients}")
            print(f"Intercept: {self.intercept}")
            print(f"Pixel Indices (P_s): {self.pixel_indices}")
        else:
            print("Model has not been fitted yet.")

    # Function to save the model to a pickle file
    def save_model_to_pickle(self, filename):
        """
        Saves the StrehlModel object to a pickle file.

        Args:
            filename (str): The file path where the model should be saved.
            model (StrehlModel): The StrehlModel instance to save.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)


class PixelWiseStrehlModel(StrehlModelGeneric):
    def __init__(
        self,
        model_description="Linear regression model fitting intensities to Strehl ratio.",
    ):
        self.m = None  # Slopes for each pixel
        self.S0 = None  # Intercepts for each pixel
        self.pixel_indices = None
        self.model_description = model_description

    def fit(self, X, y, pixel_filter):
        """
        Fits a linear model S = m_ij * I_ij + S0 for each pixel (filtered by pixel_filter) across all samples.
        The final model will just apply the average prediction from each pixel

        Args:
            X (NDArray): A 3D matrix of shape (M, N, K) where M is the number of data points,
                            and N x K is the grid of pixel intensities.
            y (NDArray): A vector of shape (M,) corresponding to the measured Strehl ratio.
            pixel_filter (NDArray): A boolean array of shape (N, K) that defines which pixels to use in the model.
        """
        # Ensure the pixel_filter has the correct shape
        assert (
            pixel_filter.shape == X[0].shape
        ), "pixel_filter must have the same shape as a single grid (N, K)"

        # X is shape (M, N, K). Flatten the N x K grid for each data point into a 1D array of length N * K.
        M, N, K = X.shape
        X_flattened = X.reshape(M, N * K)

        # Select the pixel indices based on the boolean pixel_filter (already flattened to 1D)
        self.pixel_indices = np.where(pixel_filter.ravel())[0]

        # Select only the pixels from X that pass through the filter
        X_subset = X_flattened[:, self.pixel_indices]

        # Initialize arrays to store slopes and intercepts for each selected pixel
        num_selected_pixels = len(self.pixel_indices)
        self.m = np.zeros(num_selected_pixels)
        self.S0 = np.zeros(num_selected_pixels)

        # Fit a linear model y = m_ij * X_ij + S0 for each selected pixel
        for idx, pixel_idx in enumerate(self.pixel_indices):
            I_pixel = X_subset[:, idx]  # Intensities for this pixel across all samples

            # Use least squares to fit the linear model: y = m * I_pixel + S0
            A = np.vstack(
                [I_pixel, np.ones(len(I_pixel))]
            ).T  # Design matrix [I_pixel, 1]
            m_ij, S0_ij = np.linalg.lstsq(A, y, rcond=None)[0]  # Least squares fit

            # Store the slope and intercept
            self.m[idx] = m_ij
            self.S0[idx] = S0_ij

    def apply_model(self, X):
        """
        Applies the fitted linear model to new data and returns the estimated Strehl ratio.

        Args:
            X (NDArray): A 3D matrix of shape (M_new, N, K) where M_new is the number of new data points,
                            and N x K is the grid of pixel intensities.

        Returns:
            NDArray: The predicted Strehl ratio for each new data point (M_new,).
        """
        if self.m is None or self.S0 is None or self.pixel_indices is None:
            raise ValueError("Model has not been fitted yet.")

        M_new, N, K = X.shape

        # Flatten the N x K grid for each data point into a 1D array of length N * K.
        X_flattened = X.reshape(M_new, N * K)

        # Select only the relevant pixels (from the pixel_filter used during fitting)
        X_subset = X_flattened[:, self.pixel_indices]

        # Initialize an array to store predicted Strehl values
        S_pred = np.zeros(M_new)

        # For each sample, compute the Strehl estimate by averaging over the selected pixels
        for k in range(M_new):
            # Apply the model for each selected pixel: S_ij = m_ij * I_ij + S0_ij
            S_ij = self.m * X_subset[k] + self.S0

            # Average the Strehl predictions across the selected pixels to get the final estimate
            S_pred[k] = np.mean(S_ij)

        return S_pred

    def describe(self):
        """
        Prints a description of the model.
        """
        print(f"Model Description: {self.model_description}")
        # JCR: removed block that called non-existent attributes
        print("Model has not been fitted yet.")

    # Function to save the model to a pickle file
    def save_model_to_pickle(self, filename):
        """
        Saves the StrehlModel object to a pickle file.

        Args:
            filename (str): The file path where the model should be saved.
            model (StrehlModel): The StrehlModel instance to save.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)


def reset_telemetry(zwfs_ns):
    zwfs_ns.telem = SimpleNamespace(**init_telem_dict())
    return zwfs_ns


def reset_ctrl(zwfs_ns):
    zwfs_ns.ctrl.TT_ctrl.reset()
    zwfs_ns.ctrl.HO_ctrl.reset()

    zwfs_ns.telem = init_telem_dict()


def init_telem_dict():
    # i_list is intensity measured on the detector
    # i_dm_list is intensity interpolated onto DM actuators - it is used only in zonal_interp control methods
    # s_list is processed intensity signal used in the control loop (e.g. I - I0)
    # e_* is control error signals
    # u_* is control signals (e.g. after PID control)
    # c_* is DM command signals
    telemetry_dict = {
        "i_list": [],
        "i_dm_list": [],
        "s_list": [],
        "e_TT_list": [],
        "u_TT_list": [],
        "c_TT_list": [],
        "e_HO_list": [],
        "u_HO_list": [],
        "c_HO_list": [],
        "atm_disturb_list": [],
        "dm_disturb_list": [],
        "rmse_list": [],
        "flux_outside_pupil_list": [],
        "residual_list": [],
        "field_phase": [],
        "strehl": [],
    }
    return telemetry_dict


# def save_telemetry( zwfs_ns , savename = None, overwrite=True):

#     tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

#     telem_dict = vars(zwfs_ns.telem )
#     # Create a list of HDUs (Header Data Units)
#     hdul = fits.HDUList()

#     # Add each list to the HDU list as a new extension
#     for list_name, data_list in telem_dict.items():
#         # Convert list to numpy array for FITS compatibility
#         data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

#         # Create a new ImageHDU with the data
#         hdu = fits.ImageHDU(data_array)

#         # Set the EXTNAME header to the variable name
#         hdu.header['EXTNAME'] = list_name

#         # Append the HDU to the HDU list
#         hdul.append(hdu)

#     # Write the HDU list to a FITS file
#     if savename is None:
#         savename = f'~/Downloads/telemetry_simulation_{tstamp}.fits'
#     hdul.writeto( savename, overwrite=True)

#     return hdul


def save_telemetry(telemetry_ns, savename=None, overwrite=True, return_fits=False):

    tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

    # Create a Primary HDU (Header/Data Unit)
    primary_hdu = fits.PrimaryHDU()

    # Create a list to hold the individual HDUs
    hdul = fits.HDUList([primary_hdu])

    # Iterate through the telemetry dictionary and create HDUs for each key-value pair
    for key, value in vars(telemetry_ns).items():
        if value:  # Only add non-empty data arrays/lists
            value_array = np.array(value)

            if np.iscomplexobj(
                value_array
            ):  # Check if the array contains complex numbers
                # Split into real and imaginary parts
                real_part = np.real(value_array)
                imag_part = np.imag(value_array)

                # Create HDU for the real part
                real_hdu = fits.ImageHDU(real_part, name=f"{key.upper()}_REAL")
                real_hdu.header["EXTNAME"] = f"{key}_REAL"
                real_hdu.header["COMMENT"] = "Real part of complex values"

                # Create HDU for the imaginary part
                imag_hdu = fits.ImageHDU(imag_part, name=f"{key.upper()}_IMAG")
                imag_hdu.header["EXTNAME"] = f"{key}_IMAG"
                imag_hdu.header["COMMENT"] = "Imaginary part of complex values"

                # Append both HDUs to the HDU list
                hdul.append(real_hdu)
                hdul.append(imag_hdu)
            else:
                # Handle non-complex data as usual
                hdu = fits.ImageHDU(value_array, name=key.upper())
                hdu.header["EXTNAME"] = key.upper()
                hdu.header["COMMENT"] = f"Data corresponding to {key.upper()}"
                hdul.append(hdu)

        if savename is None:
            savename = f"~/Downloads/telemetry_simulation_{tstamp}.fits"

        # Write the HDU list to a FITS file
        hdul.writeto(savename, overwrite=overwrite)

        if return_fits:
            return hdul


def roll_screen_on_dm(
    zwfs_ns,
    Nmodes_removed,
    ph_scale=0.2,
    actuators_per_iteration=0.5,
    number_of_screen_initiations=100,
    opd_internal=None,
):

    t0 = time.time()

    print(f"Rolling screen on DM with {Nmodes_removed} modes removed")
    # flux from configuration
    photon_flux_per_pixel_at_vlti = (
        zwfs_ns.throughput.vlti_throughput
        * (np.pi * (zwfs_ns.grid.D / 2) ** 2)
        / (np.pi * zwfs_ns.pyZelda.pupil_diameter / 2) ** 2
        * util.magnitude_to_photon_flux(
            magnitude=zwfs_ns.stellar.magnitude,
            band=zwfs_ns.stellar.waveband,
            wavelength=1e9 * zwfs_ns.optics.wvl0,
        )
    )

    amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil

    if opd_internal is None:
        opd_internal = 0 * amp_input

    nx_size = int(zwfs_ns.dm.Nact_x / actuators_per_iteration)

    scrn = phasescreens.PhaseScreenKolmogorov(
        nx_size=nx_size,
        pixel_scale=zwfs_ns.grid.D / nx_size,
        r0=zwfs_ns.atmosphere.r0,
        L0=zwfs_ns.atmosphere.l0,
        random_seed=None,
    )
    opd_input = 0 * amp_input
    I0 = get_I0(
        opd_input,
        amp_input,
        opd_internal,
        zwfs_ns,
        detector=zwfs_ns.detector,
        include_shotnoise=True,
        use_pyZelda=True,
    )

    N0 = get_N0(
        opd_input,
        amp_input,
        opd_internal,
        zwfs_ns,
        detector=zwfs_ns.detector,
        include_shotnoise=True,
        use_pyZelda=True,
    )

    # first stage AO
    basis_cropped = ztools.zernike.zernike_basis(
        nterms=Nmodes_removed + 2, npix=zwfs_ns.pyZelda.pupil_diameter
    )
    # we have padding around telescope pupil (check zwfs_ns.pyZelda.pupil.shape and zwfs_ns.pyZelda.pupil_diameter)
    # so we need to put basis in the same frame
    basis_template = np.zeros(zwfs_ns.pyZelda.pupil.shape)
    basis = np.array(
        [
            util.insert_concentric(np.nan_to_num(b, nan=0), basis_template)
            for b in basis_cropped
        ]
    )

    pupil_disk = basis[
        0
    ]  # we define a disk pupil without secondary - useful for removing Zernike modes later

    telemetry = {
        "I0": [I0],
        "N0": [N0],
        "dm_cmd": [],
        "i": [],
        "t_dm0": [],
        "t_dm1": [],
        "t_i0": [],
        "t_i1": [],
    }

    telem_ns = SimpleNamespace(**telemetry)

    for it in range(number_of_screen_initiations):

        print(f"Iteration {it} of {number_of_screen_initiations}")
        scrn.add_row()

        telem_ns.t_dm0.append(time.time() - t0)
        # scaling_factor=0.05, drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False
        zwfs_ns.dm.current_cmd = util.create_phase_screen_cmd_for_DM(
            scrn,
            scaling_factor=ph_scale,
            drop_indicies=[0, 11, 11 * 12, -1],
            plot_cmd=False,
        )

        opd_current_dm = zwfs_ns.compiled_dm.eval(
            zwfs_ns.dm.current_cmd * zwfs_ns.dm.opd_per_cmd
        )

        phi = (
            zwfs_ns.grid.pupil_mask * 2 * np.pi / zwfs_ns.optics.wvl0 * (opd_current_dm)
        )

        pupil_disk_cropped, atm_in_pupil = util.crop_pupil(pupil_disk, phi)

        # test project onto Zernike modes
        mode_coefficients = np.array(
            ztools.zernike.opd_expand(
                atm_in_pupil * pupil_disk_cropped,
                nterms=len(basis),
                aperture=pupil_disk_cropped,
            )
        )

        # do the reconstruction for N modes
        reco = np.sum(
            mode_coefficients[:Nmodes_removed, np.newaxis, np.newaxis]
            * basis[:Nmodes_removed, :, :],
            axis=0,
        )

        # remove N modes
        ao_1 = pupil_disk * (phi - reco)

        # convert to OPD map
        opd_map = (
            zwfs_ns.pyZelda.pupil
            * zwfs_ns.optics.wvl0
            / (2 * np.pi)
            * (ao_1 + opd_internal)
        )

        telem_ns.t_dm1.append(time.time() - t0)

        Ic = photon_flux_per_pixel_at_vlti * zwfs_ns.pyZelda.propagate_opd_map(
            opd_map, wave=zwfs_ns.optics.wvl0
        )

        telem_ns.t_i0.append(time.time() - t0)
        i = detect(
            Ic,
            binning=(zwfs_ns.detector.binning, zwfs_ns.detector.binning),
            qe=zwfs_ns.detector.qe,
            dit=zwfs_ns.detector.dit,
            ron=zwfs_ns.detector.ron,
            include_shotnoise=True,
            spectral_bandwidth=zwfs_ns.stellar.bandwidth,
        )
        telem_ns.t_i1.append(time.time() - t0)

        telem_ns.i.append(i)
        telem_ns.dm_cmd.append(zwfs_ns.dm.current_cmd)

    return telem_ns


def calibrate_strehl_model(
    zwfs_ns,
    save_results_path=None,
    train_fraction=0.6,
    correlation_threshold=0.5,
    number_of_screen_initiations=50,
    scrn_scaling_grid=np.logspace(-1, 0.2, 5),
    model_type="PixelWiseStrehlModel",
):
    """_summary_

    TO DO - did not include internal aberrations here

    Training a linear model to map a subset of pixel intensities to Strehl Ratio in a Zernike wavefront sensor.
    The model is trained by applying various instances of scaled Kolmogorov phasescreens on the DM and measuring the ZWFS intensity response
    The pixel intensities are normalized by the average clear (no phasemask) pupil intensity measured in the detector within the active pupil region.
    pixels are selected that have a Pearson R correlation with the Strehl ratio (determined by DM influience function ) > correlation_threshold

    quiet slow in simulation mode, but should be fast in real life (i.e on a real DM / camera)

    Args:
        zwfs_ns (_type_): _description_ namespace initialized from a configuration file. e.g.
        save_results_path (_type_, optional): _description_. Defaults to None. where to save the results?
            if not None a timestamped folder will be created in the path and all plots/ results save here.
        train_fraction (float, optional): _description_. Defaults to 0.6. what fraction of the data should be used for training?
        correlation_threshold (float, optional): _description_. Defaults to 0.5. what is the minimum correlation threshold
            between a pixels intensity and the Strehl ratio for it to be included in the model?
        number_of_screen_initiations (int, optional): _description_. Defaults to 50. How many unique instances of a Kolmogorov screens to initialize on
            the DM for training the model?
        scrn_scaling_grid (_type_, optional): _description_. what scaling grid do you want to apply to the phasescreen for training the model? Defaults to np.logspace(-1,0.2,5).
        model_type (str, optional): _description_. Defaults to 'PixelWiseStrehlModel'. what type of model do you want to train/apply?  model_type must be 'lin_comb' or 'PixelWiseStrehlModel'
    Returns:
        _type_: the trained Strehl model
    """

    print(f"USING ---- {model_type} ----")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
    if save_results_path is not None:
        save_results_path = os.path.join(
            save_results_path, f"strehl_model_config-{zwfs_ns.name}_{timestamp}", ""
        )
        if os.path.exists(save_results_path) == False:
            os.makedirs(save_results_path)

    ## FITTING THE MODEL
    model_description = "Linear regression model fitting intensities to Strehl ratio."

    if model_type == "lin_comb":
        model = StrehlModel(model_description)

    elif model_type == "PixelWiseStrehlModel":
        model = PixelWiseStrehlModel(model_description)
    else:
        raise ValueError(
            "invalid model_type! model_type must be 'lin_comb' or 'PixelWiseStrehlModel'"
        )

    # first stage AO
    basis_cropped = ztools.zernike.zernike_basis(
        nterms=150, npix=zwfs_ns.pyZelda.pupil_diameter
    )
    # we have padding around telescope pupil (check zwfs_ns.pyZelda.pupil.shape and zwfs_ns.pyZelda.pupil_diameter)
    # so we need to put basis in the same frame
    basis_template = np.zeros(zwfs_ns.pyZelda.pupil.shape)
    basis = np.array(
        [
            util.insert_concentric(np.nan_to_num(b, nan=0), basis_template)
            for b in basis_cropped
        ]
    )

    pupil_disk = basis[
        0
    ]  # we define a disk pupil without secondary - useful for removing Zernike modes later

    Nmodes_removed = 2  # Default will be to remove Zernike modes

    photon_flux_per_pixel_at_vlti = (
        zwfs_ns.throughput.vlti_throughput
        * (np.pi * (zwfs_ns.grid.D / 2) ** 2)
        / (np.pi * zwfs_ns.pyZelda.pupil_diameter / 2) ** 2
        * util.magnitude_to_photon_flux(
            magnitude=zwfs_ns.stellar.magnitude,
            band=zwfs_ns.stellar.waveband,
            wavelength=1e9 * zwfs_ns.optics.wvl0,
        )
    )

    scrn_list = []
    for _ in range(number_of_screen_initiations):
        # scrn = ps.PhaseScreenKolmogorov(nx_size=zwfs_ns.grid.N, pixel_scale=dx, r0=zwfs_ns.atmosphere.r0, L0=zwfs_ns.atmosphere.l0, random_seed=1)
        scrn = phasescreens.PhaseScreenKolmogorov(
            nx_size=24,
            pixel_scale=zwfs_ns.grid.D / 24,
            r0=zwfs_ns.atmosphere.r0,
            L0=zwfs_ns.atmosphere.l0,
            random_seed=None,
        )
        scrn_list.append(scrn)
        # zwfs_ns.grid.pupil_mask * util.insert_concentric( scrn.scrn, zwfs_ns.pyZelda.pupil ) )

    telemetry = {
        "I0": [],
        "N0": [],
        "scrn": [],
        "ao_1": [],
        "Ic": [],
        "i": [],
        "i_norm": [],
        "strehl": [],
        "dm_cmd": [],
        "b": [],
        "b_detector": [],
        "pupilmask_in_detector": [],
        "ao_2": [],
    }
    telem_ns = SimpleNamespace(**telemetry)

    for it in range(len(scrn_list)):

        # roll screen
        # scrn.add_row()
        for ph_scale in scrn_scaling_grid:

            # scaling_factor=0.05, drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False
            zwfs_ns.dm.current_cmd = util.create_phase_screen_cmd_for_DM(
                scrn_list[it],
                scaling_factor=ph_scale,
                drop_indicies=[0, 11, 11 * 12, -1],
                plot_cmd=False,
            )

            opd_current_dm = zwfs_ns.compiled_dm.eval(
                zwfs_ns.dm.current_cmd * zwfs_ns.dm.opd_per_cmd
            )

            phi = (
                zwfs_ns.grid.pupil_mask
                * 2
                * np.pi
                / zwfs_ns.optics.wvl0
                * (opd_current_dm)
            )

            pupil_disk_cropped, atm_in_pupil = util.crop_pupil(pupil_disk, phi)

            # test project onto Zernike modes
            mode_coefficients = np.array(
                ztools.zernike.opd_expand(
                    atm_in_pupil * pupil_disk_cropped,
                    nterms=len(basis),
                    aperture=pupil_disk_cropped,
                )
            )

            # do the reconstruction for N modes
            reco = np.sum(
                mode_coefficients[:Nmodes_removed, np.newaxis, np.newaxis]
                * basis[:Nmodes_removed, :, :],
                axis=0,
            )

            # remove N modes
            ao_1 = pupil_disk * (phi - reco)

            # add vibrations
            # TO DO

            # for calibration purposes
            print(
                f"for {Nmodes_removed} Zernike modes removed (scrn_scaling={ph_scale}),\n \
                atmospheric conditions r0= {round(zwfs_ns.atmosphere.r0,2)}m at a central wavelength {round(1e6*zwfs_ns.optics.wvl0,2)}um\n\
                    post 1st stage AO rmse [nm rms] = ",
                round(
                    1e9
                    * (zwfs_ns.optics.wvl0 / (2 * np.pi) * ao_1)[
                        zwfs_ns.pyZelda.pupil > 0.5
                    ].std()
                ),
            )

            # apply DM
            # ao1 *= DM_field

            # convert to OPD map
            opd_map = zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2 * np.pi) * ao_1

            if it == 0:

                N0_wsp = photon_flux_per_pixel_at_vlti * ztools.propagate_opd_map(
                    0 * opd_map,
                    zwfs_ns.pyZelda.mask_diameter,
                    0 * zwfs_ns.pyZelda.mask_depth,
                    zwfs_ns.pyZelda.mask_substrate,
                    zwfs_ns.pyZelda.mask_Fratio,
                    zwfs_ns.pyZelda.pupil_diameter,
                    zwfs_ns.pyZelda.pupil,
                    wave=zwfs_ns.optics.wvl0,
                    fourier_filter_diam=zwfs_ns.pyZelda.fourier_filter_diam,
                )

                I0_wsp = photon_flux_per_pixel_at_vlti * ztools.propagate_opd_map(
                    0 * opd_map,
                    zwfs_ns.pyZelda.mask_diameter,
                    zwfs_ns.pyZelda.mask_depth,
                    zwfs_ns.pyZelda.mask_substrate,
                    zwfs_ns.pyZelda.mask_Fratio,
                    zwfs_ns.pyZelda.pupil_diameter,
                    zwfs_ns.pyZelda.pupil,
                    wave=zwfs_ns.optics.wvl0,
                    fourier_filter_diam=zwfs_ns.pyZelda.fourier_filter_diam,
                )

                # bin to detector pixelspace
                I0 = detect(
                    I0_wsp,
                    binning=(zwfs_ns.detector.binning, zwfs_ns.detector.binning),
                    qe=zwfs_ns.detector.qe,
                    dit=zwfs_ns.detector.dit,
                    ron=zwfs_ns.detector.ron,
                    include_shotnoise=True,
                    spectral_bandwidth=zwfs_ns.stellar.bandwidth,
                )
                N0 = detect(
                    N0_wsp,
                    binning=(zwfs_ns.detector.binning, zwfs_ns.detector.binning),
                    qe=zwfs_ns.detector.qe,
                    dit=zwfs_ns.detector.dit,
                    ron=zwfs_ns.detector.ron,
                    include_shotnoise=True,
                    spectral_bandwidth=zwfs_ns.stellar.bandwidth,
                )

                pupilmask_in_detector = 0 < sum_subarrays(
                    zwfs_ns.pyZelda.pupil,
                    (zwfs_ns.detector.binning, zwfs_ns.detector.binning),
                )

                telem_ns.pupilmask_in_detector.append(pupilmask_in_detector)
                telem_ns.I0.append(I0)
                telem_ns.N0.append(N0)

            # caclulate Strehl ratio
            strehl = np.exp(-np.var(ao_1[zwfs_ns.pyZelda.pupil > 0.5]))

            b, _ = ztools.create_reference_wave_beyond_pupil_with_aberrations(
                opd_map,
                zwfs_ns.pyZelda.mask_diameter,
                zwfs_ns.pyZelda.mask_depth,
                zwfs_ns.pyZelda.mask_substrate,
                zwfs_ns.pyZelda.mask_Fratio,
                zwfs_ns.pyZelda.pupil_diameter,
                zwfs_ns.pyZelda.pupil,
                zwfs_ns.optics.wvl0,
                clear=np.array([]),
                sign_mask=np.array([]),
                cpix=False,
            )

            b_detector = average_subarrays(
                abs(b), (zwfs_ns.detector.binning, zwfs_ns.detector.binning)
            )

            # normalized such that np.sum( I0 ) / np.sum( N0 ) ~ 1 where N0.max() = 1.
            # do normalization by known area of the pupil and the input stellar magnitude at the given wavelength
            # represent as #photons / s / pixel / nm

            Ic = photon_flux_per_pixel_at_vlti * zwfs_ns.pyZelda.propagate_opd_map(
                opd_map, wave=zwfs_ns.optics.wvl0
            )

            i = detect(
                Ic,
                binning=(zwfs_ns.detector.binning, zwfs_ns.detector.binning),
                qe=zwfs_ns.detector.qe,
                dit=zwfs_ns.detector.dit,
                ron=zwfs_ns.detector.ron,
                include_shotnoise=True,
                spectral_bandwidth=zwfs_ns.stellar.bandwidth,
            )

            # telem_ns.ao_1.append(zwfs_ns.pyZelda.pupil * ao_1)
            telem_ns.i.append(i)

            # this is what we use for model
            telem_ns.i_norm.append(
                i / np.mean(telem_ns.N0[0][telem_ns.pupilmask_in_detector[0]])
            )

            telem_ns.Ic.append(Ic)
            telem_ns.strehl.append(strehl)
            telem_ns.b.append(b)
            telem_ns.b_detector.append(b_detector)
            telem_ns.dm_cmd.append(zwfs_ns.dm.current_cmd)

        print(f"iteration {it} done")

    def _compute_correlation_map(intensity_frames, strehl_ratios):
        # intensity_frames: k x N x M array (k frames of N x M pixels)
        # strehl_ratios: k array (Strehl ratio for each frame)

        k, N, M = intensity_frames.shape
        correlation_map = np.zeros((N, M))

        for i in range(N):
            for j in range(M):
                pixel_intensity_series = intensity_frames[:, i, j]
                correlation_map[i, j], _ = pearsonr(
                    pixel_intensity_series, strehl_ratios
                )

        return correlation_map

    correlation_map = _compute_correlation_map(
        np.array(telem_ns.i), np.array(telem_ns.strehl)
    )

    # SNR
    SNR = np.mean(telem_ns.i, axis=0) / np.std(telem_ns.i, axis=0)

    if save_results_path is not None:
        util.nice_heatmap_subplots(
            im_list=[correlation_map],
            cbar_label_list=["Pearson R"],
            savefig=save_results_path + "strehl_vs_intensity_pearson_R.png",
        )  # fig_path + 'strehl_vs_intensity_pearson_R.png' )

        util.nice_heatmap_subplots(
            im_list=[SNR / np.max(SNR)],
            cbar_label_list=["normalized SNR"],
            savefig=save_results_path + "SNR_simulation.png",
        )  # fig_path + 'SNR_simulation.png' )

    # Select top 5% of pixels with the highest correlation

    selected_pixels = correlation_map > correlation_threshold

    if save_results_path is not None:
        plt.figure()
        plt.imshow(selected_pixels)
        plt.colorbar(label="filter")
        plt.savefig(
            save_results_path + "selected_pixels.png", bbox_inches="tight", dpi=300
        )
        plt.show()

    # pixel_indices = np.where( selected_pixels )

    i_train = int(train_fraction * len(telem_ns.i))

    y_train = np.array(telem_ns.strehl)[:i_train]
    X_train = np.array(telem_ns.i_norm)[:i_train]

    y_test = np.array(telem_ns.strehl)[i_train:]
    X_test = np.array(telem_ns.i_norm)[i_train:]

    # coefficients, intercept = model.fit_linear_model(x, y)
    model.fit(X=X_train, y=y_train, pixel_filter=selected_pixels)

    y_fit = model.apply_model(X_test)

    # add the pupil in
    model.name = zwfs_ns.name  # so we know what config file was used

    model.detector_pupilmask = telem_ns.pupilmask_in_detector[
        0
    ]  # mask used to get pixels for normalization
    model.N0 = telem_ns.N0[0]  # clear pupil intensity used for normalization

    # y_fit = model.predict(x)

    # show out of sample test results
    if save_results_path is not None:
        util.plot_data_and_residuals(
            y_test,
            y_test,
            y_fit,
            xlabel=r"$\text{Strehl Ratio}$",
            ylabel=r"$\text{Predicted Strehl Ratio}$",
            residual_ylabel=r"$\Delta$",
            label_1="1:1",
            label_2="model",
            savefig=save_results_path + "strehl_linear_fit.png",
        )

    # save the model
    if save_results_path is not None:
        model.save_model_to_pickle(
            filename=save_results_path
            + f"strehl_model_config-{zwfs_ns.name}_{timestamp}.pkl"
        )

    return model


def get_theoretical_reference_pupils(
    wavelength=1.65e-6,
    F_number=21.2,
    mask_diam=1.2,
    diameter_in_angular_units=True,
    get_individual_terms=False,
    phaseshift=np.pi / 2,
    padding_factor=4,
    debug=True,
    analytic_solution=True,
):
    """
    get theoretical reference pupil intensities of ZWFS with / without phasemask
    NO ABERRATIONS

    Parameters
    ----------
    wavelength : TYPE, optional
        DESCRIPTION. input wavelength The default is 1.65e-6.
    F_number : TYPE, optional
        DESCRIPTION. The default is 21.2.
    mask_diam : phase dot diameter. TYPE, optional
            if diameter_in_angular_units=True than this has diffraction limit units ( 1.22 * f * lambda/D )
            if  diameter_in_angular_units=False than this has physical units (m) determined by F_number and wavelength
        DESCRIPTION. The default is 1.2.
    diameter_in_angular_units : TYPE, optional
        DESCRIPTION. The default is True.
    get_individual_terms : Type optional
        DESCRIPTION : if false (default) with jsut return intensity, otherwise return P^2, abs(M)^2 , phi + mu
    phaseshift : TYPE, optional
        DESCRIPTION. phase phase shift imparted on input field (radians). The default is np.pi/2.
    padding_factor : pad to change the resolution in image plane. TYPE, optional
        DESCRIPTION. The default is 4.
    debug : TYPE, optional
        DESCRIPTION. Do we want to plot some things? The default is True.
    analytic_solution: TYPE, optional
        DESCRIPTION. use analytic formula or calculate numerically? The default is True.
    Returns
    -------
    Ic, reference pupil intensity with phasemask in
    P, reference pupil intensity with phasemask out

    """
    pupil_radius = 1  # Pupil radius in meters

    # Define the grid in the pupil plane
    N = 2**9 + 1  # 256  # Number of grid points (assumed to be square)
    L_pupil = 2 * pupil_radius  # Pupil plane size (physical dimension)
    dx_pupil = L_pupil / N  # Sampling interval in the pupil plane
    x_pupil = np.linspace(-L_pupil / 2, L_pupil / 2, N)  # Pupil plane coordinates
    y_pupil = np.linspace(-L_pupil / 2, L_pupil / 2, N)
    X_pupil, Y_pupil = np.meshgrid(x_pupil, y_pupil)

    # Define a circular pupil function
    pupil = np.sqrt(X_pupil**2 + Y_pupil**2) <= pupil_radius

    # Zero padding to increase resolution
    # Increase the array size by padding (e.g., 4x original size)
    N_padded = N * padding_factor
    pupil_padded = np.zeros((N_padded, N_padded))
    start_idx = (N_padded - N) // 2
    pupil_padded[start_idx : start_idx + N, start_idx : start_idx + N] = pupil

    # Perform the Fourier transform on the padded array (normalizing for the FFT)
    pupil_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_padded)))

    # Compute the Airy disk scaling factor (1.22 * λ * F)
    airy_scale = 1.22 * wavelength * F_number

    # Image plane sampling interval (adjusted for padding)
    L_image = wavelength * F_number / dx_pupil  # Total size in the image plane
    dx_image_padded = (
        L_image / N_padded
    )  # Sampling interval in the image plane with padding

    if diameter_in_angular_units:
        x_image_padded = (
            np.linspace(-L_image / 2, L_image / 2, N_padded) / airy_scale
        )  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image / 2, L_image / 2, N_padded) / airy_scale
    else:
        x_image_padded = np.linspace(
            -L_image / 2, L_image / 2, N_padded
        )  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image / 2, L_image / 2, N_padded)

    X_image_padded, Y_image_padded = np.meshgrid(x_image_padded, y_image_padded)

    if diameter_in_angular_units:
        mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 4
    else:
        mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 4

    psi_B = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_padded)))

    b = np.fft.fftshift(np.fft.ifft2(mask * psi_B))

    if debug:

        psf = (
            np.abs(pupil_ft) ** 2
        )  # Get the PSF by taking the square of the absolute value
        psf /= np.max(psf)  # Normalize PSF intensity

        if diameter_in_angular_units:
            zoom_range = 3  # Number of Airy disk radii to zoom in on
        else:
            zoom_range = 3 * airy_scale

        extent = (-zoom_range, zoom_range, -zoom_range, zoom_range)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(
            psf,
            extent=(
                x_image_padded.min(),
                x_image_padded.max(),
                y_image_padded.min(),
                y_image_padded.max(),
            ),
            cmap="gray",
        )
        ax.contour(
            X_image_padded,
            Y_image_padded,
            mask,
            levels=[0.5],
            colors="red",
            linewidths=2,
            label="phasemask",
        )
        # ax[1].imshow( mask, extent=(x_image_padded.min(), x_image_padded.max(), y_image_padded.min(), y_image_padded.max()), cmap='gray')
        # for axx in ax.reshape(-1):
        #    axx.set_xlim(-zoom_range, zoom_range)
        #    axx.set_ylim(-zoom_range, zoom_range)
        ax.set_xlim(-zoom_range, zoom_range)
        ax.set_ylim(-zoom_range, zoom_range)
        ax.set_title("PSF")
        ax.legend()
        # ax[1].set_title('phasemask')

    # if considering complex b
    # beta = np.angle(b) # complex argunment of b
    # M = b * (np.exp(1J*theta)-1)**0.5

    # relabelling
    theta = phaseshift  # rad ,
    P = pupil_padded.copy()

    if analytic_solution:

        M = abs(b) * np.sqrt((np.cos(theta) - 1) ** 2 + np.sin(theta) ** 2)
        mu = np.angle(
            (np.exp(1j * theta) - 1)
        )  # np.arctan( np.sin(theta)/(np.cos(theta)-1) ) #

        phi = np.zeros(P.shape)  # added aberrations

        # out formula ----------
        # if measured_pupil!=None:
        #    P = measured_pupil / np.mean( P[P > np.mean(P)] ) # normalize by average value in Pupil

        Ic = P**2 + abs(M) ** 2 + 2 * P * abs(M) * np.cos(phi + mu)  # + beta)
        if not get_individual_terms:
            return (P, Ic)
        else:
            return (P, abs(M), phi + mu)
    else:

        # phasemask filter

        T_on = 1
        T_off = 1
        H = T_off * (1 + (T_on / T_off * np.exp(1j * theta) - 1) * mask)

        Ic = abs(np.fft.fftshift(np.fft.ifft2(H * psi_B))) ** 2

        return (P, Ic)


def get_grids(
    wavelength=1.65e-6,
    F_number=21.2,
    mask_diam=1.2,
    diameter_in_angular_units=True,
    N=256,
    padding_factor=4,
):
    """
    get theoretical reference pupil intensities of ZWFS with / without phasemask


    Parameters
    ----------
    wavelength : TYPE, optional
        DESCRIPTION. input wavelength The default is 1.65e-6.
    F_number : TYPE, optional
        DESCRIPTION. The default is 21.2.
    mask_diam : phase dot diameter. TYPE, optional
            if diameter_in_angular_units=True than this has diffraction limit units ( 1.22 * f * lambda/D )
            if  diameter_in_angular_units=False than this has physical units (m) determined by F_number and wavelength
        DESCRIPTION. The default is 1.2.
    diameter_in_angular_units : TYPE, optional
        DESCRIPTION. The default is True.
    get_individual_terms : Type optional
        DESCRIPTION : if false (default) with jsut return intensity, otherwise return P^2, abs(M)^2 , phi + mu
    phaseshift : TYPE, optional
        DESCRIPTION. phase phase shift imparted on input field (radians). The default is np.pi/2.
    padding_factor : pad to change the resolution in image plane. TYPE, optional
        DESCRIPTION. The default is 4.
    debug : TYPE, optional
        DESCRIPTION. Do we want to plot some things? The default is True.
    analytic_solution: TYPE, optional
        DESCRIPTION. use analytic formula or calculate numerically? The default is True.
    Returns
    -------
    Ic, reference pupil intensity with phasemask in
    P, reference pupil intensity with phasemask out

    """
    pupil_radius = 1  # Pupil radius in meters

    # Define the grid in the pupil plane
    # N = 2**9 + 1 #256  # Number of grid points (assumed to be square)
    L_pupil = 2 * pupil_radius  # Pupil plane size (physical dimension)
    dx_pupil = L_pupil / N  # Sampling interval in the pupil plane
    x_pupil = np.linspace(-L_pupil / 2, L_pupil / 2, N)  # Pupil plane coordinates
    y_pupil = np.linspace(-L_pupil / 2, L_pupil / 2, N)
    X_pupil, Y_pupil = np.meshgrid(x_pupil, y_pupil)

    # Define a circular pupil function
    pupil = np.sqrt(X_pupil**2 + Y_pupil**2) <= pupil_radius

    # Zero padding to increase resolution
    # Increase the array size by padding (e.g., 4x original size)
    N_padded = N * padding_factor
    pupil_padded = np.zeros((N_padded, N_padded))
    start_idx = (N_padded - N) // 2
    pupil_padded[start_idx : start_idx + N, start_idx : start_idx + N] = pupil

    # Perform the Fourier transform on the padded array (normalizing for the FFT)
    # pupil_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_padded)))

    # Compute the Airy disk scaling factor (1.22 * λ * F)
    airy_scale = 1.22 * wavelength * F_number

    # Image plane sampling interval (adjusted for padding)
    L_image = wavelength * F_number / dx_pupil  # Total size in the image plane
    # dx_image_padded = L_image / N_padded  # Sampling interval in the image plane with padding

    if diameter_in_angular_units:
        x_image_padded = (
            np.linspace(-L_image / 2, L_image / 2, N_padded) / airy_scale
        )  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image / 2, L_image / 2, N_padded) / airy_scale
    else:
        x_image_padded = np.linspace(
            -L_image / 2, L_image / 2, N_padded
        )  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image / 2, L_image / 2, N_padded)

    X_image_padded, Y_image_padded = np.meshgrid(x_image_padded, y_image_padded)

    mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 4

    return pupil_padded, mask


### DM actuator influence options
# @njit
def gaussian_displacement(c_i, sigma_i, x, y, x0, y0):
    """Compute Gaussian displacement for a single actuator."""
    return c_i * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma_i**2)
    # return c_i / (1 + ((x - x0)**2 + (y - y0)**2) / (0.7*sigma_i)**2) #lorentzian approximation to speed things up - good blanace accuracy and speed
    # return np.clip(c_i * (1 - ((x - x0)**2 + (y - y0)**2)/ (1.2 * sigma_i**2)), a_min=0, a_max=None) #quadratic
    # return c_i * ( (x - x0)**2 + (y - y0)**2 <= sigma_i**2 ) # box profile'


# t0 = time.time()
# OL_data = bldr.roll_screen_on_dm( zwfs_ns=zwfs_ns,  Nmodes_removed=14, ph_scale = 0.2, actuators_per_iteration = 0.5, number_of_screen_initiations= 200, opd_internal=opd_internal)
# t1 = time.time()
# print( t1- t0)
# x = np.linspace(-10,10, 100)
# c_i=1; x0=0; y = 0; y0 = 0; sigma_i = 2
# x = np.linspace(-10,10, 100)
# plt.figure();
# plt.plot( x , c_i / (1 + ((x - x0)**2 + (y - y0)**2) / (0.7*sigma_i)**2) , label='lorentizian approx')
# plt.plot( x , c_i * np.exp(-((x - x0)**2 + (y - y0)**2) / sigma_i**2) , label='gaussian')
# plt.plot( x , np.clip(c_i * (1 - ((x - x0)**2 + (y - y0)**2)/ ( sigma_i**2) ) , a_min=0, a_max=None) ,label = 'quadratic')
# plt.plot( x , c_i * ( (x - x0)**2 + (y - y0)**2 <= sigma_i**2 ) ,label = 'box profile')
# plt.legend()
# plt.show()

# WE USE GLOBAL LOOKUP TABLES TO MAKE DM INFLUENCE FUNCTION CALCULATION FASTER!!!
# RE-CONFIGURE USING  update_sigma(new_sigma) (e.g. bldr.update_sigma( zwfs_ns.dm.actuator_coupling_factor ))

# # Global parameters for the Gaussian
# SIGMA = 1.0            # Standard deviation for Gaussian
# MAX_RADIUS = 5.0 * SIGMA      # Max radius for lookup table
# RESOLUTION = 100       # Resolution of lookup table

# # Global variable for the lookup table
# LOOKUP_DICT = None

# def initialize_lookup_table():
#     """Initialize or reinitialize the global lookup table based on current parameters."""
#     global LOOKUP_DICT
#     MAX_RADIUS = 5.0 * SIGMA
#     LOOKUP_DICT = generate_gaussian_lookup_dict(SIGMA, MAX_RADIUS, RESOLUTION)

# def generate_gaussian_lookup_dict(sigma, max_radius, resolution=100):
#     """
#     Generate a dictionary-based lookup table for Gaussian values over squared distances.

#     Args:
#         sigma (float): The standard deviation of the Gaussian.
#         max_radius (float): The maximum radius to calculate values for.
#         resolution (int): Number of points in the radius grid. Higher values improve accuracy.

#     Returns:
#         dict: Dictionary with squared radius as keys and Gaussian values as values.
#     """
#     radius_values = np.linspace(0, max_radius, resolution)
#     squared_radius_values = radius_values**2
#     gaussian_values = np.exp(-squared_radius_values / (2 * sigma**2))

#     lookup_dict = {round(sq_radius, 5): value for sq_radius, value in zip(squared_radius_values, gaussian_values)}
#     return lookup_dict

# def get_nearest_gaussian_value(sq_distance):
#     """
#     Get the Gaussian value from the global lookup dictionary for a given squared distance.

#     Args:
#         sq_distance (float): Squared distance.

#     Returns:
#         float: Gaussian value for the nearest squared distance in the lookup.
#     """
#     rounded_distance = round(sq_distance, 5)
#     if rounded_distance in LOOKUP_DICT:
#         return LOOKUP_DICT[rounded_distance]
#     return LOOKUP_DICT[min(LOOKUP_DICT.keys(), key=lambda k: abs(k - rounded_distance))]

# def get_dm_displacement(command_vector, gain, X, Y, x0, y0, sigma=None):
#     """
#     Calculate a displacement map for a deformable mirror using a dictionary lookup for Gaussian values.

#     Args:
#         command_vector (1D array): Commands for each actuator.
#         gain (float): Scaling factor for commands.
#         X (2D array): X coordinates of the space you want the DM to be in (e.g. pixel space)
#         Y (2D array): Y coordinates of the space you want the DM to be in (e.g. pixel space)
#         x0 (1D array): X-coordinates of actuator centers.
#         y0 (1D array): Y-coordinates of actuator centers.
#         sigma is the standard deviation of the Gaussian. If None, use the global SIGMA value.

#         # TO DO - WE DID NOT INCLUDE IF sigma != None ...

#     Returns:
#         2D array: Displacement map of DM in X, Y space.
#     """
#     displacement_map = np.zeros(X.shape)
#     for i in range(len(command_vector)):
#         sq_distances = (X - x0[i])**2 + (Y - y0[i])**2
#         for j in range(X.shape[0]):
#             for k in range(X.shape[1]):
#                 displacement_map[j, k] += gain * command_vector[i] * get_nearest_gaussian_value(sq_distances[j, k])
#     return displacement_map

# # Initialize the lookup table once on module load
# initialize_lookup_table()

# # Example usage of changing SIGMA and reinitializing
# def update_sigma(new_sigma):
#     """Update SIGMA and regenerate the lookup table."""
#     global SIGMA
#     SIGMA = new_sigma
#     initialize_lookup_table()


def generate_dm_coordinates(Nx=12, Ny=12, spacing=1.0):
    """
    Generates the x, y coordinates of the actuators in a 12x12 grid DM with missing corners.

    Args:
        Nx, Ny: Number of actuators in the x and y directions (12x12 grid).
        spacing: The spacing between actuators (default is 1 unit).

    Returns:
        - coords: A list of tuples (x, y) representing the coordinates of the actuators.
        - flattened_indices: A dictionary that maps actuator indices (0 to 139) to (x, y) coordinates.
        - coord_to_index: A dictionary mapping (x, y) coordinates to actuator indices.
    """
    coords = []
    coord_to_index = {}
    flattened_indices = {}

    center_x = (Nx - 1) / 2  # Center of the grid in x
    center_y = (Ny - 1) / 2  # Center of the grid in y

    actuator_index = 0
    for i in range(Ny):
        for j in range(Nx):
            # Skip the missing corners
            if (
                (i == 0 and j == 0)
                or (i == 0 and j == Nx - 1)
                or (i == Ny - 1 and j == 0)
                or (i == Ny - 1 and j == Nx - 1)
            ):
                continue

            # Calculate x and y coordinates relative to the center
            x = (j - center_x) * spacing
            y = (i - center_y) * spacing

            coords.append((x, y))
            coord_to_index[(x, y)] = actuator_index
            flattened_indices[actuator_index] = (x, y)
            actuator_index += 1

    return coords, flattened_indices, coord_to_index


def actuator_to_xy(actuator_index, flattened_indices):
    """
    Given an actuator index, return the corresponding (x, y) coordinates.

    Args:
        actuator_index: The actuator number in the flattened 140-length array.
        flattened_indices: A dictionary mapping actuator indices to (x, y) coordinates.

    Returns:
        (x, y) coordinates of the actuator.
    """
    return flattened_indices.get(actuator_index)


def fit_affine_transformation_with_center(corners_dm, corners_img, intersection_img):
    """
    Fit an affine transformation from DM space to image space, using the DM center as the origin (0,0).

    Args:
        corners_dm: List of (x, y) coordinates of DM corners in DM space (relative to the DM center).
        corners_img: List of (x, y) coordinates of the corresponding points in image space.
        intersection_img: The (x, y) coordinates of the DM center in image space.

    Returns:
        - transform_matrix: A 2x3 matrix that transforms DM coordinates to pixel coordinates.
    """
    # Create arrays for the corners
    dm = np.array(corners_dm)
    img = np.array(corners_img)

    # Subtract the DM center (intersection) from the image coordinates to compute translation
    tx, ty = intersection_img

    # Now we need to solve for the linear transformation matrix (a, b, c, d)
    # We have the relationship: [x_img, y_img] = A * [x_dm, y_dm] + [tx, ty]
    # where A is the 2x2 matrix with components [a, b; c, d]

    # Create the matrix for DM space (without the translation part)
    dm_coords = np.vstack([dm.T, np.ones(len(dm))]).T

    # Subtract translation from image coordinates (image coordinates relative to DM center)
    img_coords = img - np.array([tx, ty])

    # Solve the linear system A * dm_coords = img_coords for A (a, b, c, d)
    # Solve the two systems independently for x and y
    A_x = np.linalg.lstsq(dm_coords[:, :2], img_coords[:, 0], rcond=None)[0]
    A_y = np.linalg.lstsq(dm_coords[:, :2], img_coords[:, 1], rcond=None)[0]

    # Construct the 2x3 affine transformation matrix
    transform_matrix = np.array(
        [[A_x[0], A_x[1], tx], [A_y[0], A_y[1], ty]]  # [a, b, tx]  # [c, d, ty]
    )

    return transform_matrix


def pixel_to_dm(pixel_coord, transform_matrix):
    """
    Converts pixel coordinates to DM coordinates using the inverse of the affine transformation.

    Args:
        pixel_coord: A tuple (x, y) in pixel space.
        transform_matrix: The affine transformation matrix from DM space to pixel space.

    Returns:
        Tuple (x_dm, y_dm) in DM coordinates.
    """
    A = transform_matrix[:, :2]  # 2x2 matrix part
    t = transform_matrix[:, 2]  # translation part

    # Inverse transformation
    A_inv = np.linalg.inv(A)
    pixel_coord = np.array(pixel_coord)
    dm_coord = np.dot(A_inv, pixel_coord - t)
    return tuple(dm_coord)


def dm_to_pixel(dm_coord, transform_matrix):
    """
    Converts DM coordinates to pixel coordinates using the affine transformation.

    Args:
        dm_coord: A tuple (x, y) in DM space.
        transform_matrix: The affine transformation matrix from DM space to pixel space.

    Returns:
        Tuple (x_pixel, y_pixel) in pixel coordinates.
    """
    dm_coord = np.array(dm_coord)
    pixel_coord = np.dot(transform_matrix[:, :2], dm_coord) + transform_matrix[:, 2]
    return tuple(pixel_coord)


def convert_to_serializable(obj):
    """
    Recursively converts NumPy arrays and other non-serializable objects to serializable forms.
    Also converts dictionary keys to standard types (str, int, float).
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, np.integer):
        return int(obj)  # Convert NumPy integers to Python int
    elif isinstance(obj, np.floating):
        return float(obj)  # Convert NumPy floats to Python float
    elif isinstance(obj, dict):
        return {
            str(key): convert_to_serializable(value) for key, value in obj.items()
        }  # Ensure keys are strings
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj  # Base case: return the object itself if it doesn't need conversion


### mft version takes ~ 0.2s per iteration, while fft ~ 8e-3s per iteration even with large padding.. no brainer. dont use mft
# def get_pupil_intensity_mft(
#     phi, amp, theta,
#     phasemask_diameter,      # legacy: in units of 1.22 * F * λ/D (DIAMETER)
#     phasemask_mask,          # focal-plane mask (fplane_pixels×fplane_pixels), center-aligned
#     pupil_diameter,
#     coldstop_diam=None,      # DIAMETER in the same 1.22-units as above (legacy style)
#     coldstop_mask=None,      # focal-plane cold stop mask (same grid); if given, we use it as-is
#     fplane_pixels=300,
#     pixels_across_mask=10
# ):
#     """
#     ZWFS intensity using your legacy MFT path.
#     - Uses the phase-mask-defined sampling (m1) for *both* mask and cold stop.
#     - If only coldstop_diam is given, we synthesize a disc on the SAME focal grid.
#     - If coldstop_mask is provided, we use it directly (no recentering).
#     """
#     t0 = time.time()
#     # ---- pupil field
#     psi_A = amp * np.exp(1j * phi)

#     # ---- legacy m1 (keep exactly as before)
#     array_dim    = phi.shape[0]
#     pupil_radius = pupil_diameter // 2
#     # Convert phase-mask "diameter in 1.22 units" -> DIAMETER in λ/D
#     R_mask_lD_diam = float(phasemask_diameter) / 1.22

#     # legacy: m1 = pixels_across_mask * 2 * R_mask * (array_dim / (2 * pupil_radius))
#     # (R_mask was historically used like a diameter, despite the name in comments)
#     m1 = pixels_across_mask * 2.0 * R_mask_lD_diam * (array_dim / (2.0 * pupil_radius))

#     # ---- forward MFT to focal plane
#     psi_B = mft.mft(psi_A, array_dim, fplane_pixels, m1)

#     # ---- reference arm via provided phase mask (same grid; no recentering)
#     # (Assume phasemask_mask matches fplane_pixels and is centered as in your pipeline.)
#     b = mft.imft(phasemask_mask * psi_B, fplane_pixels, array_dim, m1)

#     # ---- analytic ZWFS combine on the pupil
#     ejt_minus_1 = np.exp(1j * float(theta)) - 1.0
#     psi_c = psi_A + ejt_minus_1 * b

#     # ---- propagate to focal plane to apply cold stop
#     Psi_c = mft.mft(psi_c, array_dim, fplane_pixels, m1)

#     # ---- case A: no cold stop at all
#     if (coldstop_mask is None) and (coldstop_diam is None):
#         psi_out = mft.imft(Psi_c, fplane_pixels, array_dim, m1)
#         I_out = np.abs(psi_out)**2
#         #return I_out

#     # ---- helper: infer pixels-per-(λ/D) from the phase-mask sampling
#     # We know the phase-mask diameter in λ/D (R_mask_lD_diam) and the number of pixels across it on this grid.
#     # The grid pixels across the mask is exactly `pixels_across_mask`.
#     # => pixels per (λ/D) on this focal plane:
#     px_per_lD = float(pixels_across_mask) / max(R_mask_lD_diam, 1e-12)

#     # ---- case B: build cold stop disc from diameter (no precomputed mask)
#     if (coldstop_mask is None) and (coldstop_diam is not None):
#         # Convert cold-stop DIAMETER (in same 1.22-units) -> DIAMETER in λ/D
#         R_stop_lD_diam = float(coldstop_diam) / 1.22
#         r_stop_pix = 0.5 * R_stop_lD_diam * px_per_lD  # radius in pixels on THIS grid

#         yy, xx = np.indices((fplane_pixels, fplane_pixels))
#         c = (fplane_pixels - 1) / 2.0
#         r = np.hypot(yy - c, xx - c)
#         C = (r <= r_stop_pix).astype(float)

#         Psi_c *= C
#         psi_out = mft.imft(Psi_c, fplane_pixels, array_dim, m1)
#         I_out = np.abs(psi_out)**2

#         # ----- plot |Ψ_c|^2 with cold stop outline -----
#         I_fp = np.abs(Psi_c)**2
#         I_fp /= (I_fp.max() + 1e-12)

#         # plt.figure(figsize=(5.2, 4.6))
#         # plt.imshow(np.log10(I_fp + 1e-6), origin="lower", cmap="magma")
#         # plt.colorbar(label="log10 |Ψ_c|² (norm.)")
#         # plt.contour(C, levels=[0.5], colors="cyan", linewidths=1.2)  # cold stop outline
#         # plt.title("Focal-plane |Ψ_c|² with cold stop (cyan)")
#         # plt.tight_layout()
#         # plt.show()
#         #return I_out

#     # ---- case C: precomputed cold-stop mask is provided (use as-is)
#     if coldstop_mask is not None:
#         # No need for coldstop_diam in this branch.
#         C = np.asarray(coldstop_mask, dtype=float)
#         # Sanity: shape must match this focal grid
#         assert C.shape == (fplane_pixels, fplane_pixels), \
#             "coldstop_mask must have shape (fplane_pixels, fplane_pixels)"
#         Psi_c *= C
#         psi_out = mft.imft(Psi_c, fplane_pixels, array_dim, m1)
#         I_out = np.abs(psi_out)**2
#         #return I_out

#     # Fallback (shouldn’t hit)
#     #psi_out = mft.imft(Psi_c, fplane_pixels, array_dim, m1)
#     #I_out = np.abs(psi_out)**2
#     t1 = time.time()
#     print(t1-t0)
#     return I_out


def get_zwfs_output_field(
    phi,
    amp,
    theta,
    phasemask_diameter,
    phasemask_mask=None,
    pupil_diameter=None,
    fplane_pixels=300,
    pixels_across_mask=10,
    return_terms=False,
):
    """
    Compute the analytic ZWFS exit-pupil complex field using N'Diaye Eq. 6.

    This returns the field immediately after the ZWFS phase-mask operation,
    before any downstream cold stop, relay propagation, detector binning, etc.

    Model:
        psi_A = amp * exp(i phi)

        b = IFFT{ M * FFT{psi_A} }

        psi_C = psi_A - (1 - exp(i theta)) * b
              = psi_A + (exp(i theta) - 1) * b

    Parameters
    ----------
    phi : 2D array
        Input phase in radians.
    amp : 2D array
        Input field amplitude. Usually includes the pupil support.
    theta : float
        Phase shift of the ZWFS mask in radians.
    phasemask_diameter : float
        Phase-mask diameter in the same lambda/D convention used by
        get_pupil_intensity().
    phasemask_mask : 2D array or None
        Optional precomputed focal-plane mask. If None, build a circular
        mask using phasemask_diameter and the FFT sampling.
    pupil_diameter : float or None
        Kept for interface compatibility. Not required by this implementation.
    fplane_pixels : int
        FFT/focal-plane grid size. Internally promoted to odd size if needed.
    pixels_across_mask : float
        Fallback sampling scale if the mask/pupil support cannot be inferred.
    return_terms : bool
        If True, return a dict with intermediate fields and masks.

    Returns
    -------
    psi_C : 2D complex array
        Analytic ZWFS output field cropped back to the original phi/amp shape.

    Or, if return_terms=True, returns dict with:
        psi_C, psi_A, b, phase_disc_fp, pix_per_wvld
    """

    del pupil_diameter  # currently unused; kept for compatibility

    phi = np.asarray(phi)
    amp = np.asarray(amp)

    if phi.shape != amp.shape or phi.ndim != 2:
        raise ValueError("phi and amp must be 2D arrays with the same shape.")

    if phi.shape[0] != phi.shape[1]:
        raise ValueError("phi and amp must be square arrays.")

    N_orig = phi.shape[0]

    # Match the even-grid safety convention already used in get_pupil_intensity().
    padded_even = False
    if (N_orig % 2) == 0:
        phi_work = np.pad(phi, ((0, 1), (0, 1)), mode="constant")
        amp_work = np.pad(amp, ((0, 1), (0, 1)), mode="constant")
        N = N_orig + 1
        padded_even = True
    else:
        phi_work = phi
        amp_work = amp
        N = N_orig

    # Entrance-pupil field.
    psi_A = amp_work.astype(np.complex128) * np.exp(1j * phi_work.astype(np.float64))

    # Use an odd focal grid to keep centering unambiguous.
    Nf = int(max(fplane_pixels, N))
    if (Nf % 2) == 0:
        Nf += 1

    pad = (Nf - N) // 2

    psi_A_pad = np.zeros((Nf, Nf), dtype=np.complex128)
    psi_A_pad[pad : pad + N, pad : pad + N] = psi_A

    # Focal-plane field before phase mask.
    psi_B = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(psi_A_pad),
            norm="ortho",
        )
    )

    # Build or resize the focal-plane phase-disc mask.
    if phasemask_mask is not None:
        m = np.asarray(phasemask_mask, dtype=float)

        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("phasemask_mask must be a square 2D array.")

        if m.shape[0] == Nf:
            phase_disc = (m > 0.5).astype(float)
        else:
            phase_disc = np.zeros((Nf, Nf), dtype=float)

            Ns = m.shape[0]
            c_src = (Ns - 1) // 2
            c_dst = (Nf - 1) // 2

            src_y0 = max(0, c_src - c_dst)
            src_x0 = max(0, c_src - c_dst)
            dst_y0 = max(0, c_dst - c_src)
            dst_x0 = max(0, c_dst - c_src)

            h = min(Ns - src_y0, Nf - dst_y0)
            w = min(Ns - src_x0, Nf - dst_x0)

            phase_disc[
                dst_y0 : dst_y0 + h,
                dst_x0 : dst_x0 + w,
            ] = m[
                src_y0 : src_y0 + h,
                src_x0 : src_x0 + w,
            ]

            phase_disc = (phase_disc > 0.5).astype(float)

        # Infer pixels per lambda/D from the supplied mask.
        yy, xx = np.indices((Nf, Nf))
        c = (Nf - 1) / 2.0
        rr = np.hypot(yy - c, xx - c)

        if np.any(phase_disc > 0):
            r_pix = rr[phase_disc > 0].max()
            pix_per_wvld = (2.0 * r_pix) / float(phasemask_diameter)
        else:
            pix_per_wvld = float(pixels_across_mask)

    else:
        # THIS BRANCH IS SUPER SLOW AND GETS CALLED MULTIPLE TIMES PER
        # ITERATION, IS IT POSSIBLE TO DEFINE `pix_per_wvld` AND `phase_disc`
        # ONCE AND PASS IT IN TO THIS FUNCTION?
        # Infer pixels per lambda/D from the padded pupil support.
        amp_pad = np.zeros((Nf, Nf), dtype=float)
        amp_pad[pad : pad + N, pad : pad + N] = np.abs(amp_work)

        support = amp_pad > 0

        yy, xx = np.indices((Nf, Nf))
        c = (Nf - 1) / 2.0
        rr = np.hypot(yy - c, xx - c)

        if np.any(support):
            r_pupil_pix = rr[support].max()
            d_pupil_pix = 2.0 * r_pupil_pix
            pix_per_wvld = float(Nf) / max(d_pupil_pix, 1e-12)
        else:
            pix_per_wvld = float(pixels_across_mask)

        r_mask_pix = 0.5 * float(phasemask_diameter) * pix_per_wvld
        phase_disc = (rr <= r_mask_pix).astype(float)
        

    # Reference wave b = IFFT{M FFT{psi_A}}.
    b_pad = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(phase_disc * psi_B),
            norm="ortho",
        )
    )

    # N'Diaye Eq. 6:
    # psi_C = psi_A - (1 - exp(i theta)) b
    #       = psi_A + (exp(i theta) - 1) b
    psi_C_pad = psi_A_pad + (np.exp(1j * float(theta)) - 1.0) * b_pad

    # Crop back to the caller's original pupil/grid size.
    psi_A_crop = psi_A_pad[pad : pad + N, pad : pad + N]
    b_crop = b_pad[pad : pad + N, pad : pad + N]
    psi_C_crop = psi_C_pad[pad : pad + N, pad : pad + N]

    if padded_even:
        psi_A_crop = psi_A_crop[:N_orig, :N_orig]
        b_crop = b_crop[:N_orig, :N_orig]
        psi_C_crop = psi_C_crop[:N_orig, :N_orig]

    if return_terms:
        return {
            "psi_C": psi_C_crop,
            "psi_A": psi_A_crop,
            "b": b_crop,
            "phase_disc_fp": phase_disc,
            "pix_per_wvld": pix_per_wvld,
        }

    return psi_C_crop


def get_pupil_intensity(
    phi,
    amp,
    theta,
    phasemask_diameter,
    phasemask_mask,
    pupil_diameter,
    fplane_pixels=300,
    pixels_across_mask=10,
    *,
    coldstop_diam=None,
    coldstop_offset=(0.0, 0.0),
    coldstop_mask=None,
    include_beta=True,
    return_field=False,
    return_terms=False,
    debug=False,
) -> NDArray | Dict:
    """
    ZWFS pupil intensity with analytic output field and focal-plane cold stop.
    Tilt-safe for even-sized inputs: internally pads to odd size to avoid half-pixel centering errors,
    then crops back to the caller’s original size.

    #     Args:
    #         phi (_type_): input phase (radians)
    #         amp (_type_): input amplitude of field (sqrt of intensity)
    #         theta (_type_): phaseshift of mask (rad)
    #         phasemask_diameter ( ) : diameter in units of 1.22 * F * lambda/D of phasemask
    #         phasemask_mask ( ) : 2D array of the phaseshifting region in image plane (input to make things quicker)
    #         pupil_diameter () : diameter of pupil in pixels
    #         fplane_pixels (int) : number of pixels in focal plane
    #         pixels_across_mask (int) : number of pixels across the phase shifting region of mask in focal plane
    #     Returns:
    #         _type_: ZWFS pupil intensity

    """
    # t0 = time.time()

    # ---------------- even-grid safe: promote to odd internally ----------------
    phi = np.asarray(phi)
    amp = np.asarray(amp)
    assert phi.shape == amp.shape and phi.ndim == 2, "phi and amp must be same 2D shape"
    N_orig = amp.shape[0]
    assert amp.shape[1] == N_orig, "phi/amp must be square"

    padded_even = False
    if (N_orig % 2) == 0:
        # pad ONE row/col (bottom/right) -> internal odd grid; center aligns to a pixel
        amp_work = np.pad(amp, ((0, 1), (0, 1)), mode="constant")
        phi_work = np.pad(phi, ((0, 1), (0, 1)), mode="constant")
        N = N_orig + 1
        padded_even = True
    else:
        amp_work = amp
        phi_work = phi
        N = N_orig

    # ---------------- build pupil-plane field Ψ_A on the odd (or original odd) grid ----------------
    psi_A = amp_work.astype(np.complex128) * np.exp(1j * phi_work.astype(np.float64))

    # ---------------- choose focal-plane FFT size (prefer odd to keep centering simple) -------------
    Nf = int(max(fplane_pixels, N))
    if (Nf % 2) == 0:  # enforce odd focal grid as well
        Nf += 1

    # ---------------- center embed to focal grid ----------------
    pad = (Nf - N) // 2  # integer because both are odd
    psi_A_pad = np.zeros((Nf, Nf), dtype=np.complex128)
    psi_A_pad[pad : pad + N, pad : pad + N] = psi_A

    # ---------------- forward FFT to focal plane ----------------
    psi_B = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(psi_A_pad), norm="ortho")
    )  # norm="ortho" to maintain flux conservation

    # ---------------- phase disc (derive physical pixels-per-(λ/D) correctly) ----------------
    def _center_resize_mask(mask, tgtN):
        m = np.asarray(mask, float)
        assert m.ndim == 2 and m.shape[0] == m.shape[1], "mask must be square"
        if m.shape[0] == tgtN:
            return (m > 0.5).astype(float)
        out = np.zeros((tgtN, tgtN), float)
        c0 = (m.shape[0] - 1) // 2
        ct = (tgtN - 1) // 2
        y0s = max(0, ct - c0)
        y0d = max(0, c0 - ct)
        h = min(m.shape[0], tgtN)
        out[y0s : y0s + h, y0s : y0s + h] = m[y0d : y0d + h, y0d : y0d + h]
        return (out > 0.5).astype(float)

    def _disc_radius_in_pixels(bin_mask):
        if bin_mask.max() == 0:
            return 0.0
        yy, xx = np.indices(bin_mask.shape)
        c = (bin_mask.shape[0] - 1) / 2.0
        r = np.hypot(yy - c, xx - c)
        return r[bin_mask.astype(bool)].max()

    if phasemask_mask is not None:
        phase_disc = _center_resize_mask(phasemask_mask, Nf)
        r_pix = _disc_radius_in_pixels(phase_disc)
        if r_pix <= 0 or phasemask_diameter <= 0:
            # fallback; will be superseded by cold-stop scaling anyway
            pix_per_wvld = float(pixels_across_mask)
        else:
            pix_per_wvld = (2.0 * r_pix) / float(phasemask_diameter)
    else:
        # derive physical pixels-per-(λ/D) from the pupil diameter on the padded grid
        amp_pad = np.zeros((Nf, Nf), dtype=float)
        amp_pad[pad : pad + N, pad : pad + N] = np.abs(amp_work)
        support = amp_pad > 0
        yy, xx = np.indices((Nf, Nf))
        c = (Nf - 1) / 2.0
        r = np.hypot(yy - c, xx - c)
        if np.any(support):
            Rpix = r[support].max()
            Dp_pix = 2.0 * Rpix
            pix_per_wvld = float(Nf) / max(Dp_pix, 1e-9)
        else:
            pix_per_wvld = float(pixels_across_mask)

        r_mask_pix = 0.5 * float(phasemask_diameter) * pix_per_wvld
        phase_disc = (r <= r_mask_pix).astype(float)

    # ---------------- reference arm & analytic pupil field ----------------
    b = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(phase_disc * psi_B), norm="ortho")
    )
    ejt_minus_1 = np.exp(1j * float(theta)) - 1.0
    psi_theta_pad = psi_A_pad + ejt_minus_1 * b

    # ---------------- forward to focal plane, apply cold stop ----------------
    psi_theta_B = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(psi_theta_pad), norm="ortho")
    )

    if coldstop_mask is not None:
        C = _center_resize_mask(coldstop_mask, Nf)
    else:
        if coldstop_diam is None:
            C = np.ones((Nf, Nf), float)
        else:
            dx_wvld, dy_wvld = coldstop_offset
            dx_pix = float(dx_wvld) * pix_per_wvld
            dy_pix = float(dy_wvld) * pix_per_wvld
            r_cs_pix = 0.5 * float(coldstop_diam) * pix_per_wvld
            yy, xx = np.indices((Nf, Nf))
            c = (Nf - 1) / 2.0
            r_cs = np.hypot(yy - (c + dy_pix), xx - (c + dx_pix))
            C = (r_cs <= r_cs_pix).astype(float)

    psi_out_pad = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(C * psi_theta_B), norm="ortho")
    )

    # ---------------- crop back to caller's original size ----------------
    # First crop to the internal pupil block (N×N), then drop the extra row/col if we padded.
    psi_crop = psi_out_pad[pad : pad + N, pad : pad + N]
    if padded_even:
        psi_out = psi_crop[:N_orig, :N_orig]
    else:
        psi_out = psi_crop
    Ic = np.abs(psi_out) ** 2

    # ---------------- optional diag ----------------
    if debug:
        psf = np.abs(psi_theta_B) ** 2
        if psf.max() > 0:
            psf /= psf.max()
        plt.figure(figsize=(5, 4))
        plt.imshow(psf, origin="lower", cmap="gray")
        plt.contour(phase_disc, levels=[0.5], colors="w", linewidths=1.0)
        if coldstop_diam is not None or coldstop_mask is not None:
            plt.contour(C, levels=[0.5], colors="r", linewidths=1.0)
        plt.title("Focal-plane |Ψ_θ|² (phase disc white, cold stop red)")
        plt.tight_layout()

    # ---------------- returns ----------------
    if return_terms or return_field:
        out: Dict[str, NDArray | float] = {}
        out["Ic"] = Ic
        if return_field:
            out["field_pupil"] = psi_out
        if return_terms:
            # also return pupil-plane terms cropped back to caller’s shape for convenience
            b_crop = b[pad : pad + N, pad : pad + N]
            psi_theta_crop = psi_theta_pad[pad : pad + N, pad : pad + N]
            if padded_even:
                b_crop = b_crop[:N_orig, :N_orig]
                psi_theta_crop = psi_theta_crop[:N_orig, :N_orig]
            out.update(
                {
                    "psi_A": amp * np.exp(1j * phi),  # original-resolution pupil field
                    "b_crop": b_crop,
                    "psi_theta_crop": psi_theta_crop,
                    "phase_disc_fp": phase_disc,
                    "cold_stop_fp": C,
                    "pix_per_wvld": pix_per_wvld,
                }
            )
        return out
    # t1 = time.time()
    # print(t1-t0)
    return Ic


def update_dm_registration_wavespace(transform_matrix, zwfs_ns):
    """_summary_
    # STANDARD WAY TO UPDATE THE REGISTRATION OF THE DM IN WAVE SPACE
    # UPDATES --> zwfs_ns <--- name space !!! Only use this method to update registration

    Args:
        transform_matrix (_type_): _description_ affine transform describing the mapping from DM actuators to the wavefront
        zwfs_ns (_type_): _description_ the zwfs name space (holding configuration details)

    zwfs_ns dependancies (must have in namespace for code to work):
        zwfs_ns.dm.Nact_x  (int)
        zwfs_ns.dm.Nact_y  (int)
        zwfs_ns.dm.dm_pitch  (float)
        zwfs_ns.dm.actuator_coupling_factor (float)
        zwfs_ns.dm.current_cmd (1D array, size 140 for BMC multi-3.5 DM)

    """

    dm_coords, dm_actuator_to_coord, dm_coord_to_actuator = generate_dm_coordinates(
        Nx=zwfs_ns.dm.Nact_x, Ny=zwfs_ns.dm.Nact_y, spacing=zwfs_ns.dm.dm_pitch
    )

    # plt.figure(); plt.scatter([xx[0] for xx in dm_coords], [xx[1] for xx in dm_coords] ); plt.show();

    pixel_coord_list = np.array([dm_to_pixel(c, transform_matrix) for c in dm_coords])

    # plt.figure(); plt.scatter([xx[0] for xx in pixel_coord_list], [xx[1] for xx in pixel_coord_list] ); plt.show();

    # projecting the DM actuator space to wavespace. For convinience this is same as pixel space (before binnning)
    sigma = (
        zwfs_ns.dm.actuator_coupling_factor
        * abs(pixel_coord_list[0][0] - pixel_coord_list[1][0])
        * np.ones(len(zwfs_ns.dm.current_cmd))
    )  # coupling of actuators projected to wavespace

    x0_list = [xx[0] for xx in pixel_coord_list]
    y0_list = [yy[1] for yy in pixel_coord_list]

    dm2wavespace_registration_dict = {
        "dm_to_wavesp_transform_matrix": transform_matrix,  # affine transform from DM coordinates to wave coordinates
        "dm_actuator_to_coord": dm_actuator_to_coord,
        "dm_coord_to_actuator": dm_coord_to_actuator,
    }

    dm_coord_dict = {
        "dm_coords": dm_coords,  # DM coordinates DM space
        "dm_coord_wavesp": pixel_coord_list,
        "act_x0_list_wavesp": x0_list,  # actuator x coorindate in pixel space
        "act_y0_list_wavesp": y0_list,  # actuator y coordinate in pixel space
        "act_sigma_wavesp": sigma,
    }

    dm2wavespace_registration_ns = SimpleNamespace(**dm2wavespace_registration_dict)
    # wave_coord_ns = SimpleNamespace(**wave_coord_dict )
    dm_coord_ns = SimpleNamespace(**dm_coord_dict)

    # Add DM and wave coorindates to grid namespace
    zwfs_ns.grid.dm_coord = dm_coord_ns
    zwfs_ns.dm2wavespace_registration = dm2wavespace_registration_ns

    zwfs_ns.compiled_dm = CompiledDM(
        zwfs_ns.grid.wave_coord.X,
        zwfs_ns.grid.wave_coord.Y,
        np.array(zwfs_ns.grid.dm_coord.act_x0_list_wavesp),
        np.array(zwfs_ns.grid.dm_coord.act_y0_list_wavesp),
        np.array(zwfs_ns.grid.dm_coord.act_sigma_wavesp),
    )

    return zwfs_ns


def init_ns_from_pyZelda(z, wvl0):

    if "UT" in z.instrument.upper():
        telescope = "UT"
    elif "AT" in z.instrument.upper():
        telescope = "AT"
    else:
        telescope = "DISC"

    grid_dict = {
        "telescope": telescope,
        "D": 8,  # diameter of beam (m)
        "N": z.pupil_diameter,  # number of pixels across pupil diameter
        "dim": z.pupil_dim,  # physical size of grid (m)
        # "padding_factor" : z.pupil_dim / z.pupil_diameter, # how many pupil diameters fit into grid x axis
        # TOTAL NUMBER OF PIXELS = padding_factor * N
    }

    optics_dict = {
        "wvl0": wvl0,  # central wavelength (m)
        "F_number": z.mask_Fratio,  # F number on phasemask
        "mask_diam": z.mask_diameter
        / (
            1.22 * z.mask_Fratio * wvl0
        ),  # diameter of phaseshifting region in diffraction limit units (physical unit is mask_diam * 1.22 * F_number * lambda)
        "theta": z.mask_phase_shift(wvl0),  # phaseshift of phasemask
    }

    grid_ns = SimpleNamespace(**grid_dict)
    optics_ns = SimpleNamespace(**optics_dict)

    return grid_ns, optics_ns


def check_ns_consistency_with_pyZelda(z, zwfs_ns):

    fail_log = {}

    if not z.pupil_diameter == zwfs_ns.grid.N:
        fail_log["pupil_diameter"] = (z.pupil_diameter, zwfs_ns.grid.N)
    # if not z.pupil_dim / z.pupil_diameter == zwfs_ns.grid.padding_factor:
    #    fail_log['pupil_dim'] = (z.pupil_dim / z.pupil_diameter, zwfs_ns.grid.padding_factor)
    if not z.mask_diameter == zwfs_ns.optics.mask_diam * (
        1.22 * zwfs_ns.optics.F_number * zwfs_ns.optics.wvl0
    ):
        fail_log["mask_diameter"] = (
            z.mask_diameter,
            zwfs_ns.optics.mask_diam
            * (1.22 * zwfs_ns.optics.F_number * zwfs_ns.optics.wvl0),
        )
    if not z.mask_Fratio == zwfs_ns.optics.F_number:
        fail_log["mask_Fratio"] = (z.mask_Fratio, zwfs_ns.optics.F_number)
    if not z.mask_phase_shift(zwfs_ns.optics.wvl0) == zwfs_ns.optics.theta:
        fail_log["theta"] = (
            z.mask_phase_shift(zwfs_ns.optics.wvl0),
            zwfs_ns.optics.theta,
        )

    return fail_log


def init_zwfs_from_config_ini(config_ini, wvl0):

    ns = util.ini_to_namespace(config_ini)

    # init the pyZelda sensor
    z = zelda.Sensor(ns.instrument.pyzelda_config)

    # extract the grid and optic namespace from pyZelda object at given wavelength
    # to put in format compatible with BaldrApp code
    grid_ns, optics_ns = init_ns_from_pyZelda(z, wvl0)

    # merge all namespaces into one following standards of BaldrApp
    zwfs_ns = init_zwfs(grid_ns, optics_ns, ns.dm)

    zwfs_ns.name = ns.instrument.name

    # append the other relevant information to the namespace
    zwfs_ns.detector = ns.detector
    zwfs_ns.stellar = ns.stellar
    zwfs_ns.throughput = ns.throughput
    zwfs_ns.atmosphere = ns.atmosphere

    # also append the pyZelda object to the namespace to inherit methods etc
    zwfs_ns.pyZelda = z

    return zwfs_ns


def init_zwfs(grid_ns, optics_ns, dm_ns):
    #############
    #### GRID

    # get the pupil, ignore phasemask mask returned since only valid for fft transform
    # pupil, _ = get_grids( wavelength = optics_ns.wvl0 , F_number = optics_ns.F_number, mask_diam = optics_ns.mask_diam, diameter_in_angular_units = True, N = grid_ns.N, padding_factor = grid_ns.padding_factor )

    if hasattr(grid_ns, "telescope"):
        if grid_ns.telescope.upper() == "UT":
            pupil = aperture.baldr_UT_pupil(
                diameter=grid_ns.N, dim=int(grid_ns.dim), spiders_thickness=0.008
            )  # padding_factor = 2 )
        elif grid_ns.telescope.upper() == "AT":
            pupil = aperture.baldr_AT_pupil(
                diameter=grid_ns.N,
                dim=int(grid_ns.dim),
                spiders_thickness=0.016,
                strict=False,
                cpix=False,
            )  # , padding_factor = 2 )
        elif grid_ns.telescope.upper() == "DISC":
            pupil = aperture.disc(
                dim=int(grid_ns.dim),
                size=grid_ns.N,
                diameter=True,
                strict=False,
                center=(),
                cpix=False,
                invert=False,
                mask=False,
            )
        elif (
            grid_ns.telescope.upper() == "SOLARSTEIN"
        ):  # ASGARD internal source SOLARSTEIN which has UT like secondary with no spiders
            pupil = aperture.disc_obstructed(
                dim=int(grid_ns.dim),
                size=grid_ns.N,
                obs=1100 / 8000,
                diameter=True,
                strict=False,
            )

        else:
            raise TypeError("telescope not implemented. Try DISC, AT or UT")

    else:
        pupil = aperture.disc(
            dim=int(grid_ns.dim),
            size=grid_ns.N,
            diameter=True,
            strict=False,
            center=(),
            cpix=False,
            invert=False,
            mask=False,
        )
        # raise UserWarning("telescope not defined in grid_ns. Defaulting to disk pupil")
        print("telescope not defined in grid_ns. Defaulting to disk pupil")
    grid_ns.pupil_mask = pupil
    # grid_ns.phasemask_mask = phasemask

    # coorindates in the pupil plance
    padding_factor = grid_ns.dim / grid_ns.N
    x = np.linspace(
        -(grid_ns.D * padding_factor) // 2,
        (grid_ns.D * padding_factor) // 2,
        pupil.shape[0],
    )
    y = np.linspace(
        -(grid_ns.D * padding_factor) // 2,
        (grid_ns.D * padding_factor) // 2,
        pupil.shape[0],
    )
    X, Y = np.meshgrid(x, y)

    wave_coord_dict = {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
    }

    wave_coord_ns = SimpleNamespace(**wave_coord_dict)

    grid_ns.wave_coord = wave_coord_ns

    # focal plane sampling is hardcoded. we use mft to sample when calculating output intensity.
    # hold a fixed phasemask mask here so don't have to re-init each run
    focal_plane_sampling_dict = {"fplane_pixels": 300, "pixels_across_mask": 10}

    phasemask_mask = aperture.disc(
        focal_plane_sampling_dict["fplane_pixels"],
        focal_plane_sampling_dict["fplane_pixels"]
        // focal_plane_sampling_dict["pixels_across_mask"],
        diameter=True,
        cpix=True,
        strict=False,
    )

    # attach to grid_ns for legacy reasons (not have to make lots of edits )
    grid_ns.phasemask_mask = phasemask_mask

    # grid_ns.coldstop_mask =

    # get coordinates in focal plane
    xmin = (
        -focal_plane_sampling_dict["fplane_pixels"]
        / focal_plane_sampling_dict["pixels_across_mask"]
        * optics_ns.mask_diam
        * optics_ns.wvl0
        * optics_ns.F_number
        / 2
    )
    xmax = (
        focal_plane_sampling_dict["fplane_pixels"]
        / focal_plane_sampling_dict["pixels_across_mask"]
        * optics_ns.mask_diam
        * optics_ns.wvl0
        * optics_ns.F_number
        / 2
    )
    x = np.linspace(xmin, xmax, focal_plane_sampling_dict["fplane_pixels"])
    y = np.linspace(xmin, xmax, focal_plane_sampling_dict["fplane_pixels"])
    X, Y = np.meshgrid(x, y)

    focal_plane_sampling_dict["x"] = x
    focal_plane_sampling_dict["X"] = X
    focal_plane_sampling_dict["y"] = y
    focal_plane_sampling_dict["Y"] = Y

    focal_plane_ns = SimpleNamespace(**focal_plane_sampling_dict)

    #############
    #### DM

    if dm_ns.dm_model == "BMC-multi-3.5":
        # 12x12 with missing corners
        dm_ns.Nact_x = 12  # actuators along DM x axis
        dm_ns.Nact_y = 12  # actuators along DM y axis
        dm_ns.Nact = 140  # total number of actuators
        dm_ns.dm_flat = 0.5 + dm_ns.flat_rmse * np.random.rand(
            dm_ns.Nact
        )  # add some noise to the DM flat
        dm_ns.current_cmd = dm_ns.dm_flat  # default set to dm flat
    else:
        raise TypeError("input DM model not implemented. Try BMC-multi-3.5")

    # set so pupil covers about 10 actuators on BMC multi 3.5 DM
    # np.ptp(wave_coord_dict['y'])
    a, b, c, d = (
        12 / 10 * (grid_ns.D / 2) / (dm_ns.Nact_x / 2 - 0.5),
        0,
        0,
        12 / 10 * (grid_ns.D / 2) / (dm_ns.Nact_x / 2 - 0.5),
    )  # Parameters for affine transform (identity for simplicity)
    # set by default to be centered and overlap with pupil (pupil touches edge of DM )

    t_x, t_y = np.mean(wave_coord_dict["x"]), np.mean(
        wave_coord_dict["y"]
    )  # Translation in phase space

    # we can introduce mis-registrations by rolling input pupil

    dm_act_2_wave_space_transform_matrix = np.array([[a, b, t_x], [c, d, t_y]])

    #### Stellar Name Space
    # detect requires stellar.bandwidth input so we set as None as default
    stellar_ns = SimpleNamespace(**{"bandwidth": None})

    # nned to add
    # ZWFS NAME SPACE
    zwfs_dict = {
        "grid": grid_ns,
        "optics": optics_ns,
        "dm": dm_ns,
        "focal_plane": focal_plane_ns,
        "stellar": stellar_ns,
        # "dm2wavespace_registration" : dm2wavespace_registration_ns
    }

    zwfs_ns = SimpleNamespace(**zwfs_dict)

    # user warning: only ever use update_dm_registration_wavespace if you want a consistent update across all variables
    # this updates the zwfs_ns.grid with dm coords in DM and wavespace as well as defining dm2wavespace_registration namespace
    zwfs_ns = update_dm_registration_wavespace(
        dm_act_2_wave_space_transform_matrix, zwfs_ns
    )

    return zwfs_ns


def first_stage_ao(
    atm_scrn,
    Nmodes_removed,
    basis,
    phase_scaling_factor: float = 1,
    return_reconstructor=False,
):
    """_summary_

    simple first stage AO that perfectly reconstructs Nmodes_removed Zernike modes from the input phase screen
    returns the reconstructor

    if return_reco = True the reconstructor is also returned so AO latency can be simulated by applying reconstructor after
    rolling the phase screen a few times
    e.g.
    atm_scrn.add_row()
    ao_1 = pupil * (phase_scaling_factor * atm_scrn.scrn - reco)


    Args:
        atm_scrn (phasescreens.PhaseScreenKolmogorov): atmospheric phase screen object initialized from aotools or common/phasescreens.py
        Nmodes_removed (int): Number of Zernike modes removed in first stage ao
        basis (list of 2D arrays): Zernike basis function on the input atm_scrn pupil footprint
            IMPORTANT - basis[0] should be the pupil disk without secondary mirror
        phase_scaling_factor (float, optional): _description_. Defaults to 1. to scale the phase screen before projecting onto Zernike modes
        return_reco (bool) : return the reconstructor if you want to add latency in the simulation
    """
    pupil_disk = basis[
        0
    ]  # we define a disk pupil without secondary - so Zernike modes are orthogonal

    # crop the pupil disk and the phasescreen within it (remove padding outside pupil)
    pupil_disk_cropped, atm_in_pupil = util.crop_pupil(
        pupil_disk, phase_scaling_factor * atm_scrn.scrn
    )

    # project onto Zernike modes
    mode_coefficients = np.array(
        ztools.zernike.opd_expand(
            atm_in_pupil * pupil_disk_cropped,
            nterms=len(basis),
            aperture=pupil_disk_cropped,
        )
    )

    # do the reconstruction for N modes
    reco = np.sum(
        mode_coefficients[:Nmodes_removed, np.newaxis, np.newaxis]
        * basis[:Nmodes_removed, :, :],
        axis=0,
    )

    # remove N modes
    ao_1 = pupil_disk * (phase_scaling_factor * atm_scrn.scrn - reco)

    if return_reconstructor:
        return ao_1, reco
    else:
        return ao_1


def test_propagation(zwfs_ns):
    """_summary_
    just test propagating through the zwfs system with :
        -small (10nm rms) random internal aberations ,
        -no atmospheric aberations ,
        -the current DM state in the zwfs_ns
    Args:
        zwfs_ns (_type_): _description_

    Returns:
        _type_: phi, phi_internal, N0, I0, I
            phi is the wavefront phase (at defined central wvl) from current dm, atm, internal aberrations
            phi_internal is the wavefront phase (at defined central wvl)  from defined flat dm, internal aberrations
            N0 is intensity with flat dm, no phasemask
            I0 is intensity with flat dm,  phasemask
            I is intensity with input dm, phasemask
    """
    opd_atm = (
        0 * zwfs_ns.grid.pupil_mask * np.random.randn(*zwfs_ns.grid.pupil_mask.shape)
    )

    opd_internal = (
        10e-9
        * zwfs_ns.grid.pupil_mask
        * np.random.randn(*zwfs_ns.grid.pupil_mask.shape)
    )

    # get the OPD from the DM in the wave space.
    # the only real dynamic thing needed is the current command of the DM
    # zwfs_ns.dm.current_cmd
    opd_flat_dm = zwfs_ns.compiled_dm.eval(zwfs_ns.dm.dm_flat * zwfs_ns.dm.opd_per_cmd)
    opd_current_dm = zwfs_ns.compiled_dm.eval(
        zwfs_ns.dm.current_cmd * zwfs_ns.dm.opd_per_cmd
    )

    phi_internal = (
        2 * np.pi / zwfs_ns.optics.wvl0 * (opd_internal + opd_flat_dm)
    )  # phi_atm , phi_dm are in opd

    phi = (
        2 * np.pi / zwfs_ns.optics.wvl0 * (opd_internal + opd_current_dm + opd_atm)
    )  # phi_atm , phi_dm are in opd

    amp = 1e2 * zwfs_ns.grid.pupil_mask

    get_pupil_intensity(
        phi=phi_internal,
        amp=amp,
        theta=zwfs_ns.optics.theta,
        phasemask_diameter=zwfs_ns.optics.mask_diam,
        phasemask_mask=zwfs_ns.grid.phasemask_mask,
        pupil_diameter=zwfs_ns.grid.N,
        fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
        pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
    )

    N0 = get_pupil_intensity(
        phi=phi_internal,
        amp=amp,
        theta=0,
        phasemask_diameter=zwfs_ns.optics.mask_diam,
        phasemask_mask=zwfs_ns.grid.phasemask_mask,
        pupil_diameter=zwfs_ns.grid.N,
        fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
        pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
    )

    I0 = get_pupil_intensity(
        phi=phi_internal,
        amp=amp,
        theta=zwfs_ns.optics.theta,
        phasemask_diameter=zwfs_ns.optics.mask_diam,
        phasemask_mask=zwfs_ns.grid.phasemask_mask,
        pupil_diameter=zwfs_ns.grid.N,
        fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
        pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
    )

    Intensity = get_pupil_intensity(
        phi=phi,
        amp=amp,
        theta=zwfs_ns.optics.theta,
        phasemask_diameter=zwfs_ns.optics.mask_diam,
        phasemask_mask=zwfs_ns.grid.phasemask_mask,
        pupil_diameter=zwfs_ns.grid.N,
        fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
        pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
    )

    # get_pupil_intensity( phi = phi, theta = zwfs_ns.optics.theta, phasemask=zwfs_ns.grid.phasemask_mask, amp=amp )

    return phi, phi_internal, N0, I0, Intensity


def average_subarrays(array, block_size):
    """
    Averages non-overlapping sub-arrays of a given 2D NumPy array.

    Parameters:
    array (numpy.ndarray): Input 2D array of shape (N, M).
    block_size (tuple): Size of the sub-array blocks (height, width).

    Returns:
    numpy.ndarray: 2D array containing the averaged values of the sub-arrays.
    """
    # Check if the array dimensions are divisible by the block size
    if array.shape[0] % block_size[0] != 0 or array.shape[1] % block_size[1] != 0:
        raise ValueError("Array dimensions must be divisible by the block size.")

    # Reshape the array to isolate the sub-arrays
    reshaped = array.reshape(
        array.shape[0] // block_size[0],
        block_size[0],
        array.shape[1] // block_size[1],
        block_size[1],
    )

    # Compute the mean of the sub-arrays
    averaged_subarrays = reshaped.mean(axis=(1, 3))

    return averaged_subarrays


def sum_subarrays(array: NDArray, block_size: Tuple[int, int]) -> NDArray:
    """
    Averages non-overlapping sub-arrays of a given 2D NumPy array.

    Parameters:
    array (numpy.ndarray): Input 2D array of shape (N, M).
    block_size (tuple): Size of the sub-array blocks (height, width).

    Returns:
    numpy.ndarray: 2D array containing the averaged values of the sub-arrays.
    """
    # Check if the array dimensions are divisible by the block size
    if array.shape[0] % block_size[0] != 0 or array.shape[1] % block_size[1] != 0:
        raise ValueError("Array dimensions must be divisible by the block size.")

    # Reshape the array to isolate the sub-arrays
    reshaped = array.reshape(
        array.shape[0] // block_size[0],
        block_size[0],
        array.shape[1] // block_size[1],
        block_size[1],
    )

    # Compute the mean of the sub-arrays
    summed_subarrays = reshaped.sum(axis=(1, 3))

    return summed_subarrays


def calculate_detector_binning_factor(
    grid_pixels_across_pupil, detector_pixels_across_pupil
):
    binning = grid_pixels_across_pupil / detector_pixels_across_pupil
    return binning


def detect(
    i: NDArray,
    binning: Tuple[int, int],
    qe: float,
    dit: float,
    ron: float = 0.0,
    include_shotnoise=True,
    spectral_bandwidth: Optional[float] = None,
) -> NDArray:
    """
    Convert an input photon-rate image into a detected detector frame.

    Unit convention
    ---------------
    This detector method supports two closely related input conventions,
    controlled by the `spectral_bandwidth` argument.

    1. Monochromatic or spectral-density input

        If `spectral_bandwidth` is not None, the input image `i` is assumed
        to have units:

            photons / second / wave-space-pixel / nm

        The detector first spatially bins the image, then multiplies by:

            spectral_bandwidth * dit * qe

        to obtain detected photoelectrons.

        This is the historical BaldrApp convention and corresponds to a flat
        spectrum across the supplied spectral bandwidth.

    2. Already spectrally integrated input

        If `spectral_bandwidth` is None, the input image `i` is assumed to
        already have units:

            photons / second / wave-space-pixel

        In this case no spectral-bandwidth multiplication is applied. The
        detector only applies spatial binning, detector integration time, QE,
        optional shot noise, and read noise.

    Polychromatic simulations
    -------------------------
    For non-flat or explicitly sampled spectra, spectral integration should be
    performed before calling this detector method.

    If each monochromatic image `I_density_k` has units:

        photons / second / wave-space-pixel / nm

    then the spectral integration must include wavelength-bin widths. For a
    discrete wavelength grid this is generally:

        I_rate = sum_k I_density_k * spectral_shape_k * delta_lambda_k_nm

    where `delta_lambda_k_nm` is the wavelength-bin width associated with
    wavelength sample k, and `spectral_shape_k` is the relative spectral shape
    evaluated at that sample.

    Equivalently, define integration weights:

        weights_nm_k = spectral_shape_k * delta_lambda_k_nm

    with units of nm. Then:

        I_rate = sum_k weights_nm_k * I_density_k

    The resulting `I_rate` has units:

        photons / second / wave-space-pixel

    and should be passed to this detector method with:

        spectral_bandwidth=None

    If using fractional weights normalized such that:

        sum_k weight_k = 1

    then the full-band integration can also be written as:

        I_rate = bandwidth_nm * sum_k weight_k * I_density_k

    but this is only equivalent when:

        bandwidth_nm * weight_k

    correctly represents the wavelength-bin integration weight for sample k.
    To avoid ambiguity, higher-level polychromatic frame-generation code should
    prefer explicit `weights_nm` or `delta_lambda_k_nm` integration weights.

    Design note
    -----------
    The detector intentionally does not inspect `zwfs_ns.spectrum` or any
    instrument-level spectral configuration. Spectral weighting, chromatic ZWFS
    propagation, Fresnel propagation, and bandwidth integration should be
    handled by higher-level frame-generation functions before this detector
    method is called.

    In other words:

        - Use `spectral_bandwidth` only for the legacy flat-spectrum density
        convention.
        - Use `spectral_bandwidth=None` when the image has already been
        integrated over wavelength.

    Parameters
    ----------
    i : ndarray
        Input photon image. Units depend on `spectral_bandwidth`, as described
        above.

    include_shotnoise : bool
        If True, apply Poisson shot noise after conversion to expected detected
        photoelectrons.

    spectral_bandwidth : float, None, or "auto"
        Spectral bandwidth in nm for the legacy flat-spectrum convention.
        If None, `i` is treated as already spectrally integrated.

    Returns
    -------
    detected_image : ndarray
        Detector image after binning, detector integration time, QE, optional
        shot noise, and read noise.
    """
    """_summary_
    assumes input intensity is in photons per second per pixel per nm, 
    if spectral_bandwidth is None than returns photons per pixel per nm of input light,
    otherwise returns photons per pixel
    
    Args:
        i (2D array like): _description_ input intensity before being detected on a detector (generally higher spatial resolution than detector)
        binning (tuple): _description_ binning factor (rows to sum, columns to sum) 
        qe (scalar): _description_ quantum efficiency of detector
        dit (scalar): _description_ integration time of detector
        ron (float, optional): _description_. Defaults to 1. readout noise in electrons per pixel
        include_shotnoise (bool, optional): _description_. Defaults to True. Sample poisson distribution for each pixel (input intensity is the expected value)
        spectral_bandwidth (_type_, optional): _description_. Defaults to None. if spectral_bandwidth is None than returns photons per pixel per nm of input light,
    """

    i = sum_subarrays(array=i, block_size=binning)

    if spectral_bandwidth is None:
        i *= qe * dit
    else:
        i *= qe * dit * spectral_bandwidth

    if include_shotnoise:
        noisy_intensity = np.random.poisson(lam=i)
    else:  # no noise
        noisy_intensity = i

    if ron > 0:
        noisy_intensity += np.random.normal(0, ron, noisy_intensity.shape).astype(int)

    return noisy_intensity


def get_I0(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    use_pyZelda=True,
):
    """_summary_
    ## LEGACY - should use get_I0_configured now which can handle fresnel propagaation and polychromatic spectrum consistently from config file

    propagates the input field with phase described by opd_input and internal aberrations described by opd_internal, field flux described by amp_input
    through a Zernike wavefront sensor system described by the zwfs_ns namespace.

    I0 is the reference intensity so HERE the DM is set to the flat state (no correction)

    Args:
        opd_input (2D array like): _description_ opd map in units of meters in wavespace
        amp_input (2D array like): _description_ flux of the input field in wavespace in units of sqrt( photons / pixel / second / nm )
        opd_internal (2D array like): _description_ opd map in units of meters in wavespace for internal aberrations (somewhat redudant with opd_input.. i know)
        zwfs_ns (simple name space): _description_ namespace containing all the information about the zwfs system. Use configuration file and init_zwfs_from_config_ini to get this
        detector (_type_, optional): _description_. Defaults to None. if None returns intensity in wave space, otherwise returns intensity on detector.
            Detector should be a class or namespace with dit, qe, ron, binning, attributes. If you want to use the spectral_bandwidth attribute of zwfs_ns.stellar should also be present
        include_shotnoise (bool, optional): _description_. Defaults to True.
        use_pyZelda (bool, optional): _description_. Defaults to True. use pyZelda to propagate the opd map. If False use the get_pupil_intensity function to propagate the opd map

    Raises:
        ValueError: _description_

    Returns:
        _type_: field intensity
    """
    opd_current_dm = zwfs_ns.compiled_dm.eval(
        zwfs_ns.dm.dm_flat * zwfs_ns.dm.opd_per_cmd
    )

    opd_map = opd_input + opd_internal + opd_current_dm

    if use_pyZelda and (not hasattr(zwfs_ns, "pyZelda")):
        raise ValueError(
            "use_pyZelda= True but pyZelda not in zwfs_ns (no zwfs_ns.pyZelda namespace exists).\
            Add pyZelda to zwfs_ns namespace or Set use_pyZelda = False to use get_pupil_intensity function instead"
        )

    if use_pyZelda:
        Intensity = ztools.propagate_opd_map(
            zwfs_ns.pyZelda.pupil * opd_map,
            zwfs_ns.pyZelda.mask_diameter,
            zwfs_ns.pyZelda.mask_depth,
            zwfs_ns.pyZelda.mask_substrate,
            zwfs_ns.pyZelda.mask_Fratio,
            zwfs_ns.pyZelda.pupil_diameter,
            amp_input * zwfs_ns.pyZelda.pupil,
            wave=zwfs_ns.optics.wvl0,
            fourier_filter_diam=zwfs_ns.pyZelda.fourier_filter_diam,
        )
        # Intensity =  amp_input**2 * zwfs_ns.pyZelda.propagate_opd_map( opd_map , wave = zwfs_ns.optics.wvl0 )
        # Intensity =  amp_input**2 * ztools.propagate_opd_map( zwfs_ns.pyZelda.pupil * (opd_map), zwfs_ns.pyZelda.mask_diameter, zwfs_ns.pyZelda.mask_depth, zwfs_ns.pyZelda.mask_substrate,
        #                                    zwfs_ns.pyZelda.mask_Fratio, zwfs_ns.pyZelda.pupil_diameter, zwfs_ns.pyZelda.pupil, wave=zwfs_ns.optics.wvl0)

    else:
        phi = zwfs_ns.grid.pupil_mask * 2 * np.pi / zwfs_ns.optics.wvl0 * (opd_map)

        ## before cold stop update
        # Intensity = get_pupil_intensity( phi= phi, amp=amp_input, theta = zwfs_ns.optics.theta , phasemask_diameter = zwfs_ns.optics.mask_diam, \
        #     phasemask_mask = zwfs_ns.grid.phasemask_mask, pupil_diameter = zwfs_ns.grid.N, fplane_pixels=zwfs_ns.focal_plane.fplane_pixels, \
        #         pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask )

        # with coldstop update
        Intensity = get_pupil_intensity(
            phi=phi,
            amp=amp_input,
            theta=zwfs_ns.optics.theta,
            phasemask_diameter=zwfs_ns.optics.mask_diam,
            coldstop_diam=zwfs_ns.optics.coldstop_diam,  # new
            coldstop_offset=zwfs_ns.optics.coldstop_offset,  # new
            pupil_diameter=zwfs_ns.grid.N,
            fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
            pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
            phasemask_mask=None,
        )

    assert type(Intensity) is np.ndarray

    if detector is not None:
        if not hasattr(zwfs_ns, "stellar"):
            raise ValueError(
                "zwfs_ns must have a stellar attribute to get spectral bandwidth (zwfs_ns.stellar.bandwidth )"
            )

        Intensity = detect(
            Intensity,
            binning=(detector.binning, detector.binning),
            qe=detector.qe,
            dit=detector.dit,
            ron=detector.ron,
            include_shotnoise=include_shotnoise,
            spectral_bandwidth=zwfs_ns.stellar.bandwidth,
        )
        # average_subarrays(array=Intensity, block_size = detector)

    return Intensity


def get_I0_configured(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    use_pyZelda=False,
    spectral_bandwidth=_AUTO_SPECTRAL_BANDWIDTH,
    force_fresnel=None,
    force_polychromatic=None,
    return_intermediates=False,
):
    """
    Generate the Baldr reference intensity I0 using the configured propagation
    path.

    I0 is the phase-mask-in reference intensity. It is evaluated with the DM set
    to its flat command, zwfs_ns.dm.dm_flat, then the previous DM command is
    restored.

    This wrapper uses get_frame_configured(...), so it supports the same routing:

        non-Fresnel monochromatic
        non-Fresnel polychromatic
        Fresnel monochromatic
        Fresnel polychromatic

    This is preferred over legacy get_I0(...) for simulations using Fresnel
    propagation or explicit spectral integration.
    """
    original_cmd = zwfs_ns.dm.current_cmd.copy()

    try:
        zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()

        return get_frame_configured(
            opd_input=opd_input,
            amp_input=amp_input,
            opd_internal=opd_internal,
            zwfs_ns=zwfs_ns,
            detector=detector,
            include_shotnoise=include_shotnoise,
            use_pyZelda=use_pyZelda,
            spectral_bandwidth=spectral_bandwidth,
            force_fresnel=force_fresnel,
            force_polychromatic=force_polychromatic,
            return_intermediates=return_intermediates,
        )

    finally:
        zwfs_ns.dm.current_cmd = original_cmd


def get_N0(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    use_pyZelda=True,
) -> NDArray:
    """_summary_
    ## LEGACY - you should use get_N0_configured now now which can handle fresnel propagaation and polychromatic spectrum consistently from config file
    propagates the input field with phase described by opd_input and internal aberrations described by opd_internal, field flux described by amp_input
    through a Zernike wavefront sensor system described by the zwfs_ns namespace WITH NO PHASEMASK INSERTED (CLEAR PUPIL).

    Args:
        opd_input (2D array like): _description_ opd map in units of meters in wavespace
        amp_input (2D array like): _description_ flux of the input field in wavespace in units of sqrt( photons / wavespace_pixel / second / nm )
        opd_internal (2D array like): _description_ opd map in units of meters in wavespace for internal aberrations (somewhat redudant with opd_input.. i know)
        zwfs_ns (simple name space): _description_ namespace containing all the information about the zwfs system. Use configuration file and init_zwfs_from_config_ini to get this
        detector (_type_, optional): _description_. Defaults to None. if None returns intensity in wave space, otherwise returns intensity on detector.
            Detector should be a class or namespace with dit, qe, ron, binning, attributes. If you want to use the spectral_bandwidth attribute of zwfs_ns.stellar should also be present
        include_shotnoise (bool, optional): _description_. Defaults to True.
        use_pyZelda (bool, optional): _description_. Defaults to True. use pyZelda to propagate the opd map. If False use the get_pupil_intensity function to propagate the opd map

    Raises:
        ValueError: _description_

    Returns:
        _type_: field intensity
    """

    opd_current_dm = zwfs_ns.compiled_dm.eval(
        zwfs_ns.dm.dm_flat * zwfs_ns.dm.opd_per_cmd
    )

    opd_map = opd_input + opd_internal + opd_current_dm

    if use_pyZelda and (not hasattr(zwfs_ns, "pyZelda")):
        raise ValueError(
            "use_pyZelda= True but pyZelda not in zwfs_ns (no zwfs_ns.pyZelda namespace exists).\
            Add pyZelda to zwfs_ns namespace or Set use_pyZelda = False to use get_pupil_intensity function instead"
        )

    if use_pyZelda:
        Intensity = ztools.propagate_opd_map(
            zwfs_ns.pyZelda.pupil * opd_map,
            0 * zwfs_ns.pyZelda.mask_diameter,
            zwfs_ns.pyZelda.mask_depth,
            zwfs_ns.pyZelda.mask_substrate,
            zwfs_ns.pyZelda.mask_Fratio,
            zwfs_ns.pyZelda.pupil_diameter,
            amp_input * zwfs_ns.pyZelda.pupil,
            wave=zwfs_ns.optics.wvl0,
            fourier_filter_diam=zwfs_ns.pyZelda.fourier_filter_diam,
        )
        # Intensity =  amp_input**2 * ztools.propagate_opd_map(opd_map, zwfs_ns.pyZelda.mask_diameter, 0 * zwfs_ns.pyZelda.mask_depth, zwfs_ns.pyZelda.mask_substrate,
        #                                   zwfs_ns.pyZelda.mask_Fratio, zwfs_ns.pyZelda.pupil_diameter, zwfs_ns.pyZelda.pupil, wave=zwfs_ns.optics.wvl0)
        #

    else:
        #  convert to radians
        phi = zwfs_ns.grid.pupil_mask * 2 * np.pi / zwfs_ns.optics.wvl0 * (opd_map)

        # before coldstop update
        # Intensity = get_pupil_intensity( phi= phi, amp=amp_input, theta = 0 , phasemask_diameter = zwfs_ns.optics.mask_diam, \
        #     phasemask_mask = zwfs_ns.grid.phasemask_mask, pupil_diameter = zwfs_ns.grid.N, fplane_pixels=zwfs_ns.focal_plane.fplane_pixels, \
        #         pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask )

        # after coldstop update
        Intensity = get_pupil_intensity(
            phi=phi,
            amp=amp_input,
            theta=0,
            phasemask_diameter=zwfs_ns.optics.mask_diam,
            coldstop_diam=zwfs_ns.optics.coldstop_diam,  # new
            coldstop_offset=zwfs_ns.optics.coldstop_offset,  # new
            pupil_diameter=zwfs_ns.grid.N,
            fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
            pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
            phasemask_mask=None,
        )
    assert type(Intensity) is np.ndarray

    if detector is not None:
        if not hasattr(zwfs_ns, "stellar"):
            raise ValueError(
                "zwfs_ns must have a stellar attribute to get spectral bandwidth (zwfs_ns.stellar.bandwidth )"
            )

        Intensity = detect(
            Intensity,
            binning=(detector.binning, detector.binning),
            qe=detector.qe,
            dit=detector.dit,
            ron=detector.ron,
            include_shotnoise=include_shotnoise,
            spectral_bandwidth=zwfs_ns.stellar.bandwidth,
        )
        # average_subarrays(array=Intensity, block_size = detector)

    return Intensity


def get_N0_configured(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    use_pyZelda=False,
    spectral_bandwidth=_AUTO_SPECTRAL_BANDWIDTH,
    force_fresnel=None,
    force_polychromatic=None,
    return_intermediates=False,
):
    """
    Generate the Baldr clear-pupil reference intensity N0 using the configured
    propagation path.

    N0 is the phase-mask-out reference intensity. It is evaluated with:

        zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat
        phase-mask phase shift theta = 0
        active physical/chromatic phasemask disabled, if present

    The previous DM command and phase-mask state are restored afterwards.

    This wrapper uses get_frame_configured(...), so it supports:

        non-Fresnel monochromatic
        non-Fresnel polychromatic
        Fresnel monochromatic
        Fresnel polychromatic
    """
    original_cmd = zwfs_ns.dm.current_cmd.copy()

    # Legacy optics state.
    original_theta = getattr(zwfs_ns.optics, "theta", None)
    original_theta_mode = getattr(zwfs_ns.optics, "theta_mode", "constant")

    # New physical/chromatic phasemask state.
    had_active_phasemask = hasattr(zwfs_ns.optics, "active_phasemask")
    original_active_phasemask = (
        zwfs_ns.optics.active_phasemask if had_active_phasemask else None
    )

    # Optional runtime diagnostics / current mask label.
    had_current_mask = hasattr(zwfs_ns.optics, "current_mask")
    original_current_mask = zwfs_ns.optics.current_mask if had_current_mask else None

    # pyZELDA compatibility, if present.
    has_pyzelda = hasattr(zwfs_ns, "pyZelda")
    if has_pyzelda and hasattr(zwfs_ns.pyZelda, "_mask_depth"):
        original_mask_depth = zwfs_ns.pyZelda._mask_depth
    else:
        original_mask_depth = None

    try:
        # Use flat DM for the reference.
        zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()

        # Generic analytic/Fresnel path: theta=0 removes the phase-mask effect.
        zwfs_ns.optics.theta = 0.0
        zwfs_ns.optics.theta_mode = "constant"

        # Critical for the new physical/chromatic mask path:
        # remove active_phasemask so spec.theta_at_wavelength(...) cannot
        # recompute a non-zero physical-depth theta during polychromatic N0.
        if had_active_phasemask:
            delattr(zwfs_ns.optics, "active_phasemask")

        # Optional diagnostic label while this temporary state is active.
        zwfs_ns.optics.current_mask = ""

        # pyZELDA path: also force the physical mask depth to zero if possible.
        if has_pyzelda and original_mask_depth is not None:
            zwfs_ns.pyZelda._mask_depth = 0.0

        return get_frame_configured(
            opd_input=opd_input,
            amp_input=amp_input,
            opd_internal=opd_internal,
            zwfs_ns=zwfs_ns,
            detector=detector,
            include_shotnoise=include_shotnoise,
            use_pyZelda=use_pyZelda,
            spectral_bandwidth=spectral_bandwidth,
            force_fresnel=force_fresnel,
            force_polychromatic=force_polychromatic,
            return_intermediates=return_intermediates,
        )

    finally:
        # Restore DM command.
        zwfs_ns.dm.current_cmd = original_cmd

        # Restore legacy optics state.
        if original_theta is None:
            if hasattr(zwfs_ns.optics, "theta"):
                delattr(zwfs_ns.optics, "theta")
        else:
            zwfs_ns.optics.theta = original_theta

        zwfs_ns.optics.theta_mode = original_theta_mode

        # Restore active physical/chromatic phasemask state.
        if had_active_phasemask:
            zwfs_ns.optics.active_phasemask = original_active_phasemask
        else:
            if hasattr(zwfs_ns.optics, "active_phasemask"):
                delattr(zwfs_ns.optics, "active_phasemask")

        # Restore current-mask diagnostic label.
        if had_current_mask:
            zwfs_ns.optics.current_mask = original_current_mask
        else:
            if hasattr(zwfs_ns.optics, "current_mask"):
                delattr(zwfs_ns.optics, "current_mask")

        # Restore pyZELDA mask depth.
        if has_pyzelda and original_mask_depth is not None:
            zwfs_ns.pyZelda._mask_depth = original_mask_depth


# def get_N0_configured(
#     opd_input,
#     amp_input,
#     opd_internal,
#     zwfs_ns,
#     detector=None,
#     include_shotnoise=True,
#     use_pyZelda=False,
#     spectral_bandwidth=_AUTO_SPECTRAL_BANDWIDTH,
#     force_fresnel=None,
#     force_polychromatic=None,
#     return_intermediates=False,
# ):
#     """
#     Generate the Baldr clear-pupil reference intensity N0 using the configured
#     propagation path.

#     N0 is the phase-mask-out reference intensity. It is evaluated with:

#         zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat
#         phase-mask phase shift theta = 0

#     The previous DM command and phase-mask setting are restored afterwards.

#     This wrapper uses get_frame_configured(...), so it supports the same routing:

#         non-Fresnel monochromatic
#         non-Fresnel polychromatic
#         Fresnel monochromatic
#         Fresnel polychromatic

#     This is preferred over legacy get_N0(...) for simulations using Fresnel
#     propagation or explicit spectral integration.
#     """
#     original_cmd = zwfs_ns.dm.current_cmd.copy()
#     original_theta = zwfs_ns.optics.theta

#     # pyZELDA compatibility, if present.
#     has_pyzelda = hasattr(zwfs_ns, "pyZelda")
#     if has_pyzelda and hasattr(zwfs_ns.pyZelda, "_mask_depth"):
#         original_mask_depth = zwfs_ns.pyZelda._mask_depth
#     else:
#         original_mask_depth = None

#     try:
#         zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()

#         # For the generic analytic path, theta=0 removes the phase mask.
#         zwfs_ns.optics.theta = 0.0

#         # For pyZELDA, also force the physical mask depth to zero if possible.
#         if has_pyzelda and original_mask_depth is not None:
#             zwfs_ns.pyZelda._mask_depth = 0.0

#         return get_frame_configured(
#             opd_input=opd_input,
#             amp_input=amp_input,
#             opd_internal=opd_internal,
#             zwfs_ns=zwfs_ns,
#             detector=detector,
#             include_shotnoise=include_shotnoise,
#             use_pyZelda=use_pyZelda,
#             spectral_bandwidth=spectral_bandwidth,
#             force_fresnel=force_fresnel,
#             force_polychromatic=force_polychromatic,
#             return_intermediates=return_intermediates,
#         )

#     finally:
#         zwfs_ns.dm.current_cmd = original_cmd
#         zwfs_ns.optics.theta = original_theta

#         if has_pyzelda and original_mask_depth is not None:
#             zwfs_ns.pyZelda._mask_depth = original_mask_depth


def get_b(
    phi,
    amp,
    phasemask_diameter,
    phasemask_mask,
    fplane_pixels=300,
    pixels_across_mask=10,
    detector=None,
):
    # copy exaclt machinary from get_pupil_intensity( )
    # if detector is None, then calculate in wavespace (same space as amp and opd lives in), otherwise pixel space by detecting the field

    # ---------------- even-grid safe: promote to odd internally ----------------
    phi = np.asarray(phi)
    amp = np.asarray(amp)
    assert phi.shape == amp.shape and phi.ndim == 2, "phi and amp must be same 2D shape"
    N_orig = amp.shape[0]
    assert amp.shape[1] == N_orig, "phi/amp must be square"

    padded_even = False
    if (N_orig % 2) == 0:
        # pad ONE row/col (bottom/right) -> internal odd grid; center aligns to a pixel
        amp_work = np.pad(amp, ((0, 1), (0, 1)), mode="constant")
        phi_work = np.pad(phi, ((0, 1), (0, 1)), mode="constant")
        N = N_orig + 1
        padded_even = True
    else:
        amp_work = amp
        phi_work = phi
        N = N_orig

    # ---------------- build pupil-plane field Ψ_A on the odd (or original odd) grid ----------------
    psi_A = amp_work.astype(np.complex128) * np.exp(1j * phi_work.astype(np.float64))

    # ---------------- choose focal-plane FFT size (prefer odd to keep centering simple) -------------
    Nf = int(max(fplane_pixels, N))
    if (Nf % 2) == 0:  # enforce odd focal grid as well
        Nf += 1

    # ---------------- center embed to focal grid ----------------
    pad = (Nf - N) // 2  # integer because both are odd
    psi_A_pad = np.zeros((Nf, Nf), dtype=np.complex128)
    psi_A_pad[pad : pad + N, pad : pad + N] = psi_A

    # ---------------- forward FFT to focal plane ----------------
    psi_B = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(psi_A_pad), norm="ortho")
    )  # norm="ortho" to maintain flux conservation

    # ---------------- phase disc (derive physical pixels-per-(λ/D) correctly) ----------------
    def _center_resize_mask(mask, tgtN):
        m = np.asarray(mask, float)
        assert m.ndim == 2 and m.shape[0] == m.shape[1], "mask must be square"
        if m.shape[0] == tgtN:
            return (m > 0.5).astype(float)
        out = np.zeros((tgtN, tgtN), float)
        c0 = (m.shape[0] - 1) // 2
        ct = (tgtN - 1) // 2
        y0s = max(0, ct - c0)
        y0d = max(0, c0 - ct)
        h = min(m.shape[0], tgtN)
        out[y0s : y0s + h, y0s : y0s + h] = m[y0d : y0d + h, y0d : y0d + h]
        return (out > 0.5).astype(float)

    def _disc_radius_in_pixels(bin_mask):
        if bin_mask.max() == 0:
            return 0.0
        yy, xx = np.indices(bin_mask.shape)
        c = (bin_mask.shape[0] - 1) / 2.0
        r = np.hypot(yy - c, xx - c)
        return r[bin_mask.astype(bool)].max()

    if phasemask_mask is not None:
        phase_disc = _center_resize_mask(phasemask_mask, Nf)
        r_pix = _disc_radius_in_pixels(phase_disc)
        if r_pix <= 0 or phasemask_diameter <= 0:
            # fallback; will be superseded by cold-stop scaling anyway
            pix_per_wvld = float(pixels_across_mask)
        else:
            pix_per_wvld = (2.0 * r_pix) / float(phasemask_diameter)
    else:
        # derive physical pixels-per-(λ/D) from the pupil diameter on the padded grid
        amp_pad = np.zeros((Nf, Nf), dtype=float)
        amp_pad[pad : pad + N, pad : pad + N] = np.abs(amp_work)
        support = amp_pad > 0
        yy, xx = np.indices((Nf, Nf))
        c = (Nf - 1) / 2.0
        r = np.hypot(yy - c, xx - c)
        if np.any(support):
            Rpix = r[support].max()
            Dp_pix = 2.0 * Rpix
            pix_per_wvld = float(Nf) / max(Dp_pix, 1e-9)
        else:
            pix_per_wvld = float(pixels_across_mask)

        r_mask_pix = 0.5 * float(phasemask_diameter) * pix_per_wvld
        phase_disc = (r <= r_mask_pix).astype(float)

    # ---------------- reference arm & analytic pupil field ----------------
    b = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(phase_disc * psi_B), norm="ortho")
    )

    if detector is not None:
        b = detector.detect(b)

    return b


def estimate_clear_pupil_onsky(
    zwfs_ns,
    detector,
    *,
    # phase screen params
    ps_module,  # e.g. ps (poppy/soapy phase screen module you use)
    dx,
    r0,
    L0,
    phase_seed=43,
    # scintillation screen params
    aotools_module=None,  # provide aotools if include_scintillation=True
    r0_scint=None,
    L0_scint=None,
    scint_seed=42,
    propagation_distance=None,
    jumps_per_iter=1,
    include_scintillation=True,
    update_scintillation_fn=None,  # callable: update_scintillation(high_alt_phasescreen, pxl_scale, wavelength, final_size, jumps, propagation_distance)
    amp_input_0=1.0,  # scalar or 2D array (same shape as pupil grid)
    # AO1 residual / latency sim
    it_lag=0,
    Nmodes_removed=0,
    basis=None,
    phase_scaling_factor=1.0,
    ao1_add_rows_per_iter=1,
    # simulation loop
    N_samples=100,
    opd_internal_onsky=0.0,  # 2D OPD internal term (or 0)
    use_pyZelda=False,
    return_intermediates=False,
):
    """
    Estimate the clear-pupil intensity N0 on-sky by Monte-Carlo sampling phase (+ optional scintillation),
    while including a first-stage AO residual with a rolling latency buffer.

    Returns
    -------
    N0_est : 2D ndarray
        Mean clear-pupil intensity estimate (same shape as zwfs pupil grid)
    N0_samples : 3D ndarray, optional
        Stack of samples (N_samples, ny, nx) if return_samples=True
    screens : dict
        The instantiated independent screens (useful for debugging/repro)
    """

    # ----------------------------
    # basic validation / defaults
    # ----------------------------
    if include_scintillation:
        if aotools_module is None:
            raise ValueError("include_scintillation=True requires aotools_module.")
        if update_scintillation_fn is None:
            raise ValueError(
                "include_scintillation=True requires update_scintillation_fn."
            )
        if propagation_distance is None:
            raise ValueError(
                "include_scintillation=True requires propagation_distance."
            )
        if (r0_scint is None) or (L0_scint is None):
            raise ValueError(
                "include_scintillation=True requires r0_scint and L0_scint."
            )

    if basis is None:
        raise ValueError("basis must be provided (AO1 residual uses it).")

    # ----------------------------
    # independent, reproducible screens
    # ----------------------------
    print(
        f"initialising new (independent) phase and scintillation screens with the given input statistics"
    )
    scrn_new = ps_module.PhaseScreenKolmogorov(
        nx_size=zwfs_ns.grid.dim,
        pixel_scale=dx,
        r0=r0,
        L0=L0,
        random_seed=phase_seed,
    )

    scint_scrn_new = None
    if include_scintillation:
        assert aotools_module is not None
        scint_scrn_new = (
            aotools_module.turbulence.infinitephasescreen.PhaseScreenVonKarman(
                nx_size=zwfs_ns.grid.dim,
                pixel_scale=dx,
                r0=r0_scint,
                L0=L0_scint,
                random_seed=scint_seed,
            )
        )

    # ----------------------------
    # populate AO1 rolling buffer (latency)
    # ----------------------------
    reco_list = []
    for _ in range(int(it_lag)):
        scrn_new.add_row()
        _, reco_1 = first_stage_ao(
            atm_scrn=scrn_new,
            Nmodes_removed=Nmodes_removed,
            basis=basis,
            phase_scaling_factor=phase_scaling_factor,
            return_reconstructor=True,
        )
        reco_list.append(reco_1)

    # ensure DM starts flat for N0 estimate
    zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()

    # allocate
    N0_samples = []

    for itt in range(int(N_samples)):
        if np.mod(itt, 100):
            print(f"complete {itt}/{N_samples}")
        # --- evolve phase screen / AO1 residual ---
        for _k in range(int(ao1_add_rows_per_iter)):
            scrn_new.add_row()

        _, reco_1 = first_stage_ao(
            atm_scrn=scrn_new,
            Nmodes_removed=Nmodes_removed,
            basis=basis,
            phase_scaling_factor=phase_scaling_factor,
            return_reconstructor=True,
        )
        reco_list.append(reco_1)

        # latency: subtract an older reconstructor output
        delayed_reco = reco_list.pop(0) if len(reco_list) > 0 else 0.0
        ao_1_residual_phase = basis[0] * (
            phase_scaling_factor * scrn_new.scrn - delayed_reco
        )

        # convert phase [rad] -> OPD [m]
        opd_input = (
            phase_scaling_factor
            * zwfs_ns.optics.wvl0
            / (2.0 * np.pi)
            * ao_1_residual_phase
        )

        # --- scintillation amplitude ---
        if include_scintillation:
            for _k in range(int(jumps_per_iter)):
                scint_scrn_new.add_row()  # type: ignore
            assert update_scintillation_fn is not None
            amp_scint = update_scintillation_fn(
                high_alt_phasescreen=scint_scrn_new,
                pxl_scale=dx,
                wavelength=zwfs_ns.optics.wvl0,
                final_size=None,
                jumps=0,
                propagation_distance=propagation_distance,
            )
            amp_input = amp_input_0 * amp_scint
        else:
            amp_input = amp_input_0

        # --- DM OPD contribution (flat) ---
        # opd_dm = get_dm_displacement(
        #     command_vector=zwfs_ns.dm.current_cmd,
        #     gain=zwfs_ns.dm.opd_per_cmd,
        #     sigma=zwfs_ns.grid.dm_coord.act_sigma_wavesp,
        #     X=zwfs_ns.grid.wave_coord.X,
        #     Y=zwfs_ns.grid.wave_coord.Y,
        #     x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp,
        #     y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp,
        # )
        opd_dm = zwfs_ns.compiled_dm.eval(
            zwfs_ns.dm.current_cmd * zwfs_ns.dm.opd_per_cmd
        )

        opd_total = opd_input + opd_dm

        # --- N0 measurement (clear pupil intensity) ---
        N0 = get_N0(
            opd_total,
            amp_input,
            opd_internal_onsky,
            zwfs_ns,
            detector=detector,
            use_pyZelda=use_pyZelda,
        )
        N0_samples.append(N0)

    N0_samples = np.asarray(N0_samples)  # (N, ny, nx)
    N0_est = np.mean(N0_samples, axis=0)

    if return_intermediates:
        screens = {"phase": scrn_new, "scint": scint_scrn_new}
        return N0_est, N0_samples, screens

    return N0_samples


# def get_frame( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector=None, include_shotnoise=True , use_pyZelda = True):
def get_frame(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    use_pyZelda=True,
    spectral_bandwidth=_AUTO_SPECTRAL_BANDWIDTH,
):
    """_summary_
    propagates the input field with phase described by opd_input and internal aberrations described by opd_internal, field flux described by amp_input
    through a Zernike wavefront sensor system described by the zwfs_ns namespace.

    Args:
        opd_input (2D array like): _description_ opd map in units of meters in wavespace
        amp_input (2D array like): _description_ flux of the input field in wavespace in units of sqrt( photons / pixel / second / nm )
        opd_internal (2D array like): _description_ opd map in units of meters in wavespace for internal aberrations (somewhat redudant with opd_input.. i know)
        zwfs_ns (simple name space): _description_ namespace containing all the information about the zwfs system. Use configuration file and init_zwfs_from_config_ini to get this
        detector (_type_, optional): _description_. Defaults to None. if None returns intensity in wave space, otherwise returns intensity on detector.
            Detector should be a class or namespace with dit, qe, ron, binning, attributes. If you want to use the spectral_bandwidth attribute of zwfs_ns.stellar should also be present
        include_shotnoise (bool, optional): _description_. Defaults to True.
        use_pyZelda (bool, optional): _description_. Defaults to True. use pyZelda to propagate the opd map. If False use the get_pupil_intensity function to propagate the opd map

    Raises:
        ValueError: _description_

    Returns:
        _type_: field intensity
    """

    # I could do this outside to save time but for now just do it here
    opd_current_dm = zwfs_ns.compiled_dm.eval(
        zwfs_ns.dm.current_cmd * zwfs_ns.dm.opd_per_cmd
    )

    opd_map = opd_input + opd_internal + opd_current_dm

    if use_pyZelda and (not hasattr(zwfs_ns, "pyZelda")):
        raise ValueError(
            "use_pyZelda= True but pyZelda not in zwfs_ns (no zwfs_ns.pyZelda namespace exists).\
            Add pyZelda to zwfs_ns namespace or Set use_pyZelda = False to use get_pupil_intensity function instead"
        )

    if use_pyZelda:

        Intensity = ztools.propagate_opd_map(
            zwfs_ns.pyZelda.pupil * opd_map,
            zwfs_ns.pyZelda.mask_diameter,
            zwfs_ns.pyZelda.mask_depth,
            zwfs_ns.pyZelda.mask_substrate,
            zwfs_ns.pyZelda.mask_Fratio,
            zwfs_ns.pyZelda.pupil_diameter,
            amp_input * zwfs_ns.pyZelda.pupil,
            wave=zwfs_ns.optics.wvl0,
            fourier_filter_diam=zwfs_ns.pyZelda.fourier_filter_diam,
        )
        # amp_input**2 * zwfs_ns.pyZelda.propagate_opd_map( opd_map , wave = zwfs_ns.optics.wvl0 )

    else:

        # convert phase in radians
        phi = zwfs_ns.grid.pupil_mask * 2 * np.pi / zwfs_ns.optics.wvl0 * (opd_map)

        if not hasattr(zwfs_ns.optics, "coldstop_offset"):
            print(
                "zwfs_ns.optics has no coldstop_offset attribute, assigning aligned coldstop ( coldstop_offset=(0,0))"
            )
            zwfs_ns.optics.coldstop_offset = (0, 0)

        # update 1/11/25 got rid of phasemask_mask input since with fft method (faster) we just generate this with the  phasemask_diameter depending on grid ,fplane_pixels and pixels_across_mask.
        Intensity = get_pupil_intensity(
            phi=phi,
            amp=amp_input,
            theta=zwfs_ns.optics.theta,
            phasemask_diameter=zwfs_ns.optics.mask_diam,
            coldstop_diam=zwfs_ns.optics.coldstop_diam,  # new
            coldstop_offset=zwfs_ns.optics.coldstop_offset,  # new
            pupil_diameter=zwfs_ns.grid.N,
            fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
            pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
            phasemask_mask=None,
        )  # phasemask_mask = None since with fft we dont keep fixed fourier plane sampling and just init phasemask each iteration... its still quicker than mft!
    assert type(Intensity) is np.ndarray
    # if detector is not None:
    #     if not hasattr(zwfs_ns, 'stellar') :
    #         raise ValueError("zwfs_ns must have a stellar attribute to get spectral bandwidth (zwfs_ns.stellar.bandwidth )")

    #     Intensity = detect( Intensity, binning = (detector.binning,detector.binning) , qe= detector.qe , dit= detector.dit, ron = detector.ron, include_shotnoise=include_shotnoise, spectral_bandwidth = zwfs_ns.stellar.bandwidth  )
    #     #average_subarrays(array=Intensity, block_size = detector)

    # return Intensity

    if detector is not None:
        detector_spectral_bandwidth = _resolve_detector_spectral_bandwidth(
            zwfs_ns,
            spectral_bandwidth=spectral_bandwidth,
        )

        Intensity = detect(
            Intensity,
            binning=(detector.binning, detector.binning),
            qe=detector.qe,
            dit=detector.dit,
            ron=detector.ron,
            include_shotnoise=include_shotnoise,
            spectral_bandwidth=detector_spectral_bandwidth,
        )

    return Intensity


def classify_pupil_regions(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    pupil_diameter_scaling=1.0,
    pupil_offset=(0, 0),
    use_pyZelda=True,
    mode="bright",
    lobe_threshold=0.03,
):

    ##18-12-25 redo to be consistent with how we do it in real Baldr
    N0 = get_N0(
        opd_input,
        amp_input,
        opd_internal,
        zwfs_ns,
        detector=detector,
        use_pyZelda=use_pyZelda,
    )
    I0 = get_I0(
        opd_input,
        amp_input,
        opd_internal,
        zwfs_ns,
        detector=detector,
        use_pyZelda=use_pyZelda,
    )

    center_x, center_y, a, b, theta, pupil_mask = util.detect_pupil(
        N0, sigma=2, threshold=0.5, plot=True, savepath=None
    )

    outside_filt = ~pupil_mask

    secondary_mask = util.get_secondary_mask(pupil_mask, (center_x, center_y))

    # filter edge of pupil and out radii limit for the strehl mask
    if "bright" in mode.lower():
        pupil_edge_filter = util.filter_exterior_annulus(
            pupil_mask, inner_radius=7, outer_radius=100
        )  # to limit pupil edge pixels
        pupil_limit_filter = ~util.filter_exterior_annulus(
            pupil_mask, inner_radius=11, outer_radius=100
        )  # to limit far out pixel
    elif "faint" in mode.lower():
        pupil_edge_filter = util.filter_exterior_annulus(
            pupil_mask, inner_radius=4, outer_radius=100
        )  # to limit pupil edge pixels
        pupil_limit_filter = ~util.filter_exterior_annulus(
            pupil_mask, inner_radius=8, outer_radius=100
        )  # to limit far out pixel
    else:
        raise UserWarning("invalid mode. Must be either 'bright' or 'faint'")
    # lobe_threshold = 0.1 # percentage of mean clear pupil interior. Absolute values above this in the exterior pixels are candidates for Strehl pixels
    # exterior_filter =  ( abs( I0  - N0 )  > lobe_threshold * np.mean( N0[pupil_mask] )  ) * (~pupil_mask) * pupil_edge_filter

    # to be more aggressive we can remove ~pupil_mask in filter
    exterior_mask = (
        (abs(I0 - N0) > lobe_threshold * np.mean(N0[pupil_mask]))
        * (~pupil_mask)
        * pupil_edge_filter
        * pupil_limit_filter
    )

    ### old simplified way commented out 18-12-25
    # # We intentionally put detector as None here to keep intensities in wave space
    # # we do the math here and then bin after if user selects detector is not None

    # # currently we don't use N0 to classify, just use known pupil diameter
    # #pupil_filt = zwfs_ns.grid.pupil_mask > 0.5
    # pupil_filt = (zwfs_ns.grid.wave_coord.X - pupil_offset[0])**2 + (zwfs_ns.grid.wave_coord.Y - pupil_offset[1])**2 <= pupil_diameter_scaling * (zwfs_ns.grid.D/2)**2

    # outside_filt = ~pupil_filt

    # secondary_strehl_filt = (zwfs_ns.grid.wave_coord.X - pupil_offset[0])**2 + (zwfs_ns.grid.wave_coord.Y - pupil_offset[1])**2 < (zwfs_ns.grid.D/10)**2

    # outer_strehl_filt = ( I0 - N0 >   4.5 * np.median(I0) ) * outside_filt

    # if detector is not None:
    #     pupil_filt = average_subarrays(array= pupil_filt, block_size=(zwfs_ns.detector.binning,zwfs_ns.detector.binning)) > 0

    #     outside_filt = average_subarrays(array= outside_filt, block_size=(zwfs_ns.detector.binning,zwfs_ns.detector.binning)) > 0

    #     secondary_strehl_filt = average_subarrays( secondary_strehl_filt ,block_size=(zwfs_ns.detector.binning,zwfs_ns.detector.binning)) > 0

    #     outer_strehl_filt = average_subarrays( outer_strehl_filt ,block_size=(zwfs_ns.detector.binning,zwfs_ns.detector.binning)) > 0

    region_classification_dict = {
        "pupil_filt": pupil_mask,
        "outside_filt": outside_filt,
        "secondary_strehl_filt": secondary_mask,
        "outer_strehl_filt": exterior_mask,
        "center_x": center_x,  # fitted centers
        "center_y": center_y,  # fitted centers
        "a": a,  # semi major axis of fitted ellipse
        "b": b,  # semi minor axis of fitted ellipse
        "theta": theta,  # orientation of semimajor axis
    }

    regions_ns = SimpleNamespace(**region_classification_dict)

    zwfs_ns.pupil_regions = regions_ns

    return zwfs_ns


def process_zwfs_signal(I, I0, pupil_filt):
    """_summary_

    STANDARD WAY TO PROCESS ZWFS ERROR SIGNAL FROM INTENSITY MEASUREMENT

    Args:
        I (_type_): _description_
        I0 (_type_): _description_
        pupil_filt (_type_): _description_

    Returns:
        _type_: _description_
    """
    s = (I / np.mean(I) - I0 / np.mean(I0))[pupil_filt]
    return s.reshape(-1)


def build_IM(
    zwfs_ns,
    calibration_opd_input,
    calibration_amp_input,
    opd_internal,
    basis="Zonal_pinned",
    Nmodes=100,
    poke_amp=0.05,
    poke_method="double_sided_poke",
    normalization_method="subframe mean",
    imgs_to_mean=10,
    detector=None,
    use_pyZelda=True,
):

    # build reconstructor name space with normalized basis, IM generated, IM generation method, pokeamp
    modal_basis = DM_basis.construct_command_basis(
        basis=basis, number_of_modes=Nmodes, without_piston=True
    ).T

    IM = []  # init our raw interaction matrix

    I0_list = []
    for _ in range(imgs_to_mean):
        I0_list.append(
            get_I0(
                opd_input=calibration_opd_input,
                amp_input=calibration_amp_input,
                opd_internal=opd_internal,
                zwfs_ns=zwfs_ns,
                detector=detector,
                use_pyZelda=use_pyZelda,
            )
        )
    I0 = np.mean(I0_list, axis=0)

    N0_list = []
    for _ in range(imgs_to_mean):
        N0_list.append(
            get_N0(
                opd_input=calibration_opd_input,
                amp_input=calibration_amp_input,
                opd_internal=opd_internal,
                zwfs_ns=zwfs_ns,
                detector=detector,
                use_pyZelda=use_pyZelda,
            )
        )
    N0 = np.mean(N0_list, axis=0)

    interior_pup_filt = binary_erosion(
        zwfs_ns.pupil_regions.pupil_filt
        * (~zwfs_ns.pupil_regions.secondary_strehl_filt),
        structure=np.ones((3, 3), dtype=bool),
    )

    if poke_method == "single_sided_poke":  # just poke one side

        for i, m in enumerate(modal_basis):
            print(f"executing cmd {i}/{len(modal_basis)}")

            zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + poke_amp * m

            img_list = []
            for _ in range(imgs_to_mean):
                img_list.append(
                    get_frame(
                        calibration_opd_input,
                        calibration_amp_input,
                        opd_internal,
                        zwfs_ns,
                        detector=detector,
                        use_pyZelda=use_pyZelda,
                    )
                )  # get some frames

            Intensity = np.mean(img_list, axis=0).reshape(-1)

            # IMPORTANT : we normalize by mean over total image region (post reduction) (NOT FILTERED )...
            # Intensity *= 1/np.mean( Intensity ) # we normalize by mean over total region!

            # get intensity error signal
            errsig = (
                Intensity - I0
            )  # process_zwfs_signal( Intensity, I0, zwfs_ns.pupil_regions.pupil_filt )

            IM.append(list(errsig.reshape(-1)))  # toook out 1/poke_amp *

    elif poke_method == "double_sided_poke":
        for i, m in enumerate(modal_basis):
            print(f"executing cmd {i}/{len(modal_basis)}")
            I_plus_list = []
            I_minus_list = []
            for sign in [
                (-1) ** n for n in range(np.max([2, imgs_to_mean]))
            ]:  # [-1,1]:

                # ZWFS.dm.send_data( list( ZWFS.dm_shapes['flat_dm'] + sign * poke_amp/2 * m )  )
                zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + sign * poke_amp / 2 * m

                if sign > 0:

                    I_plus_list += [
                        list(
                            get_frame(
                                calibration_opd_input,
                                calibration_amp_input,
                                opd_internal,
                                zwfs_ns,
                                detector=detector,
                                use_pyZelda=use_pyZelda,
                            )
                        )
                    ]

                if sign < 0:

                    I_minus_list += [
                        list(
                            get_frame(
                                calibration_opd_input,
                                calibration_amp_input,
                                opd_internal,
                                zwfs_ns,
                                detector=detector,
                                use_pyZelda=use_pyZelda,
                            )
                        )
                    ]

            I_plus = np.mean(I_plus_list, axis=0).reshape(
                -1
            )  # flatten so can filter with ZWFS.pupil_pixels
            # I_plus *= 1/np.mean( I_plus )

            I_minus = np.mean(I_minus_list, axis=0).reshape(
                -1
            )  # flatten so can filter with ZWFS.pupil_pixels
            # I_minus *= 1/np.mean( I_minus )

            if normalization_method.strip().lower() == "subframe mean":
                I_plus *= 1 / np.mean(I_plus)  # sum or mean wor
                I_minus *= 1 / np.mean(I_minus)
            elif (
                normalization_method.strip().lower() == "clear pupil mean"
            ):  # this is actually the better way , more resistent to
                I_plus *= 1 / np.mean(N0[interior_pup_filt])
                I_minus *= 1 / np.mean(N0[interior_pup_filt])
            else:
                raise UserWarning(
                    "invalid normalization_method method. Try subframe mean | clear pupil mean"
                )
            # 19/12/25 added divison of pokeamp
            errsig = (
                I_plus - I_minus
            ) / poke_amp  # [np.array( zwfs_ns.pupil_regions.pupil_filt.reshape(-1) )]
            IM.append(list(errsig.reshape(-1)))  # toook out 1/poke_amp *

    else:
        raise TypeError(
            ' no matching method for building control model. Try (for example) method="single_side_poke"'
        )

    # convert to array
    IM = np.array(IM)

    reco_dict = {
        "I0": I0,
        "N0": N0,
        "M2C_0": modal_basis,
        "basis_name": basis,
        "poke_amp": poke_amp,
        "poke_method": poke_method,
        "normalization_method": normalization_method,
        "interior_pup_filt": interior_pup_filt,
        "IM": IM,
    }

    reco_ns = SimpleNamespace(**reco_dict)

    zwfs_ns.reco = reco_ns

    return zwfs_ns


def register_DM_in_pixelspace_from_IM(zwfs_ns, plot_intermediate_results=True):
    """_summary_
    uses the interaction matrix (must be constructed on a zonal basis) to register the DM in pixel space.
    uses the inner corners of the DM to estimate the DM center in pixel space and calibrate an affine transform.
    Args:
        zwfs_ns (_type_): namespace containing all the information about the zwfs system. Use configuration file and init_zwfs_from_config_ini to get this
    """

    # get info about the basis used to generate the IM
    basis_name = zwfs_ns.reco.basis_name
    if "Zonal" not in basis_name:
        raise UserWarning(
            "basis used to construct IM must be zonal (either Zonal_pinned_edges or Zonal)"
        )
    Nmodes = zwfs_ns.reco.IM.shape[0]
    M2C_0 = DM_basis.construct_command_basis(
        basis=basis_name, number_of_modes=Nmodes, without_piston=True
    ).T

    # get inner corners for estiamting DM center in pixel space (have to deal seperately with pinned actuator basis)
    if zwfs_ns.reco.IM.shape[0] == 100:  # outer actuators are pinned,
        corner_indicies = DM_registration.get_inner_square_indices(
            outer_size=10, inner_offset=3, without_outer_corners=False
        )

    elif zwfs_ns.reco.IM.shape[0] == 140:  # outer acrtuators are free
        print(140)
        corner_indicies = DM_registration.get_inner_square_indices(
            outer_size=12, inner_offset=4, without_outer_corners=True
        )
    else:
        raise ValueError(f"CASE NOT MATCHED shape = {zwfs_ns.reco.IM.shape}")

    img_4_corners = []
    dm_4_corners = []
    for i in corner_indicies:
        dm_4_corners.append(np.where(M2C_0[i])[0][0])
        # dm2px.get_DM_command_in_2D( d['M2C'].data[:,i]  # if you want to plot it

        ## !!when we filtered the pupil in the interaction matrix
        # tmp = np.zeros( zwfs_ns.pupil_regions.pupil_filt.shape )
        # tmp.reshape(-1)[zwfs_ns.pupil_regions.pupil_filt.reshape(-1)] = zwfs_ns.reco.IM[i]

        # plt.imshow( tmp ); plt.show()
        # img_4_corners.append( abs(tmp ) )

        ## when we dont filter the pupil in the interaction matrix
        img_4_corners.append(
            abs(zwfs_ns.reco.IM[i].reshape(zwfs_ns.pupil_regions.pupil_filt.shape))
        )
    # plt.imshow( np.sum( tosee, axis=0 ) ); plt.show()

    # dm_4_corners should be an array of length 4 corresponding to the actuator index in the (flattened) DM command space
    # img_4_corners should be an array of length 4xNxM where NxM are the image dimensions.
    # !!! It is very important that img_4_corners are registered in the same order as dm_4_corners !!!
    transform_dict = DM_registration.calibrate_transform_between_DM_and_image(
        dm_4_corners, img_4_corners, debug=plot_intermediate_results, fig_path=None
    )

    # before proceeding assert that the DM coordinates in transform dict match those in zwfs_ns.grid.dm_coords
    if not np.all(
        transform_dict["actuator_coord_list_dm_space"]
        == zwfs_ns.grid.dm_coord.dm_coords
    ):
        raise UserWarning(
            "actuator_coord_list_dm_space in transform_dict does not match zwfs_ns.grid.dm_coords - this could lead to inconsistent results, make sure they are consistent "
        )

    # interpolated_intensities = DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
    zwfs_ns = update_dm_registration_in_detector_space(zwfs_ns, transform_dict)

    return zwfs_ns


def update_dm_registration_in_detector_space(zwfs_ns, transform_dict):
    """_summary_

    DM registration in wavespace is done (by construction on init) in the telescope pupil coordinates.
    Depending on detector (zwfs_ns.detector) binning this naturally dictates the detector
    pixel space coordinates (in pixels). Hence from construction we can get the DM registration in
    detector space from the wavespace simply by interpolation onto the detector pixel grid.

    However in the real system we cannot measure DM registration directly, so we need to measure it on the detector space
    This is what transform_dict has in it ( generated from DM_registration.calibrate_transform_between_DM_and_image(..)
    - so here we just standardize the relevant information to extract from here and append to zwfs namespace.

    on the detector
    Args:
        zwfs_ns (_type_): namespace containing all the information about the zwfs system. Use configuration file and init_zwfs_from_config_ini to get this
        transform_dict (_type_): generated from DM_registration.calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners, debug=plot_intermediate_results, fig_path = None )
    """

    dm_reg_dict = {
        "dm_to_pixel_transform_matrix": transform_dict["actuator_to_pixel_matrix"],
        "DM_center_pixel_space": transform_dict["DM_center_pixel_space"],
        "actuator_coord_list_pixel_space": transform_dict[
            "actuator_coord_list_pixel_space"
        ],
    }

    zwfs_ns.dm2pix_registration = SimpleNamespace(**dm_reg_dict)

    return zwfs_ns


# added 18-Jan-2026 to be more consistent with Baldr methods on sky
# use this over other reconstruction methods
def reco_method(
    zwfs_ns,
    LO=2,
    LO_inv_method="eigen",  # pinv | eigen | map
    HO_inv_method="eigen",  # pinv | eigen | zonal | map
    project_out_of_LO=None,  # we should never really project things out of LO basis, perhaps scintillation in advanced cases
    project_out_of_HO="lo_intensity",  # lo_intensity | lo_command |
    what_space="pix",  # what space does I2M operate in ( pix | dm  )
    **kwargs,
):
    """
    Construct ZWFS reconstruction operators for low-order (LO) and high-order (HO)
    channels, in either pixel measurement space ("pix") or DM-sampled measurement
    space ("dm").

    The function builds:
      - I2M_LO, I2M_HO : measurement -> modal coefficient estimators
      - M2C_LO, M2C_HO : modal coefficients -> DM command mappings
    optionally followed by projection operators that suppress LO leakage in the HO
    channel.

    The interaction matrix is taken from:
      - zwfs_ns.reco.IM : array, shape (Nmodes_total, Npix_flat)
        Each row is the measured response (flattened pixel signal) to a unit poke
        of a corresponding DM command in zwfs_ns.reco.M2C_0.

    The poke-mode DM command basis is taken from:
      - zwfs_ns.reco.M2C_0 : array, shape (Nmodes_total, Nact=140)
        Row m is the DM command used to generate IM[m].

    A DM-sampled interaction matrix is built internally by interpolating each
    IM row from pixel space onto the DM actuator sampling grid defined by
    zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space.

    --------------------------------------------------------------------------
    Parameters
    --------------------------------------------------------------------------
    zwfs_ns : namespace-like
        Must contain at least:
          - zwfs_ns.reco.IM, zwfs_ns.reco.I0, zwfs_ns.reco.M2C_0
          - zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space
        and whatever DM_registration.interpolate_pixel_intensities requires.

    LO : int, default=2
        Number of low-order modes (typically tip/tilt). Modes [0:LO) are LO and
        modes [LO:] are HO.

    LO_inv_method : {"eigen","pinv","map"}, default="eigen"
        Method used to build I2M_LO.
          - "pinv" / "eigen": uses np.linalg.pinv(IM_LO, rcond=eps)
          - "map"          : uses a Gaussian MAP estimator with (Ca_LO_*, Cn_LO_*)

    HO_inv_method : {"eigen","map","zonal"}, default="eigen"
        Method used to build I2M_HO (and an internal representation of M2C_HO).
          - "eigen": truncated SVD of IM_HO
          - "map"  : Gaussian MAP estimator in a truncated SVD subspace
                    (truncation_idx controls retained singular directions)
          - "zonal": diagonal actuator-wise inverse in *DM measurement space only*
                    (i.e. only returns a "dm" entry in reco_dict)

    project_out_of_LO : None
        Currently not implemented. Any non-None value raises.

    project_out_of_HO : {None,"lo_intensity","lo_command"}, default="lo_intensity"
        Optional suppression of LO content from the HO control path:
          - None:
              no projection
          - "lo_intensity":
              projects the HO measurement vector onto the orthogonal complement of
              the LO measurement subspace (built from IM_LO via SVD). This modifies
              I2M_HO only (measurement-side projection).
          - "lo_command":
              projects HO commands into the null-space of the LO command subspace
              spanned by M2C_0[:LO]. This modifies M2C_HO only (command-side projection).
              If filter_dm_pupil is provided, the LO command basis is weighted by it.

    --------------------------------------------------------------------------
    Keyword arguments (kwargs)
    --------------------------------------------------------------------------
    eps : float, default=1e-12
        Small diagonal regularisation used in inversions and SVD rank decisions.

    truncation_idx : int, optional
        Truncation parameter used for:
          - HO_inv_method="eigen": number of singular modes retained
          - HO_inv_method="map"  : number of singular directions retained for the
                                  reduced MAP solve (no extra tuning variable used)
        If omitted and HO_inv_method="eigen", a default is chosen in the function.

    filter_dm_pupil : array_like, shape (140,), optional
        Actuator weighting mask (0..1) representing the DM pupil footprint.
        Used for:
          - HO_inv_method="zonal": to down-weight / zero actuators outside footprint
          - project_out_of_HO="lo_command": to weight the LO command subspace before
            building its projector

    MAP covariance inputs (all optional; defaults fall back to scaled identity):
      - Cn_LO_px, Cn_LO_dm : (Nmeas,Nmeas) noise covariance for LO in pix/dm space
      - Ca_LO_px, Ca_LO_dm : (LO,LO) prior covariance on LO coefficients
      - Cn_HO_px, Cn_HO_dm : (Nmeas,Nmeas) noise covariance for HO in pix/dm space
      - Ca_HO_px, Ca_HO_dm : (Nho,Nho) prior covariance on HO coefficients
      - sigma_n : float, default=1.0
          Used only if Cn_* is not provided (Cn = sigma_n^2 I)
      - sigma_a : float, default=1.0
          Used only if Ca_* is not provided (Ca = sigma_a^2 I)

    Notes on MAP implementation:
      This implementation uses the equivalent “measurement form”:
          I2M = Ca H^T (H Ca H^T + Cn)^(-1)
      and for HO it optionally performs the solve in a truncated SVD subspace
      controlled by truncation_idx (to avoid poorly-sensed singular directions).

    --------------------------------------------------------------------------
    Returns
    --------------------------------------------------------------------------
    reco_dict : dict
        Keys are measurement spaces: "pix" and/or "dm".
        Each entry is a dict containing:
          - "I2M_LO" : ndarray, shape (LO, Nmeas)
          - "M2C_LO" : ndarray, shape (140, LO)  (from zwfs_ns.reco.M2C_0[:LO].T)
          - "I2M_HO" : ndarray, shape (Nho, Nmeas)
          - "M2C_HO" : ndarray, shape (140, Nho) or (140, k_eff) depending on method
                       (for MAP/zonal HO the mapping comes from zwfs_ns.reco.M2C_0[LO:].T)
          - "IM_LO"  : ndarray, shape (Nmeas, LO)
          - "IM_HO"  : ndarray, shape (Nmeas, Nho)
          - "LO_intermediates" / "HO_intermediates" : dicts with diagnostic matrices
            (SVD factors, covariances, projectors, effective truncation, etc.)

        Special case:
          - If HO_inv_method="zonal", only the "dm" space is returned (pixel-space
            reconstruction is skipped by design).

    --------------------------------------------------------------------------
    Assumptions / conventions
    --------------------------------------------------------------------------
    - zwfs_ns.reco.IM rows correspond 1:1 with rows of zwfs_ns.reco.M2C_0.
    - LO corresponds to the first LO poke modes in that ordering.
    - The DM has 140 actuators (BMC 3.5 layout); filter_dm_pupil is expected to
      have length 140 if provided.
    - Sign conventions are not enforced here; this function only produces linear
      operators. Closed-loop sign and gain selection are handled by the caller.
    """

    # internal helpers for if user provides nuicence vectors to project out
    def _orthonormal_basis(B, eps=1e-12):
        """
        B: (N, K) columns spanning a subspace
        returns Q: (N, r) orthonormal basis
        """
        B = np.atleast_2d(np.array(B, dtype=float))
        if B.shape[0] == 1 and B.shape[1] > 1:
            # user gave row vector; treat as single column
            B = B.T
        # SVD gives stable rank decision
        U, S, _ = np.linalg.svd(B, full_matrices=False)
        r = int(np.sum(S > eps))
        return U[:, :r], S, r

    def _projector_from_vectors(B, N, eps=1e-12):
        """
        Returns P = Q Q^T onto span(B). If B is None, returns zeros.
        """
        if B is None:
            return np.zeros((N, N)), None
        Q, S, r = _orthonormal_basis(B, eps=eps)
        P = Q @ Q.T
        info = {"S": S, "r": r, "Q": Q}
        return P, info

    # ---- kwargs handling ----
    truncation_idx = kwargs.get("truncation_idx", None)
    if truncation_idx is not None:
        truncation_idx = int(truncation_idx)

    if (truncation_idx is None) and ("eigen" in HO_inv_method.lower()):
        print("no truncation_idx for HO_inv_method, using defaul = 30\n")
        truncation_idx = 30.0

    eps = float(kwargs.get("eps", 1e-12))

    filter_dm_pupil = kwargs.get(
        "filter_dm_pupil", None
    )  # used to filter the cmd space for pupil inprint which is important if projecting modes out of command space

    if filter_dm_pupil is not None:
        if np.array(filter_dm_pupil).shape[0] != 140:
            print(filter_dm_pupil.shape, "is not BMC 3.5 size of 140! ")
    else:
        print("no user input for DM pupil filter")

    # user-provided nuisance subspaces
    B_meas = kwargs.get("project_out_HO_meas_vectors", None)  # (Nmeas, K)
    B_cmd = kwargs.get("project_out_HO_cmd_vectors", None)  # (140, K)

    LO = int(LO)  # lower order mode index definition

    if LO < 0:
        raise ValueError("LO must be >= 0")

    if LO == 0:
        if project_out_of_HO is not None:
            raise ValueError(
                "project_out_of_HO is not valid when LO=0 "
                "(no LO subspace exists to project out)"
            )

    # ---- sanity checks / expectations ----
    if not hasattr(zwfs_ns, "reco") or not hasattr(zwfs_ns.reco, "IM"):
        raise ValueError(
            "zwfs_ns.reco.IM not found. Did you run build_IM / populate zwfs_ns.reco?"
        )

    IM_px_modes_by_meas = np.array(
        zwfs_ns.reco.IM, dtype=float
    )  # expected shape (Nmodes_total, Npix_flat)
    if IM_px_modes_by_meas.ndim != 2:
        raise ValueError(
            f"zwfs_ns.reco.IM must be 2D, got shape {IM_px_modes_by_meas.shape}"
        )

    Nmodes_total, Npix_flat = IM_px_modes_by_meas.shape
    if LO < 0 or LO >= Nmodes_total:
        raise ValueError(
            f"LO must satisfy 0 <= LO < Nmodes_total. Got LO={LO}, Nmodes_total={Nmodes_total}"
        )

    # ---- build DM-sampled version of the interaction matrix ----
    IM_dm_list = []
    for ii in IM_px_modes_by_meas:  # ii shape (Npix_flat,)
        i_dm = DM_registration.interpolate_pixel_intensities(
            image=ii.reshape(zwfs_ns.reco.I0.shape),
            pixel_coords=zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space,
        )
        IM_dm_list.append(i_dm)

    IM_dm_modes_by_meas = np.array(
        IM_dm_list, dtype=float
    )  # shape (Nmodes_total, Nact_samples)
    if IM_dm_modes_by_meas.ndim != 2:
        raise ValueError(f"IM_dm must be 2D, got shape {IM_dm_modes_by_meas.shape}")

    # ---- transpose to (Nmeas, Nmodes) ----
    if LO == 0:
        IM_px_LO = None
        IM_dm_LO = None
        IM_px_HO = IM_px_modes_by_meas.T
        IM_dm_HO = IM_dm_modes_by_meas.T
    else:
        IM_px_LO = IM_px_modes_by_meas[:LO, :].T  # (Npix, LO)
        IM_px_HO = IM_px_modes_by_meas[LO:, :].T  # (Npix, Nmodes_total-LO)

        IM_dm_LO = IM_dm_modes_by_meas[:LO, :].T  # (Nact_samples, LO)
        IM_dm_HO = IM_dm_modes_by_meas[LO:, :].T  # (Nact_samples, Nmodes_total-LO)

    # LO in DM command basis
    # B_LO = zwfs_ns.reco.M2C_0[:LO, :].T   # (140, LO)

    reco_dict = {}
    inv_HO_dict = {}
    inv_LO_dict = {}
    if "zonal" in HO_inv_method.lower():
        print(
            "---------------\n\n Since you selected 'zonal' for HO inversion method, we can only do this on the DM interpolated space, so do not consider pixel space!\n\n"
        )
        space_labels = ["dm"]
        interaction_matricies = [(IM_dm_LO, IM_dm_HO)]
    else:
        space_labels = ["pix", "dm"]
        interaction_matricies = [(IM_px_LO, IM_px_HO), (IM_dm_LO, IM_dm_HO)]
    for space, (IM_LO, IM_HO) in zip(space_labels, interaction_matricies):
        # -------------------------
        # HO INVERSION (raw)
        # -------------------------
        if "eigen" in HO_inv_method.lower():
            k = truncation_idx
            U, S, Vt = np.linalg.svd(IM_HO, full_matrices=False)

            k_eff = min(k, S.shape[0])  # guard against k > rank dims
            U_k = U[:, :k_eff]
            S_k = S[:k_eff]
            Vt_k = Vt[:k_eff, :]
            V_k = Vt_k.T

            I2M_HO_raw = U_k.T
            M2C_HO_raw = V_k @ np.diag(1.0 / (S_k + eps))

            inv_HO_dict = {
                "method": "eigen",
                "truncation_idx": k,
                "truncation_idx_effective": k_eff,
                "U": U,
                "S": S,
                "Vt": Vt,
                "U_k": U_k,
                "S_k": S_k,
                "Vt_k": Vt_k,
                "I2M_HO_raw": I2M_HO_raw,
                "M2C_HO_raw": M2C_HO_raw,
            }

        elif "map" in HO_inv_method.lower():

            # get raw object first (could be None)
            if space == "pix":
                Ca = kwargs.get("Ca_HO_px", None)
                Cn = kwargs.get("Cn_HO_px", None)
            elif space == "dm":
                Ca = kwargs.get("Ca_HO_dm", None)  # (Nho, Nho)
                Cn = kwargs.get("Cn_HO_dm", None)  # (Nmeas, Nmeas)
            else:
                raise ValueError(space, " not a valid space")

            # ---- shapes ----
            H = IM_HO  # (Nmeas, Nho)
            n_meas, n_modes = H.shape

            # ---- defaults ----
            if truncation_idx is None:
                print("no truncation_idx for HO MAP inversion, using default = 54\n")
                truncation_idx = 54
            k = int(min(truncation_idx, min(n_meas, n_modes)))

            # ---- Measurement noise covariance defaults ----
            if Cn is None:
                print(
                    "no input noise covariance - using Identity with sigma_n input, if not sigma_n input kwargs then sigma_n=1"
                )
                sigma_n = kwargs.get("sigma_n", 1.0)
                Cn = (float(sigma_n) ** 2) * np.eye(n_meas)
            else:
                Cn = np.array(Cn, dtype=float)
                if Cn.shape != (n_meas, n_meas):
                    raise ValueError(f"Cn must be ({n_meas},{n_meas})")

            # ---- Prior covariance defaults ----
            if Ca is None:
                print(
                    "no input phase covariance prior - using Identity with sigma_a input, if no sigma_a input kwargs then sigma_a=1"
                )
                sigma_a = kwargs.get("sigma_a", 1.0)
                Ca = (float(sigma_a) ** 2) * np.eye(n_modes)
            else:
                Ca = np.array(Ca, dtype=float)
                if Ca.shape != (n_modes, n_modes):
                    raise ValueError(f"Ca must be ({n_modes},{n_modes})")

            # ---------------------------------------------------------
            # SVD truncation using truncation_idx (NO extra variables)
            # ---------------------------------------------------------
            U, S, Vt = np.linalg.svd(H, full_matrices=False)
            U_k = U[:, :k]  # (Nmeas, k)
            S_k = S[:k]  # (k,)
            V_k = Vt[:k, :].T  # (Nho, k)

            # H_eff = U_k diag(S_k)
            H_eff = U_k @ np.diag(S_k)  # (Nmeas, k)

            # project Ca into truncated coordinates
            Ca_eff = V_k.T @ Ca @ V_k  # (k, k)

            # MAP in truncated subspace, then lift back
            A = H_eff @ Ca_eff @ H_eff.T + Cn  # (Nmeas, Nmeas)
            Ainv = np.linalg.solve(A + eps * np.eye(n_meas), np.eye(n_meas))
            I2M_red = Ca_eff @ H_eff.T @ Ainv  # (k, Nmeas)
            I2M_HO_raw = V_k @ I2M_red  # (Nho, Nmeas)

            # HO command basis in actuator space (140 x Nho)
            M2C_0 = np.array(zwfs_ns.reco.M2C_0, dtype=float)  # (Nmodes_total, 140)
            M2C_HO_raw = M2C_0[LO:, :].T  # (140, Nho)

            inv_HO_dict = {
                "method": "map",
                "truncation_idx": truncation_idx,
                "truncation_idx_effective": k,
                "Cn": Cn,
                "Ca": Ca,
                "S": S,
                "S_k": S_k,
                "I2M_HO_raw": I2M_HO_raw,
                "M2C_HO_raw": M2C_HO_raw,
            }

        elif "zonal" in HO_inv_method.lower():

            if filter_dm_pupil is None:
                # then we estimate it from the input HO
                # filter for pupil footprint on DM
                # look at std over IM response , normalized (0-1)
                dm_im_std = np.std(zwfs_ns.reco.IM[LO:, :].T, axis=0)
                filter_dm_pupil = (dm_im_std - np.min(dm_im_std)) / (
                    np.max(dm_im_std) - np.min(dm_im_std)
                )
                filter_dm_pupil[filter_dm_pupil < 0.2] = (
                    0  # normalize 0-1, anything below 0.2 force to 0
                )

            # diagonal "inverse" with pupil weighting
            I2M_HO_raw = np.diag(
                np.array(
                    [
                        (
                            filter_dm_pupil[i] / IM_HO[i][i]
                            if np.isfinite(1.0 / IM_HO[i][i])
                            else 0.0
                        )
                        for i in range(IM_HO.shape[0])
                    ],
                    dtype=float,
                )
            )

            M2C_0 = np.array(zwfs_ns.reco.M2C_0, dtype=float)  # (Nmodes_total, 140)
            M2C_HO_raw = M2C_0[LO:, :].T

            inv_HO_dict = {
                "method": "zonal",
                "filter_dm_pupil": filter_dm_pupil,
                "I2M_HO_raw": I2M_HO_raw,
                "M2C_HO_raw": M2C_HO_raw,
            }
        else:
            raise UserWarning("no valid HO_inv_method implemented (expected 'eigen')")

        # -------------------------
        # LO INVERSION (raw)
        # -------------------------

        I2M_LO_raw = None
        M2C_LO_raw = None
        if LO == 0:  # we only have one reconstructor
            I2M_LO = None
            M2C_LO = None
            inv_LO_dict = {"method": None, "note": "LO disabled (LO=0)"}
        else:  # we do LO matricies
            assert IM_LO is not None
            if ("eigen" in LO_inv_method.lower()) or ("pinv" in LO_inv_method.lower()):
                # physical LO amplitudes from measurement: e_LO = pinv(IM_LO) @ s
                # IM_LO is (Nmeas, LO) -> pinv is (LO, Nmeas)
                I2M_LO_raw = np.linalg.pinv(IM_LO, rcond=eps)

                inv_LO_dict = {
                    "method": "pinv(IM_LO)",
                    "I2M_LO_raw": I2M_LO_raw,
                    "eps": eps,
                }

            elif "map" in LO_inv_method.lower():
                if space == "pix":
                    Ca = kwargs.get("Ca_LO_px", None)
                    Cn = kwargs.get("Cn_LO_px", None)
                elif space == "dm":
                    Ca = kwargs.get("Ca_LO_dm", None)  # (Nho, Nho)
                    Cn = kwargs.get("Cn_LO_dm", None)  # (Nmeas, Nmeas)
                else:
                    raise ValueError(space, " not a valid space")

                # ---- Measurement noise covariance defaults----
                n_meas, n_modes = IM_LO.shape
                if Cn is None:
                    print(
                        "no input noise covariance - using Identity with sigma_n input, if not signa_n input kwargs then signa_n=1"
                    )
                    sigma_n = kwargs.get("sigma_n", 1.0)
                    Cn = (sigma_n**2) * np.eye(n_meas)
                else:
                    Cn = np.array(Cn, dtype=float)
                    if Cn.shape != (n_meas, n_meas):
                        raise ValueError(f"Cn must be ({n_meas},{n_meas})")

                # ---- Prior covariance defaults ----
                if Ca is None:
                    print(
                        "no input phase covariance prior - using Identity with sigma_a input, if no signa_a input kwargs then signa_a=1"
                    )
                    sigma_a = kwargs.get("sigma_a", 1.0)
                    Ca = (sigma_a**2) * np.eye(n_modes)
                else:
                    Ca = np.array(Ca, dtype=float)
                    if Ca.shape != (n_modes, n_modes):
                        raise ValueError(f"Ca must be ({n_modes},{n_modes})")
                assert type(Ca) is np.ndarray
                # H = IM_LO is (Nmeas, LO)
                H = IM_LO

                # I2M is (LO, Nmeas)
                A = H @ Ca @ H.T + Cn
                Ainv = np.linalg.solve(A + eps * np.eye(A.shape[0]), np.eye(A.shape[0]))
                I2M_LO_raw = Ca @ H.T @ Ainv

                # LO command basis in actuator space (140 x LO)
                M2C_0 = np.array(zwfs_ns.reco.M2C_0, dtype=float)  # (Nmodes_total, 140)
                M2C_LO_raw = M2C_0[:LO, :].T  # (140, LO)

                inv_LO_dict = {
                    "method": "map",
                    "Cn": Cn,
                    "Ca": Ca,
                    "I2M_LO_raw": I2M_LO_raw,
                    "M2C_LO_raw": M2C_LO_raw,
                }

            else:
                raise UserWarning(
                    "no valid LO_inv_method implemented (expected 'eigen')"
                )

        # -------------------------
        # PROJECTIONS
        # -------------------------
        if LO == 0:
            I2M_HO = I2M_HO_raw
            M2C_HO = M2C_HO_raw
            inv_HO_dict["proj"] = None
        else:
            assert IM_LO is not None
            if project_out_of_LO is not None:
                raise UserWarning("project_out_of_LO not implemented yet")

            if project_out_of_HO is not None:
                proj_key = project_out_of_HO.lower().strip()

                if "lo_intensity" in proj_key:
                    U_lo, S_lo, Vt_lo = np.linalg.svd(IM_LO, full_matrices=False)
                    r_lo = int(np.sum(S_lo > eps))
                    U_lo_r = U_lo[:, :r_lo]
                    P_LO_intensity = U_lo_r @ U_lo_r.T

                    I_eye = np.eye(IM_HO.shape[0])
                    I2M_HO = I2M_HO_raw @ (I_eye - P_LO_intensity)
                    M2C_HO = M2C_HO_raw

                    inv_HO_dict["proj"] = "lo_intensity"
                    inv_HO_dict["r_lo_intensity"] = r_lo
                    inv_HO_dict["P_LO_intensity"] = P_LO_intensity

                elif "lo_command" in proj_key:
                    # Build LO projector in DM actuator command space (140x140) using the known poke-mode commands
                    if not hasattr(zwfs_ns.reco, "M2C_0"):
                        raise ValueError(
                            "zwfs_ns.reco.M2C_0 not found, cannot build LO command subspace projector."
                        )

                    M2C_0 = np.array(
                        zwfs_ns.reco.M2C_0, dtype=float
                    )  # expected (Nmodes_total, 140)
                    if M2C_0.ndim != 2 or M2C_0.shape[1] != 140:
                        raise ValueError(
                            f"Expected zwfs_ns.reco.M2C_0 shape (Nmodes_total, 140). Got {M2C_0.shape}"
                        )

                    if M2C_0.shape[0] < LO:
                        raise ValueError(
                            f"M2C_0 has only {M2C_0.shape[0]} modes, but LO={LO}"
                        )

                    # CRITICAL IS TO FILTER THE PUPIL PROJECTED ON THE DM FOR THIS
                    # # look at std over IM response , normalized (0-1)
                    # dm_im_std = np.std( IM_HO ,axis=0)
                    # pup_filt_dm = (dm_im_std - np.min(dm_im_std)) / (np.max( dm_im_std ) - np.min( dm_im_std ))
                    # pup_filt_dm[pup_filt_dm < 0.2] = 0 # normalize 0-1, anything below 0.2 force to 0
                    # B_LO = pup_filt_dm[:,np.newaxis] * M2C_0[:LO, :].T

                    if filter_dm_pupil is not None:
                        B_LO = filter_dm_pupil[:, np.newaxis] * M2C_0[:LO, :].T
                    else:
                        B_LO = M2C_0[:LO, :].T

                    # Orthonormalize and build projector
                    Uc, Sc, Vtc = np.linalg.svd(B_LO, full_matrices=False)
                    rc = int(np.sum(Sc > eps))
                    Q = Uc[:, :rc]  # (140, rc)

                    P_LO_command = Q @ Q.T
                    P_LO_command_null = np.eye(P_LO_command.shape[0]) - P_LO_command

                    I2M_HO = I2M_HO_raw

                    # Apply projector to HO command mapping
                    # In your observed case M2C_HO_raw is (140, k_eff)
                    if M2C_HO_raw.shape[0] != P_LO_command_null.shape[0]:
                        raise ValueError(
                            f"Expected M2C_HO_raw axis0 to match actuator dimension {P_LO_command_null.shape[0]}. "
                            f"Got M2C_HO_raw shape={M2C_HO_raw.shape}"
                        )

                    M2C_HO = P_LO_command_null @ M2C_HO_raw

                    inv_HO_dict["proj"] = "lo_command"
                    inv_HO_dict["rc_command"] = rc
                    inv_HO_dict["Sc_command"] = Sc
                    inv_HO_dict["P_LO_command"] = P_LO_command
                    inv_HO_dict["P_LO_command_null"] = P_LO_command_null

                else:
                    raise UserWarning(
                        "no valid project_out_of_HO option (use 'lo_intensity', 'lo_command', or None)"
                    )
            else:
                I2M_HO = I2M_HO_raw
                M2C_HO = M2C_HO_raw
                inv_HO_dict["proj"] = None

        if LO == 0:
            I2M_LO = None
            M2C_LO = None
            inv_LO_dict = {"method": None, "note": "LO disabled (LO=0)"}
        else:
            assert type(I2M_LO_raw) is np.ndarray
            # define LO outputs consistently (no projection here yet)
            I2M_LO = I2M_LO_raw
            # M2C_LO = M2C_LO_raw

            # LO command basis (actuator-space)
            M2C_0 = np.array(zwfs_ns.reco.M2C_0, dtype=float)  # (Nmodes_total, 140)
            M2C_LO = M2C_0[:LO, :].T  # (140, LO)

        ## FINALLY we project any user defined nuisance vectors out of the spaces!
        # measurement-side nuisance vectors
        if B_meas is not None:
            Bm = np.atleast_2d(np.array(B_meas, dtype=float))
            if Bm.shape[0] == 1 and Bm.shape[1] == IM_HO.shape[0]:
                Bm = Bm.T  # treat as single column vector

            if Bm.shape[0] != IM_HO.shape[0]:
                print(
                    f"[reco_method][{space}] "
                    f"Skipping measurement-side HO projection: "
                    f"B_meas has length {Bm.shape[0]}, expected {IM_HO.shape[0]}"
                )
            else:
                P_bad_meas, info_meas = _projector_from_vectors(
                    Bm, N=IM_HO.shape[0], eps=eps
                )
                I2M_HO = I2M_HO @ (np.eye(IM_HO.shape[0]) - P_bad_meas)

                inv_HO_dict["proj_user_meas"] = {"space": space, **(info_meas or {})}
                inv_HO_dict["P_bad_meas"] = P_bad_meas

        # command-side nuisance vectors
        if B_cmd is not None:
            Bc = np.atleast_2d(np.array(B_cmd, dtype=float))
            if Bc.shape[0] == 1 and Bc.shape[1] == IM_HO.shape[0]:
                Bc = Bc.T  # treat as single column vector
            if Bc.shape[0] != M2C_HO.shape[0]:
                print(
                    f"[reco_method][{space}] "
                    f"Skipping command-side HO projection: "
                    f"B_cmd has length {Bc.shape[0]}, expected {M2C_HO.shape[0]}"
                )
            else:
                P_bad_cmd, info_cmd = _projector_from_vectors(
                    Bc, N=M2C_HO.shape[0], eps=eps
                )
                M2C_HO = (np.eye(M2C_HO.shape[0]) - P_bad_cmd) @ M2C_HO

                inv_HO_dict["proj_user_cmd"] = info_cmd
                inv_HO_dict["P_bad_cmd"] = P_bad_cmd

        reco_dict[space] = {
            "space": space,
            "LO": LO,
            "LO_inv_method": LO_inv_method,
            "HO_inv_method": HO_inv_method,
            "project_out_of_HO": project_out_of_HO,
            "truncation_idx": truncation_idx,
            "eps": eps,
            "HO_intermediates": inv_HO_dict,
            "LO_intermediates": inv_LO_dict,
            "I2M_HO": I2M_HO,
            "M2C_HO": M2C_HO,
            "I2M_LO": I2M_LO,
            "M2C_LO": M2C_LO,
            "IM_LO": IM_LO,
            "IM_HO": IM_HO,
        }

    # APPEND ALL THIS TO zwfs_ns (as the previous one did)
    dict2append = {  # legacy names
        "I2M_TT": reco_dict[what_space]["I2M_LO"],
        "I2M_HO": reco_dict[what_space]["I2M_HO"],
        "M2C_LO": reco_dict[what_space]["M2C_LO"],
        "M2C_HO": reco_dict[what_space]["M2C_HO"],
        "info": {
            # auxillary info
            "space": what_space,
            "LO": LO,
            "LO_inv_method": LO_inv_method,
            "HO_inv_method": HO_inv_method,
            "project_out_of_HO": project_out_of_HO,
            "truncation_idx": truncation_idx,
            "eps": eps,
            "HO_intermediates": inv_HO_dict,
            "LO_intermediates": inv_LO_dict,
        },
    }
    reco_dict_current = vars(zwfs_ns.reco)
    updated_reco_dict = reco_dict_current | dict2append
    zwfs_ns.reco = SimpleNamespace(
        **updated_reco_dict
    )  # add it to the current reco namespace with

    return reco_dict


def plot_eigenmodes(
    zwfs_ns, save_path=None, descr_label=None, crop_image_modes=[None, None, None, None]
):
    """_summary_

    Args:
        zwfs_ns (namespace ): _description_ namespace with interaction matrix
        save_path (string, optional): _description_. Defaults to None. path to save images
        descr_label (string, optional): _description_. Defaults to None. descriptive label to add to default plot names when saving
        crop_image_modes (list, optional): _description_. Defaults to [None, None, None, None]. crop the image modes to show only a portion of the image (i.e. pupil)
    """
    tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

    U, S, Vt = np.linalg.svd(zwfs_ns.reco.IM, full_matrices=True)

    # singular values
    plt.figure(1)
    plt.semilogy(S)  # /np.max(S))
    # plt.axvline( np.pi * (10/2)**2, color='k', ls=':', label='number of actuators in pupil')
    plt.legend()
    plt.xlabel("mode index")
    plt.ylabel("singular values")

    if save_path is not None:
        plt.savefig(
            save_path + f"singularvalues_{descr_label}_{tstamp}.png",
            bbox_inches="tight",
            dpi=200,
        )
    # plt.show()

    # THE IMAGE MODES
    n_row = round(np.sqrt(zwfs_ns.reco.M2C_0.shape[0])) - 1
    fig, ax = plt.subplots(n_row, n_row, figsize=(30, 30))

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    for i, axx in enumerate(np.array(ax).reshape(-1)):
        assert type(axx) is Axes
        # we filtered circle on grid, so need to put back in grid
        tmp = zwfs_ns.pupil_regions.pupil_filt.copy()
        vtgrid = np.zeros(tmp.shape)
        vtgrid[tmp] = Vt[i]
        r1, r2, c1, c2 = crop_image_modes  # 10,-10,10,-10
        axx.imshow(
            vtgrid.reshape(zwfs_ns.reco.I0.shape)[r1:r2, c1:c2]
        )  # cp_x2-cp_x1,cp_y2-cp_y1) )
        # axx.set_title(f'\n\n\nmode {i}, S={round(S[i]/np.max(S),3)}',fontsize=5)
        #
        axx.text(10, 10, f"{i}", color="w", fontsize=4)
        axx.text(10, 20, f"S={round(S[i]/np.max(S),3)}", color="w", fontsize=4)
        axx.axis("off")
        # plt.legend(ax=axx)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path + f"det_eignmodes_{descr_label}_{tstamp}.png",
            bbox_inches="tight",
            dpi=100,
        )
    # plt.show()

    # THE DM MODES

    # NOTE: if not zonal (modal) i might need M2C to get this to dm space
    # if zonal M2C is just identity matrix.
    fig, ax = plt.subplots(n_row, n_row, figsize=(30, 30))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    for i, axx in enumerate(np.array(ax).reshape(-1)):
        axx.imshow(util.get_DM_command_in_2D(zwfs_ns.reco.M2C_0.T @ U.T[i]))
        # axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
        axx.text(1, 2, f"{i}", color="w", fontsize=6)
        axx.text(1, 3, f"S={round(S[i]/np.max(S),3)}", color="w", fontsize=6)
        axx.axis("off")
        # plt.legend(ax=axx)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(
            save_path + f"dm_eignmodes_{descr_label}_{tstamp}.png",
            bbox_inches="tight",
            dpi=100,
        )
    # plt.show()


# Should use newer reconstruction matrix builder def reco_method()
def construct_ctrl_matricies_from_IM(
    zwfs_ns, method="Eigen_TT-HO", Smax=50, TT_vectors=DM_basis.get_tip_tilt_vectors()
):

    print("! OUTDATE - you should use new reconstruction method 'reco_method()'")

    M2C_0 = zwfs_ns.reco.M2C_0
    poke_amp = zwfs_ns.reco.poke_amp

    if method == "Eigen_TT-HO":
        U, S, Vt = np.linalg.svd(zwfs_ns.reco.IM, full_matrices=False)

        R = (Vt.T * [1 / ss if i < Smax else 0 for i, ss in enumerate(S)]) @ U.T

        # TT_vectors = DM_basis.get_tip_tilt_vectors()

        TT_space = M2C_0 @ TT_vectors

        U_TT, S_TT, Vt_TT = np.linalg.svd(TT_space, full_matrices=False)

        I2M_TT = U_TT.T @ R.T

        M2C_TT = (
            poke_amp * M2C_0.T @ U_TT
        )  # since pinned need M2C to go back to 140 dimension vector

        R_HO = (np.eye(U_TT.shape[0]) - U_TT @ U_TT.T) @ R.T

        # go to Eigenmodes for modal control in higher order reconstructor
        U_HO, S_HO, Vt_HO = np.linalg.svd(R_HO, full_matrices=False)
        I2M_HO = Vt_HO
        M2C_HO = (
            poke_amp * M2C_0.T @ (U_HO * S_HO)
        )  # since pinned need M2C to go back to 140 dimension vector
        # plt.plot( M2C_HO @ I2M_HO @ IM[63] ) ; plt.show()
        reco_dict_current = vars(zwfs_ns.reco)
        reco_dict_2append = {
            "U": U,
            "S": S,
            "Vt": Vt,
            "Smax": Smax,
            "R": R,
            "U_TT": U_TT,
            "S_TT": S_TT,
            "Vt_TT": Vt_TT,
            "U_HO": U_HO,
            "S_HO": S_HO,
            "Vt_HO": Vt_HO,
            "I2M_TT": I2M_TT,
            "M2C_TT": M2C_TT,
            "I2M_HO": I2M_HO,
            "M2C_HO": M2C_HO,
        }

        reco_dict = reco_dict_current | reco_dict_2append
        zwfs_ns.reco = SimpleNamespace(
            **reco_dict
        )  # add it to the current reco namespace with

    elif method == "Eigen_HO":  # just look at eigen basis in HO - no projection onto TT
        U, S, Vt = np.linalg.svd(zwfs_ns.reco.IM.T, full_matrices=False)

        R = (Vt.T * [1 / ss if i < Smax else 0 for i, ss in enumerate(S)]) @ U.T

        #
        TT_space = M2C_0 @ TT_vectors
        U_TT, S_TT, Vt_TT = np.linalg.svd(TT_space, full_matrices=False)

        I2M_TT = np.zeros(
            [2, R.shape[1]]
        )  # set to zero - we only consider HO in eigenmodes
        M2C_TT = TT_vectors

        U_HO, S_HO, Vt_HO = U.copy(), S.copy(), Vt.copy()
        I2M_HO = U.T  # R.T
        M2C_HO = (
            poke_amp
            * M2C_0.T
            @ (Vt.T * [1 / ss if i < Smax else 0 for i, ss in enumerate(S)])
        )
        # plt.plot( M2C_HO @ I2M_HO @ IM[63] ) ; plt.show()
    else:
        raise TypeError("construct_ctrl_matricies_from_IM method name NOT FOUND!!!!")

    reco_dict_current = vars(zwfs_ns.reco)
    reco_dict_2append = {
        "U": U,
        "S": S,
        "Vt": Vt,
        "Smax": Smax,
        "R": R,
        "U_TT": U_TT,
        "S_TT": S_TT,
        "Vt_TT": Vt_TT,
        "U_HO": U_HO,
        "S_HO": S_HO,
        "Vt_HO": Vt_HO,
        "I2M_TT": I2M_TT,
        "M2C_TT": M2C_TT,
        "I2M_HO": I2M_HO,
        "M2C_HO": M2C_HO,
    }
    reco_dict = reco_dict_current | reco_dict_2append
    zwfs_ns.reco = SimpleNamespace(
        **reco_dict
    )  # add it to the current reco namespace with

    return zwfs_ns


def add_controllers_for_zonal_interp_no_projection(
    zwfs_ns, HO="leaky", return_controller=False
):

    N = np.sum(zwfs_ns.reco.linear_zonal_model.act_filt_recommended)
    if HO == "leaky":
        ki_leak = 0 * np.ones(N)
        kp_leak = 0 * np.ones(N)
        lower_limit_leak = -100 * np.ones(N)
        upper_limit_leak = 100 * np.ones(N)

        HO_ctrl = LeakyIntegrator(
            ki=ki_leak,
            kp=kp_leak,
            lower_limit=lower_limit_leak,
            upper_limit=upper_limit_leak,
        )

    elif HO == "PID":
        kp = 0.0 * np.ones(N)
        ki = 0.0 * np.ones(N)
        kd = 0.0 * np.ones(N)
        setpoint = np.zeros(N)
        lower_limit_pid = -100 * np.ones(N)
        upper_limit_pid = 100 * np.ones(N)

        HO_ctrl = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)
    else:
        raise ValueError(f"HO control type unsupported: {HO}")

    controller_dict = {"HO_ctrl": HO_ctrl}

    if not return_controller:  # then we append to the zwfs_ns and return it

        control_ns = SimpleNamespace(**controller_dict)

        telemetry_dict = init_telem_dict()

        tele_ns = SimpleNamespace(**telemetry_dict)

        zwfs_ns.ctrl = control_ns
        zwfs_ns.telem = tele_ns

        zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat

        return zwfs_ns

    else:
        return controller_dict


def add_controllers_for_MVM_TT_HO(
    zwfs_ns, TT="PID", HO="leaky", return_controller=False
):

    if HO == "leaky":
        ki_leak = 0 * np.ones(zwfs_ns.reco.I2M_HO.shape[0])
        kp_leak = 0 * np.ones(zwfs_ns.reco.I2M_HO.shape[0])
        lower_limit_leak = -100 * np.ones(zwfs_ns.reco.I2M_HO.shape[0])
        upper_limit_leak = 100 * np.ones(zwfs_ns.reco.I2M_HO.shape[0])

        HO_ctrl = LeakyIntegrator(
            ki=ki_leak,
            kp=kp_leak,
            lower_limit=lower_limit_leak,
            upper_limit=upper_limit_leak,
        )

    elif HO == "PID":
        kp = 0 * np.ones(zwfs_ns.reco.I2M_HO.shape[0])
        ki = 0 * np.ones(zwfs_ns.reco.I2M_HO.shape[0])
        kd = 0.0 * np.ones(zwfs_ns.reco.I2M_HO.shape[0])
        setpoint = np.zeros(zwfs_ns.reco.I2M_HO.shape[0])
        lower_limit_pid = -100 * np.ones(zwfs_ns.reco.I2M_HO.shape[0])
        upper_limit_pid = 100 * np.ones(zwfs_ns.reco.I2M_HO.shape[0])

        HO_ctrl = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)
    else:
        raise ValueError(f"HO control type unsupported: {HO}")

    if TT == "leaky":
        ki = 0 * np.ones(zwfs_ns.reco.I2M_TT.shape[0])
        kp_leak = 0 * np.ones(zwfs_ns.reco.I2M_TT.shape[0])
        lower_limit_leak = -100 * np.ones(zwfs_ns.reco.I2M_TT.shape[0])
        upper_limit_leak = 100 * np.ones(zwfs_ns.reco.I2M_TT.shape[0])

        TT_ctrl = LeakyIntegrator(
            ki=ki,
            kp=kp_leak,
            lower_limit=lower_limit_leak,
            upper_limit=upper_limit_leak,
        )

    elif TT == "PID":
        kp = 0 * np.ones(zwfs_ns.reco.I2M_TT.shape[0])
        ki = 0 * np.ones(zwfs_ns.reco.I2M_TT.shape[0])
        kd = 0.0 * np.ones(zwfs_ns.reco.I2M_TT.shape[0])
        setpoint = np.zeros(zwfs_ns.reco.I2M_TT.shape[0])
        lower_limit_pid = -100 * np.ones(zwfs_ns.reco.I2M_TT.shape[0])
        upper_limit_pid = 100 * np.ones(zwfs_ns.reco.I2M_TT.shape[0])

        TT_ctrl = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)

    else:
        raise ValueError(f"TT control type unsupported: {TT}")
    controller_dict = {"TT_ctrl": TT_ctrl, "HO_ctrl": HO_ctrl}

    if not return_controller:  # then we append to the zwfs_ns and return it

        control_ns = SimpleNamespace(**controller_dict)

        telemetry_dict = init_telem_dict()

        tele_ns = SimpleNamespace(**telemetry_dict)

        zwfs_ns.ctrl = control_ns
        zwfs_ns.telem = tele_ns

        zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat

        return zwfs_ns

    else:
        return controller_dict


def AO_iteration(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    dm_disturbance=np.zeros(140),
    record_telemetry=True,
    method="MVM-TT-HO",
    detector=None,
    obs_intermediate_field=True,
    use_pyZelda=True,
    include_shotnoise=True,
    **kwargs,
):
    # THis is old and i think outdated ...

    # got rid of I0 and should get rid of detector (since it is in zwfs_ns
    # single iteration of AO in closed loop

    # propagates opd over DM and get intensity

    if (
        obs_intermediate_field
    ):  # we onbserve the actual field (only valid in simulation to test results)
        # opd in wavespace
        opd_current_dm = zwfs_ns.compiled_dm.eval(
            zwfs_ns.dm.current_cmd * zwfs_ns.dm.opd_per_cmd
        )

        phi = (
            zwfs_ns.grid.pupil_mask
            * 2
            * np.pi
            / zwfs_ns.optics.wvl0
            * (opd_input + opd_internal + opd_current_dm)
        )

        #
        i = get_frame(
            opd_input=opd_input,
            amp_input=amp_input,
            opd_internal=opd_internal,
            zwfs_ns=zwfs_ns,
            detector=detector,
            use_pyZelda=use_pyZelda,
            include_shotnoise=include_shotnoise,
        )

        # i = get_frame(  opd_input  = opd_input + opd_current_dm ,   amp_input = amp_input,\
        #         opd_internal = opd_internal,  zwfs_ns= zwfs_ns , detector= detector, use_pyZelda = use_pyZelda , include_shotnoise=include_shotnoise)

        # if use_pyZelda :
        #     i = get_frame(  opd_input  = opd_input + opd_current_dm ,   amp_input = amp_input,\
        #         opd_internal = opd_internal,  zwfs_ns= zwfs_ns , detector= detector, use_pyZelda =True)
        # else:
        #     # i = get_pupil_intensity( phi= phi, amp=amp_input, theta = zwfs_ns.optics.theta , phasemask_diameter = zwfs_ns.optics.mask_diam, \
        #     # phasemask_mask = zwfs_ns.grid.phasemask_mask, pupil_diameter = zwfs_ns.grid.N, fplane_pixels=zwfs_ns.focal_plane.fplane_pixels, \
        #     #     pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask )

        # if detector is not None:
        #    i = average_subarrays(array=i, block_size = detector)

        strehl = np.exp(-np.var(phi[zwfs_ns.grid.pupil_mask > 0]))

    else:  # we just see intensity
        i = get_frame(
            opd_input=opd_input,
            amp_input=amp_input,
            opd_internal=opd_internal,
            zwfs_ns=zwfs_ns,
            detector=detector,
            use_pyZelda=use_pyZelda,
        )
        phi = None
        strehl = None

    # kwargs <--- contains controllers , what is required in this dictionary depends on the method used
    delta_cmd = process_zwfs_intensity(
        i, zwfs_ns, method=method, record_telemetry=record_telemetry, **kwargs
    )

    # put this in function specific for how to process the signal and apply controller
    # sig = process_zwfs_signal( i, I0, zwfs_ns.pupil_regions.pupil_filt ) # I0_theory/ np.mean(I0_theory) #

    # e_TT = zwfs_ns.reco.I2M_TT @ sig

    # u_TT = zwfs_ns.ctrl.TT_ctrl.process( e_TT )

    # c_TT = zwfs_ns.reco.M2C_TT @ u_TT

    # e_HO = zwfs_ns.reco.I2M_HO @ sig

    # u_HO = zwfs_ns.ctrl.HO_ctrl.process( e_HO )

    # c_HO = zwfs_ns.reco.M2C_HO @ u_HO

    # # safety
    # if np.max( c_TT + c_HO ) > 0.8:
    #     print( ' going badly.. ')

    # delta_cmd  = c_TT + c_HO

    # SEND DM COMMAND
    zwfs_ns.dm.current_cmd = (
        zwfs_ns.dm.dm_flat + dm_disturbance - delta_cmd
    )  # c_HO - c_TT
    # zwfs_ns.dm.current_cmd += dm_disturbance - delta_cmd

    # only measure residual in the registered pupil on DM
    residual = dm_disturbance - delta_cmd  # c_HO - c_TT)
    rmse = np.nanstd(residual)

    # telemetry
    if record_telemetry:
        ## these other ones get don in process_zwfs_intensity
        # zwfs_ns.telem.i_list.append( i )
        # zwfs_ns.telem.s_list.append( sig )
        # zwfs_ns.telem.e_TT_list.append( e_TT )
        # zwfs_ns.telem.u_TT_list.append( u_TT )
        # zwfs_ns.telem.c_TT_list.append( c_TT )

        # zwfs_ns.telem.e_HO_list.append( e_HO )
        # zwfs_ns.telem.u_HO_list.append( u_HO )
        # zwfs_ns.telem.c_HO_list.append( c_HO )

        # atm_disturb_list.append( scrn.scrn )
        zwfs_ns.telem.dm_disturb_list.append(dm_disturbance)

        zwfs_ns.telem.residual_list.append(residual)
        zwfs_ns.telem.rmse_list.append(rmse)

        if obs_intermediate_field:
            zwfs_ns.telem.field_phase.append(phi)
            zwfs_ns.telem.strehl.append(strehl)

    return i


class my_lin_fit:
    act_filt_recommended: NDArray
    N0_dm: NDArray
    I0_dm: NDArray
    pearson_R_dm: NDArray

    # Rows are samples, columns are features
    def __init__(self, model_name="pixelwise_first"):
        """
        Initialize the linear fit model.

        Parameters:
        - model_name: str, the name/type of model (currently supports 'pixelwise_first')
        """
        self.model_name = model_name
        self.models = None

    def process_signal(self, i_dm, N0_dm, act_filt):
        """_summary_

        Args:
            i_dm (_type_): ZWFS intensity interpolated onto DM actuator space
            N0_dm (_type_): clear pupil (no phasemask) intensity interpolated onto DM actuator space
            act_filt (_type_): filter for active pupil on DM (in DM space).

        Returns:
            _type_: signal that should be used for fitting the model
        """
        # i_dm = DM_registration.interpolate_pixel_intensities(image = image, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space)

        sig = i_dm / np.mean(N0_dm[act_filt])

        return sig

    def fit(self, X, Y):
        """
        Fit the model based on the input features X and target Y.

        Parameters:
        - X: NDArray, shape (N, P), input data matrix (N samples, P features)
        - Y: NDArray, shape (N, P), target data matrix (same shape as X)

        Returns:
        - coe: list of model coefficients for each feature
        """
        if self.model_name == "pixelwise_first":
            coe = []
            # Fit a first-order polynomial (linear) for each feature (each column)
            for v in range(X.shape[1]):
                coe.append(
                    np.polyfit(X[:, v], Y[:, v], 1)
                )  # Linear fit for each feature
            self.models = coe
            return coe

    def apply(self, X):
        """
        Apply the fitted model to new input data X to make predictions.

        Parameters:
        - X: NDArray, input data for which to predict Y.

        Returns:
        - Y_pred: NDArray, predicted values based on the fitted models
        """
        if self.model_name == "pixelwise_first":
            Y_pred = []
            # Apply the model to each feature
            assert self.models is not None
            for v in range(len(self.models)):
                a_i, b_i = self.models[v]
                if len(X.shape) == 1:
                    # X is 1D (single sample)
                    assert len(X) == len(
                        self.models
                    ), "Dimension mismatch: X does not match model dimensions."
                    Y_pred.append(a_i * X[v] + b_i)
                elif len(X.shape) == 2:
                    # X is 2D (multiple samples)
                    assert X.shape[1] == len(
                        self.models
                    ), "Dimension mismatch: X columns do not match model dimensions."
                    Y_pred.append(a_i * X[:, v] + b_i)
            return np.array(Y_pred).T  # Transpose to match the input shape
        else:
            raise ValueError(f"unsupported model: {self.model_name}")


def fit_linear_zonal_model(
    zwfs_ns,
    opd_internal,
    iterations=100,
    photon_flux_per_pixel_at_vlti=200,
    pearson_R_threshold=0.6,
    use_R_threshold=True,
    phase_scaling_factor=0.2,
    plot_intermediate=True,
    fig_path=None,
):

    # zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space
    # zwfs_ns.grid
    # zwfs_ns.detector
    # zwfs_ns.dm
    # zwfs_ns.pyZelda

    # init phase screen object
    dx = zwfs_ns.grid.D / zwfs_ns.grid.N
    # This screen is to put on the DM : assumes diameter covers entire DM - encoded in pixelscale
    scrn = phasescreens.PhaseScreenKolmogorov(
        nx_size=2 * zwfs_ns.dm.Nact_x,
        pixel_scale=zwfs_ns.grid.D / (2 * zwfs_ns.dm.Nact_x),
        r0=zwfs_ns.atmosphere.r0,
        L0=zwfs_ns.atmosphere.l0,
        random_seed=1,
    )

    opd_flat_dm = zwfs_ns.compiled_dm.eval(zwfs_ns.dm.dm_flat * zwfs_ns.dm.opd_per_cmd)

    b0_wsp, _ = ztools.create_reference_wave_beyond_pupil_with_aberrations(
        opd_internal + opd_flat_dm,
        zwfs_ns.pyZelda.mask_diameter,
        zwfs_ns.pyZelda.mask_depth,
        zwfs_ns.pyZelda.mask_substrate,
        zwfs_ns.pyZelda.mask_Fratio,
        zwfs_ns.pyZelda.pupil_diameter,
        zwfs_ns.pyZelda.pupil,
        zwfs_ns.optics.wvl0,
        clear=np.array([]),
        sign_mask=np.array([]),
        cpix=False,
    )

    # to put in pixel space (we just average with the same binning as the bldr detector)
    b0 = average_subarrays(
        abs(b0_wsp), (zwfs_ns.detector.binning, zwfs_ns.detector.binning)
    )

    I0 = get_I0(
        opd_input=0 * zwfs_ns.pyZelda.pupil,
        amp_input=photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil,
        opd_internal=zwfs_ns.pyZelda.pupil * (opd_internal + opd_flat_dm),
        zwfs_ns=zwfs_ns,
        detector=zwfs_ns.detector,
        include_shotnoise=True,
        use_pyZelda=True,
    )

    N0 = get_N0(
        opd_input=0 * zwfs_ns.pyZelda.pupil,
        amp_input=photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil,
        opd_internal=zwfs_ns.pyZelda.pupil * (opd_internal + opd_flat_dm),
        zwfs_ns=zwfs_ns,
        detector=zwfs_ns.detector,
        include_shotnoise=True,
        use_pyZelda=True,
    )

    b0_dm = DM_registration.interpolate_pixel_intensities(
        image=I0,
        pixel_coords=zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space,
    )  # DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
    I0_dm = DM_registration.interpolate_pixel_intensities(
        image=b0,
        pixel_coords=zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space,
    )  # DM_registration.interpolate_pixel_intensities(image = b0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
    N0_dm = DM_registration.interpolate_pixel_intensities(
        image=N0,
        pixel_coords=zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space,
    )  # DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

    telemetry = {
        "I0": [I0],
        "I0_dm": [I0_dm],
        "N0": [N0],
        "N0_dm": [N0_dm],
        "b0": [b0],
        "b0_dm": [b0_dm],
        "dm_cmd": [],
        "b": [],
        "b_est": [],
        "b_dm_est": [],
        "i": [],
        "Ic": [],
        "i_dm": [],
        "s": [],
        "strehl_0": [],
        "strehl_1": [],
        "strehl_2": [],
        "strehl_2_est": [],
    }

    telem_ns = SimpleNamespace(**telemetry)

    zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()

    for it in range(iterations):

        print(it)

        # roll screen
        for _ in range(10):
            scrn.add_row()

        zwfs_ns.dm.current_cmd = util.create_phase_screen_cmd_for_DM(
            scrn,
            scaling_factor=phase_scaling_factor,
            drop_indicies=[0, 11, 11 * 12, -1],
            plot_cmd=False,
        )

        # add BALDR DM OPD (onto wavespace)
        opd_current_dm = zwfs_ns.compiled_dm.eval(
            zwfs_ns.dm.current_cmd * zwfs_ns.dm.opd_per_cmd
        )

        # sum all opd contributions in the Baldr input pupil plane
        bldr_opd_map = np.sum([opd_internal, opd_current_dm], axis=0)

        # get the real strehl ratio applied by DM
        Strehl_0 = np.exp(
            -np.var(
                2
                * np.pi
                / zwfs_ns.optics.wvl0
                * bldr_opd_map[zwfs_ns.pyZelda.pupil > 0.5]
            )
        )  # atmospheric strehl

        # propagate to the detector plane
        Ic = photon_flux_per_pixel_at_vlti * zwfs_ns.pyZelda.propagate_opd_map(
            bldr_opd_map, wave=zwfs_ns.optics.wvl0
        )

        # detect the intensity
        i = detect(
            Ic,
            binning=(zwfs_ns.detector.binning, zwfs_ns.detector.binning),
            qe=zwfs_ns.detector.qe,
            dit=zwfs_ns.detector.dit,
            ron=zwfs_ns.detector.ron,
            include_shotnoise=True,
            spectral_bandwidth=zwfs_ns.stellar.bandwidth,
        )

        # interpolate signals onto registered actuator grid
        i_dm = DM_registration.interpolate_pixel_intensities(
            image=i,
            pixel_coords=zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space,
        )

        # get the optical gain (just incase we want to look - not required)
        # b, _ = ztools.create_reference_wave_beyond_pupil_with_aberrations(bldr_opd_map, zwfs_ns.pyZelda.mask_diameter, zwfs_ns.pyZelda.mask_depth, zwfs_ns.pyZelda.mask_substrate, zwfs_ns.pyZelda.mask_Fratio,
        #                                        zwfs_ns.pyZelda.pupil_diameter, zwfs_ns.pyZelda.pupil, zwfs_ns.optics.wvl0, clear=np.array([]),
        #                                        sign_mask=np.array([]), cpix=False)

        telem_ns.i.append(i)
        telem_ns.Ic.append(Ic)
        telem_ns.i_dm.append(i_dm)
        telem_ns.strehl_0.append(Strehl_0)
        # telem_ns.b.append( b )
        telem_ns.dm_cmd.append(zwfs_ns.dm.current_cmd)

    # save fits
    # plot the  interpolated intensity on DM and the DM command
    # if save_telemetry:
    #    save_telemetry( telem_ns , savename = fig_path + f'telem_with_dm_interactuator_coupling-{zwfs_ns.dm.actuator_coupling_factor}.fits', overwrite=True, return_fits = False)

    if plot_intermediate:
        # let have a dybnamic plot of the telemetry
        image_lists = [
            [util.get_DM_command_in_2D(a) for a in telem_ns.i_dm],
            [util.get_DM_command_in_2D(a) for a in telem_ns.dm_cmd],
            telem_ns.Ic,
        ]

        util.display_images_with_slider(
            image_lists=image_lists,
            plot_titles=["intensity interp dm", "dm cmd", "intensity wavespace"],
            cbar_labels=None,
        )

        # make a movie
        # util.display_images_as_movie( image_lists = image_lists,\
        #    plot_titles=['intensity interp dm', 'dm cmd', 'intensity wavespace'], cbar_labels=None, save_path = fig_path + 'zonal_model_calibration_dm_interactuator_coupling-{zwfs_ns.dm.actuator_coupling_factor}.mp4', fps=5)

        act = 65
        plt.figure()
        plt.plot(np.array(telem_ns.dm_cmd).T[act], np.array(telem_ns.i_dm).T[act], ".")
        plt.xlabel("dm cmd")
        plt.ylabel("intensity interp dm")

        if fig_path is not None:
            plt.savefig(
                fig_path
                + f"dmcmd_vs_dmIntensity_actuator-{act}_dm_interactuator_coupling-{zwfs_ns.dm.actuator_coupling_factor}.png"
            )
        plt.show()

    # look at the correlation between the DM command and the interpolated intensity (Pearson R)
    R_list = []
    for act in range(140):
        R_list.append(
            pearsonr(
                [a[act] for a in telem_ns.i_dm], [a[act] for a in telem_ns.dm_cmd]
            ).statistic
        )

    if plot_intermediate:
        plt.figure()
        plt.imshow(util.get_DM_command_in_2D(R_list))
        plt.colorbar(label="Pearson R")
        plt.title(
            "Pearson R between DM command and \ninterpolated intensity onto DM actuator space"
        )
        # plt.savefig(fig_path + f'pearson_r_dmcmd_dmIntensity_dm_interactuator_coupling-{zwfs_ns.dm.actuator_coupling_factor}.png')
        plt.show()

    # we filter a little tighter (4 actuator radius) because edge effects are bad
    if use_R_threshold:
        act_filt = (
            np.array(R_list) > pearson_R_threshold
        )  # * np.array( [x**2 + y**2 < 4**2 for x,y in zwfs_ns.grid.dm_coord.dm_coords])
    else:
        print(
            "ALERT: use_R_threshold=False. Using set DM actuator radius = 4 actuators for control region"
        )
        act_filt = util.get_circle_DM_command(radius=4, Nx_act=12).astype(bool)
    # we could also have a set filter (e.g 10 actuator diameter )
    # so standardise the testing with differenet scenarios / parameters

    telem_ns.act_filt = act_filt
    telem_ns.pearson_R = np.array(R_list)

    if plot_intermediate:
        util.nice_heatmap_subplots([util.get_DM_command_in_2D(act_filt)])
        plt.show()

    # Initialize the linear fit model
    model_1 = my_lin_fit(model_name="pixelwise_first")

    # Assuming telem_ns contains the necessary data
    # note act_filt just used for filtering what pixels to use to calculate average of N0_dm
    X = model_1.process_signal(
        np.array(telem_ns.i_dm), N0_dm, act_filt
    )  # np.array(telem_ns.i_dm / np.mean(N0_dm[ N0_dm > np.mean( N0_dm ) ] ) ) # / ( np.array(telem_ns.b_dm_est) * np.mean( N0_dm )   - I0_dm/ b0_dm)  # Input features (samples x features)

    Y = np.array(telem_ns.dm_cmd)  # Target values (samples x features)

    # Fit the model to X and Y
    model_1.fit(X=X, Y=Y)

    model_1.act_filt_recommended = act_filt
    model_1.N0_dm = N0_dm
    model_1.I0_dm = I0_dm
    model_1.pearson_R_dm = np.array(R_list)

    if plot_intermediate:
        # Apply the model to make predictions
        Y_pred = model_1.apply(X)
        # Select an actuator/feature to plot
        act = 65  # Example actuator/feature index
        # Plot the true values vs. the model predictions for the selected feature
        plt.plot(X[:, act], Y_pred[:, act], ".", label="Model Prediction")
        plt.plot(X[:, act], Y[:, act], ".", label="True Data")
        plt.xlabel("I0/<N0>")
        plt.ylabel("DM Command")
        plt.legend()
        plt.show()

    zwfs_ns.reco.linear_zonal_model = model_1

    return zwfs_ns


def process_zwfs_intensity(i, zwfs_ns, method, record_telemetry=False, **kwargs):
    ### NOTE here we don't use zwfs_ns.ctrl namespace , instead we append controllers to kwargs depending on method!!
    if method == "MVM-TT-HO":

        I0 = kwargs["I0"]
        TT_ctrl = kwargs["TT_ctrl"]  # pid or leakyintegrator
        HO_ctrl = kwargs["HO_ctrl"]  # pid or leakyintegrator

        # matrix vector multiplication for TT and HO control on some modal basis
        sig = process_zwfs_signal(
            i, I0, zwfs_ns.pupil_regions.pupil_filt
        )  # I0_theory/ np.mean(I0_theory) #

        e_TT = zwfs_ns.reco.I2M_TT @ sig

        u_TT = TT_ctrl.process(e_TT)

        c_TT = zwfs_ns.reco.M2C_TT @ u_TT

        e_HO = zwfs_ns.reco.I2M_HO @ sig

        u_HO = HO_ctrl.process(e_HO)

        c_HO = zwfs_ns.reco.M2C_HO @ u_HO

        delta_cmd = c_TT + c_HO
        # safety
        if np.max(delta_cmd) > 0.8:
            print(" going badly.. ")

        # telemetry
        if record_telemetry:
            zwfs_ns.telem.i_list.append(i)
            zwfs_ns.telem.s_list.append(sig)
            zwfs_ns.telem.e_TT_list.append(e_TT)
            zwfs_ns.telem.u_TT_list.append(u_TT)
            zwfs_ns.telem.c_TT_list.append(c_TT)

            zwfs_ns.telem.e_HO_list.append(e_HO)
            zwfs_ns.telem.u_HO_list.append(u_HO)
            zwfs_ns.telem.c_HO_list.append(c_HO)

        return delta_cmd

    elif method == "zonal_interp_no_projection":

        N0_dm = kwargs["N0_dm"]
        HO_ctrl = kwargs["HO_ctrl"]  # pid or leakyintegrator

        i_dm = DM_registration.interpolate_pixel_intensities(
            image=i,
            pixel_coords=zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space,
        )

        # this should really be repeated each time its called - could be input
        # N0_dm = DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space)
        # act_filt_recommended here is used to filter N0 pixels to calculate average for normalization
        sig = zwfs_ns.reco.linear_zonal_model.process_signal(
            i_dm, N0_dm, zwfs_ns.reco.linear_zonal_model.act_filt_recommended
        )  # reco.i_dm / np.mean( N0[ N0 > np.mean( N0 ) ] )

        e_HO = zwfs_ns.reco.linear_zonal_model.apply(sig)[
            zwfs_ns.reco.linear_zonal_model.act_filt_recommended
        ]

        u_HO = HO_ctrl.process(e_HO)

        # forcefully remove piston
        u_HO -= np.mean(u_HO)

        ## how best to build mode to command matrix for interpolation method when using select actuators (what happens to the rest?)
        delta_cmd = np.zeros(len(zwfs_ns.dm.dm_flat))
        delta_cmd[zwfs_ns.reco.linear_zonal_model.act_filt_recommended] = u_HO
        ### -- we shoudl aim for something like this:::
        # zwfs_ns.reco.M2C_HO @ u_HO

        # telemetry
        if record_telemetry:
            zwfs_ns.telem.i_list.append(i)
            zwfs_ns.telem.i_dm_list.append(i_dm)
            zwfs_ns.telem.s_list.append(sig)
            zwfs_ns.telem.e_TT_list.append(np.zeros(len(e_HO)))
            zwfs_ns.telem.u_TT_list.append(np.zeros(len(e_HO)))
            zwfs_ns.telem.c_TT_list.append(np.zeros(len(delta_cmd)))

            zwfs_ns.telem.e_HO_list.append(e_HO)
            zwfs_ns.telem.u_HO_list.append(u_HO)
            zwfs_ns.telem.c_HO_list.append(delta_cmd)

        return delta_cmd

    else:
        raise TypeError("process_zwfs_intensity method name NOT FOUND!!!!")


##########################################
##########################################
# Adding new configuration input via json and initialization of Fresnel propagation
##########################################
##########################################


def init_zwfs_from_json(
    json_config,
    derive_fresnel=True,
    derive_spectrum=True,
    instantiate_detector=True,
):
    """
    Initialise a Baldr ZWFS namespace directly from a generic JSON
    configuration file.

    This initializer does not use pyZELDA ini files. It is intended for generic
    BaldrApp simulations where the optical model, Fresnel relay, spectrum, and
    optional simulator metadata are described by a JSON file.

    Required JSON sections
    ----------------------
    The JSON file must contain:

        grid
        optics
        dm

    These are passed to the existing init_zwfs(grid_ns, optics_ns, dm_ns)
    function.

    Optional derived sections
    -------------------------
    fresnel_relay
        If present and derive_fresnel is True, derived physical relay
        quantities are added, including:

            D_phys
            M_nominal
            s_object_nominal
            s_image_nominal
            z_mirror_to_lens
            s_object
            s_image
            z_focus_to_detector_nominal
            z_focus_to_detector
            M_pupil
            D_detector_predicted

    spectrum
        If present and derive_spectrum is True, this section is passed to
        baldrapp.common.spectrum.derive_spectrum(...). The returned namespace
        is attached as zwfs_ns.spectrum.

        If absent, a backwards-compatible monochromatic spectrum is attached:

            wavelength = optics.wvl0
            weight = 1

    detector
        If present, the raw detector config is attached as
        zwfs_ns.detector_config. If instantiate_detector is True, a BaldrApp
        detector object is also created as zwfs_ns.detector.

    Optional metadata/runtime sections
    ----------------------------------
    The following sections are attached as namespaces if present, but are not
    interpreted by this initializer:

        meta
        source
        stellar
        throughput
        internal_aberrations
        atmosphere
        first_stage_ao
        simulator_runtime

    Design note
    -----------
    This function configures the ZWFS/instrument namespace. It should not
    generate atmospheric screens, scintillation screens, DM commands, or
    detector frames. Those should be handled by higher-level simulation code or
    frame-generation wrappers.

    Example
    -------
    zwfs_ns = init_zwfs_from_json("baldr_config.json")

    fr = zwfs_ns.fresnel_relay
    sp = zwfs_ns.spectrum
    """
    json_config = Path(json_config)
    cfg = cfghelp.load_json_config(json_config)

    for section in ["grid", "optics", "dm"]:
        if section not in cfg:
            raise ValueError(f"JSON config must contain a '{section}' section.")

    # Core sections needed by the existing init_zwfs(...)
    grid_ns = cfghelp.dict_to_namespace_recursive(cfg["grid"])
    optics_ns = cfghelp.dict_to_namespace_recursive(cfg["optics"])
    dm_ns = cfghelp.dict_to_namespace_recursive(cfg["dm"])

    optics_ns = cfghelp.ensure_optics_defaults(optics_ns)

    zwfs_ns = init_zwfs(grid_ns, optics_ns, dm_ns)

    # Keep provenance/debug information.
    zwfs_ns.config_json = cfg
    zwfs_ns.config_path = str(json_config)

    # Attach optional metadata/runtime sections.
    zwfs_ns = cfghelp.attach_optional_config_sections(zwfs_ns, cfg)

    # Optional runtime source profiles.
    # This is metadata for higher-level simulators. It is not interpreted here,
    # so old configs remain fully backwards compatible.
    if "source_profiles" in cfg:
        zwfs_ns.source_profiles = cfghelp.dict_to_namespace_recursive(
            cfg["source_profiles"]
        )

    # Detector config/object.
    # Detector instantiation stays here, not in config_helper.py, because the
    # detector class is defined in baldr_core.py.
    if "detector" in cfg:
        det_cfg = cfghelp.dict_to_namespace_recursive(cfg["detector"])
        zwfs_ns.detector_config = det_cfg

        if instantiate_detector and getattr(det_cfg, "enabled", True):
            zwfs_ns.detector = detector(
                binning=int(getattr(det_cfg, "binning", 1)),
                dit=float(getattr(det_cfg, "dit", 1.0)),
                ron=float(getattr(det_cfg, "ron", 0.0)),
                qe=float(getattr(det_cfg, "qe", 1.0)),
            )

    # # Spectrum config.
    # # Always attach zwfs_ns.spectrum when derive_spectrum=True, even if the
    # # JSON has no spectrum section. This keeps downstream wrappers simple and
    # # backwards-compatible.
    # if derive_spectrum:
    #     if "spectrum" in cfg:
    #         spectrum_ns = spec.derive_spectrum(
    #             cfg["spectrum"],
    #             default_wvl0=zwfs_ns.optics.wvl0,
    #         )
    #     else:
    #         spectrum_ns = spec.default_monochromatic_spectrum(
    #             zwfs_ns.optics.wvl0
    #         )

    #     spectrum_ns = cfghelp.complete_spectrum_integration_fields(
    #         spectrum_ns
    #     )
    #     zwfs_ns.spectrum = spectrum_ns

    # Stellar / spectrum config.
    # Preferred user config is now cfg["stellar"]["spectrum"].
    # Backwards-compatible fallback is top-level cfg["spectrum"].
    spectrum_cfg = cfghelp.get_spectrum_config_from_cfg(cfg)

    stellar_ns = getattr(zwfs_ns, "stellar", None)
    stellar_ns, bandwidth_source = cfghelp.ensure_stellar_bandwidth(
        stellar_ns,
        spectrum_cfg=spectrum_cfg,
        default_nm=1.0,
        verbose=True,
    )
    zwfs_ns.stellar = stellar_ns

    cfghelp.check_stellar_bandwidth_consistency(
        zwfs_ns.stellar,
        spectrum_cfg=spectrum_cfg,
        relative_tolerance=1e-3,
        verbose=True,
    )

    if derive_spectrum:
        if spectrum_cfg is not None:
            spectrum_ns = spec.derive_spectrum(
                spectrum_cfg,
                default_wvl0=zwfs_ns.optics.wvl0,
            )
        else:
            spectrum_ns = spec.default_monochromatic_spectrum(zwfs_ns.optics.wvl0)

        spectrum_ns = cfghelp.complete_spectrum_integration_fields(spectrum_ns)

        spectrum_ns = cfghelp.apply_stellar_bandwidth_to_spectrum(
            spectrum_ns,
            stellar_ns=zwfs_ns.stellar,
            warn=True,
        )

        zwfs_ns.spectrum = spectrum_ns

    # Fresnel relay config.
    if "fresnel_relay" in cfg:
        fresnel_ns = cfghelp.dict_to_namespace_recursive(cfg["fresnel_relay"])

        if derive_fresnel:
            fresnel_ns = cfghelp.derive_fresnel_relay_ns(fresnel_ns)

        zwfs_ns.fresnel_relay = fresnel_ns

    return zwfs_ns


def get_zwfs_output_field_from_opd(
    opd,
    amp,
    wavelength,
    zwfs_ns,
    theta=None,
    phasemask_diameter=None,
    phasemask_mask=None,
    return_terms=True,
):
    """
    Convenience wrapper around get_zwfs_output_field(...) that accepts OPD in
    metres instead of phase in radians.

    This is useful for chromatic propagation, where the same OPD map must be
    converted to phase separately at each wavelength.

    Parameters
    ----------
    opd : ndarray
        Optical path difference in metres, on the BaldrApp wave-space grid.

    amp : ndarray
        Input field amplitude. If amp**2 has units photons / s / pixel / nm,
        then the output intensity from this function has the same spectral
        density convention.

    wavelength : float
        Wavelength in metres.

    zwfs_ns : SimpleNamespace
        Baldr ZWFS namespace.

    theta : float or None
        ZWFS phase-mask phase shift in radians. If None, use
        zwfs_ns.optics.theta.

    phasemask_diameter : float or None
        Phase-mask diameter in the convention expected by
        get_zwfs_output_field(...). If None, use zwfs_ns.optics.mask_diam.

    phasemask_mask : ndarray or None
        Optional precomputed phase-mask support mask.

    return_terms : bool
        Passed through to get_zwfs_output_field(...).

    Returns
    -------
    output : dict or ndarray
        Output from get_zwfs_output_field(...).
    """
    pupil = zwfs_ns.grid.pupil_mask

    if theta is None:
        theta = zwfs_ns.optics.theta

    if phasemask_diameter is None:
        phasemask_diameter = zwfs_ns.optics.mask_diam

    phi = pupil * 2.0 * np.pi / wavelength * opd

    return get_zwfs_output_field(
        phi=phi,
        amp=amp,
        theta=theta,
        phasemask_diameter=phasemask_diameter,
        phasemask_mask=phasemask_mask,
        pupil_diameter=zwfs_ns.grid.N,
        fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
        pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
        return_terms=return_terms,
    )


####################################
# get frame wrappers for polychromatic and fresnel versions
def get_frame_polychromatic(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    use_pyZelda=False,
    return_intermediates=False,
):
    """
    Generate a polychromatic Baldr ZWFS frame using the existing non-Fresnel
    propagation path, i.e. get_frame(...).

    This function loops over zwfs_ns.spectrum, computes one monochromatic
    spectral-density image per wavelength using get_frame(..., detector=None),
    integrates those images over wavelength using weights_nm, and then applies
    the detector once at the end if detector is provided.

    Unit convention
    ---------------
    This function assumes that amp_input**2 has units:

        photons / second / wave-space-pixel / nm

    Therefore each monochromatic image returned by get_frame(..., detector=None)
    has units:

        photons / second / wave-space-pixel / nm

    Spectral integration is performed as:

        I_rate = sum_k weights_nm_k * I_density_k

    where weights_nm_k has units of nm. The integrated image therefore has
    units:

        photons / second / wave-space-pixel

    If detector is provided, detector.detect(...) is called with:

        spectral_bandwidth=None

    because the spectral integration has already been done explicitly.

    Notes
    -----
    This function temporarily updates:

        zwfs_ns.optics.wvl0
        zwfs_ns.optics.theta
        zwfs_ns.optics.mask_diam

    inside the wavelength loop, then restores their original values.

    For the current generic JSON/config path, use_pyZelda=False is recommended.
    The pyZELDA path may require additional care because phase-mask chromaticity
    is encoded through pyZELDA mask depth/material rather than directly through
    theta.
    """
    # Backwards-compatible fallback if no spectrum namespace exists.
    if not hasattr(zwfs_ns, "spectrum"):
        wavelengths = np.array([zwfs_ns.optics.wvl0], dtype=float)
        weights_nm = np.array([1.0], dtype=float)
    else:
        wavelengths = np.asarray(zwfs_ns.spectrum.wavelengths, dtype=float)

        if hasattr(zwfs_ns.spectrum, "weights_nm"):
            weights_nm = np.asarray(zwfs_ns.spectrum.weights_nm, dtype=float)
        else:
            if hasattr(zwfs_ns.spectrum, "weights_normalized"):
                weights_normalized = np.asarray(
                    zwfs_ns.spectrum.weights_normalized,
                    dtype=float,
                )
            else:
                weights_normalized = np.ones_like(wavelengths, dtype=float) / len(
                    wavelengths
                )

            bandwidth_nm = float(getattr(zwfs_ns.spectrum, "bandwidth_nm", 1.0))

            weights_nm = bandwidth_nm * weights_normalized

    if wavelengths.ndim != 1:
        raise ValueError("zwfs_ns.spectrum.wavelengths must be one-dimensional.")

    if weights_nm.ndim != 1:
        raise ValueError("zwfs_ns.spectrum.weights_nm must be one-dimensional.")

    if wavelengths.shape != weights_nm.shape:
        raise ValueError(
            "zwfs_ns.spectrum.wavelengths and weights_nm must have the same shape."
        )

    if len(wavelengths) < 1:
        raise ValueError("Spectrum must contain at least one wavelength sample.")

    if np.any(wavelengths <= 0):
        raise ValueError("All spectrum wavelengths must be positive.")

    if np.any(weights_nm < 0):
        raise ValueError("All spectrum weights_nm must be non-negative.")

    if np.sum(weights_nm) <= 0:
        raise ValueError("Spectrum weights_nm must have positive sum.")

    # Save original optical settings so we can restore them after the loop.
    original_wvl0 = zwfs_ns.optics.wvl0
    original_theta = zwfs_ns.optics.theta
    original_mask_diam = zwfs_ns.optics.mask_diam

    I_rate = None
    per_wavelength_outputs = []

    try:
        for wavelength, weight_nm in zip(wavelengths, weights_nm):

            theta = spec.theta_at_wavelength(
                zwfs_ns.optics,
                wavelength,
            )

            phasemask_diameter = spec.phasemask_diameter_at_wavelength(
                zwfs_ns.optics,
                wavelength_m=wavelength,
                default_wvl0=original_wvl0,
            )

            # Temporarily update the namespace so get_frame(...) uses this
            # wavelength and phase-mask setting internally.
            zwfs_ns.optics.wvl0 = float(wavelength)
            zwfs_ns.optics.theta = float(theta)
            zwfs_ns.optics.mask_diam = float(phasemask_diameter)

            I_density = get_frame(
                opd_input=opd_input,
                amp_input=amp_input,
                opd_internal=opd_internal,
                zwfs_ns=zwfs_ns,
                detector=None,
                include_shotnoise=False,
                use_pyZelda=use_pyZelda,
            )

            if I_rate is None:
                I_rate = float(weight_nm) * I_density
            else:
                I_rate = I_rate + float(weight_nm) * I_density

            if return_intermediates:
                per_wavelength_outputs.append(
                    {
                        "wavelength": float(wavelength),
                        "weight_nm": float(weight_nm),
                        "theta": float(theta),
                        "phasemask_diameter": float(phasemask_diameter),
                        "intensity_density": I_density,
                    }
                )

    finally:
        # Always restore original settings, even if an error occurs.
        zwfs_ns.optics.wvl0 = original_wvl0
        zwfs_ns.optics.theta = original_theta
        zwfs_ns.optics.mask_diam = original_mask_diam

    # if detector is not None:
    #     intensity = detector.detect(
    #         I_rate,
    #         include_shotnoise=include_shotnoise,
    #         spectral_bandwidth=None,
    #     )
    # else:
    #     intensity = I_rate
    if detector is not None:
        assert I_rate is not None
        intensity = detect(
            I_rate,
            binning=(detector.binning, detector.binning),
            qe=detector.qe,
            dit=detector.dit,
            ron=detector.ron,
            include_shotnoise=include_shotnoise,
            spectral_bandwidth=None,
        )
    else:
        intensity = I_rate

    if not return_intermediates:
        return intensity

    return {
        "intensity": intensity,
        "intensity_pre_detector": I_rate,
        "wavelengths": wavelengths,
        "weights_nm": weights_nm,
        "weights_nm_sum": float(np.sum(weights_nm)),
        "per_wavelength_outputs": per_wavelength_outputs,
    }


def get_frame_fresnel(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    spectral_bandwidth: Optional[float | str] = _AUTO_SPECTRAL_BANDWIDTH,
    wavelength=None,
    theta=None,
    phasemask_diameter=None,
    return_intermediates=False,
):
    """
    Generate a monochromatic detector-plane Baldr frame using the analytic ZWFS
    output field followed by the explicit Fresnel relay.

    This function does not perform spectral integration. For polychromatic
    propagation, call a separate polychromatic wrapper that loops over
    zwfs_ns.spectrum and integrates the monochromatic spectral-density images.

    Unit convention
    ---------------
    The pre-detector intensity has the same photon-rate convention as
    amp_input**2. For example, if amp_input**2 is in:

        photons / second / wave-space-pixel / nm

    then intensity_pre_detector is a monochromatic spectral-density image.
    """
    if not hasattr(zwfs_ns, "fresnel_relay"):
        raise ValueError(
            "zwfs_ns has no fresnel_relay attribute. "
            "Initialise from a config with a fresnel_relay section."
        )

    fr = zwfs_ns.fresnel_relay

    if wavelength is None:
        wavelength = zwfs_ns.optics.wvl0

    # pupil = zwfs_ns.grid.pupil_mask
    # opd_total = pupil * (opd_input + opd_internal)

    pupil = zwfs_ns.grid.pupil_mask

    opd_current_dm = zwfs_ns.compiled_dm.eval(
        zwfs_ns.dm.current_cmd * zwfs_ns.dm.opd_per_cmd
    )

    opd_total = pupil * (opd_input + opd_internal + opd_current_dm)

    if theta is None:
        theta = spec.theta_at_wavelength(zwfs_ns.optics, wavelength)

    if phasemask_diameter is None:
        phasemask_diameter = spec.phasemask_diameter_at_wavelength(
            zwfs_ns.optics,
            wavelength_m=wavelength,
            default_wvl0=zwfs_ns.optics.wvl0,
        )

    zwfs_terms = get_zwfs_output_field_from_opd(
        opd=opd_total,
        amp=amp_input,
        wavelength=wavelength,
        zwfs_ns=zwfs_ns,
        theta=theta,
        phasemask_diameter=phasemask_diameter,
        phasemask_mask=None,
        return_terms=True,
    )

    psi_C = zwfs_terms["psi_C"]

    dx_phys = fr.D_phys / zwfs_ns.grid.N

    X_phys, Y_phys = fresnel.make_coordinate_grid(
        psi_C.shape,
        dx_phys,
    )

    # 1. Propagate from post-collimator ZWFS pupil plane to D/knife-edge mirror.
    psi_mirror = fresnel.propagate(
        psi_C,
        wavelength=wavelength,
        dx=dx_phys,
        z=fr.z_to_mirror,
        method="angular_spectrum",
    )

    mirror_mask_full = fresnel.circular_aperture(
        psi_mirror.shape,
        dx=dx_phys,
        radius=fr.D_mirror / 2.0,
    ).astype(float)

    Xr = X_phys * np.cos(fr.edge_angle) + Y_phys * np.sin(fr.edge_angle)

    mirror_mask_D = (Xr >= fr.edge_offset).astype(float) * mirror_mask_full

    psi_after_mirror = psi_mirror * mirror_mask_D

    # 2. Propagate mirror to re-imaging lens.
    field_at_lens = fresnel.propagate(
        psi_after_mirror,
        wavelength=wavelength,
        dx=dx_phys,
        z=fr.z_mirror_to_lens,
        method="angular_spectrum",
    )

    # 3. Apply re-imaging lens.
    field_after_lens = fresnel.apply_thin_lens(
        field_at_lens,
        wavelength=wavelength,
        dx=dx_phys,
        focal_length=fr.f_imaging,
    )

    # 4. Propagate lens to cold-stop / star-image plane.
    field_coldstop_plane, dx_cold, dy_cold = fresnel.fresnel_one_step_propagate(
        field_after_lens,
        wavelength=wavelength,
        dx=dx_phys,
        z=fr.f_imaging,
        include_global_phase=False,
    )

    X_cold, Y_cold = fresnel.make_coordinate_grid(
        field_coldstop_plane.shape,
        dx=dx_cold,
        dy=dy_cold,
    )

    R_cold = np.hypot(
        X_cold - fr.coldstop_x_offset,
        Y_cold - fr.coldstop_y_offset,
    )

    coldstop_mask_phys = (R_cold <= fr.D_coldstop / 2.0).astype(float)

    field_after_coldstop = field_coldstop_plane * coldstop_mask_phys

    # 5. Propagate cold-stop / star-image plane to detector pupil plane.
    field_detector_pupil, dx_detector, dy_detector = fresnel.fresnel_one_step_propagate(
        field_after_coldstop,
        wavelength=wavelength,
        dx=dx_cold,
        dy=dy_cold,
        z=fr.z_focus_to_detector,
        include_global_phase=False,
    )

    intensity_pre_detector = np.abs(field_detector_pupil) ** 2

    # Convert Fresnel detector-plane irradiance samples back to the historical
    # BaldrApp photon-rate-per-wave-space-pixel convention used by detect(...).
    #
    # Fresnel propagation changes the physical pixel scale from dx_phys to
    # dx_detector, dy_detector. Raw array sums therefore do not represent the same
    # physical photon rate unless pixel area is accounted for.
    fresnel_pixel_area_scale = (dx_detector * dy_detector) / (dx_phys**2)
    intensity_pre_detector = intensity_pre_detector * fresnel_pixel_area_scale

    # if detector is not None:
    #     intensity = detector.detect(
    #         intensity_pre_detector,
    #         include_shotnoise=include_shotnoise,
    #         spectral_bandwidth=spectral_bandwidth,
    #     )
    # else:
    #     intensity = intensity_pre_detector

    if detector is not None:
        detector_spectral_bandwidth = _resolve_detector_spectral_bandwidth(
            zwfs_ns,
            spectral_bandwidth=spectral_bandwidth,
        )

        # intensity = detector.detect(
        #     intensity_pre_detector,
        #     include_shotnoise=include_shotnoise,
        #     spectral_bandwidth=detector_spectral_bandwidth,
        # )
        intensity = detect(
            intensity_pre_detector,
            binning=(detector.binning, detector.binning),
            qe=detector.qe,
            dit=detector.dit,
            ron=detector.ron,
            include_shotnoise=include_shotnoise,
            spectral_bandwidth=detector_spectral_bandwidth,
        )

    else:
        intensity = intensity_pre_detector

    if not return_intermediates:
        return intensity

    return {
        "intensity": intensity,
        "intensity_pre_detector": intensity_pre_detector,
        "zwfs_terms": zwfs_terms,
        "psi_C": psi_C,
        "psi_mirror": psi_mirror,
        "mirror_mask_D": mirror_mask_D,
        "field_at_lens": field_at_lens,
        "field_after_lens": field_after_lens,
        "field_coldstop_plane": field_coldstop_plane,
        "coldstop_mask_phys": coldstop_mask_phys,
        "field_after_coldstop": field_after_coldstop,
        "field_detector_pupil": field_detector_pupil,
        "dx_phys": dx_phys,
        "dx_cold": dx_cold,
        "dy_cold": dy_cold,
        "dx_detector": dx_detector,
        "dy_detector": dy_detector,
        "wavelength": wavelength,
        "theta": theta,
        "phasemask_diameter": phasemask_diameter,
    }


def get_frame_fresnel_polychromatic(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    return_intermediates=False,
):
    """
    Generate a polychromatic detector-plane Baldr frame using the analytic ZWFS
    output field followed by the explicit Fresnel relay.

    This function loops over zwfs_ns.spectrum, computes one monochromatic
    Fresnel relay frame per wavelength, and integrates the resulting
    spectral-density images over wavelength before detector detection.

    Unit convention
    ---------------
    This function assumes that amp_input**2 has units:

        photons / second / wave-space-pixel / nm

    Therefore each monochromatic image returned by get_frame_fresnel(...) has
    units:

        photons / second / detector-plane-pixel / nm

    Spectral integration is then performed as:

        I_rate = sum_k weights_nm_k * I_density_k

    where weights_nm_k has units of nm.

    The integrated pre-detector image therefore has units:

        photons / second / detector-plane-pixel

    If detector is provided, detector.detect(...) is called with:

        spectral_bandwidth=None

    because the spectral bandwidth has already been integrated explicitly.

    Parameters
    ----------
    opd_input : ndarray
        Input OPD in metres on the BaldrApp wave-space grid.

    amp_input : ndarray
        Input field amplitude. amp_input**2 is assumed to be a photon spectral
        density image in photons / second / wave-space-pixel / nm.

    opd_internal : ndarray
        Static/internal OPD in metres on the BaldrApp wave-space grid.

    zwfs_ns : SimpleNamespace
        Baldr ZWFS namespace. Must contain fresnel_relay. Should contain
        spectrum. If no spectrum is present, a single-wavelength 1 nm
        monochromatic spectrum is used.

    detector : detector or None
        Optional BaldrApp detector object. If provided, the spectrally
        integrated photon-rate image is passed through detector.detect(...).

    include_shotnoise : bool
        Passed to detector.detect(...) if detector is provided.

    return_intermediates : bool
        If False, return only the final image.
        If True, return a dictionary containing the final image, pre-detector
        integrated image, wavelength grid, integration weights, and optionally
        per-wavelength intermediate outputs.

    Returns
    -------
    intensity : ndarray
        If return_intermediates is False, returns the final image. If detector
        is None, this is the spectrally integrated pre-detector photon-rate
        image. If detector is not None, this is the detected image.

    output : dict
        If return_intermediates is True.
    """
    if not hasattr(zwfs_ns, "fresnel_relay"):
        raise ValueError(
            "zwfs_ns has no fresnel_relay attribute. "
            "Initialise from a config with a fresnel_relay section."
        )

    # Backwards-compatible fallback if zwfs_ns was not initialised from the
    # new JSON/config path.
    if not hasattr(zwfs_ns, "spectrum"):
        wavelengths = np.array([zwfs_ns.optics.wvl0], dtype=float)
        weights_nm = np.array([1.0], dtype=float)
    else:
        wavelengths = np.asarray(zwfs_ns.spectrum.wavelengths, dtype=float)

        if hasattr(zwfs_ns.spectrum, "weights_nm"):
            weights_nm = np.asarray(zwfs_ns.spectrum.weights_nm, dtype=float)
        else:
            # Robust fallback. Prefer explicit weights_nm from config_helper,
            # but allow older derived spectrum namespaces to work.
            if hasattr(zwfs_ns.spectrum, "weights_normalized"):
                weights_normalized = np.asarray(
                    zwfs_ns.spectrum.weights_normalized,
                    dtype=float,
                )
            else:
                weights_normalized = np.ones_like(wavelengths, dtype=float) / len(
                    wavelengths
                )

            bandwidth_nm = float(getattr(zwfs_ns.spectrum, "bandwidth_nm", 1.0))

            weights_nm = bandwidth_nm * weights_normalized

    if wavelengths.ndim != 1:
        raise ValueError("zwfs_ns.spectrum.wavelengths must be one-dimensional.")

    if weights_nm.ndim != 1:
        raise ValueError("zwfs_ns.spectrum.weights_nm must be one-dimensional.")

    if wavelengths.shape != weights_nm.shape:
        raise ValueError(
            "zwfs_ns.spectrum.wavelengths and weights_nm must have the same shape."
        )

    if len(wavelengths) < 1:
        raise ValueError("Spectrum must contain at least one wavelength sample.")

    if np.any(wavelengths <= 0):
        raise ValueError("All spectrum wavelengths must be positive.")

    if np.any(weights_nm < 0):
        raise ValueError("All spectrum weights_nm must be non-negative.")

    if np.sum(weights_nm) <= 0:
        raise ValueError("Spectrum weights_nm must have positive sum.")

    I_rate = None
    per_wavelength_outputs = []

    for wavelength, weight_nm in zip(wavelengths, weights_nm):

        theta = spec.theta_at_wavelength(
            zwfs_ns.optics,
            wavelength,
        )

        phasemask_diameter = spec.phasemask_diameter_at_wavelength(
            zwfs_ns.optics,
            wavelength_m=wavelength,
            default_wvl0=zwfs_ns.optics.wvl0,
        )

        out = get_frame_fresnel(
            opd_input=opd_input,
            amp_input=amp_input,
            opd_internal=opd_internal,
            zwfs_ns=zwfs_ns,
            detector=None,
            include_shotnoise=False,
            spectral_bandwidth=None,
            wavelength=float(wavelength),
            theta=theta,
            phasemask_diameter=phasemask_diameter,
            return_intermediates=return_intermediates,
        )

        if return_intermediates:
            I_density = out["intensity_pre_detector"]
            out["wavelength"] = float(wavelength)
            out["weight_nm"] = float(weight_nm)
            out["theta"] = float(theta)
            out["phasemask_diameter"] = float(phasemask_diameter)
            per_wavelength_outputs.append(out)
        else:
            I_density = out

        assert type(I_density) is np.ndarray
        if I_rate is None:
            I_rate = float(weight_nm) * I_density
        else:
            I_rate = I_rate + float(weight_nm) * I_density

    # if detector is not None:
    #     intensity = detector.detect(
    #         I_rate,
    #         include_shotnoise=include_shotnoise,
    #         spectral_bandwidth=None,
    #     )
    # else:
    #     intensity = I_rate
    if detector is not None:
        assert I_rate is not None
        intensity = detect(
            I_rate,
            binning=(detector.binning, detector.binning),
            qe=detector.qe,
            dit=detector.dit,
            ron=detector.ron,
            include_shotnoise=include_shotnoise,
            spectral_bandwidth=None,
        )
    else:
        intensity = I_rate

    if not return_intermediates:
        return intensity

    return {
        "intensity": intensity,
        "intensity_pre_detector": I_rate,
        "wavelengths": wavelengths,
        "weights_nm": weights_nm,
        "weights_nm_sum": float(np.sum(weights_nm)),
        "per_wavelength_outputs": per_wavelength_outputs,
    }


def get_frame_configured(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    use_pyZelda=False,
    spectral_bandwidth=_AUTO_SPECTRAL_BANDWIDTH,
    force_fresnel=None,
    force_polychromatic=None,
    return_intermediates=False,
):
    """
    Generate a Baldr frame using the propagation mode configured in zwfs_ns.

    This is a convenience dispatcher around the explicit frame-generation
    functions:

        get_frame(...)
        get_frame_polychromatic(...)
        get_frame_fresnel(...)
        get_frame_fresnel_polychromatic(...)

    It does not implement new propagation physics. It only chooses the
    appropriate already-tested path based on the configuration.

    Routing logic
    -------------
    Fresnel propagation is enabled if:

        zwfs_ns.fresnel_relay.enabled is True

    unless overridden by force_fresnel.

    Polychromatic propagation is enabled if:

        zwfs_ns.spectrum.enabled is True
        and len(zwfs_ns.spectrum.wavelengths) > 1

    unless overridden by force_polychromatic.

    Unit convention
    ---------------
    For monochromatic legacy paths, if spectral_bandwidth is not None, the
    detector interprets the input intensity as:

        photons / second / pixel / nm

    and multiplies by spectral_bandwidth.

    For polychromatic paths, the spectrum is integrated before detector
    detection using weights_nm, and the detector is called with:

        spectral_bandwidth=None

    to avoid double-counting bandwidth.

    Parameters
    ----------
    opd_input : ndarray
        Input OPD in metres.

    amp_input : ndarray
        Input field amplitude.

    opd_internal : ndarray
        Internal/static OPD in metres.

    zwfs_ns : SimpleNamespace
        Configured Baldr ZWFS namespace.

    detector : detector or None
        Optional detector object. If None and zwfs_ns.detector exists, this
        function uses zwfs_ns.detector.

    include_shotnoise : bool
        Whether to include shot noise if a detector is used.

    use_pyZelda : bool
        Passed only to the existing non-Fresnel get_frame(...) path and
        get_frame_polychromatic(...). For the generic JSON/config path,
        use_pyZelda=False is recommended.

    spectral_bandwidth :  float, None, or "auto"
        Used only for monochromatic detector paths. For polychromatic paths,
        this is ignored and detector.detect(...) is called with
        spectral_bandwidth=None.

    force_fresnel : bool or None
        If None, use zwfs_ns.fresnel_relay.enabled. If True/False, override.

    force_polychromatic : bool or None
        If None, use zwfs_ns.spectrum.enabled and the number of wavelengths.
        If True/False, override.

    return_intermediates : bool
        Only supported by the Fresnel and polychromatic wrapper paths. For the
        plain get_frame(...) path, this must be False.

    Returns
    -------
    frame : ndarray or dict
        Frame output from the selected propagation path.
    """
    if detector is None and hasattr(zwfs_ns, "detector"):
        detector = zwfs_ns.detector

    # Decide Fresnel mode.
    if force_fresnel is None:
        use_fresnel = hasattr(zwfs_ns, "fresnel_relay") and getattr(
            zwfs_ns.fresnel_relay, "enabled", False
        )
    else:
        use_fresnel = bool(force_fresnel)

    # Decide spectral mode.
    if force_polychromatic is None:
        if hasattr(zwfs_ns, "spectrum"):
            spectrum_enabled = getattr(zwfs_ns.spectrum, "enabled", True)
            wavelengths = np.asarray(zwfs_ns.spectrum.wavelengths, dtype=float)
            use_polychromatic = bool(spectrum_enabled) and len(wavelengths) > 1
        else:
            use_polychromatic = False
    else:
        use_polychromatic = bool(force_polychromatic)

    # ------------------------------------------------------------
    # Fresnel + polychromatic
    # ------------------------------------------------------------
    if use_fresnel and use_polychromatic:
        return get_frame_fresnel_polychromatic(
            opd_input=opd_input,
            amp_input=amp_input,
            opd_internal=opd_internal,
            zwfs_ns=zwfs_ns,
            detector=detector,
            include_shotnoise=include_shotnoise,
            return_intermediates=return_intermediates,
        )

    # ------------------------------------------------------------
    # Fresnel + monochromatic
    # ------------------------------------------------------------
    if use_fresnel and not use_polychromatic:
        return get_frame_fresnel(
            opd_input=opd_input,
            amp_input=amp_input,
            opd_internal=opd_internal,
            zwfs_ns=zwfs_ns,
            detector=detector,
            include_shotnoise=include_shotnoise,
            spectral_bandwidth=spectral_bandwidth,
            wavelength=zwfs_ns.optics.wvl0,
            return_intermediates=return_intermediates,
        )

    # ------------------------------------------------------------
    # Non-Fresnel + polychromatic
    # ------------------------------------------------------------
    if (not use_fresnel) and use_polychromatic:
        return get_frame_polychromatic(
            opd_input=opd_input,
            amp_input=amp_input,
            opd_internal=opd_internal,
            zwfs_ns=zwfs_ns,
            detector=detector,
            include_shotnoise=include_shotnoise,
            use_pyZelda=use_pyZelda,
            return_intermediates=return_intermediates,
        )

    # ------------------------------------------------------------
    # Non-Fresnel + monochromatic legacy path
    # ------------------------------------------------------------
    if return_intermediates:
        raise ValueError(
            "return_intermediates=True is not supported for the plain "
            "non-Fresnel monochromatic get_frame(...) path."
        )

    return get_frame(
        opd_input=opd_input,
        amp_input=amp_input,
        opd_internal=opd_internal,
        zwfs_ns=zwfs_ns,
        detector=detector,
        include_shotnoise=include_shotnoise,
        spectral_bandwidth=spectral_bandwidth,
        use_pyZelda=use_pyZelda,
    )


class Actuator(ABC):
    @abstractmethod
    def sample(self, x: float, y: float) -> float:
        pass

    @abstractmethod
    def sample_vec(self, xx: NDArray, yy: NDArray) -> NDArray:
        pass

    def evaluate(self, com: float, xx: NDArray, yy: NDArray) -> NDArray:
        zz = com * self.sample_vec(xx.flatten(), yy.flatten())
        return zz.reshape(xx.shape)


class DM(ABC):
    @abstractmethod
    def actuator(self, idx: int) -> Actuator:
        pass

    @property
    @abstractmethod
    def nact(self) -> int:
        pass

    def sample(self, command: NDArray, x: float, y: float) -> float:
        z: float = 0.0
        for i in range(self.nact):
            z += command[i] * self.actuator(i).sample(x, y)
        return z

    def evaluate(self, command: NDArray, xx: NDArray, yy: NDArray) -> NDArray:
        zz = (xx * 0.0).flatten()
        for i, (x, y) in enumerate(zip(xx.ravel(), yy.ravel())):
            zz[i] += self.sample(command, x, y)
        return zz.reshape(xx.shape)


class GaussianActuator(Actuator):
    sigma: float
    x0: float
    y0: float

    def __init__(self, sigma: float, x0: float, y0: float):
        self.sigma = sigma
        self.x0 = x0
        self.y0 = y0

    def sample(self, x: float, y: float) -> float:
        return np.exp(-((x - self.x0) ** 2 + (y - self.y0) ** 2) / self.sigma**2)

    def sample_vec(
        self,
        xx: NDArray,
        yy: NDArray,
    ) -> NDArray:
        return np.exp(-((xx - self.x0) ** 2 + (yy - self.y0) ** 2) / self.sigma**2)


class GaussianDM(DM):
    _nact: int
    actuators: List[GaussianActuator]

    def __init__(self, x0: NDArray, y0: NDArray, sigma: NDArray):
        self._nact = x0.shape[0]
        actuators: List[GaussianActuator] = []
        for x, y, sig in zip(x0, y0, sigma):
            actuators.append(GaussianActuator(sig, x, y))
        self.actuators = actuators

    @property
    def nact(self) -> int:
        return self._nact

    def actuator(self, idx: int) -> Actuator:
        return self.actuators[idx]


class CompiledDM(GaussianDM):
    """This object constructs the DM based on static parameters, and allows
    efficient evaluation (once initialised)"""

    influ: NDArray  # array of influence functions (nact, ncoordx, ncoordy)

    def __init__(self, xx: NDArray, yy: NDArray, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        influ = []
        for c in range(self.nact):
            influ.append(self.actuator(c).evaluate(1.0, xx, yy))
        self.influ = np.array(influ)

    def eval(self, com: NDArray) -> NDArray:
        return np.einsum("i...,i->...", self.influ, com)

import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime 
from astropy import units as u
from matplotlib.widgets import Slider
import matplotlib.animation as animation
import math
from configparser import ConfigParser
from types import SimpleNamespace
import scipy.ndimage as ndimage
from scipy.integrate import quad 
from scipy.optimize import least_squares
import pandas as pd 
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter,  median_filter
from scipy.optimize import leastsq
from scipy.stats import pearsonr
# Utils for co-aligning (center and rotate) frames from a measured pupil, and theoretical pupil
from scipy.ndimage import shift as nd_shift
from scipy.ndimage import rotate as nd_rotate
#from skimage.transform import warp_polar


def ini_to_namespace(ini_file):
    # convert ini file to python namespace
    
    # Create a ConfigParser object and read the INI file
    config = ConfigParser()
    config.read(ini_file)

    # Initialize an empty SimpleNamespace
    namespace = SimpleNamespace()

    # Iterate through sections and keys to populate the namespace
    for section in config.sections():
        section_namespace = SimpleNamespace()
        for key, value in config.items(section):
            setattr(section_namespace, key, value)
        
        # Set the section as an attribute of the main namespace
        setattr(namespace, section, section_namespace)

    return namespace


from configparser import ConfigParser
from types import SimpleNamespace

def ini_to_namespace(ini_file):
    # Create a ConfigParser object and read the INI file
    config = ConfigParser()
    config.read(ini_file)

    # Initialize an empty SimpleNamespace
    namespace = SimpleNamespace()

    # Iterate through sections and keys to populate the namespace
    for section in config.sections():
        section_namespace = SimpleNamespace()
        for key, value in config.items(section):
            # Try automatic type conversion using ConfigParser methods
            if config.has_option(section, key):
                # First attempt to convert to an integer
                try:
                    converted_value = config.getint(section, key)
                except ValueError:
                    # If it's not an int, try converting to a float
                    try:
                        converted_value = config.getfloat(section, key)
                    except ValueError:
                        # If it's not a float, check if it's a boolean
                        try:
                            converted_value = config.getboolean(section, key)
                        except ValueError:
                            # Fallback to original string if no conversion works
                            converted_value = value
            
            setattr(section_namespace, key, converted_value)
        
        # Set the section as an attribute of the main namespace
        setattr(namespace, section, section_namespace)

    return namespace



def get_DM_command_in_2D(cmd,Nx_act=12):
    # function so we can easily plot the DM shape (since DM grid is not perfectly square raw cmds can not be plotted in 2D immediately )
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM.
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(Nx_act,Nx_act) )




def get_circle_DM_command(radius, Nx_act=12):
    """
    Generates a DM command that forms a circular shape of the given radius.

    Parameters:
        radius (float): Desired radius in actuator units.
        Nx_act (int, optional): Number of actuators per side of the DM (default 12).

    Returns:
        cmd (ndarray): A 140-length DM command vector with a circular shape.
    """
    # Generate actuator coordinate grid
    x = np.arange(Nx_act)
    y = np.arange(Nx_act)
    X, Y = np.meshgrid(x, y)

    # Compute distances from the center of the DM grid
    center = (Nx_act - 1) / 2  # DM is 12x12, so center is at (5.5, 5.5)
    distances = np.sqrt((X - center) ** 2 + (Y - center) ** 2)

    # Mask actuators inside the desired radius
    mask = distances <= radius

    # Flatten the mask and remove corner actuators
    mask_flattened = mask.flatten()
    corner_indices = [0, Nx_act-1, Nx_act*(Nx_act-1), Nx_act*Nx_act-1]
    mask_flattened = np.delete(mask_flattened, corner_indices)

    # Create the DM command vector of length 140
    cmd = np.zeros(140)
    cmd[mask_flattened] = 1  # Set selected actuators to 1

    return cmd


def insert_concentric(smaller_array, larger_array):
    # Get the shapes of both arrays
    N, M = smaller_array.shape
    P, Q = larger_array.shape

    # Check if the smaller array can fit in the larger array
    if N > P or M > Q:
        raise ValueError("Smaller array must have dimensions less than or equal to the larger array.")

    # Find the starting indices to center the smaller array in the larger array
    start_row = (P - N) // 2
    start_col = (Q - M) // 2

    # Create a copy of the larger array to avoid modifying the input directly
    result_array = larger_array.copy()

    # Insert the smaller array into the larger array
    result_array[start_row:start_row + N, start_col:start_col + M] = smaller_array

    return result_array


def get_phasemask_phaseshift(
    wvl,
    depth,
    dot_material="N_1405",
    extrapolate=True,
):
    """
    Return the phase shift of a physical phase mask.

    Parameters
    ----------
    wvl : float or ndarray
        Wavelength in micrometers.
    depth : float
        Physical phase-mask depth in micrometers.
    dot_material : str
        Material name. Currently supports "N_1405".
    extrapolate : bool
        If True, linearly extrapolate the material index outside the available
        tabulated wavelength range.

    Notes
    -----
    The material optical-constants table is read only once per material /
    extrapolation mode and cached as an interp1d object.
    """
    if not hasattr(get_phasemask_phaseshift, "_interp_cache"):
        get_phasemask_phaseshift._interp_cache = {}

    cache = get_phasemask_phaseshift._interp_cache

    cache_key = (str(dot_material), bool(extrapolate))

    if cache_key not in cache:
        optical_constants_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "Exposed_Ma-N_1405_optical_constants.txt"
        )

        if dot_material == "N_1405":
            df = pd.read_csv(optical_constants_path, sep=r"\s+", header=1)

            fill_value = "extrapolate" if extrapolate else np.nan

            cache[cache_key] = interp1d(
                df["Wavelength(nm)"],
                df["n"],
                kind="linear",
                bounds_error=False,
                fill_value=fill_value,
            )
        else:
            raise TypeError(
                "No corresponding dot material for given input. Try N_1405."
            )

    f_n = cache[cache_key]

    wvl = np.asarray(wvl, dtype=float)
    depth = float(depth)

    n = f_n(wvl * 1e3)  # input um -> nm
    phaseshift = 2 * np.pi / wvl * depth * (n - 1)

    return phaseshift

# def get_phasemask_phaseshift( wvl, depth, dot_material = 'N_1405', extrapolate=True ):
#     """
#     wvl is wavelength in micrometers
#     depth is the physical depth of the phasemask in micrometers
#     dot material is the material of phaseshifting object

#     it is assumed phasemask is in air (n=1).
#     N_1405 is photoresist used for making phasedots in Sydney
#     """
    

#     optical_constants_path = (
#         Path(__file__).resolve().parent.parent
#         / "data"
#         / "Exposed_Ma-N_1405_optical_constants.txt"
#     )


#     #print( 'reminder wvl input should be um!')
#     if dot_material == 'N_1405':
#         # wavelengths in csv file are in nanometers
#         #df = pd.read_csv(optical_constants_path, sep='\s+', header=1)
#         df = pd.read_csv(optical_constants_path, sep=r'\s+', header=1)
#         if extrapolate:
#             f = interp1d(df['Wavelength(nm)'], df['n'], kind='linear',bounds_error=False, fill_value="extrapolate" ) #fill_value=np.nan, bounds_error=False)
#         else:
#             f = interp1d(df['Wavelength(nm)'], df['n'], kind='linear',fill_value=np.nan, bounds_error=False)
#         n = f( wvl * 1e3 ) # convert input wavelength um - > nm
#         phaseshift = 2 * np.pi/ wvl  * depth * (n -1)
#         return( phaseshift )
    
#     else:
#         raise TypeError('No corresponding dot material for given input. Try N_1405.')



# Planck's law function for spectral radiance
def planck_law(wavelength, T):
    """Returns spectral radiance (Planck's law) at a given wavelength and temperature."""
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    return (2 * h * c**2) / (wavelength**5) / (np.exp(h * c / (wavelength * k * T)) - 1)

# Function to find the weighted average wavelength (central wavelength)
def find_central_wavelength(lambda_cut_on, lambda_cut_off, T):
    # Define integrands for energy and weighted wavelength
    def _integrand_energy(wavelength):
        return planck_law(wavelength, T)

    def _integrand_weighted(wavelength):
        return planck_law(wavelength, T) * wavelength

    # Integrate to find total energy and weighted energy
    total_energy, _ = quad(_integrand_energy, lambda_cut_on, lambda_cut_off)
    weighted_energy, _ = quad(_integrand_weighted, lambda_cut_on, lambda_cut_off)
    
    # Calculate the central wavelength as the weighted average wavelength
    central_wavelength = weighted_energy / total_energy
    return central_wavelength

def get_theoretical_reference_pupils( wavelength = 1.65e-6 ,F_number = 21.2, mask_diam = 1.2, coldstop_diam=None, eta=0, diameter_in_angular_units = True, get_individual_terms=False, phaseshift = np.pi/2 , padding_factor = 4, debug= True, analytic_solution = True ) :
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
    coldstop_diam : diameter in lambda / D of focal plane coldstop
    eta : ratio of secondary obstruction radius (r_2/r_1), where r2 is secondary, r1 is primary. 0 meams no secondary obstruction
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
    N = 2**9+1  # for parity (to not introduce tilt) works better ODD!  # Number of grid points (assumed to be square)
    L_pupil = 2 * pupil_radius  # Pupil plane size (physical dimension)
    dx_pupil = L_pupil / N  # Sampling interval in the pupil plane
    x_pupil = np.linspace(-L_pupil/2, L_pupil/2, N)   # Pupil plane coordinates
    y_pupil = np.linspace(-L_pupil/2, L_pupil/2, N) 
    X_pupil, Y_pupil = np.meshgrid(x_pupil, y_pupil)
    
    


    # Define a circular pupil function
    pupil = (np.sqrt(X_pupil**2 + Y_pupil**2) > eta*pupil_radius) & (np.sqrt(X_pupil**2 + Y_pupil**2) <= pupil_radius)

    # Zero padding to increase resolution
    # Increase the array size by padding (e.g., 4x original size)
    N_padded = N * padding_factor
    if (N % 2) != (N_padded % 2):  
        N_padded += 1  # Adjust to maintain parity
        
    pupil_padded = np.zeros((N_padded, N_padded))
    #start_idx = (N_padded - N) // 2
    #pupil_padded[start_idx:start_idx+N, start_idx:start_idx+N] = pupil

    start_idx_x = (N_padded - N) // 2
    start_idx_y = (N_padded - N) // 2  # Explicitly ensure symmetry

    pupil_padded[start_idx_y:start_idx_y+N, start_idx_x:start_idx_x+N] = pupil

    # Perform the Fourier transform on the padded array (normalizing for the FFT)
    pupil_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_padded)))
    
    # Compute the Airy disk scaling factor (1.22 * λ * F)
    airy_scale = 1.22 * wavelength * F_number

    # Image plane sampling interval (adjusted for padding)
    L_image = wavelength * F_number / dx_pupil  # Total size in the image plane
    dx_image_padded = L_image / N_padded  # Sampling interval in the image plane with padding
    

    if diameter_in_angular_units:
        x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale
    else:
        x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded)  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) 
        
    X_image_padded, Y_image_padded = np.meshgrid(x_image_padded, y_image_padded)

    if diameter_in_angular_units:
        mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 4
    else: 
        mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 4
        
    if coldstop_diam is not None:
        coldmask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= coldstop_diam / 4
    else:
        coldmask = np.ones(X_image_padded.shape)

    pupil_ft = np.fft.fft2(np.fft.ifftshift(pupil_padded))  # Remove outer fftshift
    pupil_ft = np.fft.fftshift(pupil_ft)  # Shift only once at the end

    psi_B = coldmask * pupil_ft
                            
    b = np.fft.fftshift( np.fft.ifft2( mask * psi_B ) ) 

    
    if debug: 
        
        psf = np.abs(pupil_ft)**2  # Get the PSF by taking the square of the absolute value
        psf /= np.max(psf)  # Normalize PSF intensity
        
        if diameter_in_angular_units:
            zoom_range = 3  # Number of Airy disk radii to zoom in on
        else:
            zoom_range = 3 * airy_scale 
            
        extent = (-zoom_range, zoom_range, -zoom_range, zoom_range)

        fig,ax = plt.subplots(1,1)
        ax.imshow(psf, extent=(x_image_padded.min(), x_image_padded.max(), y_image_padded.min(), y_image_padded.max()), cmap='gray')
        ax.contour(X_image_padded, Y_image_padded, mask, levels=[0.5], colors='red', linewidths=2, label='phasemask')
        #ax[1].imshow( mask, extent=(x_image_padded.min(), x_image_padded.max(), y_image_padded.min(), y_image_padded.max()), cmap='gray')
        #for axx in ax.reshape(-1):
        #    axx.set_xlim(-zoom_range, zoom_range)
        #    axx.set_ylim(-zoom_range, zoom_range)
        ax.set_xlim(-zoom_range, zoom_range)
        ax.set_ylim(-zoom_range, zoom_range)
        ax.set_title( 'PSF' )
        ax.legend() 
        #ax[1].set_title('phasemask')


    
    # if considering complex b 
    # beta = np.angle(b) # complex argunment of b 
    # M = b * (np.exp(1J*theta)-1)**0.5
    
    # relabelling
    theta = phaseshift # rad , 
    P = pupil_padded.copy() 
    
    if analytic_solution :
        
        M = abs( b ) * np.sqrt((np.cos(theta)-1)**2 + np.sin(theta)**2)
        mu = np.angle((np.exp(1J*theta)-1) ) # np.arctan( np.sin(theta)/(np.cos(theta)-1) ) #
        
        phi = np.zeros( P.shape ) # added aberrations 
        
        # out formula ----------
        #if measured_pupil!=None:
        #    P = measured_pupil / np.mean( P[P > np.mean(P)] ) # normalize by average value in Pupil
        
        Ic = ( P**2 + abs(M)**2 + 2* P* abs(M) * np.cos(phi + mu) ) #+ beta)
        if not get_individual_terms:
            return( P, Ic )
        else:
            return( P, abs(M) , phi+mu )
    else:
        
        # phasemask filter 
        
        T_on = 1
        T_off = 1
        H = T_off*(1 + (T_on/T_off * np.exp(1j * theta) - 1) * mask  ) 
        
        Ic = abs( np.fft.fftshift( np.fft.ifft2( H * psi_B ) ) ) **2 
    
        return( P, Ic )


def interpolate_pupil_to_measurement(original_pupil, original_image, M, N, m, n, x_c, y_c, new_radius):
    """
    Interpolate the pupil onto a new grid, centering the original pupil at (x_c, y_c) 
    and giving it a specified radius in the new grid.
    
    Parameters:
    - pupil: Original MxN pupil array.
    - original_image: original image (i.e intensity with phasemask in) corresponding to the pupil (phasemask out)
    - M, N: Size of the original grid.
    - n, m: Size of the new grid.
    - x_c, y_c: Center of the pupil in the new grid (in pixels).
    - new_radius: The desired radius of the pupil in the new grid (in pixels).
    
    Returns:
    - new_pupil: The pupil interpolated onto the new grid (nxm).
    """
    # Original grid coordinates (centered at the middle)
    x_orig = np.linspace(-M/2, M/2, M)
    y_orig = np.linspace(-N/2, N/2, N)
    #X_orig, Y_orig = np.meshgrid(x_orig, y_orig)
    
    # Create the new grid coordinates (centered)
    x_new = np.linspace(-m/2, m/2, m)  # New grid should also be centered
    y_new = np.linspace(-n/2, n/2, n)
    X_new, Y_new = np.meshgrid(x_new, y_new)

    # Find the actual radius of the original pupil in terms of grid size (not M/2)
    orig_radius = np.sum( original_pupil/np.pi )**0.5 #np.sqrt((X_orig**2 + Y_orig**2).max())

    # Map new grid coordinates to the original grid
    scale_factor = new_radius / orig_radius  # Correct scaling factor based on actual original radius
    X_new_mapped = (X_new - x_c + m/2) / scale_factor + M/2
    Y_new_mapped = (Y_new - y_c + n/2) / scale_factor + N/2

    # Perform interpolation using map_coordinates
    new_pupil = ndimage.map_coordinates(original_image, [Y_new_mapped.ravel(), X_new_mapped.ravel()], order=1, mode='constant', cval=0)
    
    # Reshape the interpolated result to the new grid size
    new_pupil = new_pupil.reshape(n, m)

    return new_pupil

def crop_pupil(pupil, image):
    """
    Detects the boundary of a pupil in a binary mask (with pupil = 1 and background = 0)
    and crops both the pupil mask and the corresponding image to contain just the pupil.
    
    Parameters:
    - pupil: A 2D NumPy array (binary) representing the pupil (1 inside the pupil, 0 outside).
    - image: A 2D NumPy array of the same shape as 'pupil' representing the image to be cropped.
    
    Returns:
    - cropped_pupil: The cropped pupil mask.
    - cropped_image: The cropped image based on the pupil's bounding box.
    """
    # Ensure both arrays have the same shape
    if pupil.shape != image.shape:
        raise ValueError("Pupil and image must have the same dimensions.")

    # Sum along the rows (axis=1) to find the non-zero rows (pupil region)
    row_sums = np.sum(pupil, axis=1)
    non_zero_rows = np.where(row_sums > 0)[0]

    # Sum along the columns (axis=0) to find the non-zero columns (pupil region)
    col_sums = np.sum(pupil, axis=0)
    non_zero_cols = np.where(col_sums > 0)[0]

    # Get the bounding box of the pupil by identifying the min and max indices
    row_start, row_end = non_zero_rows[0], non_zero_rows[-1] + 1
    col_start, col_end = non_zero_cols[0], non_zero_cols[-1] + 1

    # Crop both the pupil and the image
    cropped_pupil = pupil[row_start:row_end, col_start:col_end]
    cropped_image = image[row_start:row_end, col_start:col_end]

    return cropped_pupil, cropped_image



def get_secondary_mask(image, center):
    """
    Create a boolean mask with the same shape as `image` that is True 
    for a 3x3 patch centered at the given (x,y) coordinate (floats)
    and False elsewhere. x,y is rounded to nearet integer

    Designed for identifying pixels shaddowed by secondary obstruction. 
    Use detect_pupil() function to get the center! 
    
    Parameters:
        image (np.ndarray): 2D array (image).
        center (tuple): (x, y) coordinate (floats) of the patch center.
                        x is column, y is row.
    
    Returns:
        mask (np.ndarray): Boolean mask array with True in the 3x3 patch.
    """
    # Initialize a boolean mask of the same shape as the image with all False
    mask = np.zeros_like(image, dtype=bool)
    
    # Unpack the center coordinates and round to nearest integer
    x, y = center
    col = int(round(x))
    row = int(round(y))
    
    # Set the 3x3 patch to True.
    # Note: This simple example assumes the patch is fully contained in the image.
    mask[row-1:row+2, col-1:col+2] = True
    
    return mask


def filter_exterior_annulus(pupil_mask, inner_radius, outer_radius):
    """
    Generate a boolean mask that filters pixels exterior to the circular pupil
    but within the specified inner and outer radii.
    """
    # Get the image shape
    ny, nx = pupil_mask.shape

    # Compute the pupil center (mean of True pixels)
    y_indices, x_indices = np.where(pupil_mask)
    center_x = np.mean(x_indices)
    center_y = np.mean(y_indices)

    # Generate a coordinate grid
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    # Compute the Euclidean distance of each pixel from the pupil center
    distance_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    # Create an annular mask where pixels are within the given inner and outer radius
    annular_mask = (distance_from_center >= inner_radius) & (distance_from_center <= outer_radius)

    return annular_mask


def detect_pupil(image, sigma=2, threshold=0.5, plot=True, savepath=None):
    """
    Detects an elliptical pupil (with possible rotation) in a cropped image using edge detection 
    and least-squares fitting. Returns both the ellipse parameters and a pupil mask.

    The ellipse is modeled by:

        ((x - cx)*cos(theta) + (y - cy)*sin(theta))^2 / a^2 +
        (-(x - cx)*sin(theta) + (y - cy)*cos(theta))^2 / b^2 = 1

    Parameters:
        image (2D array): Cropped grayscale image containing a single pupil.
        sigma (float): Standard deviation for Gaussian smoothing.
        threshold (float): Threshold factor for edge detection.
        plot (bool): If True, displays the image with the fitted ellipse overlay.
        savepath (str): If provided, the plot is saved to this path.

    Returns:
        (center_x, center_y, a, b, theta, pupil_mask)
          where (center_x, center_y) is the ellipse center,
                a and b are the semimajor and semiminor axes,
                theta is the rotation angle in radians,
                pupil_mask is a 2D boolean array (True = inside ellipse).
    """
    # Normalize the image
    image = image / image.max()
    
    # Smooth the image
    smoothed_image = gaussian_filter(image, sigma=sigma)
    
    # Compute gradients (Sobel-like edge detection)
    grad_x = np.gradient(smoothed_image, axis=1)
    grad_y = np.gradient(smoothed_image, axis=0)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold edges to create a binary mask
    binary_edges = edges > (threshold * edges.max())
    
    # Get edge pixel coordinates
    y_coords, x_coords = np.nonzero(binary_edges)
    
    # Initial guess: center from mean, radius from average distance, and theta = 0.
    def initial_guess(x, y):
        center_x = np.mean(x)
        center_y = np.mean(y)
        r_init = np.sqrt(np.mean((x - center_x)**2 + (y - center_y)**2))
        return center_x, center_y, r_init, r_init, 0.0  # (cx, cy, a, b, theta)
    
    # Ellipse model function with rotation.
    def ellipse_model(params, x, y):
        cx, cy, a, b, theta = params
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_shift = x - cx
        y_shift = y - cy
        xp =  cos_t * x_shift + sin_t * y_shift
        yp = -sin_t * x_shift + cos_t * y_shift
        # Model: xp^2/a^2 + yp^2/b^2 = 1 => residual = sqrt(...) - 1
        return np.sqrt((xp/a)**2 + (yp/b)**2) - 1.0

    # Fit via least squares.
    guess = initial_guess(x_coords, y_coords)
    result, _ = leastsq(ellipse_model, guess, args=(x_coords, y_coords))
    center_x, center_y, a, b, theta = result
    
    # Create a boolean pupil mask for the fitted ellipse
    yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_shift = xx - center_x
    y_shift = yy - center_y
    xp = cos_t * x_shift + sin_t * y_shift
    yp = -sin_t * x_shift + cos_t * y_shift
    pupil_mask = (xp/a)**2 + (yp/b)**2 <= 1

    if plot:
        # Overlay for visualization
        overlay = np.zeros_like(image)
        overlay[pupil_mask] = 1
        
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap="gray", origin="upper")
        plt.contour(binary_edges, colors="cyan", linewidths=1)
        plt.contour(overlay, colors="red", linewidths=1)
        plt.scatter(center_x, center_y, color="blue", marker="+")
        plt.title("Detected Pupil with Fitted Ellipse")
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()
    
    return center_x, center_y, a, b, theta, pupil_mask



def create_phase_screen_cmd_for_DM(scrn,  scaling_factor=0.1, drop_indicies = None, plot_cmd=False):
    """
    aggregate a scrn (aotools.infinitephasescreen object) onto a DM command space. phase screen is normalized by
    between +-0.5 and then scaled by scaling_factor. Final DM command values should
    always be between -0.5,0.5 (this should be added to a flat reference so flat reference + phase screen should always be bounded between 0-1). phase screens are usually a NxN matrix, while DM is MxM with some missing pixels (e.g. 
    corners). drop_indicies is a list of indicies in the flat MxM DM array that should not be included in the command space. 
    """

    #print('----------\ncheck phase screen size is multiple of DM\n--------')
    
    Nx_act = 12 #number of actuators across DM diameter
    
    scrn_array = ( scrn.scrn - np.min(scrn.scrn) ) / (np.max(scrn.scrn) - np.min(scrn.scrn)) - 0.5 # normalize phase screen between -0.5 - 0.5 
    
    size_factor = int(scrn_array.shape[0] / Nx_act) # how much bigger phase screen is to DM shape in x axis. Note this should be an integer!!
    
    # reshape screen so that axis 1,3 correspond to values that should be aggregated 
    scrn_to_aggregate = scrn_array.reshape(scrn_array.shape[0]//size_factor, size_factor, scrn_array.shape[1]//size_factor, size_factor)
    
    # now aggreagate and apply the scaling factor 
    scrn_on_DM = scaling_factor * np.mean( scrn_to_aggregate, axis=(1,3) ).reshape(-1) 

    #If DM is missing corners etc we set these to nan and drop them before sending the DM command vector
    #dm_cmd =  scrn_on_DM.to_list()
    if drop_indicies is not None:
        for i in drop_indicies:
            scrn_on_DM[i]=np.nan
             
    if plot_cmd: #can be used as a check that the command looks right!
        fig,ax = plt.subplots(1,2,figsize=(12,6))
        im0 = ax[0].imshow( scrn_on_DM.reshape([Nx_act,Nx_act]) )
        ax[0].set_title('DM command (averaging offset)')
        im1 = ax[1].imshow(scrn.scrn)
        ax[1].set_title('original phase screen')
        plt.colorbar(im0, ax=ax[0])
        plt.colorbar(im1, ax=ax[1]) 
        plt.show() 

    dm_cmd =  list( scrn_on_DM[np.isfinite(scrn_on_DM)] ) #drop non-finite values which should be nan values created from drop_indicies array
    return(dm_cmd) 





def magnitude_to_photon_flux(magnitude, band, wavelength):
    """
    Convert stellar magnitude in a given band to photon flux (photons / s / m^2 / nm).
    
    ***EXPERIMENTAL  - need to verify results 
    
    Parameters:
    - magnitude: The magnitude of the star.
    - band: The name of the filter (e.g., 'V', 'J', 'H').
    - wavelength: The central wavelength of the filter in nm.
    
    Returns:
    - photon_flux: The number of photons per second per square meter per nanometer.
    """

    from astropy.constants import h, c
    # Zero points in energy flux for different bands (in erg/s/cm^2/Å)
    zero_point_flux = {
        'V': 3.63e-9 * u.erg / (u.cm**2 * u.s * u.AA),  # V-band zero point
        'J': 3.13e-10 * u.erg / (u.cm**2 * u.s * u.AA), # J-band zero point
        'H': 1.16e-10 * u.erg / (u.cm**2 * u.s * u.AA), # H-band zero point
        # Add more bands as needed
    }
    
    if band not in zero_point_flux:
        raise ValueError(f"Unknown band: {band}. Available bands are {list(zero_point_flux.keys())}")
    
    # Convert magnitude to energy flux density (f_lambda in erg/s/cm^2/Å)
    f_lambda = zero_point_flux[band] * 10**(-0.4 * magnitude)
    
    # Convert wavelength to meters
    wavelength_m = (wavelength * u.nm).to(u.m)
    
    # Convert energy flux density to W/m^2/nm
    f_lambda_si = f_lambda.to(u.W / (u.m**2 * u.nm), equivalencies=u.spectral_density(wavelength_m))
    
    # Calculate the energy per photon (in joules) at the given wavelength
    energy_per_photon = (h * c / wavelength_m).to(u.J)  # Energy per photon at this wavelength
    
    # Calculate photon flux (photons/s/m^2/nm)
    photon_flux = f_lambda_si / energy_per_photon.value  # Explicitly divide by the scalar value of energy_per_photon
    
    # Return photon flux in the appropriate units (photon/s/m^2/nm)
    return photon_flux.value





###### MODELLING MIRROR SCRATCHES


def apply_parabolic_scratches(array, dx, dy, list_a, list_b, list_c, width_list, depth_list):
    """
    Apply multiple parabolic scratches to a 2D array based on input parameters.

    Parameters:
    array (2D numpy array): The input 2D array to which the scratches will be applied.
    dx (float): Pixel scale in the x direction.
    dy (float): Pixel scale in the y direction.
    list_a, list_b, list_c (lists of floats): Lists of a, b, c coefficients for each parabola (y = a*x^2 + b*x + c).
    width_list (list of floats): List of widths for each scratch around the parabolic contour.
    depth_list (list of floats): List of depths for each scratch.

    Returns:
    Modified 2D numpy array with the parabolic scratches applied.
    """
    num_pixels_y, num_pixels_x = array.shape

    # Generate x and y coordinates corresponding to pixel locations
    x_vals = np.linspace(-num_pixels_x/2 * dx, num_pixels_x/2 * dx, num_pixels_x)
    y_vals = np.linspace(-num_pixels_y/2 * dy, num_pixels_y/2 * dy, num_pixels_y)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Apply each parabolic scratch
    for a, b, c, width, depth in zip(list_a, list_b, list_c, width_list, depth_list):
        # Compute the parabolic contour y = a*x^2 + b*x + c for each scratch
        parabolic_curve_y = a * X**2 + b * X + c
        
        # Apply the scratch around the parabolic curve with constant depth and width
        for i in range(num_pixels_y):
            for j in range(num_pixels_x):
                # Compute the distance to the parabolic contour
                distance_to_parabola = np.abs(Y[i, j] - parabolic_curve_y[i, j])
                
                # If the point is within the width of the scratch, modify the array
                if distance_to_parabola <= width / 2:
                    array[i, j] -= depth

    return array

# # Example usage
# num_pixels_x, num_pixels_y = 100, 100  # Size of the array (100x100)
# dx, dy = 0.1, 0.1  # Pixel scale in the x and y directions
# input_array = np.full((num_pixels_y, num_pixels_x), 10)  # Constant background array

# # Lists of parabolic parameters, widths, and depths for the scratches
# list_a = [0.5, 0.7]
# list_b = [0, 0]
# list_c = [0, 2]
# width_list = [0.5, 0.2]  # Width of the scratches
# depth_list = [2, 3]  # Depth of the scratches

# # Apply the scratches
# modified_array = apply_parabolic_scratches(input_array, dx, dy, list_a, list_b, list_c, width_list, depth_list)

# # Visualize the result
# import matplotlib.pyplot as plt
# plt.imshow(modified_array, cmap='hot', extent=[-num_pixels_x/2*dx, num_pixels_x/2*dx, -num_pixels_y/2*dy, num_pixels_y/2*dy])
# plt.colorbar(label='Value')
# plt.title('2D Array with Multiple Parabolic Scratches')
# plt.show()





#### PLOTTING 


def nice_heatmap_subplots( im_list , xlabel_list=None, ylabel_list=None, title_list=None, cbar_label_list=None, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None):

    n = len(im_list)
    fs = fontsize
    fig = plt.figure(figsize=(5*n, 5))

    for a in range(n) :
        ax1 = fig.add_subplot(int(f'1{n}{a+1}'))

        if vlims is not None:
            im1 = ax1.imshow(  im_list[a] , vmin = vlims[a][0], vmax = vlims[a][1])
        else:
            im1 = ax1.imshow(  im_list[a] )
        if title_list is not None:
            ax1.set_title( title_list[a] ,fontsize=fs)
        if xlabel_list is not None:
            ax1.set_xlabel( xlabel_list[a] ,fontsize=fs) 
        if ylabel_list is not None:
            ax1.set_ylabel( ylabel_list[a] ,fontsize=fs) 
        ax1.tick_params( labelsize=fs ) 

        if axis_off:
            ax1.axis('off')
        divider = make_axes_locatable(ax1)
        if cbar_orientation == 'bottom':
            cax = divider.append_axes('bottom', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
                
        elif cbar_orientation == 'top':
            cax = divider.append_axes('top', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
                
        else: # we put it on the right 
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='vertical')  
        
        if cbar_label_list is not None:
            cbar.set_label( cbar_label_list[a], rotation=0,fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)
    if savefig is not None:
        plt.savefig( savefig , bbox_inches='tight', dpi=300) 

    #plt.show()
    
    
def nice_heatmap_subplots_3rows(
    im_grid,                       # list of rows; each row is list of images (2D arrays)
    row_titles=None,               # e.g. ["TT recon", "HO recon", "Residual (inj - recon)"]
    col_titles=None,               # e.g. ["injection", "command", "intensity", "None"]
    fontsize=16,
    axis_off=True,
    cbar_orientation="bottom",     # "bottom" | "top" | "right"
    vlims_row=None,                # list length nrows: [(vmin,vmax), ...] OR None
    vlims_row_mode="manual",       # "manual" | "std"  (if "std": vlim = +-nsig*std over that row)
    nsig=2.0,
    cmap="viridis",
    savefig=None,
    dpi=300,
    figscale=5.0,                  # controls overall size
):
    """
    3-row (or general n-row) heatmap grid with:
      - same vlims across columns within each row
      - different vlims between rows
      - per-panel colorbars (like your existing helper)
    """
    nrows = len(im_grid)
    if nrows < 1:
        raise ValueError("im_grid must have at least 1 row")
    ncols = len(im_grid[0])
    if any(len(r) != ncols for r in im_grid):
        raise ValueError("All rows in im_grid must have the same number of columns")

    fs = fontsize
    fig = plt.figure(figsize=(figscale * ncols, figscale * nrows))

    # --- decide row-wise vlims ---
    if vlims_row_mode.lower() == "std":
        vlims_row_eff = []
        for r in range(nrows):
            vals = np.concatenate([np.asarray(im_grid[r][c]).ravel() for c in range(ncols)])
            s = np.std(vals)
            vlims_row_eff.append((-nsig * s, nsig * s))
    else:
        vlims_row_eff = vlims_row

    if vlims_row_eff is None:
        vlims_row_eff = [None] * nrows
    if len(vlims_row_eff) != nrows:
        raise ValueError(f"vlims_row must have length nrows={nrows}, got {len(vlims_row_eff)}")

    # --- plot ---
    for r in range(nrows):
        for c in range(ncols):
            ax = fig.add_subplot(nrows, ncols, r * ncols + c + 1)

            vlim = vlims_row_eff[r]
            if vlim is None:
                im = ax.imshow(im_grid[r][c], cmap=cmap)
            else:
                im = ax.imshow(im_grid[r][c], vmin=vlim[0], vmax=vlim[1], cmap=cmap)

            # column titles only on first row
            if (col_titles is not None) and (r == 0):
                ax.set_title(col_titles[c], fontsize=fs)

            # row titles only on first column
            if (row_titles is not None) and (c == 0):
                # Put row title on y-label for nice alignment
                ax.set_ylabel(row_titles[r], fontsize=fs)

            ax.tick_params(labelsize=fs)
            if axis_off:
                ax.axis("off")

            divider = make_axes_locatable(ax)
            if cbar_orientation == "bottom":
                cax = divider.append_axes("bottom", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            elif cbar_orientation == "top":
                cax = divider.append_axes("top", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            else:  # right
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation="vertical")

            cbar.ax.tick_params(labelsize=fs)

    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight", dpi=dpi)

    return fig

def nice_DM_plot( data, savefig=None ): #for a 140 actuator BMC 3.5 DM
    fig,ax = plt.subplots(1,1)
    if len( np.array(data).shape ) == 1: 
        ax.imshow( get_DM_command_in_2D(data) )
    else: 
        ax.imshow( data )
    #ax.set_title('poorly registered actuators')
    ax.grid(True, which='minor',axis='both', linestyle='-', color='k', lw=2 )
    ax.set_xticks( np.arange(12) - 0.5 , minor=True)
    ax.set_yticks( np.arange(12) - 0.5 , minor=True)
    if savefig is not None:
        plt.savefig( savefig , bbox_inches='tight', dpi=300) 



def plot_data_and_residuals(x, y_meas, y_model, xlabel, ylabel, residual_ylabel, label_1=None, label_2=None, savefig=None):
    # Calculate residuals
    residuals = y_meas - y_model

    # Create a figure with two subplots: one for the data and one for the residuals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # First subplot: measured and modeled data
    if label_1 is None:
        ax1.plot(x, y_meas, '-', label='Measured Data', color='blue', markersize=2, alpha =0.3)
    else: 
        ax1.plot(x, y_meas, '-', label=label_1, color='blue', markersize=2, alpha =0.3)
    if label_2 is None:    
        ax1.plot(x, y_model, '.', label='Modeled Data', color='red', linewidth=2, alpha =0.3)
    else:
        ax1.plot(x, y_model, '.', label=label_2, color='red', linewidth=2, alpha =0.3)
    #ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.grid(True)

    # Second subplot: residuals
    ax2.plot(x, residuals, '.', color='green', markersize=5, alpha =0.3)
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')  # Horizontal line at zero
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(residual_ylabel)
    ax2.grid(True)

    # Adjust layout for better spacing between subplots
    plt.tight_layout()
    
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight', dpi=300  )
    # Show the plot
    plt.show()
    
    

def create_telem_mosaic(image_list, image_title_list, image_colorbar_list, 
                  plot_list, plot_title_list, plot_xlabel_list, plot_ylabel_list):
    """
    Creates a 3-row mosaic layout with:
    - First row: images with colorbars below
    - Second and third rows: plots with titles and axis labels
    
    Parameters:
    - image_list: List of image data for the first row (4 images)
    - image_title_list: List of titles for the first row images
    - image_colorbar_list: List of colorbars (True/False) for each image in the first row
    - plot_list: List of plot data for second and third rows (4 plots, 2 per row)
    - plot_title_list: List of titles for each plot
    - plot_xlabel_list: List of x-axis labels for each plot
    - plot_ylabel_list: List of y-axis labels for each plot
    """
    
    # Create a figure with constrained layout and extra padding
    fig = plt.figure(constrained_layout=True, figsize=(10, 8))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Create GridSpec with 3 rows and different numbers of columns
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1])
    
    # Top row: 4 columns with colorbars
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        img = image_list[i]
        im = ax.imshow(img, cmap='viridis')  # Modify colormap if needed
        ax.set_title(image_title_list[i])
        
        # Optionally add a colorbar below the image
        if image_colorbar_list[i]:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='5%', pad=0.2)
            fig.colorbar(im, cax=cax, orientation='horizontal')
    
    # Middle row: 2 columns, each spanning 2 grid columns
    for i in range(2):
        ax = fig.add_subplot(gs[1, 2*i:2*i+2])
        data = plot_list[i]
        ax.plot(data)
        ax.set_title(plot_title_list[i])
        ax.set_xlabel(plot_xlabel_list[i])
        ax.set_ylabel(plot_ylabel_list[i])

    # Bottom row: 2 columns, each spanning 2 grid columns
    for i in range(2, 4):
        ax = fig.add_subplot(gs[2, 2*(i-2):2*(i-2)+2])
        data = plot_list[i]
        
        ax.plot(data)
        ax.set_title(plot_title_list[i])
        ax.set_xlabel(plot_xlabel_list[i])
        ax.set_ylabel(plot_ylabel_list[i])
    
    # Show the plot
    plt.show()



def plot_eigenmodes( zwfs_ns , save_path = None ):
    
    tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

    U,S,Vt = np.linalg.svd( zwfs_ns.reco.IM, full_matrices=True)

    #singular values
    plt.figure() 
    plt.semilogy(S) #/np.max(S))
    #plt.axvline( np.pi * (10/2)**2, color='k', ls=':', label='number of actuators in pupil')
    plt.legend() 
    plt.xlabel('mode index')
    plt.ylabel('singular values')

    if save_path is not None:
        plt.savefig(save_path +  f'singularvalues_{tstamp}.png', bbox_inches='tight', dpi=200)
    plt.show()
    
    # THE IMAGE MODES 
    n_row = round( np.sqrt( zwfs_ns.reco.M2C_0.shape[0]) ) - 1
    fig,ax = plt.subplots(n_row  ,n_row ,figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        # we filtered circle on grid, so need to put back in grid
        tmp =  zwfs_ns.pupil_regions.pupil_filt.copy()
        vtgrid = np.zeros(tmp.shape)
        vtgrid[tmp] = Vt[i]
        r1,r2,c1,c2 = 10,-10,10,-10
        axx.imshow( vtgrid.reshape(zwfs_ns.reco.I0.shape )[r1:r2,c1:c2] ) #cp_x2-cp_x1,cp_y2-cp_y1) )
        #axx.set_title(f'\n\n\nmode {i}, S={round(S[i]/np.max(S),3)}',fontsize=5)
        #
        axx.text( 10,10, f'{i}',color='w',fontsize=4)
        axx.text( 10,20, f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=4)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path + f'det_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=200)
    plt.show()
    
    # THE DM MODES 

    # NOTE: if not zonal (modal) i might need M2C to get this to dm space 
    # if zonal M2C is just identity matrix. 
    fig,ax = plt.subplots(n_row, n_row, figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        axx.imshow( get_DM_command_in_2D( zwfs_ns.reco.M2C_0.T @ U.T[i] ) )
        #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
        axx.text( 1,2,f'{i}',color='w',fontsize=6)
        axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path +  f'dm_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=200)
    plt.show()




def display_images_with_slider(image_lists, plot_titles=None, cbar_labels=None, row_col = None):
    """
    Displays multiple images or 1D plots from a list of lists with a slider to control the shared index.
    
    Parameters:
    - image_lists: list of lists where each inner list contains either 2D arrays (images) or 1D arrays (scalars).
                   The inner lists must all have the same length.
    - plot_titles: list of strings, one for each subplot. Default is None (no titles).
    - cbar_labels: list of strings, one for each colorbar. Default is None (no labels).
    """
    
    # Check that all inner lists have the same length
    assert all(len(lst) == len(image_lists[0]) for lst in image_lists), "All inner lists must have the same length."
    
    # Number of rows and columns based on the number of plots
    if row_col is None:
        num_plots = len(image_lists)
        ncols = math.ceil(math.sqrt(num_plots))  # Number of columns for grid
        nrows = math.ceil(num_plots / ncols)     # Number of rows for grid
    else:
        num_plots = len(image_lists)
        try:
            nrows, ncols = row_col
        except:
            raise UserWarning('cannot cast "nrows, ncols = row_col"')
        
        assert nrows * ncols == num_plots 
        
    num_frames = len(image_lists[0])

    # Create figure and axes
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    plt.subplots_adjust(bottom=0.2)

    # Flatten axes array for easier iteration
    axes = axes.flatten() if num_plots > 1 else [axes]

    # Store the display objects for each plot (either imshow or line plot)
    img_displays = []
    line_displays = []
    
    # Get max/min values for 1D arrays to set static axis limits
    max_values = [max(lst) if not isinstance(lst[0], np.ndarray) else None for lst in image_lists]
    min_values = [min(lst) if not isinstance(lst[0], np.ndarray) else None for lst in image_lists]

    for i, ax in enumerate(axes[:num_plots]):  # Only iterate over the number of plots
        # Check if the first item in the list is a 2D array (an image) or a scalar
        if isinstance(image_lists[i][0], np.ndarray) and image_lists[i][0].ndim == 2:
            # Use imshow for 2D data (images)
            img_display = ax.imshow(image_lists[i][0], cmap='viridis')
            img_displays.append(img_display)
            line_displays.append(None)  # Placeholder for line plots
            
            # Add colorbar if it's an image
            cbar = fig.colorbar(img_display, ax=ax)
            if cbar_labels and i < len(cbar_labels) and cbar_labels[i] is not None:
                cbar.set_label(cbar_labels[i])

        else:
            # Plot the list of scalar values up to the initial index
            line_display, = ax.plot(np.arange(len(image_lists[i])), image_lists[i], color='b')
            line_display.set_data(np.arange(1), image_lists[i][:1])  # Start with only the first value
            ax.set_xlim(0, len(image_lists[i]))  # Set x-axis to full length of the data
            ax.set_ylim(min_values[i], max_values[i])  # Set y-axis to cover the full range
            line_displays.append(line_display)
            img_displays.append(None)  # Placeholder for image plots

        # Set plot title if provided
        if plot_titles and i < len(plot_titles) and plot_titles[i] is not None:
            ax.set_title(plot_titles[i])

    # Remove any unused axes
    for ax in axes[num_plots:]:
        ax.remove()

    # Slider for selecting the frame index
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    frame_slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        index = int(frame_slider.val)  # Get the selected index from the slider
        for i, (img_display, line_display) in enumerate(zip(img_displays, line_displays)):
            if img_display is not None:
                # Update the image data for 2D data
                img_display.set_data(image_lists[i][index])
            if line_display is not None:
                # Update the line plot for scalar values (plot up to the selected index)
                line_display.set_data(np.arange(index), image_lists[i][:index])
        fig.canvas.draw_idle()  # Redraw the figure

    # Connect the slider to the update function
    frame_slider.on_changed(update)

    plt.show()



def display_images_as_movie(image_lists, plot_titles=None, cbar_labels=None, save_path="output_movie.mp4", fps=5,row_col = None):
    """
    Creates an animation from multiple images or 1D plots from a list of lists and saves it as a movie.
    
    Parameters:
    - image_lists: list of lists where each inner list contains either 2D arrays (images) or 1D arrays (scalars).
                   The inner lists must all have the same length.
    - plot_titles: list of strings, one for each subplot. Default is None (no titles).
    - cbar_labels: list of strings, one for each colorbar. Default is None (no labels).
    - save_path: path where the output movie will be saved.
    - fps: frames per second for the output movie.
    """
    
    # Check that all inner lists have the same length
    assert all(len(lst) == len(image_lists[0]) for lst in image_lists), "All inner lists must have the same length."
    
    # # Number of rows and columns based on the number of plots
    # num_plots = len(image_lists)
    # ncols = math.ceil(math.sqrt(num_plots))  # Number of columns for grid
    # nrows = math.ceil(num_plots / ncols)     # Number of rows for grid
    
    # num_frames = len(image_lists[0])

    # Number of rows and columns based on the number of plots
    if row_col is None:
        num_plots = len(image_lists)
        ncols = math.ceil(math.sqrt(num_plots))  # Number of columns for grid
        nrows = math.ceil(num_plots / ncols)     # Number of rows for grid
    else:
        num_plots = len(image_lists)
        try:
            nrows, ncols = row_col
        except:
            raise UserWarning('cannot cast "nrows, ncols = row_col"')
        
        assert nrows * ncols == num_plots 
        
    num_frames = len(image_lists[0])


    # Create figure and axes
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
    plt.subplots_adjust(bottom=0.2)

    # Flatten axes array for easier iteration
    axes = axes.flatten() if num_plots > 1 else [axes]

    # Store the display objects for each plot (either imshow or line plot)
    img_displays = []
    line_displays = []
    
    # Get max/min values for 1D arrays to set static axis limits
    max_values = [max(lst) if not isinstance(lst[0], np.ndarray) else None for lst in image_lists]
    min_values = [min(lst) if not isinstance(lst[0], np.ndarray) else None for lst in image_lists]

    for i, ax in enumerate(axes[:num_plots]):  # Only iterate over the number of plots
        # Check if the first item in the list is a 2D array (an image) or a scalar
        if isinstance(image_lists[i][0], np.ndarray) and image_lists[i][0].ndim == 2:
            # Use imshow for 2D data (images)
            img_display = ax.imshow(image_lists[i][0], cmap='viridis')
            img_displays.append(img_display)
            line_displays.append(None)  # Placeholder for line plots
            
            # Add colorbar if it's an image
            cbar = fig.colorbar(img_display, ax=ax)
            if cbar_labels and i < len(cbar_labels) and cbar_labels[i] is not None:
                cbar.set_label(cbar_labels[i])

        else:
            # Plot the list of scalar values up to the initial index
            line_display, = ax.plot(np.arange(len(image_lists[i])), image_lists[i], color='b')
            line_display.set_data(np.arange(1), image_lists[i][:1])  # Start with only the first value
            ax.set_xlim(0, len(image_lists[i]))  # Set x-axis to full length of the data
            ax.set_ylim(min_values[i], max_values[i])  # Set y-axis to cover the full range
            line_displays.append(line_display)
            img_displays.append(None)  # Placeholder for image plots

        # Set plot title if provided
        if plot_titles and i < len(plot_titles) and plot_titles[i] is not None:
            ax.set_title(plot_titles[i])

    # Remove any unused axes
    for ax in axes[num_plots:]:
        ax.remove()

    # Function to update the frames
    def update_frame(frame_idx):
        for i, (img_display, line_display) in enumerate(zip(img_displays, line_displays)):
            if img_display is not None:
                # Update the image data for 2D data
                img_display.set_data(image_lists[i][frame_idx])
            if line_display is not None:
                # Update the line plot for scalar values (plot up to the current index)
                line_display.set_data(np.arange(frame_idx), image_lists[i][:frame_idx])
        return img_displays + line_displays

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, blit=False, repeat=False)

    # Save the animation as a movie file
    ani.save(save_path, fps=fps, writer='ffmpeg')

    plt.show()



#######
# Some utils for updating scintillation (used originally in ASGARD/paranal_onsky_comissioning/simulation_experiments/)
def upsample_by_factor(ar: np.ndarray, f: int | tuple[int, int]) -> np.ndarray:
    """
    Upsample 2D array by integer factor(s) via block replication (nearest-neighbour).
    If f is an int, uses the same factor on both axes. If f=(fy, fx), uses per-axis factors.
    """
    ar = np.asarray(ar)
    if ar.ndim != 2:
        raise ValueError("ar must be 2D")
    fy, fx = (f, f) if isinstance(f, int) else f
    if fy < 1 or fx < 1:
        raise ValueError("factors must be positive integers")
    return np.repeat(np.repeat(ar, fy, axis=0), fx, axis=1)


def pad_to_shape_edge(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Grow a 2D array to target_shape by copying its edges (last row/col).
    Equivalent to np.pad(..., mode='edge').
    """
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    m, n = arr.shape
    M, N = target_shape
    if M < m or N < n:
        raise ValueError("target_shape must be >= current shape in both dims")

    pad_rows = M - m
    pad_cols = N - n
    return np.pad(arr, ((0, pad_rows), (0, pad_cols)), mode="edge")


def upsample( ar, target_size ):
    out_almost = upsample_by_factor(ar, target_size//len(ar) ) # 
    if np.mod( target_size, len(ar) ) != 0: # not even divisor, we just pad! 
        out = pad_to_shape_edge( out_almost , target_shape = (target_size,target_size))
    else :
        out = out_almost
    return( out )


# e.g. 
# def update_scintillation( high_alt_phasescreen , pxl_scale, wavelength, final_size = None,jumps = 1,propagation_distance=10000):
#     for _ in range(jumps):
#         high_alt_phasescreen.add_row()
#     wavefront = np.exp(1J *  high_alt_phasescreen.scrn ) # amplitude mean ~ 1 
#     propagated_screen = aotools.opticalpropagation.angularSpectrum(inputComplexAmp=wavefront,
#                                                                z=propagation_distance, 
#                                                                wvl=wavelength, 
#                                                                inputSpacing = pxl_scale, 
#                                                                outputSpacing = pxl_scale
#                                                                )
#     #print("upsample it scintillation screen")
#     if final_size is not None:
#         amp = util.upsample(propagated_screen, final_size ) # This oversamples to nearest multiple size, and then pads the rest with repeated rows, not the most accurate but fastest. Negligible if +1 from even number
#     else:
#         amp = propagated_screen

#     return( abs(amp) ) # amplitude of field, not intensity (amp^2)! rotate 90 degrees so not correlated with phase 




#######
# for fitting OPD model from aggregated ZWFS pixels exterior from pupil (but can be used generally)
def compute_correlation_map(intensity_frames, strehl_ratios):
    # intensity_frames: k x N x M array (k frames of N x M pixels)
    # strehl_ratios: k array (Strehl ratio for each frame)
    
    k, N, M = intensity_frames.shape
    correlation_map = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            pixel_intensity_series = intensity_frames[:, i, j]
            correlation_map[i, j], _ = pearsonr(pixel_intensity_series, strehl_ratios)
    
    return correlation_map


def piecewise_continuous(x, interc, slope_1, slope_2, x_knee):
    # piecewise linear (hinge) model 
    return interc + slope_1 * x + slope_2 * np.maximum(0.0, x - x_knee)


def fit_piecewise_continuous(x, y, n_grid=60, trim=0.1): 
    # fits a piecewise linear (hinge) model , typically used for OPD model from ZWFS pixels exterior to pupil

    def _fit_hinge_given_x0(x, y, x0):
        h = np.maximum(0.0, x - x0)

        # linear in (a, b, d): y ≈ a + b*x + d*h
        A = np.column_stack([np.ones_like(x), x, h])

        # robust solve via least_squares on linear params
        def resid(p):
            return (A @ p) - y

        p0 = np.linalg.lstsq(A, y, rcond=None)[0]
        res = least_squares(resid, p0, loss="soft_l1", f_scale=np.std(y) + 1e-12)
        a, b, d = res.x
        return a, b, d, res.cost ##return a, b, d, np.mean(res.fun**2) #<- this could be more robust?

    def _fit_hinge_gridsearch(x, y, n_grid=60, trim=0.1):
        # grid search to fit a reasonable knee point 
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        # candidate x0 values within central range (avoid extremes)
        xs = np.sort(x)
        lo = xs[int(trim * len(xs))]
        hi = xs[int((1-trim) * len(xs))]

        x0_grid = np.linspace(lo, hi, n_grid)

        best = None
        for x0 in x0_grid:
            a, b, d, cost = _fit_hinge_given_x0(x, y, x0)
            if (best is None) or (cost < best["cost"]):
                #best = dict(a=a, b_left=b, b_right=b+d, d=d, x0=x0, cost=cost)
                best = dict(
                            interc=a,
                            slope_1=b,
                            slope_2=d,
                            #slope_left=b,
                            #slope_right=b + d,
                            x_knee=x0,
                            cost=cost,
                        )
        return best

    params = _fit_hinge_gridsearch(x,y, n_grid=80, trim=0.15)

    return params



def lucky_img( I0_meas,  image_processing_fn, performance_model, model_params,  quantile_threshold = 0.05 , keep = "<threshold" ):
    """
    Select a subset of reference ZWFS pupil intensities based on a
    performance metric (e.g. OPD, Strehl), using a model based function on the measured intensities and a quantile cut
    ("lucky imaging" style selection).
    
    :param I0_meas: list of measured (aggregated) reference zwfs pupil intensities (generally filtered in a particular region, e.g. exterior pixels for Strehl or OPD model)
    :param image_processing_fn: callable function that takes an single item in I0_meas list (e.g. a 2D image) and converts it to a signal that is converted to performance metric, i.e. input, to performance_model
    :param performance_model: callable model for performance metric (e.g opd or strehl model), signature = performance_model( I0_meas, **model_params)
        the performance model must take a list-like array containing measurements as input to convert it to some performance metric (e.g. opd or strehl)
    :param model_param: (dictionary) Description dictionary of the performance_model parameters to pass 
    :param quantile_threshold: (float) threshold in the performance_model quantile 
    :param keep: (string) do filter for where performance is <threshold or >threshold (input = '<threshold' | '>threshold' )
    """

    img_signal = np.array([image_processing_fn( ii ) for ii in I0_meas])
    perf_est = performance_model(img_signal, **model_params)

    perf_threshold = np.quantile( perf_est, quantile_threshold )
    
    if keep == "<threshold": 
        I0_lucky = np.array( I0_meas )[ perf_est < perf_threshold ]
    elif keep == ">threshold": 
        I0_lucky = np.array( I0_meas )[ perf_est > perf_threshold ]
    else: 
        raise UserWarning("invalid keep input. try either '<threshold' or '>threshold'")
    return I0_lucky



# Util for co-aligning (center and rotate) frames from a measured pupil, and theoretical pupil


# atempt 3.5 
import numpy as np
from scipy.ndimage import shift as nd_shift
from scipy.ndimage import rotate as nd_rotate
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.morphology import binary_erosion, disk

def align_prior_to_meas_using_spiders_matched_filter(
    I_meas,
    I_prior,
    N0_meas,
    N0_prior,
    pupil_mask=None,
    detect_pupil_fn=None,   # must return (cx, cy, ...)
    sigma_meas_hp=2.0,
    sigma_prior_hp=5.0,
    rotate_order=3,
    shift_order=3,
    # spider feature knobs
    mask_erosion_px=2,
    use_edge_term=True,
    edge_weight=0.5,
    # matched filter knobs
    angle_search_deg=(-180.0, 180.0),
    angle_step_deg=0.5,
    refine=True,
    refine_half_width_deg=2.0,
    refine_step_deg=0.05,
    # optional radial weighting to favor mid-pupil (spiders live there)
    radial_weight=True,
    radial_weight_power=0.5,
    debug_plot=False,
):
    """
    Align prior -> meas by:
      1) detect & center pupils (translation)
      2) build spider feature maps on centered N0 images (DoG + optional Sobel)
      3) matched filter: rotate prior spider-map over a grid and maximize NCC score
      4) rotate centered prior (I_prior and N0_prior) by best dtheta

    Returns dict with aligned images + diagnostics.
    """

    def _image_center(shape):
        ny, nx = shape
        return (ny - 1) / 2.0, (nx - 1) / 2.0

    def _shift_bool_mask(mask, shift):
        m = nd_shift(mask.astype(float), shift=shift, order=0, mode="constant", cval=0.0)
        return m > 0.5

    def _make_centered_mask_from_detect(N0_centered):
        if detect_pupil_fn is None:
            raise ValueError("detect_pupil_fn must be provided if pupil_mask is None.")

        cx, cy, *rest = detect_pupil_fn(N0_centered, plot=False)

        r_guess = None
        if len(rest) >= 2 and np.isfinite(rest[0]) and np.isfinite(rest[1]):
            a, b = rest[0], rest[1]
            r_guess = 0.5 * (abs(a) + abs(b))

        if (r_guess is None) or (not np.isfinite(r_guess)) or (r_guess <= 1):
            yy, xx = np.indices(N0_centered.shape)
            w = np.clip(N0_centered - np.median(N0_centered), 0, None)
            wsum = np.sum(w) + 1e-18
            mx = np.sum(xx * w) / wsum
            my = np.sum(yy * w) / wsum
            rr = np.sqrt((xx - mx) ** 2 + (yy - my) ** 2)
            r_guess = np.quantile(rr[w > np.quantile(w, 0.7)], 0.9)

        yy, xx = np.indices(N0_centered.shape)
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = rr <= r_guess

        if mask_erosion_px and mask_erosion_px > 0:
            mask_e = binary_erosion(mask, disk(mask_erosion_px))
            if np.any(mask_e):
                mask = mask_e

        return mask, float(r_guess)

    def _spider_features(N0_c, mask_c, sigma_hp):
        """
        Spider-friendly feature map:
          - normalize by median in mask
          - high-pass (subtract Gaussian)
          - invert so dark spiders become positive
          - optional Sobel edges to emphasize spider boundaries
          - restrict to mask
        """
        N0_c = np.asarray(N0_c, float)
        med = np.median(N0_c[mask_c]) if np.any(mask_c) else np.median(N0_c)
        Nn = N0_c / (med + 1e-18)

        low = gaussian_filter(Nn, sigma=sigma_hp)
        hp = Nn - low

        feat = np.clip(-hp, 0.0, None)  # spiders dark -> positive

        if use_edge_term:
            e = sobel(Nn)
            e *= mask_c.astype(float)
            escale = np.median(e[mask_c]) + 1e-18
            e = e / escale
            feat = feat + edge_weight * np.clip(e, 0.0, None)

        feat *= mask_c.astype(float)
        return feat

    def _make_radial_weight(mask_c, cy0, cx0, power=0.5):
        yy, xx = np.indices(mask_c.shape)
        rr = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2)

        # estimate pupil radius from mask
        rmax = np.max(rr[mask_c]) if np.any(mask_c) else np.max(rr)
        r = rr / (rmax + 1e-18)

        # bump in mid radii: (r(1-r))^power
        w = (r * (1.0 - r))
        w = np.clip(w, 0.0, None) ** power
        w *= mask_c.astype(float)
        w /= (np.max(w) + 1e-18)
        return w

    def _ncc_score(A, B, W=None):
        """
        Normalized cross-correlation (cosine similarity) with optional weights.
        """
        if W is None:
            a = A.ravel()
            b = B.ravel()
        else:
            a = (A * W).ravel()
            b = (B * W).ravel()

        a = a - np.mean(a)
        b = b - np.mean(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18)
        return float(np.dot(a, b) / denom)

    # -----------------------
    I_meas  = np.asarray(I_meas, dtype=float)
    I_prior = np.asarray(I_prior, dtype=float)
    N0_meas = np.asarray(N0_meas, dtype=float)
    N0_prior= np.asarray(N0_prior, dtype=float)

    if detect_pupil_fn is None:
        raise ValueError("detect_pupil_fn must be provided.")

    cy0, cx0 = _image_center(I_meas.shape)

    # --- detect and center ---
    cx_m, cy_m, *_ = detect_pupil_fn(N0_meas, plot=False)
    cx_p, cy_p, *_ = detect_pupil_fn(N0_prior, plot=False)

    dy_m, dx_m = (cy0 - cy_m), (cx0 - cx_m)
    dy_p, dx_p = (cy0 - cy_p), (cx0 - cx_p)

    I_meas_c   = nd_shift(I_meas,   shift=(dy_m, dx_m), order=shift_order, mode="constant", cval=0.0)
    I_prior_c  = nd_shift(I_prior,  shift=(dy_p, dx_p), order=shift_order, mode="constant", cval=0.0)
    N0_meas_c  = nd_shift(N0_meas,  shift=(dy_m, dx_m), order=shift_order, mode="constant", cval=0.0)
    N0_prior_c = nd_shift(N0_prior, shift=(dy_p, dx_p), order=shift_order, mode="constant", cval=0.0)

    # --- mask in centered coords ---
    if pupil_mask is None:
        mask_c, r_guess = _make_centered_mask_from_detect(N0_meas_c)
    else:
        m = np.asarray(pupil_mask, bool)
        if m.shape != N0_meas.shape:
            raise ValueError(f"pupil_mask shape {m.shape} != N0_meas shape {N0_meas.shape}")
        mask_c = _shift_bool_mask(m, shift=(dy_m, dx_m))

        if mask_erosion_px and mask_erosion_px > 0:
            mask_e = binary_erosion(mask_c, disk(mask_erosion_px))
            if np.any(mask_e):
                mask_c = mask_e

        if not np.any(mask_c):
            mask_c, r_guess = _make_centered_mask_from_detect(N0_meas_c)
        else:
            yy, xx = np.indices(mask_c.shape)
            rr = np.sqrt((xx - cx0)**2 + (yy - cy0)**2)
            r_guess = float(np.max(rr[mask_c]))

    # --- spider feature maps ---
    sp_meas  = _spider_features(N0_meas_c,  mask_c, sigma_hp=sigma_meas_hp)
    sp_prior = _spider_features(N0_prior_c, mask_c, sigma_hp=sigma_prior_hp)

    # optional radial weights (helps suppress center + edge artifacts)
    W = _make_radial_weight(mask_c, cy0, cx0, power=radial_weight_power) if radial_weight else None

    # --- matched-filter rotation scan ---
    a0, a1 = angle_search_deg
    angles = np.arange(a0, a1 + angle_step_deg/2, angle_step_deg, dtype=float)

    scores = np.zeros_like(angles)
    for i, ang in enumerate(angles):
        sp_rot = nd_rotate(sp_prior, angle=ang, reshape=False, order=1, mode="constant", cval=0.0)
        scores[i] = _ncc_score(sp_meas, sp_rot, W=W)

    i_best = int(np.argmax(scores))
    best_ang = float(angles[i_best])
    best_score = float(scores[i_best])

    # --- local refinement around best angle ---
    if refine:
        a_lo = best_ang - refine_half_width_deg
        a_hi = best_ang + refine_half_width_deg
        fine_angles = np.arange(a_lo, a_hi + refine_step_deg/2, refine_step_deg, dtype=float)
        fine_scores = np.zeros_like(fine_angles)

        for i, ang in enumerate(fine_angles):
            sp_rot = nd_rotate(sp_prior, angle=ang, reshape=False, order=1, mode="constant", cval=0.0)
            fine_scores[i] = _ncc_score(sp_meas, sp_rot, W=W)

        j = int(np.argmax(fine_scores))
        best_ang = float(fine_angles[j])
        best_score = float(fine_scores[j])

        # replace coarse arrays with refined, for output clarity
        angles_used = fine_angles
        scores_used = fine_scores
    else:
        angles_used = angles
        scores_used = scores

    # convention: dtheta is what we apply to PRIOR to match MEAS
    dtheta_deg = best_ang
    dtheta_rad = np.deg2rad(dtheta_deg)

    # --- rotate prior intensity images by best angle ---
    I_prior_aligned = nd_rotate(I_prior_c, angle=dtheta_deg, reshape=False,
                                order=rotate_order, mode="constant", cval=0.0)
    N0_prior_aligned = nd_rotate(N0_prior_c, angle=dtheta_deg, reshape=False,
                                 order=rotate_order, mode="constant", cval=0.0)

    # What we usually want is the theory always in the measurement frame (fastest for an RTC)
    shift_meas=(dy_m, dx_m)
    shift_prior=(dy_p, dx_p)
    dy_net = shift_meas[0] - shift_prior[0]
    dx_net = shift_meas[1] - shift_prior[1]
    net_shift_prior_to_meas = (-dy_net, -dx_net)

    I_prior_aligned_in_meas_frame = nd_shift(I_prior, shift=net_shift_prior_to_meas,
                                            order=shift_order, mode="constant", cval=0.0)
    I_prior_aligned_in_meas_frame = nd_rotate(I_prior_aligned_in_meas_frame, angle=dtheta_deg,
                                            reshape=False, order=rotate_order, mode="constant", cval=0.0)

    N0_prior_aligned_in_meas_frame = nd_shift(N0_prior, shift=net_shift_prior_to_meas,
                                            order=shift_order, mode="constant", cval=0.0)
    N0_prior_aligned_in_meas_frame = nd_rotate(N0_prior_aligned_in_meas_frame, angle=dtheta_deg,
                                            reshape=False, order=rotate_order, mode="constant", cval=0.0)

    return dict(
        cy0=cy0, cx0=cx0,
        shift_meas=shift_meas,
        shift_prior=shift_prior,
        mask_centered=mask_c,
        r_guess=r_guess,
        dtheta_rad=dtheta_rad,
        dtheta_deg=dtheta_deg,
        best_score=best_score,
        angles=angles_used,
        scores=scores_used,
        I_meas_centered=I_meas_c,
        I_prior_centered=I_prior_c,
        I_prior_aligned=I_prior_aligned,
        I_prior_aligned_in_meas_frame=I_prior_aligned_in_meas_frame,

        N0_meas_centered=N0_meas_c,
        N0_prior_centered=N0_prior_c,
        N0_prior_aligned=N0_prior_aligned,
        N0_prior_aligned_in_meas_frame=N0_prior_aligned_in_meas_frame,
        spiders_meas=sp_meas,
        spiders_prior=sp_prior,
        weight=W,
    )

# import numpy as np
# from scipy.ndimage import shift as nd_shift
# from scipy.ndimage import rotate as nd_rotate
# from scipy.ndimage import gaussian_filter
# from skimage.transform import warp_polar
# from skimage.filters import sobel
# from skimage.morphology import binary_erosion, disk

# # second attempt 
# def align_prior_to_meas_using_spiders(
#     I_meas,
#     I_prior,
#     N0_meas,
#     N0_prior,
#     pupil_mask=None,          # can be None; if provided it must match input frame coords
#     detect_pupil_fn=None,     # must return (cx, cy, ...)
#     sigma_meas_hp=2.0,
#     sigma_prior_hp=5.0,
#     polar_radius=None,
#     polar_output_shape=None,
#     refine_peak=True,
#     rotate_order=3,
#     shift_order=3,
#     debug_plot=False,
#     # new knobs (safe defaults)
#     mask_erosion_px=2,        # erode to focus on interior (spiders are dark inside)
#     use_edge_term=True,       # add Sobel edge map to spider features
#     edge_weight=0.5,          # 0..1 typical
#     radial_weight=True,       # downweight center / edge where artifacts live
# ):
#     """
#     Align prior -> measured by:
#       (1) detect & center pupils (translation)
#       (2) build spider feature maps on centered N0s (DoG + optional Sobel)
#       (3) convert to polar around image center
#       (4) compute angular correlation => dtheta
#       (5) rotate prior images by chosen dtheta (with sign check)
#     """

#     def _image_center(shape):
#         ny, nx = shape
#         return (ny - 1) / 2.0, (nx - 1) / 2.0

#     def _shift_bool_mask(mask, shift):
#         # nearest-neighbour shift for boolean masks
#         m = nd_shift(mask.astype(float), shift=shift, order=0, mode="constant", cval=0.0)
#         return m > 0.5

#     def _make_centered_mask_from_detect(N0_centered):
#         # robust-ish mask: use detect_pupil ellipse center/radius if your detect returns them,
#         # otherwise fall back to simple threshold on normalized N0.
#         if detect_pupil_fn is None:
#             raise ValueError("detect_pupil_fn must be provided if pupil_mask is None.")

#         cx, cy, *rest = detect_pupil_fn(N0_centered, plot=False)

#         # crude radius guess from rest if available; otherwise estimate from second moments
#         # If your detect_pupil returns (cx, cy, a, b, theta, ...) you can use a/b here.
#         r_guess = None
#         if len(rest) >= 2 and np.isfinite(rest[0]) and np.isfinite(rest[1]):
#             a, b = rest[0], rest[1]
#             # interpret a,b as semi-axes in pixels if that's what your detect returns
#             r_guess = 0.5 * (abs(a) + abs(b))
#         if (r_guess is None) or (not np.isfinite(r_guess)) or (r_guess <= 1):
#             # moment-based fallback
#             yy, xx = np.indices(N0_centered.shape)
#             w = np.clip(N0_centered - np.median(N0_centered), 0, None)
#             wsum = np.sum(w) + 1e-18
#             mx = np.sum(xx * w) / wsum
#             my = np.sum(yy * w) / wsum
#             rr = np.sqrt((xx - mx) ** 2 + (yy - my) ** 2)
#             # pick radius at, say, 95% energy of w as rough pupil radius
#             r_guess = np.quantile(rr[w > np.quantile(w, 0.7)], 0.9)

#         yy, xx = np.indices(N0_centered.shape)
#         rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
#         mask = rr <= r_guess

#         if mask_erosion_px and mask_erosion_px > 0:
#             mask = binary_erosion(mask, disk(mask_erosion_px))

#         return mask

#     def _spider_features(N0_c, mask_c, sigma_hp):
#         """
#         Spider-friendly feature map:
#           - normalize inside mask
#           - DoG high-pass
#           - invert so dark spiders are positive
#           - optional edge term (Sobel) for spider boundaries
#         """
#         N0_c = np.asarray(N0_c, float)

#         med = np.median(N0_c[mask_c]) if np.any(mask_c) else np.median(N0_c)
#         Nn = N0_c / (med + 1e-18)

#         # DoG: emphasizes thin-ish dark structures and suppresses smooth gradients
#         low = gaussian_filter(Nn, sigma=sigma_hp)
#         hp = Nn - low

#         # spiders are dark: invert -> spiders become positive
#         feat = np.clip(-hp, 0.0, None)

#         if use_edge_term:
#             # edges on the *normalized* image, restricted to mask
#             e = sobel(Nn)
#             e *= mask_c.astype(float)
#             # normalize edge term to comparable scale
#             escale = np.median(e[mask_c]) + 1e-18
#             e = e / escale
#             feat = feat + edge_weight * np.clip(e, 0.0, None)

#         # restrict to mask (avoid outside junk)
#         feat *= mask_c.astype(float)

#         return feat

#     def _angular_score(polar_meas, polar_prior, r_weight=None):
#         """
#         Compute normalized angular similarity per angle bin then sum over radius.
#         polar_* shape: (Nr, Ntheta)
#         """
#         A = polar_meas
#         B = polar_prior

#         if r_weight is not None:
#             A = A * r_weight[:, None]
#             B = B * r_weight[:, None]

#         # sum over radius -> vectors over theta
#         a = np.sum(A, axis=0)
#         b = np.sum(B, axis=0)

#         # circular correlation via FFT (normalized)
#         a0 = a - np.mean(a)
#         b0 = b - np.mean(b)
#         denom = (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-18)

#         corr = np.fft.ifft(np.fft.fft(a0) * np.conj(np.fft.fft(b0))).real / denom
#         # corr[k] corresponds to shift by k bins
#         return corr

#     def _refine_peak_parabola(corr, idx):
#         n = corr.size
#         i = np.array([(idx - 1) % n, idx % n, (idx + 1) % n], dtype=float)
#         y = corr[i.astype(int)]
#         a, b, _c = np.polyfit(i, y, 2)
#         if abs(a) < 1e-18:
#             return float(idx)
#         i_peak = -b / (2 * a)
#         return float(i_peak % n)

#     # --------------------------
#     # Inputs
#     I_meas  = np.asarray(I_meas, dtype=float)
#     I_prior = np.asarray(I_prior, dtype=float)
#     N0_meas = np.asarray(N0_meas, dtype=float)
#     N0_prior= np.asarray(N0_prior, dtype=float)

#     cy0, cx0 = _image_center(I_meas.shape)

#     # --- center estimates from detect_pupil on raw frames ---
#     if detect_pupil_fn is None:
#         raise ValueError("detect_pupil_fn must be provided.")

#     cx_m, cy_m, *_ = detect_pupil_fn(N0_meas, plot=False)
#     cx_p, cy_p, *_ = detect_pupil_fn(N0_prior, plot=False)

#     dy_m, dx_m = (cy0 - cy_m), (cx0 - cx_m)
#     dy_p, dx_p = (cy0 - cy_p), (cx0 - cx_p)

#     # --- center frames ---
#     I_meas_c   = nd_shift(I_meas,   shift=(dy_m, dx_m), order=shift_order, mode="constant", cval=0.0)
#     I_prior_c  = nd_shift(I_prior,  shift=(dy_p, dx_p), order=shift_order, mode="constant", cval=0.0)
#     N0_meas_c  = nd_shift(N0_meas,  shift=(dy_m, dx_m), order=shift_order, mode="constant", cval=0.0)
#     N0_prior_c = nd_shift(N0_prior, shift=(dy_p, dx_p), order=shift_order, mode="constant", cval=0.0)

#     # --- make/shift mask in the *centered* coordinate system ---
#     # if pupil_mask is None:
#     #     mask_c = _make_centered_mask_from_detect(N0_meas_c)
#     # else:
#     #     # shift the provided mask by the same measured shift so it matches centered frames
#     #     mask_c = _shift_bool_mask(np.asarray(pupil_mask, bool), shift=(dy_m, dx_m))
#     #     if mask_erosion_px and mask_erosion_px > 0:
#     #         mask_c = binary_erosion(mask_c, disk(mask_erosion_px))
#     # --- make/shift mask in the *centered* coordinate system ---
#     if pupil_mask is None:
#         mask_c = _make_centered_mask_from_detect(N0_meas_c)
#     else:
#         m = np.asarray(pupil_mask, bool)

#         # HARD GUARD: must match image shape
#         if m.shape != N0_meas.shape:
#             raise ValueError(
#                 f"pupil_mask shape {m.shape} does not match N0_meas shape {N0_meas.shape}. "
#                 "Pass a mask in the same pixel grid as N0_meas/N0_prior."
#             )

#         # shift provided mask into centered frame (measured shift)
#         mask_c = _shift_bool_mask(m, shift=(dy_m, dx_m))

#         # optional erosion (but don't allow it to destroy the mask)
#         if mask_erosion_px and mask_erosion_px > 0:
#             mask_e = binary_erosion(mask_c, disk(mask_erosion_px))
#             if np.any(mask_e):
#                 mask_c = mask_e  # only accept erosion if non-empty

#         # FALLBACK: if shifting killed it, rebuild a mask from the centered N0_meas
#         if not np.any(mask_c):
#             mask_c = _make_centered_mask_from_detect(N0_meas_c)
#     # --- spider feature maps ---
#     sp_meas  = _spider_features(N0_meas_c,  mask_c, sigma_hp=sigma_meas_hp)
#     sp_prior = _spider_features(N0_prior_c, mask_c, sigma_hp=sigma_prior_hp)

#     # --- polar transform ---
#     warp_kwargs = {}
#     if polar_radius is not None:
#         warp_kwargs["radius"] = polar_radius
#     if polar_output_shape is not None:
#         warp_kwargs["output_shape"] = polar_output_shape

#     sp_meas_polar  = warp_polar(sp_meas,  center=(cy0, cx0), **warp_kwargs)
#     sp_prior_polar = warp_polar(sp_prior, center=(cy0, cx0), **warp_kwargs)

#     # --- optional radial weighting (downweight center + very edge) ---
#     r_weight = None
#     if radial_weight:
#         Nr = sp_meas_polar.shape[0]
#         r = np.linspace(0.0, 1.0, Nr)
#         # bump in mid-radii where spiders live most clearly (tweak if needed)
#         r_weight = (r * (1 - r)) ** 0.5
#         r_weight = r_weight / (np.max(r_weight) + 1e-18)

#     # --- circular correlation for angle shift ---
#     corr = _angular_score(sp_meas_polar, sp_prior_polar, r_weight=r_weight)
#     idx0 = int(np.argmax(corr))
#     idx_peak = _refine_peak_parabola(corr, idx0) if refine_peak else float(idx0)

#     dtheta = 2.0 * np.pi * (idx_peak / corr.size)  # radians

#     # wrap to [-pi, pi)
#     if dtheta >= np.pi:
#         dtheta -= 2.0 * np.pi

#     # --- SIGN CHECK: try +dtheta and -dtheta, pick best on feature correlation ---
#     def _score_for_angle(theta_rad):
#         sp_prior_rot = nd_rotate(
#             sp_prior, angle=np.degrees(theta_rad),
#             reshape=False, order=1, mode="constant", cval=0.0
#         )
#         sp_prior_rot_p = warp_polar(sp_prior_rot, center=(cy0, cx0), **warp_kwargs)
#         c = _angular_score(sp_meas_polar, sp_prior_rot_p, r_weight=r_weight)
#         return np.max(c)

#     score_plus  = _score_for_angle(+dtheta)
#     score_minus = _score_for_angle(-dtheta)
#     if score_minus > score_plus:
#         dtheta = -dtheta

#     # --- rotate prior intensity images by chosen angle ---
#     I_prior_aligned = nd_rotate(
#         I_prior_c, angle=np.degrees(dtheta),
#         reshape=False, order=rotate_order, mode="constant", cval=0.0
#     )
#     N0_prior_aligned = nd_rotate(
#         N0_prior_c, angle=np.degrees(dtheta),
#         reshape=False, order=rotate_order, mode="constant", cval=0.0
#     )

#     if debug_plot:
#         # Keep your own plotting utilities here if you want
#         pass

#     return {
#         "cy0": cy0,
#         "cx0": cx0,
#         "shift_meas": (dy_m, dx_m),
#         "shift_prior": (dy_p, dx_p),
#         "mask_centered": mask_c,
#         "dtheta_rad": dtheta,
#         "dtheta_deg": np.degrees(dtheta),
#         "I_meas_centered": I_meas_c,
#         "I_prior_centered": I_prior_c,
#         "I_prior_aligned": I_prior_aligned,
#         "N0_meas_centered": N0_meas_c,
#         "N0_prior_centered": N0_prior_c,
#         "N0_prior_aligned": N0_prior_aligned,
#         "spiders_meas": sp_meas,
#         "spiders_prior": sp_prior,
#         "corr": corr,
#         "score_plus": score_plus,
#         "score_minus": score_minus,
#     }

# ## first attempt below - rotation correlation mask didnt work well 
# # def align_prior_to_meas_using_spiders(
# #     I_meas,
# #     I_prior,
# #     N0_meas,          # e.g. clear_pup (measured pupil image used to see spiders)
# #     N0_prior,         # e.g. zwfs_ns_AT.reco.N0 (theoretical pupil image used to see spiders)
# #     pupil_mask,       # boolean mask in the (centered) coordinate system
# #     detect_pupil_fn,  # util.detect_pupil
# #     sigma_meas_hp=2.0,
# #     sigma_prior_hp=5.0,
# #     polar_radius=None,      # None => default skimage
# #     polar_output_shape=None,# None => default skimage
# #     refine_peak=True,
# #     rotate_order=3,
# #     shift_order=3,
# #     debug_plot=False,
# # ):
# #     """
# #     Pipeline:
# #       1) find pupil centers in measured and prior using detect_pupil_fn
# #       2) shift meas/prior (and their N0s) so pupil centers land at image center
# #       3) build spider emphasis maps (high-pass)
# #       4) convert spider maps to polar coordinates about image center
# #       5) angular correlation => best rotation dtheta
# #       6) rotate *prior* intensity image by dtheta (about image center)

# #     Returns dict with aligned images + diagnostics.
# #     """


# #     def _image_center(shape):
# #         """Return (cy0, cx0) for an image of given shape."""
# #         ny, nx = shape
# #         return (ny - 1) / 2.0, (nx - 1) / 2.0


# #     def _spider_map(I, pupil_mask):
# #         """
# #         Build a 'spider emphasis' map:
# #         - normalize by median inside pupil
# #         - invert so dark features become bright
# #         """
# #         I = np.asarray(I, dtype=float)
# #         med = np.median(I[pupil_mask])
# #         I_n = I / (med + 1e-18)
# #         return np.clip(1.0 - I_n, 0.0, None)


# #     def _angular_correlation(sp_meas_polar, sp_prior_polar):
# #         """Compute 1D angular correlation by summing over radius."""
# #         return np.sum(sp_meas_polar * sp_prior_polar, axis=0)


# #     def _refine_peak_parabola(corr, idx):
# #         """
# #         Sub-bin refinement of argmax using parabola fit on (idx-1, idx, idx+1).
# #         Returns fractional index (float) in [0, n).
# #         """
# #         n = corr.size
# #         i = np.array([(idx - 1) % n, idx % n, (idx + 1) % n], dtype=float)
# #         y = corr[i.astype(int)]
# #         # Fit y = a*i^2 + b*i + c
# #         a, b, _c = np.polyfit(i, y, 2)
# #         if abs(a) < 1e-18:
# #             return float(idx)
# #         i_peak = -b / (2 * a)
# #         return float(i_peak % n)

# #     I_meas = np.asarray(I_meas, dtype=float)
# #     I_prior = np.asarray(I_prior, dtype=float)
# #     N0_meas = np.asarray(N0_meas, dtype=float)
# #     N0_prior = np.asarray(N0_prior, dtype=float)

# #     cy0, cx0 = _image_center(I_meas.shape)

# #     # ---  centers from ellipse fit (translation only) ---
# #     cx_m, cy_m, *_ = detect_pupil_fn(N0_meas, plot=False)
# #     cx_p, cy_p, *_ = detect_pupil_fn(N0_prior, plot=False)

# #     dy_m, dx_m = (cy0 - cy_m), (cx0 - cx_m)
# #     dy_p, dx_p = (cy0 - cy_p), (cx0 - cx_p)

# #     # ---  center measured and prior frames (and their N0s) ---
# #     I_meas_c  = nd_shift(I_meas,  shift=(dy_m, dx_m), order=shift_order, mode="constant", cval=0.0)
# #     I_prior_c = nd_shift(I_prior, shift=(dy_p, dx_p), order=shift_order, mode="constant", cval=0.0)

# #     N0_meas_c  = nd_shift(N0_meas,  shift=(dy_m, dx_m), order=shift_order, mode="constant", cval=0.0)
# #     N0_prior_c = nd_shift(N0_prior, shift=(dy_p, dx_p), order=shift_order, mode="constant", cval=0.0)

# #     # --- spider emphasis maps (high-pass) ---
# #     sp_meas  = _spider_map(N0_meas_c,  pupil_mask)
# #     sp_prior = _spider_map(N0_prior_c, pupil_mask)

# #     sp_meas  = sp_meas  - gaussian_filter(sp_meas,  sigma=sigma_meas_hp)
# #     sp_prior = sp_prior - gaussian_filter(sp_prior, sigma=sigma_prior_hp)

# #     # ---  polar transform around the *image center* ---
# #     warp_kwargs = {}
# #     if polar_radius is not None:
# #         warp_kwargs["radius"] = polar_radius
# #     if polar_output_shape is not None:
# #         warp_kwargs["output_shape"] = polar_output_shape

# #     sp_meas_polar  = warp_polar(sp_meas,  center=(cy0, cx0), **warp_kwargs)
# #     sp_prior_polar = warp_polar(sp_prior, center=(cy0, cx0), **warp_kwargs)

# #     # --- angular correlation to find dtheta ---
# #     corr = _angular_correlation(sp_meas_polar, sp_prior_polar)
# #     idx0 = int(np.argmax(corr))

# #     if refine_peak:
# #         idx_peak = _refine_peak_parabola(corr, idx0)
# #     else:
# #         idx_peak = float(idx0)

# #     dtheta = 2.0 * np.pi * (idx_peak / corr.size)  # radians

# #     # wrap to [-pi, pi) for convenience
# #     if dtheta >= np.pi:
# #         dtheta -= 2.0 * np.pi

# #     # --- rotate the prior intensity image (and optionally N0_prior/spiders) ---
# #     I_prior_aligned = nd_rotate(
# #         I_prior_c,
# #         angle=np.degrees(dtheta),
# #         reshape=False,
# #         order=rotate_order,
# #         mode="constant",
# #         cval=0.0
# #     )

# #     # You may also want aligned N0_prior for diagnostics:
# #     N0_prior_aligned = nd_rotate(
# #         N0_prior_c,
# #         angle=np.degrees(dtheta),
# #         reshape=False,
# #         order=rotate_order,
# #         mode="constant",
# #         cval=0.0
# #     )

# #     if debug_plot:
# #         # Quick checks

# #         nice_heatmap_subplots(
# #             im_list=[I_meas_c, I_prior_c, I_prior_aligned],
# #             title_list=["meas centered", "prior centered", f"prior aligned (dθ={np.degrees(dtheta):.3f}°)"],
# #         )
# #         plt.show()

# #         nice_heatmap_subplots(
# #             im_list=[sp_meas, sp_prior, _spider_map(N0_prior_aligned, pupil_mask) - gaussian_filter(_spider_map(N0_prior_aligned, pupil_mask), 5)],
# #             title_list=["spiders meas", "spiders prior", "spiders prior aligned"],
# #         )
# #         plt.show()

# #         plt.figure()
# #         plt.plot(corr)
# #         plt.title("Angular correlation")
# #         plt.xlabel("angle bin")
# #         plt.ylabel("corr")
# #         plt.show()

# #     return {
# #         "cy0": cy0,
# #         "cx0": cx0,
# #         "shift_meas": (dy_m, dx_m),
# #         "shift_prior": (dy_p, dx_p),
# #         "dtheta_rad": dtheta,
# #         "dtheta_deg": np.degrees(dtheta),
# #         "I_meas_centered": I_meas_c,
# #         "I_prior_centered": I_prior_c,
# #         "I_prior_aligned": I_prior_aligned,
# #         "N0_meas_centered": N0_meas_c,
# #         "N0_prior_centered": N0_prior_c,
# #         "N0_prior_aligned": N0_prior_aligned,
# #         "spiders_meas": sp_meas,
# #         "spiders_prior": sp_prior,
# #         "corr": corr,
# #     }
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def apply_fill_factor_with_oversampling(mask: np.ndarray, oversample_factor: int = 20, fill_factor: float=0.85) -> np.ndarray:
    Nx, Ny = mask.shape
    # Oversample by repeating pixels
    mask_os = np.repeat(np.repeat(mask, oversample_factor, axis=0), oversample_factor, axis=1)
    # Create fill factor pixel mask (active modulation area)
    active_size = int(oversample_factor * fill_factor)
    start_idx = (oversample_factor - active_size) // 2
    pixel_mask = np.zeros((oversample_factor, oversample_factor), dtype=np.bool)
    pixel_mask[start_idx:start_idx+active_size, start_idx:start_idx+active_size] = True
    # Tile over entire mask
    tiled_mask = np.tile(pixel_mask, (Nx, Ny))
    # Apply fill factor by zeroing dead regions
    filled_mask = mask_os * tiled_mask
    return filled_mask

def crop_oversampling(oversampled_array: np.ndarray, oversample_factor: int = 20) -> np.ndarray:
    '''------------------------------------------------------------------------------------
    Crop an oversampled array back to its original size without binning
    Inputs:
        oversampled_array (ndarray): the array to crop back to its original size.
        oversample_factor (int): the factor by which the original array wa oversampled.
    Outputs:
       cropped_array (ndarray): the array cropped back to its original size.
    ------------------------------------------------------------------------------------'''

    Nx, Ny = oversampled_array.shape
    NxOg, NyOg = Nx // oversample_factor, Ny // oversample_factor
    return oversampled_array[Nx//2 - NxOg//2 : Nx//2 + NxOg//2, Ny//2 - NyOg//2 : Ny//2 + NyOg//2]

    

def downsample_phase(phase_os: np.ndarray, slm_mask: np.ndarray, oversample_factor: int = 20):
    Nx_os, Ny_os = phase_os.shape
    Nx, Ny = Nx_os // oversample_factor, Ny_os // oversample_factor

    # Reshape arrays into blocks representing original pixels and oversampled subpixels
    phase_blocks = phase_os.reshape(Nx, oversample_factor, Ny, oversample_factor)
    mask_blocks = slm_mask.reshape(Nx, oversample_factor, Ny, oversample_factor)

    # Convert phase to complex representation, multiply by mask
    complex_phase = np.exp(1j * phase_blocks) * mask_blocks

    # Sum complex values and count active pixels in each block
    sum_complex = complex_phase.sum(axis=(1, 3))
    count_active = mask_blocks.sum(axis=(1, 3))

    # Avoid divide by zero
    count_active[count_active == 0] = 1

    # Average phase by angle of complex vector sum
    avg_phase = np.angle(sum_complex / count_active)
    return avg_phase


def zero_pad(array: np.ndarray, pad_factor: int = 2) -> np.ndarray:
    '''------------------------------------------------------------------------------------
    zero pads array in a centred manner

    Inputs:
        array (ndarray): input array to be zero padded
        pad_factor (int): factor by which to pad the array. Default is 2.
    Returns:
        padded_array: ndarray, zero-padded array
    To do:
    TODO: Input checks
    ------------------------------------------------------------------------------------'''
    Nx, Ny = array.shape
    Nx_padded = Nx * pad_factor
    Ny_padded = Ny * pad_factor

    padded_array = np.zeros((Nx_padded, Ny_padded), dtype=array.dtype)
    # Insert the original array into the center of the padded array
    start_x = (Nx_padded - Nx) // 2
    start_y = (Ny_padded - Ny) // 2

    padded_array[start_x:start_x+Nx, start_y:start_y+Ny] = array
    return padded_array

def depad(padded_array: np.ndarray, pad_factor: int = 2) -> np.ndarray:
    '''------------------------------------------------------------------------------------
    Depad a zero-padded array to its original
    Inputs:
        padded_array (ndarray): zero-padded array
        pad_factor (int): factor by which the array is padded. Default is 2
    Returns:
        cropped_array (ndarray): depaded array with original dimensions
    TODO: Input checks
    ------------------------------------------------------------------------------------'''
    Nx_padded, Ny_padded = padded_array.shape
    Nx = Nx_padded // pad_factor
    Ny = Ny_padded // pad_factor

    start_x = (Nx_padded - Nx) // 2
    start_y = (Ny_padded - Ny) // 2

    cropped_array = padded_array[start_x:start_x+Nx, start_y:start_y+Ny]
    return cropped_array


def create_circular_aperture(Nx: int, Ny: int, diameter: int) -> np.ndarray:
    '''------------------------------------------------------------------------------------
    Create a circular aperture with a given diameter.
    Inputs:
        Nx (int): number of pixels in x dimension of the array.
        Ny (int): number of pixels in y dimension of the array.
        diameter (int): diameter of the circular aperture in px
    Returns:
        aperture_mask (ndarray): 2D array with the circular aperture.
    TODO: Input checks
    ------------------------------------------------------------------------------------'''
    x = np.arange(Nx) - Nx // 2
    y = np.arange(Ny) - Ny // 2
    X, Y = np.meshgrid(x, y, indexing='ij')
    radius = diameter / 2
    aperture_mask = ((X**2 + Y**2) <= radius**2).astype(np.int8)
    return aperture_mask

def gerchberg_saxton(target_amplitude: np.ndarray, aperture_amplitude: np.ndarray, iter: int = 50, pad: int = 2) -> np.ndarray:
    '''------------------------------------------------------------------------------------
    Create a circular aperture with a given diameter.
    Inputs:
        target_amplitude (ndarray): 2D array representing the amplitude of the target.
        aperture_amplitude (ndarray): 2D array representing the amplitude of the aperture.
        iter (int): number of iterations to perform.
        pad (int): padding factor to correctly represent frequenzs in the Fourier domain. Default is 2.
    Returns:
        phi (ndarray): 2D array representing the reconstructed phase.
    TODO: Input checks
    ------------------------------------------------------------------------------------'''
    phase = np.exp(1j * 2 * np.pi * np.random.rand(*aperture_amplitude.shape))
    field =  zero_pad(aperture_amplitude * phase, pad_factor=pad)
    target_amplitude = zero_pad(target_amplitude, pad_factor=pad)
    aperture_amplitude = zero_pad(aperture_amplitude, pad_factor=pad)

    for _ in range(iter):
        far_field = fft2((field))
        far_field_phase = np.angle(far_field)
        far_field = target_amplitude * np.exp(1j * far_field_phase)
        field = ifft2(far_field)
        field = aperture_amplitude * np.exp(1j * np.angle(field)) # reminder that aperture_amplitude is a mask that contains 0 in the dead-zones
    return depad(np.angle((field))*aperture_amplitude, pad_factor=pad)

def generate_far_field_intensity(phase: np.ndarray, aperture_amplitude: np.ndarray, pad: int = 2) -> np.ndarray:
    '''------------------------------------------------------------------------------------
    Computes the far field intensity of a given phase and amplitude distribution.
    Inputs:
        phase (ndarray): 2D array of the phase
        aperture_amplitude (ndarray): 2D array . Must be the same size as the phase array
        pad (int): padding factor to correctly represent frequenzs in the Fourier domain. Default is 2.
    TODO: Input checks
    ------------------------------------------------------------------------------------'''
    # TODO add (/wl*F) after fft2 to include F/D ?
    field = aperture_amplitude * np.exp(1j * phase)
    far_field = depad(fft2((zero_pad(field, pad_factor=pad))), pad_factor=pad)
    intensity = np.abs(far_field)**2
    return (intensity / intensity.max())

    # field = aperture_amplitude * np.exp(1j * phase)
    # far_field = fft2(zero_pad(field))
    # intensity = np.abs(far_field)**2
    # return depad(fftshift(intensity / intensity.max()))

def compute_weighted_phase(phi_corr: np.ndarray, phi_target: np.ndarray, aperture_ampl: np.ndarray, C_corr: float = 0.35, C_target: float = 1) -> np.ndarray:
    '''------------------------------------------------------------------------------------
    Performs the complex phase addition.
    Inputs:
        phi_corr (ndarray): 2D array of the 1 of the phase to add
        phi_target (ndarray): 2D array of the other phase to add
        aperture_ampl (ndarray): 2D array of the amplitude distribution
        C_corr (float): coefficient for the first phase. Default is 0.35.
        C_target (float): coefficient for the second phase. Default is 1.
    Returns:
        phi (ndarray): 2D array of the resulting phase
    TODO: Input checks
    ------------------------------------------------------------------------------------'''
    U_corr = aperture_ampl * np.exp(1j * phi_corr)
    U_target = aperture_ampl * np.exp(1j * phi_target)
    U_slm = C_corr * U_corr + C_target * U_target
    return np.angle(U_slm)
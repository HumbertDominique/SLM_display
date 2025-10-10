import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define the parameters for the calibration frames

# N = 256 # Number of calibration frames to generate
# filepath = "data/"


def generate_frames(N, filepath, resolution = (800, 600), value_type = np.uint8):
    """Generate N grayscale .bpm files
    Inputs:
       N (int): number of calibration frames to generate
       filepath: str, the filepath where the .bpm files will be saved. Default is 'calibration
       resolution: tuple, the resolution of the calibration frames (width, height). Default is (800, 600)
       value_type: numpy type for the bit depth of the calibration frames. Default is 8 bits.
    Returns:
        frames: ndarray, N x width x height array containing the grayscale values for each frame.
    """
    
    frames = np.ones((N, resolution[0], resolution[1]), dtype=value_type)
    greys = np.linspace(0, np.iinfo(value_type).max, N)

    frames = frames*greys[:, None, None]

    i = 0
    for i in range(N):
        img = Image.fromarray(frames[i,:,:].astype(value_type))
        img.save(filepath+"calibration"+str(i)+".bmp")




# generate_frames(N, filepath)





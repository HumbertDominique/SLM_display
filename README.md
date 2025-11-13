# SLM display

Suite of tools for use the HOLOEYE LC2002 Spacial light modulator.

> Some tools were abandoned in favor of a better solution.

## Tools
1. Calibration
    1. Calibration.ipynb: displays greyscale images and acquires them
    2. preprocessing.ipynb: preprocess camera data to make the processing easier,
    Flat/dark processing is lacking
    3. processing.ipynb: processes the interferograms to extract the phase shift between gray levels,
    4. phase2gray.ipynb: converts phase screen images to the corresponding greyscale, 

2. Zeroth Order Diffraction (ZOD):
    1. ZOD_removal: computes the phase necessary to remove the ZOD and adds it to a target phase.
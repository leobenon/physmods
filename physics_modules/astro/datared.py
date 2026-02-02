import numpy as mp
from astropy.io import fits
import matplotlib.pyplot as plt


def load_image(filepath):
    """
    Loading the fits data
    ----------
    Parameters
    -------------
    filepath: 
        Path of the fits file.
    -----------
    Returns
    ---------
    data:
        Image data
    exposure_time: 
        exposure time of the image
    """
    with fits.open(filepath) as hdul:
        header = hdul[0].header
        exposure_time = header["EXPOSURE"] 
        data = hdul[0].data
    return data, exposure_time


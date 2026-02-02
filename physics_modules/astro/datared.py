import numpy as np
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

def get_overview(filenames, verbose = False):
    """
    
    
    
    
    """
    list_of_exp = []
    for x in filenames:
        _, exposure_time = load_image(x)
        list_of_exp.append(exposure_time)
    list_of_exp = np.array(list_of_exp) 
    list_of_exp = np.unique(list_of_exp) # gives all unique exposure times
    if verbose:
        print(f"possible exposure times: {list_of_exp} s")
    return list_of_exp

def calibrate_science_images(science_path, dark_path, flat_path, dark_flat_path, bias_path, exposure_science: float = 30, exposure_flat: float = 2.5, verbose: bool = False, add_bias: bool = False):
    """
    
    
    """
    list_of_exp = get_overview(science_path, verbose=True)
    if exposure_science not in list_of_exp:
        raise ValueError(f"The chosen exposure time, {exposure_science} s, is not included in the data, please choose a new exposure from: {list_of_exp} s")
    
# building darks
    darks = []
    for x in dark_path:
        dark_data, exp_time_dark = load_image(x)
        if exp_time_dark  == exposure_science :
            darks.append(dark_data)
    darks = np.stack(darks)

# building raw science images 
    raw_images = []
    science_filenames = []
    for x in science_path:
        raw_data, exp_time_raw = load_image(x)
        if exp_time_raw == exposure_science:
            raw_images.append(raw_data)
            science_filenames.append(x.split('/')[-1])
    raw_images = np.stack(raw_images)
# building flats
    flats = []
    for x in flat_path:
        flat_data, exp_time_flat = load_image(x)
        if exp_time_flat == exposure_flat:
            flats.append(flat_data)
    flats = np.stack
# building darkflats
    darkflats = []
    for x in dark_flat_path:
        darkflat_data, exp_time_darkflat = load_image(x)
        if exp_time_darkflat == exposure_flat:
            darkflats.append(darkflat_data)
    darkflats = np.stack

# building the masters of each dataset
    master_d = np.median(darks)

    master_f = np.median(flats)

    master_df = np.median(darkflats)

    f_norm = (master_f - master_df)/np.median(master_f- master_df) # normalized corrected flats

    calibrated_images = []

    for image in raw_images:
        cal_image = (image - master_d)/f_norm
        calibrated_images.append(cal_image)
    return np.stack(calibrated_images), science_filenames, darks, flats, darkflats
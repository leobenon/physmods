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
        exposure_time = header.get("EXPOSURE", header.get("EXPTIME"))
        if exposure_time is None:
            raise KeyError("No EXPOSURE/EXPTIME keyword found in header.")
        data = hdul[0].data.astype(np.float32)
    return data, exposure_time

def get_overview(filenames, verbose = False):
    """
    Getting overview on possible exposure times within the dataset
    -------
    Parameters
    -------
    filenames: Path to all FITS images of a certain type
    --------
    Returns
    ------
    list_of_exp: ndarray
        Gives all unique exposure times in the dataset
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

def calibrate_science_images(science_path, dark_path, flat_path, dark_flat_path,exposure_science: float = 30, exposure_flat: float = 2.5, verbose: bool = False,):
    """
    Calibrating science images through data reduction
    -------
    Parameters
    -------
    science_path: Path of the raw science images
    dark_path: Path of the dark images using the same exposure time of the science images
    flat_path: Path of the flat images using a dedicated exposure time for flats
    dark_flat_path: Path of the darkflat images using the same exposure time of the flats
    exposure_science: float
        Exposure time of the science images (default is set to 30s).
    exposure_flat: float
        Exposure time of the flat images (default is set to 2.5s).
    --------
    Returns
    ------
    calibrated_images: ndarray
        Calibrated science images 
    science_filenames: ndarray
        File names of the science images
    darks: ndarray
        All dark images before reduction
    flats: ndarray
        All flat images before reduction
    darkflats: ndarray
        All darkflat images before reduction
    master_d: ndarray
        Master-dark images for reduction
    master_f: ndarray 
        Master-flat images for reduction
    master_df: ndarray
        Master-darkflat images for reduction
    f_norm: ndarray
        Master-flat corrected and normalized images for reduction
    
    """
    list_of_exp = get_overview(science_path, verbose=verbose)
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
    flats = np.stack(flats)
# building darkflats
    darkflats = []
    for x in dark_flat_path:
        darkflat_data, exp_time_darkflat = load_image(x)
        if exp_time_darkflat == exposure_flat:
            darkflats.append(darkflat_data)
    darkflats = np.stack(darkflats)

# building the masters of each dataset
    master_d = np.median(darks, axis=0)

    master_f = np.median(flats,axis=0)

    master_df = np.median(darkflats, axis=0)
    
    f_corr = master_f - master_df # corrected flats
    f_norm = (f_corr)/np.median(f_corr) # normalized corrected flats

    calibrated_images = []

    for image in raw_images:
        cal_image = (image - master_d)/f_norm
        calibrated_images.append(cal_image)
    calibrated_images = np.stack(calibrated_images)
    return  calibrated_images,science_filenames, darks, flats, darkflats, master_d, master_f, master_df, f_norm
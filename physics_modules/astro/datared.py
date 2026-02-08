from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
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
        exposure_time = float(exposure_time)
        data = hdul[0].data.astype(np.float32)
    return data, exposure_time


def save_image(
    data: np.ndarray,
    out_path: Union[str, Path],
    header: Optional[fits.Header] = None,
    overwrite: bool = False,
    dtype: Optional[np.dtype] = np.float32,
) -> Path:
    """
    Save a 2D (or 3D stack) numpy array as a FITS file.

    Parameters
    ----------
    data : np.ndarray
        Image data to write. (ny, nx) or (n, ny, nx).
    out_path : str or Path
        Output filename (should end with .fits/.fit).
    header : astropy.io.fits.Header, optional
        FITS header to write. If provided, it will be copied and updated.
    overwrite : bool
        Overwrite existing file if True.
    dtype : numpy dtype or None
        Cast data to this dtype before writing. Use None to keep current dtype.

    Returns
    -------
    Path
        Path to the written FITS file.
    """
    out_path = Path(out_path)

    arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)

    hdr = (header.copy() if header is not None else fits.Header())
    for key in ["FOCALLEN", "APERTURE", "TEMPERAT", "EXPOSURE", "EXPTIME"]:
        if key in hdr:
            try:
                hdr[key] = float(hdr[key])
            except Exception:
                pass
    # Minimal provenance metadata (useful later)
    hdr["HISTORY"] = "Written by physic_modules.save_image"
    hdr["BUNIT"] = hdr.get("BUNIT", "adu")  # optional default

    hdu = fits.PrimaryHDU(arr, header=hdr)
    hdul = fits.HDUList([hdu])
    hdul.writeto(out_path, overwrite=overwrite, output_verify="silentfix")

    return out_path

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

def calibrate_science_images(science_path, dark_path, flat_path, dark_flat_path = None ,exposure_science: float = 30, exposure_flat: float = 2.5, verbose: bool = False):
    """
    Calibrating science images through data reduction
    -------
    Parameters
    -------
    science_path: Path of the raw science images.
    dark_path: Path of the dark images using the same exposure time of the science images.
    flat_path: Path of the flat images using a dedicated exposure time for flats.
    dark_flat_path: Path of the darkflat images using the same exposure time of the flats.
    exposure_science: float
        Exposure time of the science images (default is set to 30s).
    exposure_flat: float
        Exposure time of the flat images (default is set to 2.5s).
    df_check: bool
        If there are separate darkflats frames, enter True otherwise False.
    --------
    Returns
    ------
    calibrated_images: ndarray
        Calibrated science images. 
    science_filenames: list[str]
        File names of the science images.
    darks: ndarray
        All dark images before reduction.
    flats: ndarray
        All flat images before reduction.
    darkflats: ndarray
        All darkflat images before reduction.
    master_d: ndarray
        Master-dark images for reduction.
    master_f: ndarray 
        Master-flat images for reduction.
    master_df: ndarray
        Master-darkflat images for reduction.
    f_norm: ndarray
        Master-flat corrected and normalized images for reduction.
    
    """
    list_of_exp = get_overview(science_path, verbose=verbose)
    if exposure_science not in list_of_exp:
        raise ValueError(f"The chosen exposure time, {exposure_science} s, is not included in the data, please choose a new exposure from: {list_of_exp} s")
    
    def exp_match(t,target, tol = 1e-6):
        return np.isclose(t,target, rtol=0.0, atol=tol)

# building darks
    darks = []
    for x in dark_path:
        dark_data, exp_time_dark = load_image(x)
        if exp_match(exp_time_dark,exposure_science) :
            darks.append(dark_data)
    if len(darks) == 0:
        raise ValueError(f"No darks found with exposure {exposure_science}s")
    darks = np.stack(darks)

# building raw science images 
    raw_images = []
    science_filenames = []
    for x in science_path:
        raw_data, exp_time_raw = load_image(x)
        if exp_match(exp_time_raw,exposure_science):
            raw_images.append(raw_data)
            science_filenames.append(x.split('/')[-1])
    if len(raw_images) == 0:
        raise ValueError(f"No raw science images found with exposure {exposure_science}s")
    raw_images = np.stack(raw_images)
# building flats
    flats = []
    for x in flat_path:
        flat_data, exp_time_flat = load_image(x)
        if exp_match(exp_time_flat,exposure_flat):
            flats.append(flat_data)
    if len(flats) == 0:
        raise ValueError(f"No flats found with exposure {exposure_flat}s")
    flats = np.stack(flats)

    # shape check of the pixels of each frame
    sci_shape = raw_images[0].shape

    if darks.shape[1:] != sci_shape:
        raise ValueError(f"Dark shape {darks.shape[1:]} does not match science shape {sci_shape}")

    if flats.shape[1:] != sci_shape:
        raise ValueError(f"Flat shape {flats.shape[1:]} does not match science shape {sci_shape}")
    
    # building the masters of each dataset
    master_d = np.median(darks, axis=0)

    master_f = np.median(flats,axis=0)
    
# building darkflats if df_check is True
    if dark_flat_path is not None:
        darkflats = []
        for x in dark_flat_path:
            darkflat_data, exp_time_darkflat = load_image(x)
            if exp_match(exp_time_darkflat,exposure_flat):
                darkflats.append(darkflat_data)
        if len(darkflats) == 0:
            raise ValueError(f"No darkflats found with exposure {exposure_flat}s")
        darkflats = np.stack(darkflats)
        if darkflats.shape[1:] != flats.shape[1:]:
            raise ValueError(f"Dark-flat shape {darkflats.shape[1:]} does not match flat shape {flats.shape[1:]}")
        master_df = np.median(darkflats, axis=0)
        f_corr = master_f - master_df # corrected flats

    else:
        f_corr = master_f - exposure_flat*master_d/exposure_science
    
    
    med_corr = np.median(f_corr) # median of corrected flats for normalization

    if not np.isfinite(med_corr) or med_corr <=0: # error check whether the median is either not finite or is less or equal to zero
        raise ValueError("Flat correction median is not finite or <=0. Check flats/darkflats for saturation or mismatch.")
    
    f_norm = f_corr/med_corr # normalized corrected flats

    calibrated_images = []

    for image in raw_images:
        cal_image = (image - master_d)/f_norm
        calibrated_images.append(cal_image)
    calibrated_images = np.stack(calibrated_images)
    if dark_flat_path is not None :
        return  calibrated_images,science_filenames, raw_images,darks, flats, darkflats, master_d, master_f, master_df, f_norm
    else:
        return  calibrated_images,science_filenames, raw_images,darks, flats, master_d, master_f,f_norm



# Comparison plots of calibrated images vs the raw science images. 
# (calibrated images don't necessarily have to be the calibrated science images, one can also compare the darks, flats, etc. with the raw images)
def PlotCalibVsRaw(calibrated_images, raw_images, names, kcalib, kraw):
    """
    :param calibrated_images: Calibrated science images
    :param raw_images: Raw science images
    :param names: Names of both columns
    :param kcalib: Scaling of the calibrated images
    :param kraw: Scaling of the raw images
    :return:
    Null

    """
    if (kcalib<=0) or (kraw<=0):
        raise ValueError("kcalib and kraw must be greater than 0.")
    if raw_images.shape[0] != calibrated_images.shape[0]:
        raise ValueError("raw_images and calibrated_images must have the same number of frames")
    nb_images = calibrated_images.shape[0]
    fig, ax = plt.subplots(nb_images, 2, figsize=(15, 5*nb_images), squeeze=False)

    ax[0][0].set_title(f"{names[0]}")
    ax[0][1].set_title(f"{names[1]}")
    for i in range(nb_images):

      # i'th calibrated image
        vmin, vmax = (np.median(calibrated_images[i])-kcalib*np.std(calibrated_images[i]),np.median(calibrated_images[i])+kcalib*np.std(calibrated_images[i]))
        im = ax[i][0].imshow(calibrated_images[i], cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(im)
        ax[i][0].axis('off')


      #i'th raw image 
        vmin, vmax = (np.median(raw_images[i])-kraw*np.std(raw_images[i]),np.median(raw_images[i])+kraw*np.std(raw_images[i]))
        im = ax[i][1].imshow(raw_images[i], cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(im)
        ax[i][1].axis('off')

    plt.tight_layout()
    plt.show()
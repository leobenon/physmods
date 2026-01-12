import numpy as np

def savgol_filter(mess: np.array , polydeg : int, window_length: int): 
    '''
    Savitzky-Golay filtering with explicit edge handling.
    
    Parameters
    ----
    mess: ndarray, shape(N,)
        Measurements to be filtered.
    polydeg: int
        Degree of the polynomial to be used for the fit.
    window_length: int
        Window size, must be odd and >= 3.
    
    Returns
    ----
    smooth: ndarray
        Smoothed signal.
    res: ndarray
        Residuals, 'mess - smooth'.

    '''

    mess = np.asarray(mess,dtype=float)
    N = mess.size

    # Validation
    if not isinstance(window_length, int) or window_length<3 or window_length%2 == 0:
        raise ValueError(f"Window length must be an odd integer >=3. ")
    if not (0 <= polydeg < window_length):
        raise ValueError(f"The degree of the polynomial must be <= 0 and less than the window length.")
    if N < window_length:
        raise ValueError(f"The signal length ({N}) must be >= window length ({window_length})")

    half = window_length//2
    A = np.empty([window_length,polydeg+1])  # arbitrary design matrix to be later modified
    for i in range(window_length):
        for k in range(polydeg + 1):
            A[i,k] = (i-half)**k 
    C_mat = np.linalg.pinv(A)
    # the coefficients of the approximating polynomial
    coeff = C_mat[0,:] 
    # filtered values before filtering 
    smooth = np.zeros(len(mess)) 
    res = np.full(len(mess), np.nan)
    for k in range(half,len(mess)-half):
        smooth[k] = np.dot(coeff,mess[k -half: k + half + 1])
        # high pass filtered portion
        res[k] = mess[k] - smooth[k]  
    # coefficients of the polynomial
    coeff_x_init = C_mat@mess[:window_length]  
    coeff_x_end = C_mat@mess[-(window_length):]
    # row wise design matrix for the initial and end half width values 
    local_A = np.empty([polydeg+1]) 
    for m in range(half):
        for l in range(polydeg + 1):
            local_A[l] = (m-half)**l
        # boundary terms of the filtered values.
        smooth[m] = np.dot(local_A, coeff_x_init)  
        smooth[len(mess)-half+m] = np.dot(local_A, coeff_x_end)
        # residuals of the boundary values
        res[m] = mess[m]- smooth[m]    
        res[len(mess)-half+m] = mess[len(mess)-half+m]- smooth[len(mess)-half+m]
    return smooth, res
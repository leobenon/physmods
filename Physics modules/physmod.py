import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import astropy as ap
#   --------- To be implemented:  Outlier elimination, least squares method, error propagation numerical and symbolic, numerical integration (trapezodial etc), 
# numerical differentiation, solving differential equations numerically, symbolic differential equations ..........
class DataAnalysis:
    def __init__(self,measurements, n):
        self.measurements = measurements
        self.n = n
    def outlier_elimination(self):  # n: length of measurements 
        """
        Elimination of the outliers from the dataset to be analysed.
        -----------
        Parameters:
        -----------
        n: int
            The number of measurements/observations (must be a positive intiger).
        measurements: ndarray
            The dataset to be analysed.
        
        Returns:
        -------
        clean_measurements: ndarray
            The dataset cleansed of the outliers.
        """
        median = np.median(self.measurements[:,1])  # median of the measurements
        dev_median = abs(self.measurements[:,1]-median)  # deviation of the masurements from the median 
        sorted_dev_median = np.sort(dev_median) # sorted in ascending order
        std_idx = int(self.n*0.683) -1 # -1 because pyhton starts with the index 0 
        std_measurements = sorted_dev_median[std_idx]
        print(f'the standard deviation calculated from the deviation from the median is: {std_measurements}')
        del_idx_list = []
        for i in range(n):
            if self.measurements[i,1]< (median-4*std_measurements) or self.measurements[i,1]> (median +4*std_measurements):
                del_idx_list.append(i)
        clean_measurements = np.delete(self.measurements,del_idx_list,axis=0)
        return clean_measurements
    # fix the variables and the inputs later.
    def least_squares(self,n,u,var,obs, weight): # n: number of observations , u: number of parameters ==> degree of polynom = u -1, var: independent variable matrix of the system, obs: the real observation matrix which is dependent on var, weight: weight matrix
        """
        Least squares method
        -----------
        Parameters:
        -----------
        n: int
            The number of clean measurements/observations(must be a positive intiger).
        u: int
            The number of parameters to be determined (must be a positive intiger).
        var: arraylike
            The independent variable of the system (has to have a row count of `n`).
        obs: arraylike
            The actual observations/measurements dependent on `var` (has to have a row count of `n`).
        weight: arraylike
            The weight matrix of the observations/mesurements (array of size `n`x`n`).
    
        Returns:
        -------
        parameter: ndarray
            The determined parameters.
        obs_fitted: ndarray
            The observation/measurement array calculated with the determined parameters.
        residuals: ndarray
            The residuals of the real data after comparing with the fitted data.
        """
        des_mat = np.empty((n,u))  
        for i in range(n):
            for k in range(u):
                des_mat[i,k] = var[i]**k
        if (weight== np.eye(n)).all():
            parameter = np.linalg.inv(np.transpose(des_mat)@des_mat)@np.transpose(des_mat)@obs
        else:
            parameter = np.linalg.inv(np.transpose(des_mat)@weight@des_mat)@np.transpose(des_mat)@weight@obs
        obs_fitted =des_mat@parameter
        residuals = obs - obs_fitted
        return parameter, obs_fitted, residuals
    #def error_propagation(self,):
        
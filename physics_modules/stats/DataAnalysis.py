import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import astropy as ap
from typing import List
#   --------- To be implemented: DFT, FFT

#    TODO---- Make the explanations of the functions clearer!!!!!!
class DataAnalysis:
    def __init__(self,measurements):
        """
        Initializing the class.
        -----------
        Parameters
        -----------
        measurements: ndarray
            The dataset to be analysed as one column.
        errors: ndarray
            The corresponding errors of the measurements.
        """
        self.measurements = measurements
        self.n = len(measurements)  # number of data points to be analised.
    def outlier_elimination(self,scale=4, return_outliers = False):
        """
        Elimination of the outliers from the dataset to be analysed.
        -----------
        Parameters
        -----------
        measurements: ndarray
            The dataset to be analysed.
        scale: float
            The factor to scale the standard deviation with and should be determined by the user according to the data to be analysed. (By default scale=4)
        Returns
        -------
        clean_measurements: ndarray
            The dataset cleansed of the outliers.
        """
        if scale<=0:
            raise ValueError("Scale can not be non-positive")
        median = np.median(self.measurements)  # median of the measurements
        dev_median = np.abs(self.measurements-median)  # deviation of the masurements from the median 
        sorted_dev_median = np.sort(dev_median) # sorted in ascending order
        std_idx = int(self.n*0.683) -1 # -1 because pyhton starts with the index 0 
        std_like = sorted_dev_median[std_idx]
        print(f'the standard deviation calculated from the deviation from the median is: {std_like}')
        lower_bound = median-scale*std_like
        upper_bound = median +scale*std_like
        mask = (self.measurements >=lower_bound) & (self.measurements<=upper_bound)
        clean_measurements = self.measurements[mask]
        ind_outliers = np.where(~mask)[0]
        ind_nonoutliers = np.where(mask)[0]
        if return_outliers:
            outliers = self.measurements[~mask]
            return clean_measurements,ind_nonoutliers, outliers, ind_outliers
        return clean_measurements, ind_outliers
    
    def build_weight_matrix(self,errors=None,std_0=1):
        """
        Weight Matrix
        --------------------

        Parameters
        -----------
        std_0: float
            The error of the measurement with unit variance ratio. ===>>> std_i**2/std_0**2 = 1 (default is std_0 = 1)
        errors: arraylike or None
            The set of errors of the corresponding measurement, if None returns the identity matrix. (Optional, default is errors=None)
        
        Returns
        -------
        weight_mat: ndarray
            The weight matrix of the dataset.
        """
        if errors is None:
            weight_mat = np.eye(self.n)
            return weight_mat
        errors = np.array(errors)
        if len(errors)!=self.n:
            raise ValueError("The length of errors must match the number of measurements.")
        if np.any(errors<=0):
            raise ValueError("All errors values must be positive.")
        weight_mat = np.diag(std_0**2/errors**2)
        return weight_mat
    
    @staticmethod
    def linear_LS(n,u,obs,var,weight, err_par_return: bool = False, cov_mat_par_return: bool= False, m0_squared_return:bool = False):     # add optional returns such as m_0, covariance/cofactor matrix etc. add a MODEL TEST----
        """
        Least squares method
        -----------
        Parameters
        -----------
        n: int
            The number of clean measurements/observations(must be a positive integer).
        u: int
            The number of parameters to be determined (==> degree of polynomial = u -1) (must be a positive integer).        
        obs: arraylike
            The actual observations/measurements dependent on `var` (has to have a row count of `n`).
        var: arraylike
            The independent variable of the system (has to have a row count of `n`).
        weight: ndarray
            The weight matrix of the observations/measurements (array of size `n`x`n`).
    
        Returns
        -------
        parameter: ndarray
            The determined parameters.
        obs_fitted: ndarray
            The observation/measurement array calculated with the determined parameters.
        residuals: ndarray
            The residuals of the real data after comparing with the fitted data.
        """
        var = np.array(var)
        obs = np.array(obs)
        weight = np.array(weight)
        if not isinstance(n,int) or n<=0 :
            raise ValueError("The argument n must be a positive integer")
        if not isinstance(u,int) or u<=0 :
            raise ValueError("The argument u must be a positive integer")
        if len(var)!=n:
            raise ValueError("The length of variables must be equal to n")
        if len(obs)!=n:
            raise ValueError("The length of measurements must be equal to n")
        if weight.shape != (n,n):
            raise ValueError(f"The size of the weight matrix must be {n} x {n}")

        des_mat = np.empty((n,u))  
        for i in range(n):
            for k in range(u):
                des_mat[i,k] = var[i]**k
        if np.allclose(weight, np.eye(n)):
            parameter = np.linalg.inv(des_mat.T@des_mat)@des_mat.T@obs
            obs_fitted =des_mat@parameter
            residuals = obs - obs_fitted
            m_0_squared =(residuals.T @residuals)/(n-u)
            m_0 = np.sqrt(m_0_squared)
            cov_mat_par = m_0_squared* np.linalg.inv(des_mat.T@des_mat)
            err_par = np.sqrt(np.diag(cov_mat_par))
            cov_mat_obs_fitted = des_mat@cov_mat_par@des_mat.T
            err_obs_fitted = np.sqrt(np.diag(cov_mat_obs_fitted))

        else:
            parameter = np.linalg.inv(des_mat.T@weight@des_mat)@des_mat.T@weight@obs
            obs_fitted =des_mat@parameter
            residuals = obs - obs_fitted
            m_0_squared =(residuals.T @ weight @ residuals)/(n-u)
            m_0 = np.sqrt(m_0_squared)
            cov_mat_par = m_0_squared* np.linalg.inv(des_mat.T@weight@des_mat)
            err_par = np.sqrt(np.diag(cov_mat_par))
            cov_mat_obs_fitted = des_mat@cov_mat_par@des_mat.T
            err_obs_fitted = np.sqrt(np.diag(cov_mat_obs_fitted))
        if cov_mat_par_return==True and err_par_return==False and m0_squared_return == False:
            return parameter, obs_fitted, residuals,err_obs_fitted,cov_mat_par
        
        if err_par_return==True and m0_squared_return == False and cov_mat_par_return == False:
            return parameter, obs_fitted, residuals,err_obs_fitted,err_par
        
        if err_par_return==True and m0_squared_return == True and cov_mat_par_return == False:
            return parameter, obs_fitted, residuals,err_obs_fitted,err_par,m_0_squared
        
        if err_par_return==True and cov_mat_par_return==True and m0_squared_return==True:
            return parameter, obs_fitted, residuals, err_obs_fitted, err_par, cov_mat_par, m_0_squared
        
        else:
            return parameter, obs_fitted, residuals, err_obs_fitted
        
         
    
    @staticmethod
    def nonlinear_LS(x0: np.ndarray, data: np.ndarray, obs : np.ndarray, model_func, design_matrix_func, 
                     tol : float ,weight_matrix : np.ndarray = None, plot_func = None, diagonal_weight_matrix : bool = False, prior_info= None, max_iter : int = 25):
        """
        Performs an iterative non-linear Least squares to find an appropriate fit to a given model and 
        design matrix (jacobian) function until a difference of 'tol' is reached between the model function and 'obs'.
        
        Parameters
        ----
        x0: ndarray
            The initial parameter values to start the iteration.
        data: ndarray
            Values of the independent variable of the model function for different measurements.
        obs: ndarray
            The dependent observations where they represent the experimental measurements of the model function.
        model_func: function
            Function representing the model for the observations.
        design_matrix_func: function
            Function to compute the design matrix for current parameters.
        tol: float
            The criterion to stop the iteration.
        weight_matrix: ndarray
            The weigth matrix of the observation.
        plot_func: function (optional)
            The function to plot data during iterations if needed. (Default is None).

        Returns
        ---
        x: ndarray
            The final parameter estimates for the model fit.
        dx: ndarray
            The final parameter change to check if the process was succesful.
        n_iter: int
            Number of iterations performed .
        Q: ndarray
            The cofactor matrix to calculate errors and covariances if needed.
        m0_array: list
            Array containing every m0 values obtained during the iteration
        v: ndarray
            The linearized residuals.
        
        """
        n = len(obs)
        n_iter = 0
        dx = 10
        m0_array = []
        x = x0
        obs = np.asarray(obs,dtype=float)
        data = np.asarray(data,dtype=float)

        if weight_matrix is None:
            P = np.eye(len(obs))
        else:
            P = np.array(weight_matrix)
        if P.shape != (n,n):
            raise ValueError(f"P must be shape ({n},{n}), got {P.shape}.")
        if len(obs) != len(data) or len(obs) != len(model_func(x,data)):
            raise ValueError(f"The lengths of 'obs', 'data' and 'model func()' must be same")
        



        while True:
            # data residuals
            dl_data = obs - model_func(x, data)
            A_data  = design_matrix_func(x, data)

            # ----- soft prior block -----
            if prior_info is not None and len(prior_info) > 0:
            # prior_info = list of tuples like [(index, mean, sigma), ...]
                dl_prior = []           # will hold each prior residual(o-c)
                A_prior = []         # will hold each prior jacobian row
                for (idx, mean, sigma) in prior_info:
                    dl_prior.append(mean - x[idx])
                    A_prior.append(np.eye(len(x))[idx])   # derivative ∂r_prior/∂x = -1 for that parameter
                dl_prior = np.array(dl_prior) / np.array([p[2] for p in prior_info])
                A_prior = np.array(A_prior) / np.array([p[2] for p in prior_info])[:, None]

                # combine data + priors
                dl = np.concatenate([dl_data, dl_prior])
                A  = np.vstack([A_data, A_prior])
                A = np.array(A)
                if A.shape[0] != len(dl) or A.shape[1] != len(x):
                    raise ValueError(f"A must have {len(dl)} rows and {len(x)} columns, got {A.shape}.")
            
                if diagonal_weight_matrix is True:
                    w_data = np.sqrt(np.diag(P))  # weights diogonals
                    w_prior = np.ones(len(dl) - len(w_data))
                    w = np.concatenate([w_data,w_prior])
                    Wa = A * w[:,None]     # weighted jacobian
                    Wr = dl * w      # weighted o-c

                    Nw = Wa.T @ Wa  # weighted N 
                    Q = np.linalg.pinv(Nw,rcond=1e-12)  # cofactor

                    dx, *_ = np.linalg.lstsq(Wa,Wr, rcond=None) # without inversing
                    x = x+dx
                else:
                    # Normal equation system matrix for the current iteration
                    N = A.T @ P @ A
                    # Cofactor matrix for the current iteration
                    Q =  np.linalg.inv(N)
                    # Calculating the parameter change for the current iteration
                    dx = np.linalg.solve(N, A.T @ P @ dl)
                    # assigning the new parameters for the next iteration
                    x = x + dx

                   # iteration count
                    n_iter += 1
            
            
                # calculating the mean error of the weigth unit a posteriori 
                Wv = Wa @ dx - Wr   # weighted residuals
                dof = (len(dl)- len(x)) # degrees of freedom
                if dof<=0:
                    raise ValueError(f"Non-positive degrees of freedom: dof={dof}")
                m0_squared = Wv.T @ Wv/dof
                v = Wv
                m0 = np.sqrt(m0_squared)
                m0_array.append(m0)
            else:

                # observed - computed
                dl = dl_data
                # computing the design matrix for the current parameters
                A = A_data
                A = np.array(A)
                if A.shape[0] != len(dl) or A.shape[1] != len(x):
                    raise ValueError(f"A must have {len(dl)} rows and {len(x)} columns, got {A.shape}.")
            
                if diagonal_weight_matrix is True:
                    w = np.sqrt(np.diag(P))  # weights diogonals
                    Wa = A * w[:,None]     # weighted jacobian
                    Wr = dl * w      # weighted o-c

                    Nw = Wa.T @ Wa  # weighted N 
                    Q = np.linalg.pinv(Nw,rcond=1e-12)  # cofactor

                    dx, *_ = np.linalg.lstsq(Wa,Wr, rcond=None) # without inversing
                    x = x+dx
                else:
                    # Normal equation system matrix for the current iteration
                    N = A.T @ P @ A
                    # Cofactor matrix for the current iteration
                    Q =  np.linalg.pinv(N, rcond=1e-12)
                    # Calculating the parameter change for the current iteration
                    dx = Q @ (A.T @ P @ dl)
                    # assigning the new parameters for the next iteration
                    x = x + dx
            
                    # iteration count
                    n_iter += 1
            
            
                # calculating the mean error of the weigth unit a posteriori 
                v = np.array(A@dx - dl)
                m0_squared = v.T@ P @ v/(len(obs)- len(x))
                m0 = np.sqrt(m0_squared)
                m0_array.append(m0)
                obs_fitted = model_func(x,data)
            # plots the data if the function is provided
            if plot_func is not None:
                plot_func(data,obs,x,Q, m0_array[n_iter-1],n_iter-1)
            
            # stopping the iteration
            if np.all(np.abs(dx) <=tol) or n_iter>max_iter:
                return x,dx,n_iter,Q,m0_squared,v,obs_fitted
        


    @staticmethod
    def num_error_propagation(coeff_matrix : np.array, covariance_matrix : np.array): 
        """
        Propagation of Error
        --------------------
        Parameters
        -----------
        coeff_matrix: arraylike, shape (m,k)
            The coefficient matrix of the to be propagated matrix . The matrix `A` in the following expression ==>> B = Ax
        covariance_matrix: arraylike, shape (k,k)
            The covariance(/cofactor, depending on the problem at hand) matrix of the to be propagated matrix `x` in the above shown expression. 
    
        Returns
        -------
        cov_mat_new: ndarray
            The covariance or cofactor matrix of the calculated matrix `B` in the above shown expression.

        Note:
        -------
        If the expression is non-linear, linearize and use the jacobian matrix instead of the matrix `A`.

        """
        coeff_matrix = np.atleast_2d(np.array(coeff_matrix, dtype=float))
        covariance_matrix = np.array(covariance_matrix, dtype=float)
        if covariance_matrix.ndim == 1:
            covariance_matrix = np.diag(covariance_matrix)
        m,k = coeff_matrix.shape
        if covariance_matrix.shape != (k,k):
            raise ValueError(f"Covariance matrix must be of shape ({k},{k}) instead of {covariance_matrix.shape}.")
        cov_mat_new = coeff_matrix@covariance_matrix@coeff_matrix.T
        return cov_mat_new
    
    @staticmethod
    def symbolic_independent_err_prop(func: sp.Expr ,variables: List[str],errors: List[str], relative: bool = False):
        """
        Symbolic error propagation for independent variables.

        Parameters
        ----------
        func : sympy expression
            The function whose uncertainty is to be propagated.
        variables : list of str
            List of variable names as strings.
        errors : list of str
            List of associated errors of the variables as strings.
        relative : bool, optional
            If True, propagate relative errors instead of absolute.

        Returns
        -------
        ...
        """
        if not all(isinstance(v,str) for v in variables):
            raise TypeError("All elements of argument 'variables' must be strings.")
        if not all(isinstance(e,str) for e in errors):
            raise TypeError("All elements of argument 'errors' must be strings.")
        if len(errors) != len(variables):
            raise ValueError("Lengths of 'errors' and 'variables' must match.")
        syms = [sp.Symbol(v) for v in variables]
        partials = sp.Matrix([sp.diff(func,v) for v in syms])
        err_syms = sp.Matrix(sp.symbols(errors))
        if not relative:
            squared_terms = sp.Matrix(partials.multiply_elementwise(err_syms)).applyfunc(lambda t: t**2)
            sigma_f = sp.sqrt(squared_terms.sum())
            return sp.simplify(sigma_f)
        else:
            f = func
            rel_factors = sp.Matrix([(xi/f) * dfi for xi, dfi in zip(syms,partials)])
            rel_terms = sp.Matrix(rel_factors.multiply_elementwise(err_syms)).applyfunc(lambda t : t**2)
            sigma_f_rel = sp.sqrt(rel_terms.sum())
            return sp.simplify(sigma_f_rel)


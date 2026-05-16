import numpy as np

def get_init_parameters(path,id,frac_day_last):
    data = np.loadtxt(path, comments='#')
    t = data[:,0]
    v = data[:,1:4]
    r = data[:,4:7]
    w = data[:,7:10]
    q = data[:,10:14]
    print(f"Initial angular velocity components: {np.rad2deg(w[id])} rad/s")
    print(f"Initial velocity components: {v[id]/1000} km/s")
    print(f"Initial position components: {r[id]/1000} km")
    print(f"Initial time: {t[id]} s")
    print(f"Initial quaternions: {q[id]} ")
    q = q[id]
    t_init = t[id] # in seconds
    C_mat = np.array([[1 - 2*(q[2]**2 + q[3]**2), 2*(q[1]*q[2] + q[3]*q[0]), 2*(q[1]*q[3] - q[2]*q[0]) ],
                    [2*(q[2]*q[1] - q[3]*q[0]) , 1 - 2*(q[1]**2 + q[3]**2) , 2*(q[3]*q[2] + q[1]*q[0]) ],
                    [2*(q[1]*q[3] + q[2]*q[0]), 2*(q[2]*q[3] - q[1]*q[0]) , 1 - 2*(q[1]**2 + q[2]**2)]])
    print(f"C matrix computed with quaternions:\n {C_mat}")

    tan_psi = C_mat[0,1]/C_mat[0,0]
    psi = np.atan2(C_mat[0,1],C_mat[0,0])
    psi = np.rad2deg(psi)

    tan_phi = C_mat[1,2]/C_mat[2,2]
    phi = np.atan2(C_mat[1,2],C_mat[2,2])
    phi = np.rad2deg(phi)

    sin_theta = - C_mat[0,2]
    theta = - np.arcsin(C_mat[0,2])
    theta = np.rad2deg(theta)


    print(tan_psi,"(IRF) first angle=",psi)
    print(sin_theta, "(IRF) second angle=", theta)
    print(tan_phi, "(IRF) third angle=",phi)


    t_day = t_init/(60*60*24) + frac_day_last # days from initial year
    year2add  = t_day/365
    integer_part = int(year2add)
    fractional_part = year2add % 1
    print(f"Years from initial year: {year2add} years") # years from initial year
    print(f"Years from initial year (integer): {integer_part} years")
    print(f"Fractional day of the integer year: {365*fractional_part} days") # fractional day 
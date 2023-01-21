
import numpy as np

def vanderpol_dynamics(x, t, mu):
    
    x_1 = x[0]
    x_2 = x[1]
    
    dx_x_1_dt = x_2
    dx_x_2_dt = -1*x_1 + mu*(1 - x_1**2)*x_2
    
    return np.array([dx_x_1_dt, dx_x_2_dt])

def smd_dynamics(x, t, k, c):
    
    x_1 = x[0]
    x_2 = x[1]
    
    dx_x_1_dt = x_2
    dx_x_2_dt = -c*x_2 -  k*x_1
    
    return np.array([dx_x_1_dt, dx_x_2_dt])

    
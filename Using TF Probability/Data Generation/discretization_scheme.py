
import numpy as np
from dynamics_data_generation import smd_dynamics, vanderpol_dynamics

def forward_euler(dynamics, X0, T, n_total, k=0, c=0, mu= 0, beta = 0, n = 0, gamma = 0):
    
    if(dynamics== "smd_dynamics"):
        
        t = np.zeros(n_total + 1)
        x = np.zeros((2, n_total + 1))
        y = np.zeros(n_total + 1)
        
        x[:, 0] = X0
        t[0] = 0
        dt = T/n_total
        y[0] = x[0,0]
        
        for l in range(n_total):
            
            t[l+1] = t[l] + dt
            x[:, l+1] = x[:, l] + dt*smd_dynamics(x[:, l], t[l], k, c)
            y[l+1] = x[0, l+1]
            
            
    elif(dynamics== "vanderpol_dynamics"):
        
        t = np.zeros(n_total + 1)
        x = np.zeros((2, n_total + 1))
        y = np.zeros(n_total + 1)
        
        x[:, 0] = X0
        t[0] = 0
        dt = T/n_total
        y[0] = x[0,0]
        
        for l in range(n_total):
            
            t[l+1] = t[l] + dt
            x[:, l+1] = x[:, l] + dt*vanderpol_dynamics(x[:, l], t[l], mu)
            y[l+1] = x[0, l+1]
            
    elif(dynamics== "heat_eqn_dynamics"):
        
        t = np.zeros(n_total + 1)
        x = np.zeros((n_total + 1))
        y = np.zeros(n_total + 1)
        
        x[0] = X0
        t[0] = 0
        dt = T/n_total
        y[0] = x[0]

        for l in range(n_total): 
            
            t[l+1] = t[l] + dt
            x[l+1] = (((1/k)*((1/2.71)-1))-1)*t[l+1] + (1 + (1/k)) - (2.71**(-t[l+1]))/k 
            y[l+1] = x[l+1]
            
    else: 
        raise ValueError("Unknown Dynamics")
        
    return x, y, t
        
        
        
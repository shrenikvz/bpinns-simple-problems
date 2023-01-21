
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio

from dynamics_data_generation import vanderpol_dynamics, smd_dynamics 

from discretization_scheme import forward_euler

low_noise = sio.loadmat('low_noise.mat')
low_noise = np.transpose(low_noise['low_noise'])

high_noise = sio.loadmat('high_noise.mat')
high_noise = np.transpose(high_noise['high_noise'])

dyn = input("Dynamics you want to generate data for? ")

if dyn == "vanderpol_dynamics":
    
    mu = float(input("What should be the value of mu? "))
    
    # Sampling time is set as 0.01
    Ts = 0.01
    
    T = float(input("What should be the total duration? :"))
    
    n_total = int(T/Ts)
    
    time = np.linspace(0, T, n_total+1)
    
    x_init = np.array([1,0])
    
    x, y, t = forward_euler(dyn, X0 = x_init, T = T, n_total = n_total, mu = mu)
 
    y_meas_low_noise = y + low_noise[0:n_total + 1, 0]
    y_meas_high_noise = y + high_noise[0:n_total + 1, 0]
    y_meas = y
    
    np.savetxt('vanderpol_dataset.csv', (np.transpose(y_meas), np.transpose(y_meas_low_noise), np.transpose(y_meas_high_noise), t), delimiter = ',')
    
    plt.figure(dpi = 300)
    plt.plot(t, np.transpose(y_meas))
    plt.xlabel("Time (sec)")
    plt.ylabel('Output')
    plt.title('Van der Pol Oscillator')
    plt.grid()
    plt.savefig("output_vanderpol_no_noise.pdf")
    plt.show
    
    plt.figure(dpi = 300)
    plt.plot(t, np.transpose(y_meas_low_noise))
    plt.xlabel("Time (sec)")
    plt.ylabel('Output (low noise)')
    plt.title('Van der Pol Oscillator')
    plt.grid()
    plt.savefig("output_vanderpol_low_noise.pdf")
    plt.show
    
    plt.figure(dpi = 200)
    plt.plot(t, np.transpose(y_meas_high_noise))
    plt.xlabel("Time (sec)")
    plt.ylabel('Output (high noise)')
    plt.title('Van der Pol Oscillator')
    plt.grid()
    plt.savefig("output_vanderpol_high_noise.pdf")
    plt.show
    
elif dyn == "smd_dynamics":
    
    c = float(input("What should be the value of c? "))
    
    k = float(input("What should be the value of k? "))
    
    # Sampling time is set as 0.01
    Ts = 0.01
    
    T = float(input("What should be the total duration? :"))
    
    n_total = int(T/Ts)
    
    x_init = np.array([1,0])
    
    x, y, t = forward_euler(dyn, X0 = x_init, T = T, n_total = n_total, k = k, c = c)
    
    y_meas_low_noise = y + low_noise[0:n_total + 1, 0]
    y_meas_high_noise = y + high_noise[0:n_total + 1, 0]
    y_meas = y
    
    np.savetxt('smd_dataset.csv', (np.transpose(y_meas), np.transpose(y_meas_low_noise), np.transpose(y_meas_high_noise), t), delimiter = ',')
    
    plt.figure(dpi = 300)
    plt.plot(t, np.transpose(y_meas))
    plt.xlabel("Time (sec)")
    plt.ylabel('Output')
    plt.title('SMD System')
    plt.grid()
    plt.savefig("output_smd_no_noise.pdf")
    plt.show
    
    plt.figure(dpi = 300)
    plt.plot(t, np.transpose(y_meas_low_noise))
    plt.xlabel("Time (sec)")
    plt.ylabel('Output (low noise)')
    plt.title('SMD System')
    plt.grid()
    plt.savefig("output_smd_low_noise.pdf")
    plt.show
    
    plt.figure(dpi = 300)
    plt.plot(t, np.transpose(y_meas_high_noise))
    plt.xlabel("Time (sec)")
    plt.ylabel('Output (high noise)')
    plt.title('SMD System')
    plt.grid()
    plt.savefig("output_smd_high_noise.pdf")
    plt.show
    
    
elif dyn == "heat_eqn_dynamics":

    k = float(input("Value of thermal conductivity? "))
    
    # Sampling length is set as 0.01
    Ls = 0.01
    
    # Length of the rod is set as 1
    L = 1
    
    n_total = int(L/Ls)
    
    T_init = 1
    
    T, T_meas, x = forward_euler(dyn, X0 = T_init, T = L, n_total = n_total, k = k)
    
    T_meas_low_noise = T_meas + 0.1*low_noise[0:n_total + 1, 0]
    T_meas_high_noise = T_meas + 0.1*high_noise[0:n_total + 1, 0]
    
    np.savetxt('heat_eqn_dataset.csv', (np.transpose(T_meas), np.transpose(T_meas_low_noise), np.transpose(T_meas_high_noise), x), delimiter = ',')
    
    plt.figure(dpi = 200)
    plt.plot(x, np.transpose(T_meas))
    plt.xlabel("Length")
    plt.ylabel('Output')
    plt.title('1D Heat Eqn')
    plt.grid()
    plt.savefig("output_heat_eqn_no_noise.pdf")
    plt.show
    
    plt.figure(dpi = 200)
    plt.plot(x, np.transpose(T_meas_low_noise))
    plt.xlabel("Length")
    plt.ylabel('Output (low noise)')
    plt.title('1D Heat Eqn')
    plt.grid()
    plt.savefig("output_heat_eqn_low_noise.pdf")
    plt.show
    
    plt.figure(dpi = 200)
    plt.plot(x, np.transpose(T_meas_high_noise))
    plt.xlabel("Length")
    plt.ylabel('Output (high noise)')
    plt.title('1D Heat Eqn')
    plt.grid()
    plt.savefig("output_heat_eqn_high_noise.pdf")
    plt.show
    
else:
    
    raise ValueError("Unknown Dynamics")
    

    
    
    
    
    
    
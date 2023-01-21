'''
Some functions for plotting
    
'''

# Importing required packages

import matplotlib.pyplot as plt
import numpy as np

# Some functions for plotting

def plot_time_history_smd(N, T, Y_no_noise, Y, mu, std):

    color_light_green = [0.0, 0.6, 0.0]

    plt.figure(dpi=300)

    plt.plot(T[:N, :].numpy().flatten(), Y[:N, :].numpy().flatten(), linewidth=1.5, linestyle = "None", marker='1', markersize=3, color = (0.0, 0.0, 0.9), label=r'Training Data')
    plt.plot(T.numpy().flatten(), Y_no_noise.numpy().flatten(), linewidth=2, linestyle="dashed",  color=(0, 0, 0), label='Test Data')
    plt.fill_between(T.numpy().flatten(), (mu + 1.97*std).flatten(), (mu - 1.97*std).flatten(), zorder=1, alpha=0.50, color=color_light_green)
    plt.plot(T.numpy().flatten(), mu.flatten(), color=(0.8, 0.0, 0.0), linewidth=2, linestyle="--", zorder=3, alpha=1.0, label=r'B-PINN (95% Credible Interval)')

    plt.legend(loc='lower left')
    plt.xlabel("t (sec)", fontsize=20)
    plt.ylabel("x(t)", fontsize=20)
    plt.title('S-M-D System', fontsize=20) 
    plt.tight_layout()
    plt.grid()
    plt.savefig('BPINN_result_smd.pdf')
    plt.show()


def plot_time_history_1d_heat(N, T, Y_no_noise, Y, mu, std):

    color_light_green = [0.0, 0.6, 0.0]

    plt.figure(dpi=300)

    plt.plot(T[:N, :].numpy().flatten(), Y[:N, :].numpy().flatten(), linewidth=1.5, linestyle = "None", marker='1', markersize=3, color = (0.0, 0.0, 0.9), label=r'Training Data')
    plt.plot(T.numpy().flatten(), Y_no_noise.numpy().flatten(), linewidth=2, linestyle="dashed",  color=(0, 0, 0), label='Test Data')
    plt.fill_between(T.numpy().flatten(), (mu + 1.97*std).flatten(), (mu - 1.97*std).flatten(), zorder=1, alpha=0.50, color=color_light_green)
    plt.plot(T.numpy().flatten(), mu.flatten(), color=(0.8, 0.0, 0.0), linewidth=2, linestyle="--", zorder=3, alpha=1.0, label=r'B-PINN (95% Credible Interval)')

    plt.legend(loc='lower left')
    plt.xlabel("x", fontsize=20)
    plt.ylabel("T(x)", fontsize=20)
    plt.title('1-D Heat Eqn', fontsize=20)
    plt.tight_layout()
    plt.grid()
    plt.savefig('BPINN_result_1d_heat.pdf')
    plt.show()
    
def plot_time_history_vanderpol(N, T, Y_no_noise, Y, mu, std):

    color_light_green = [0.0, 0.6, 0.0]

    plt.figure(dpi=300)

    plt.plot(T[:N, :].numpy().flatten(), Y[:N, :].numpy().flatten(), linewidth=1.5, linestyle = "None", marker='1', markersize=3, color = (0.0, 0.0, 0.9), label=r'Training Data')
    plt.plot(T.numpy().flatten(), Y_no_noise.numpy().flatten(), linewidth=2, linestyle="dashed",  color=(0, 0, 0), label='Test Data')
    plt.fill_between(T.numpy().flatten(), (mu + 1.97*std).flatten(), (mu - 1.97*std).flatten(), zorder=1, alpha=0.50, color=color_light_green)
    plt.plot(T.numpy().flatten(), mu.flatten(), color=(0.8, 0.0, 0.0), linewidth=2, linestyle="--", zorder=3, alpha=1.0, label=r'B-PINN (95% Credible Interval)')

    plt.legend(loc='upper right')
    plt.xlabel("t (sec)", fontsize=20)
    plt.ylabel("x(t)", fontsize=20)
    plt.title('Van der Pol Oscillator', fontsize=20)
    #plt.tight_layout()
    plt.grid()
    plt.savefig('BPINN_result_vanderpol.pdf')
    plt.show()
    
    
# Histogram Plots

def histogram_smd(prior, posterior, variable, color, num_bin):

    plt.figure(dpi=300)
    
    plt.hist(posterior, bins=num_bin, density=True, color=color, alpha=0.5)
    mean = np.round(np.mean(posterior), 3)
    std = np.round(np.std(posterior), 2)
    plt.title(variable + ' = ' + str(mean) + '$\pm$' + str(std), fontsize=20, color=color)
    plt.ylabel('Posterior of ' + variable, fontsize=20)
    plt.grid()   
    
    if variable == "$c$":
        plt.savefig("posterior_c_histogram_smd.pdf")
    elif variable == "$k$":
        plt.savefig("posterior_k_histogram_smd.pdf", format= 'pdf')
        
        plt.show()
    
def histogram_1d_heat(prior, posterior, variable, color, num_bin):

    plt.figure(dpi=300)
    
    plt.hist(posterior.T, bins=num_bin, density=True, color=color, alpha=0.5)
    mean = np.round(np.mean(posterior.T), 3)
    std = np.round(np.std(posterior.T), 2)
    plt.title(variable + ' = ' + str(mean) + '$\pm$' + str(std), fontsize=20, color=color)
    plt.ylabel('Posterior of ' + variable, fontsize=20)
    plt.grid()
    plt.savefig('posterior_k_histogram_1d_heat.pdf')
    plt.show()
    
    
def histogram_vanderpol(prior, posterior, variable, color, num_bin):

    plt.figure(dpi=300)
    
    plt.hist(posterior, bins=num_bin, density=True, color=color, alpha=0.5)
    mean = np.round(np.mean(posterior), 4)
    std = np.round(np.std(posterior), 4)
    plt.title(variable + ' = ' + str(mean) + '$\pm$' + str(std), fontsize=20, color=color)
    plt.ylabel('Posterior of ' + variable, fontsize=20)
    plt.grid()        
    plt.show()
    
    

    

    

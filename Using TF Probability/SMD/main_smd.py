'''
Main File for BPINNs on S-M-D System

'''
# TOY PROBLEM: Spring Mass Damper System

# Loading required packages

import numpy as np
import tensorflow as tf

import bayesian_neural_network
import bayesian_smd
import sampler
from dynamics import smd_dynamics
from plotting_func import plot_time_history_smd, histogram_smd

# To prevent some errors
import collections
collections.Iterable = collections.abc.Iterable

# Loading the data
Data = np.genfromtxt('smd_dataset.csv', delimiter = ',')

# Output with no noise
Y_no_noise = tf.constant(np.transpose(Data[0,1:]).reshape(-1,1), dtype=tf.float32)
# Output with low noise
Y_low_noise = tf.constant(np.transpose(Data[1,1:]).reshape(-1,1), dtype=tf.float32)
# Output with high noise
Y_high_noise = tf.constant(np.transpose(Data[2,1:]).reshape(-1,1), dtype=tf.float32)
# Time (Ts = 0.01)
T = tf.constant(np.transpose(Data[3,1:]).reshape(-1,1), dtype=tf.float32)

Y = Y_no_noise   ##### <--------- Change me depending on the training data you want to use 

def main(N):

    BNN = bayesian_neural_network.BNN(layers=[1, 32, 32, 1])        # 2 internal layers with 32 hidden units each
    
    noise_ode = 0.1    
    noise_data = 0.1
    num_collocation = 1000            # For physics likelihood
    # Create Bayesian Model
    model = bayesian_smd.Physics_Informed_Bayesian(
        x_u=T[:N,:],                     # Output Data
        y_u=Y[:N,:],                     # Input Data
        x_pde=T[:num_collocation],                         # Input Dynamics
        pde_fn=smd_dynamics,             # Dynamics
        L=3,                             # Number of Internal Layers - 1
        noise_data=noise_data,           # Standard deviation for likelihood of observation
        c_init_mean = 0.3,               # Log Normal Mean of Prior of the damping constant   
        k_init_mean = 8.0,              # Log Normal Mean of Prior of the spring constant   
        noise_pde=noise_ode,             # Standard deviations for likelihood of PDE's/ODE's
        sigma_network=2,                 # Prior standard deviation for weight and biased of the network
        sigma_physical_parameters = 0.5  # Prior standard deviation for physical parameters
    )
    # Log Posterior
    log_posterior = model.posterior(BNN.bnn_fn)

    # HMC sampler
    hmc_kernel = sampler.AdaptiveHMC(
        target_log_prob_fn=log_posterior,                   # Log of Posterior
        init_state=BNN.variables+model.additional_inits,
        num_results=3000,                                   # Number of samples to estimate the parameters
        num_burnin=4000,                                    # Number of samples to tune the sampler
        num_leapfrog_steps=50,
    )
    
    samples, results = hmc_kernel.run_chain()
    
    u_pred = BNN.bnn_infer_fn(T, samples[:6])
    
    mu = tf.reduce_mean(u_pred, axis=0).numpy()
    
    std = tf.math.reduce_std(u_pred, axis=0).numpy()

    return model, samples, u_pred, mu, std


if __name__ == '__main__':
    
    N=1000                       # Maximum number of data points for calculating the likelihood of the data
    
    model, samples, u_pred, mu, std = main(N)
    
    plot_time_history_smd(N, T, Y_no_noise, Y, mu, std)

log_c, log_k = samples[-2:]

c, k = tf.exp(log_c+model.log_c_init), tf.exp(log_k+model.log_k_init)

# Histogram of the Posterior of Physical Parameters

num_bin = 30

s = model.additional_priors[0].sample(3000)
prior = (tf.exp(s + model.log_c_init)).numpy()
posterior = c.numpy()
variable = '$c$'
histogram_smd(prior, posterior, variable, 'tab:purple', num_bin)

s = model.additional_priors[1].sample(3000)
prior = (tf.exp(s + model.log_k_init)).numpy()
posterior = k.numpy()
variable = '$k$'
histogram_smd(prior, posterior, variable, 'tab:purple', num_bin)


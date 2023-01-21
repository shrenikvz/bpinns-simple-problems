'''
Main File for BPINNs on 1D Heat Eqn

'''
# TOY PROBLEM: 1D Heat Equation

# Loading required packages

import numpy as np
import tensorflow as tf

import bayesian_neural_network
import bayesian_1d_heat
import sampler
from dynamics import heat_eqn_dynamics
from plotting_func import plot_time_history_1d_heat, histogram_1d_heat

# To prevent some errors
import collections
collections.Iterable = collections.abc.Iterable

Data = np.genfromtxt('heat_eqn_dataset.csv', delimiter = ',')

Y_no_noise = tf.constant(np.transpose(Data[0,1:]).reshape(-1,1), dtype=tf.float32)
Y_low_noise = tf.constant(np.transpose(Data[1,1:]).reshape(-1,1), dtype=tf.float32)
Y_high_noise = tf.constant(np.transpose(Data[2,1:]).reshape(-1,1), dtype=tf.float32)
T = tf.constant(np.transpose(Data[3,1:]).reshape(-1,1), dtype=tf.float32)


Y = Y_high_noise   ##### <--------- Change me depending on the training data you want to use 


def main(N):

    BNN = bayesian_neural_network.BNN(layers=[1, 32, 32, 1])
    
    noise_ode = 0.1
    noise_data = 0.1
    num_collocation = 1
    # create Bayesian model
    model = bayesian_1d_heat.Physics_Informed_Bayesian(
        x_u=T[:N,:],
        y_u=Y[:N,:],            
        x_pde=T[:num_collocation],
        pde_fn=heat_eqn_dynamics,
        L=3,                    # Number of Layers - 1
        noise_data=noise_data,  # Standard deviation for likelihood of observation
        k_init_mean = 0.25,
        noise_pde=noise_ode,    # Standard deviations for likelihood of PDE's/ODE's
        sigma_network=2,          # Standard deviation for weight and biased of the network
        sigma_physical_parameters = 0.5 # Prior standard deviation for physical parameters
    )
    # Log Posterior
    log_posterior = model.posterior(BNN.bnn_fn)

    # HMC sampler
    hmc_kernel = sampler.AdaptiveHMC(
        target_log_prob_fn=log_posterior,       # Log of Posterior
        init_state=BNN.variables+model.additional_inits,
        num_results=3000,           # Number of samples to estimate the parameters
        num_burnin=4000,            # Number of samples to tune the sampler
        num_leapfrog_steps=50,
    )
    
    samples, results = hmc_kernel.run_chain()
    
    Acceptance_rate = np.mean(results.inner_results.is_accepted.numpy())
    
    print('Acceptance Rate: ', Acceptance_rate)
    
    print(results.inner_results.accepted_results.step_size[0].numpy())
    
    u_pred = BNN.bnn_infer_fn(T, samples[:6])
    
    mu = tf.reduce_mean(u_pred, axis=0).numpy()
    
    std = tf.math.reduce_std(u_pred, axis=0).numpy()

    return model, samples, u_pred, mu, std, Acceptance_rate


if __name__ == '__main__':
    
    N=60
    
    model, samples, u_pred, mu, std, Acceptance_rate = main(N)
    
    plot_time_history_1d_heat(N, T, Y_no_noise, Y, mu, std)

log_k = samples[-1:]

k = tf.exp(log_k+model.log_k_init)

# Histogram of the Posterior of Parameters

num_bin = 30

s = model.additional_priors[0].sample(3000)
prior = (tf.exp(s + model.log_k_init)).numpy()
posterior = k.numpy()
variable = '$k$'
histogram_1d_heat(prior, posterior, variable, 'tab:blue', num_bin)


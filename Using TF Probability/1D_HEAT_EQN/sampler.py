'''
HMC Sampler

'''
# Importing required packages

import tensorflow as tf
import tensorflow_probability as tfp

# To prevent some errors
import collections
collections.Iterable = collections.abc.Iterable

# Hamitonian Monte Carlo Sampler

class AdaptiveHMC:
    
    def __init__(
        self,
        target_log_prob_fn,     # Target Distribution
        init_state,             # Initial state of the chain
        num_results=1000,       # Number of samples to generate
        num_burnin=1000,        # Number of samples to tune
        num_leapfrog_steps=30,
        step_size=0.1,          # Initial step size to use for HMC kernel
    ):
        self.target_log_prob_fn = target_log_prob_fn
        self.init_state = init_state       # Initializes the class and set ups the MCMC transition kernel
        self.kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.target_log_prob_fn,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=step_size,
            ),
            num_adaptation_steps=int(0.8 * num_burnin),
            target_accept_prob=0.75,
        )
        self.num_results = num_results
        self.num_burnin = num_burnin

    @tf.function
    def run_chain(self):                # Runs MCMC chain to sample from the target distribution
        samples, results = tfp.mcmc.sample_chain(
            num_results=self.num_results,
            num_burnin_steps=self.num_burnin,
            current_state=self.init_state,
            kernel=self.kernel,
        )
        return samples, results



'''
Class for performing Physics Informed Bayesian Inference


'''
# Importing required packages

import tensorflow as tf
import tensorflow_probability as tfp


class Physics_Informed_Bayesian:        
    
    def __init__(
        self,
        x_u,
        y_u,
        x_pde,
        pde_fn,
        L=3,
        noise_data=0.05,
        c_init_mean = 0.5,
        k_init_mean = 10.0,
        noise_pde=0.05,
        sigma_network=1.0,
        sigma_physical_parameters=0.5
    ):
        
        self.x_u = x_u
        self.y_u = y_u
        self.x_pde = x_pde

        self.pde_fn = pde_fn
        self.L = L
        self.noise_data = noise_data
        self.noise_pde = noise_pde
        self.sigma_network = sigma_network
        self.sigma_physical_parameters = sigma_physical_parameters
        
        self.c_init_mean = c_init_mean
        self.k_init_mean = k_init_mean

        self.log_c_init = tf.math.log(self.c_init_mean)
        self.log_k_init = tf.math.log(self.k_init_mean)

        self.additional_inits = [self.log_c_init, self.log_k_init]
        self.additional_priors = [
            tfp.distributions.Normal(0, scale=self.sigma_physical_parameters),
            tfp.distributions.Normal(0, scale=self.sigma_physical_parameters),
        ]

    def posterior(self, bnn_fn):
        
        y_u = tf.constant(self.y_u, dtype=tf.float32)

        def _fn(*variables):

            # Spliting the input list into variables for neural networks and additional variables
            variables_nn = variables[: 2 * self.L]
            log_c, log_k= variables[2 * self.L :]
            c, k = (
                tf.exp(log_c + self.log_c_init),
                tf.exp(log_k + self.log_k_init),
            )
            
            # Explicitly creates a tf.Tensor, for input to neural networks to avoid bugs
            x_u = tf.constant(self.x_u, dtype=tf.float32)
            x_pde = tf.constant(self.x_pde, dtype=tf.float32)

            # Making Inference
            _fn = lambda x: bnn_fn(x, variables_nn)
            y_u_pred = _fn(x_u)                                 # Output of the neural network
            pde_pred = self.pde_fn(x_pde, _fn, [c, k])          # Physics residual

            # Constructs Prior Distributions and Likelihood Distributions
            u_likeli = tfp.distributions.Normal(loc=y_u, scale=self.noise_data * tf.ones_like(y_u))   # Likelihood of data 
            noise_pde = self.noise_pde
            num_collocation = pde_pred.shape[0]
            pde_likeli = tfp.distributions.Normal(                          
                loc=tf.zeros([num_collocation, 1]), scale=noise_pde * tf.ones([num_collocation, 1])
            )


            prior = tfp.distributions.Normal(loc=0, scale=self.sigma_network)    # Prior for weight and biases of network

            # Log Prior
            log_prior = tf.reduce_sum(
                [tf.reduce_sum(prior.log_prob(var)) for var in variables_nn]
            ) + tf.reduce_sum(
                [
                    dist.log_prob(v)
                    for v, dist in zip([log_c, log_k], self.additional_priors)
                ]
            )

                
            # Log Likelihood
            
            log_likelihood = (
                tf.reduce_sum(u_likeli.log_prob(y_u_pred))
                + tf.reduce_sum(pde_likeli.log_prob(pde_pred))
            )
            
            # Computing Log Posterior, by adding Log Prior and Log Likelihood
            return log_prior + log_likelihood

        return _fn

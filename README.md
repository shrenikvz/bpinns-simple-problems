# Bayesian physics informed neural networks for dynamical systems (B-PINNs)

This repository includes code for evaluating the effectiveness of B-PINNs in simulating various simple problems using data with noise.

## Simple problems considered

- Spring Mass Damper System

- 1D Steady State Heat Equation 

Different Measurement Noise Levels used in the training data.

- No Noise

- Low Noise (S: 0.001, $\omega_\text{cut,in}$: 3.14 rad/s, $\omega_\text{cut,off}$: 50 rad/s)

- High Noise (S: 0.01, $\omega_\text{cut,in}$: 3.14 rad/s, $\omega_\text{cut,off}$: 50 rad/s)

In this repo, we integrate data, physics, and uncertainties by combining neural networks, physics informed modeling, and Bayesian inference to improve the predictive potential of traditional neural network models. 






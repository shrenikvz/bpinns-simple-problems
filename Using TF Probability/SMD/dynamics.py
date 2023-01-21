'''
Functions for defining the physics of different dynamics
    
'''

# Importing required packages

import tensorflow as tf


# Spring Mass Damper Dynamics

# m*x_tt + k*x + c*x_t = 0 (m = 1)

def smd_dynamics(t, fn, additional_variables):
    c, k = additional_variables
    with tf.GradientTape() as g_tt:
        g_tt.watch(t)
        with tf.GradientTape() as g_t:
            g_t.watch(t)
            x = fn(t)
        x_t = g_t.gradient(x, t)
    x_tt = g_tt.gradient(x_t, t)
    f = 1/k*x_tt + c/k*x_t + x
    return f


# 1D Heat Equation Dynamics

# k*T_xx + exp(-x) = 0


def heat_eqn_dynamics(t, fn, additional_variable):      # t here is the distance and not time
    
    k = additional_variable     # Thermal conductivity
    
    with tf.GradientTape() as g_tt:
        g_tt.watch(t)
        with tf.GradientTape() as g_t:
            g_t.watch(t)
            x = fn(t)
        x_t = g_t.gradient(x, t)
    x_tt = g_tt.gradient(x_t, t)
    f = k*x_tt + tf.math.exp(-1*t)

    return f

# Van der Pol Dynamics

# m*x_tt - mu*(1 - x^2)*x_t + x = 0

def vanderpol_dynamics(t, fn, additional_variable):
    
    mu_parameter = additional_variable
    
    with tf.GradientTape() as g_tt:
        g_tt.watch(t)
        with tf.GradientTape() as g_t:
            g_t.watch(t)
            x = fn(t)
        x_t = g_t.gradient(x, t)
    x_tt = g_tt.gradient(x_t, t)
    f = x_tt - mu_parameter*(1 - x**2)*x_t + x

    return f
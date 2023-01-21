'''
Defines the Bayesian Neural Network

'''


# Importing required packages

import tensorflow as tf

# To prevent some errors

import collections
collections.Iterable = collections.abc.Iterable

# Class for Bayesian Neural Network

class BNN:
    
    def __init__(self, layers, activation=tf.tanh):
        self.L = len(layers) - 1
        self.variables = self.init_network(layers)
        self.bnn_fn = self.bnn()
        self.bnn_infer_fn = self.infer()
        self.activation = activation

    def init_network(self, layers):
        W, b = [], []
        init = tf.zeros
        for i in range(self.L):
            W += [init(shape=[layers[i], layers[i + 1]], dtype=tf.float32)]
            b += [tf.zeros(shape=[1, layers[i + 1]], dtype=tf.float32)]
        return W + b

    def bnn(self):
        def _fn(x, variables):
            
            W = variables[: len(variables) // 2]
            b = variables[len(variables) // 2 :]
            y = x
            for i in range(self.L - 1):
                y = self.activation(tf.matmul(y, W[i]) + b[i])
            return tf.matmul(y, W[-1]) + b[-1]

        return _fn

    def infer(self):
        def _fn(x, variables):
            
            W = variables[: len(variables) // 2]
            b = variables[len(variables) // 2 :]
            batch_size = W[0].shape[0]
            y = tf.tile(x[None, :, :], [batch_size, 1, 1])
            for i in range(self.L - 1):
                y = self.activation(tf.einsum("Nij,Njk->Nik", y, W[i]) + b[i])
            return tf.einsum("Nij,Njk->Nik", y, W[-1]) + b[-1]

        return _fn

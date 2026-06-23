import pandas as pd 
import numpy as np

"""
FFN:

Takes input
Projects up to a higher dim with a learnt weight matrix
Adds Bias
Applies a nonlinearity
Projects down to a the same original dim with another learnt weight matrix
Adds thes second Bias

It is called a feed forward neural network, due to its similarity of struture



"""



class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = np.random.randn(d_model, d_ff) * 0.01 #size up
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01 #size down
        self.b2 = np.zeros(d_model)
    def forward(self, x):
        self.x = x
        z1 = np.dot(x, self.W1) + self.b1
        self.z1 = z1
        a1 = np.maximum(0, z1) #Non linearity (ReLU Rectified Linear Unit Function)
        self.a1 = a1
        z2 = np.dot(a1, self.W2) + self.b2
        self.z2 = z2
        return z2 
    def backward(self, dout):
        d_a1 = dout @ self.W2.T #z2 = a1 @ W + b2
        self.d_W2 = self.a1.reshape(-1, self.d_ff).T @ dout.reshape(-1, self.d_model)
        self.d_b2 = np.sum(dout, axis=(0, 1)) #bias shared across batch

        d_z1 = d_a1 * (self.z1 > 0) #derivative of ReLU, causes dying ReLU problem but more simple
        d_x = d_z1 @ self.W1.T #z1 = x @ W + b1
        self.d_W1 = self.x.reshape(-1, self.d_model).T @ d_z1.reshape(-1, self.d_ff)
        self.d_b1 = np.sum(d_z1, axis=(0, 1))

        return d_x

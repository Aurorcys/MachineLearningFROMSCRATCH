import pandas as pd 
import numpy as np 


"""
LayerNorm:
take mean
take var
take normalize
times gamma + beta


learnt:
gamma
beta

"""

class LayerNorm:
    def __init__(self, d_model):
        self.d_model = d_model
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self.var = var
        x_norm = (x - mean) / np.sqrt(var + 1e-5) #add epsilon to avoid division by zero
        self.norm = x_norm
        return self.gamma * x_norm + self.beta
    def backward(self, dout):
        """
        Easy chain rule work
        """
        d_gamma = np.sum(self.norm * dout, axis=(0,1)) #sum over batch and seq
        d_beta = np.sum(dout, axis=(0,1)) #its a np.sum as gamma and beta is same fo every eg
        self.d_gamma = d_gamma
        self.d_beta = d_beta
        d_x_norm = dout * self.gamma
        std = np.sqrt(self.var + 1e-5)
        d_x = (1.0 / std) * (
            d_x_norm
            - np.mean(d_x_norm, axis=-1, keepdims=True)
            - self.norm * np.mean(d_x_norm * self.norm, axis=-1, keepdims=True) #not sum all cuz dependant on input
        )
        return d_x

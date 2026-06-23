import pandas as pd 
import numpy as np 

class LayerNorm:
    def __init__(self, d_model):
        self.d_model = d_model
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        self.var = var
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        self.norm = x_norm
        return self.gamma * x_norm + self.beta
    
    
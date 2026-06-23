import pandas as pd 
import numpy as np 


#projects higher dim, non-linearity, lower dim, done

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
        # mn * nm = mm, dd * df = df * fd = dd

    def forward(self, X):
        self.x = X
        z1 = np.dot(X, self.W1) + self.b1
        self.z1 = z1
        a1 = np.maximum(0, z1)
        self.a1 = a1
        z2 = np.dot(a1, self.W2) + self.b2
        self.z2 = z2
        return z2
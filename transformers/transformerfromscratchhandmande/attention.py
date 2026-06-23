import pandas as pd 
import numpy as np 

class Attention:
    def __init__(self, d_model, num_heads=4, max_seq_len=100):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01

        self.casual_mask = np.triu(np.ones((max_seq_len, max_seq_len)) * -np.inf, k=1) #prevent attending to future tokens

    def forward(self, x, token_ids):
        batch_size, seq_len, _ = x.shape
        self.x = x

        pad_mask = (token_ids != 0) #pad stuff

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3) #now all three (batch, heads, seq, dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        self.Q_heads = Q
        self.K_heads = K
        self.V_heads = V

        attn_output = self._scaled_dot_product_attention(Q, K, V, pad_mask)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model) #recombine
        self.concat_outputs = attn_output
        return attn_output @ self.W_o   #puts the information together from separate heads
    
    def _scaled_dot_product_attention(self, Q, K, V, pad_mask):
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim) # (batch, heads, seq, seq)

        scores = scores + self.casual_mask[:scores.shape[2], :scores.shape[3]]  # Apply casual mask

        pad_mask = pad_mask[:, np.newaxis, np.newaxis, :]  # (batch, 1, 1, seq)
        scores = np.where(pad_mask, scores, -np.inf) #condition, if true, if false

        attn_weights = self.softmax(scores)
        self.scores = scores
        self.attn_weights = attn_weights
        self.pad_mask_expanded = pad_mask
        return attn_weights @ V


    def softmax(self, X):
        X_max = np.max(X, axis=-1, keepdims=True)
        # Handle all -inf rows
        X = np.where(np.isinf(X_max), 0.0, X - X_max)
        X = np.exp(X)
        denom = np.sum(X, axis=-1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        return X / denom
    
    def backward(self, d_out):
        batch_size, seq_len, _ = d_out.shape

        d_concat = d_out @ self.W_o.T
        self.d_W_o = self.concat_outputs.reshape(-1, self.d_model).T @ d_out.reshape(-1, self.d_model)

        d_heads = d_concat.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        d_heads = d_heads.transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

        self.d_attn_weights = d_heads @ self.V_heads.transpose(0, 1, 3, 2)
        d_V = self.attn_weights.transpose(0, 1, 3, 2) @ d_heads  # (batch, heads, seq, head_dim)

        s = self.attn_weights
        d_scores = s * (self.d_attn_weights - np.sum(self.d_attn_weights * s, axis =-1, keepdims=True))
        d_scores = np.where(np.isfinite(self.scores), d_scores, 0.0)

        d_Q = d_scores @ self.K_heads / np.sqrt(self.head_dim) #going backwards
        d_K = d_scores.transpose(0, 1, 3, 2) @ self.Q_heads / np.sqrt(self.head_dim)
        
        #reshaping
        d_Q = d_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        d_K = d_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        d_V = d_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)


        #Linear projections
        self.d_W_q = self.x.reshape(-1, self.d_model).T @ d_Q.reshape(-1, self.d_model)
        self.d_W_k = self.x.reshape(-1, self.d_model).T @ d_K.reshape(-1, self.d_model)
        self.d_W_v = self.x.reshape(-1, self.d_model).T @ d_V.reshape(-1, self.d_model) #-1 mean to figure it out, NOT LAST DIM

        #d_x = sum of gradients from Q K V paths
        d_x = d_Q @ self.W_q.T + d_K @ self.W_k.T + d_V @ self.W_v.T
        return d_x
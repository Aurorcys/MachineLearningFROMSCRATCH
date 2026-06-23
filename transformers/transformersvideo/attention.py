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

        self.casual_mask = np.triu(np.ones((max_seq_len, max_seq_len)) * -np.inf, k=1)

    def forward(self, x, token_ids):
        batch_size, seq_len, _ = x.shape #batch, seq, d_model
        self.x = x

        pad_mask = (token_ids != 0) #pad = 0 if pad, then false, if not pad then true

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        self.Q_heads = Q
        self.K_heads = K
        self.V_heads = V

        attn_output = self._scaled_dot_product_attention(Q, K, V, pad_mask)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model) #concatenate
        self.concat_outputs = attn_output
        return attn_output @ self.W_o
    
    def _scaled_dot_product_attention(self, Q, K, V, pad_mask):
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)

        scores = scores + self.casual_mask[:scores.shape[2], :scores.shape[3]]

        pad_mask = pad_mask[:, np.newaxis, np.newaxis, :]

        scores = np.where(pad_mask, scores, -np.inf)

        attn_weights = self.softmax(scores)
        self.scores = scores
        self.attn_weights = attn_weights
        self.pad_mask_expanded = pad_mask
        return attn_weights @ V
    
    def softmax(self, X):
        X_max = np.max(X, axis=-1, keepdims=True)
        X = np.where(np.isinf(X_max), 0.0, X - X_max)
        X = np.exp(X)
        denom = np.sum(X, axis=-1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        return X / denom


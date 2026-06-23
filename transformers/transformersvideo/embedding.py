import pandas as pd 
import numpy as np


class NumpyEmbedding:
    def __init__(self, vocab_size, embedding_dim):
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.inputs = None
    def forward(self, inputs):
        self.inputs = np.asarray(inputs, dtype=np.int32)
        return self.weights[self.inputs]





class Embedding:
    def __init__(self, d_model, vocab_size, max_seq_len):
        self.word_embeddings = NumpyEmbedding(vocab_size, d_model)
        self.pe = self._positional_embeddings(max_seq_len, d_model)
    def _positional_embeddings(self, max_seq_len, d_model):
        pe = np.zeros((max_seq_len, d_model))
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        #e^lnx = x e^(-ln(10000)*i / d_model)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
    def forward(self, inputs):
        word_embeddings = self.word_embeddings.forward(inputs)
        return word_embeddings + self.pe[:inputs.shape[1], :]
import pandas as pd 
import numpy as np


class EntropyLoss:
    def forward(self, logits, target):
        self.logits = logits
        self.target = target

        shifted = logits - np.max(logits, axis=-1, keepdims=True) #numerical stability
        exp_logits = np.exp(shifted)
        self.probs = exp_logits / np.sum(exp_logits,  axis=-1, keepdims=True)

        batch_size, seq_len = target.shape
        self.batch_size = batch_size
        self.seq_len = seq_len

        loss = -np.log(self.probs[np.arange(batch_size)[:, None], np.arange(seq_len), target] + 1e-9)

        mask = (target!=0) #boolean matrix
        loss = loss * mask
        return np.sum(loss) / np.sum(mask)
    def backward(self):
        one_hot = np.zeros_like(self.probs)
        one_hot[np.arange(self.batch_size)[:, None], np.arange(self.seq_len), self.target] = 1

        d_logits = self.probs - one_hot

        mask = (self.target != 0)[:, :, None] #batch_size, seq_len, 1
        d_logits = d_logits * mask
        d_logits /= np.sum(mask) #normalize by number of non-pad tokens
        return d_logits #basically the first needed thing for back prop to work
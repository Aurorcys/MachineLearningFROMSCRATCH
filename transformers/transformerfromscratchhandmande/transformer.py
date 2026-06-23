import pandas as pd 
import numpy as np

"""
Transformer Block: a block
Transformer: basically a stack of these blocks

"""

from attention import Attention
from feedforwardnn import FeedForward
from layernorm import LayerNorm
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, max_seq_len):
        self.attention = Attention(d_model, num_heads, max_seq_len)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
    def forward(self, x, token_ids):
        attn_output = self.attention.forward(x, token_ids)
        x = x + attn_output
        x = self.ln1.forward(x)

        ffn_output = self.ffn.forward(x)
        x = x + ffn_output
        x = self.ln2.forward(x)
        return x
    def backward(self, d_x):
        d_x = self.ln2.backward(d_x)

        d_ffn = self.ffn.backward(d_x)
        d_x = d_x + d_ffn

        d_x = self.ln1.backward(d_x)
        d_attn = self.attention.backward(d_x)
        d_x = d_x + d_attn

        return d_x

class Transformer:
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, num_blocks, vocab_size=28):
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff, max_seq_len) for _ in range(num_blocks)]
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.01
    def forward(self, x, token_ids):
        for block in self.blocks:
            x = block.forward(x, token_ids)
        self.x = x
        return x @ self.output_proj #(batch, seq_len, vocab_size)
    def backward(self, d_logits):
        self.d_output_proj = np.tensordot(self.x, d_logits, axes=([0, 1], [0, 1])) #equivalent to a 2d transpose

        d_x = d_logits @ self.output_proj.T

        for blocks in reversed(self.blocks):
            d_x = blocks.backward(d_x)

        return d_x 
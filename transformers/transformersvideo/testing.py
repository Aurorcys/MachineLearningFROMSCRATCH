import pandas as pd 
import numpy as np

with open('Random English Sentences.txt', 'r') as f:
    sentences = [line.strip() for line in f if line.strip()]

def sentences_to_vec(sentence):
    vec = []
    for char in sentence.lower():
        if char.isalpha():
            vec.append(ord(char) - ord('a') + 1)
        elif char == ' ':
            vec.append(27) #space = 27

    return vec


vectors = [sentences_to_vec(s) for s in sentences]

max_length = 100

padded = []
for vec in vectors:
    if len(vec) > max_length:
        padded.append(vec[:max_length])
    else:
        padded.append(vec + [0] * (max_length - len(vec)))

token_ids = np.array(padded)

d_model = 64
num_heads = 4
d_ff = 256
num_blocks = 2
vocab_size = 28
batch_size  = token_ids.shape[0]

from embedding import Embedding
from transformer import Transformer

embedding = Embedding(d_model, vocab_size, max_seq_len=max_length)
transformer = Transformer(d_model, num_heads, d_ff, max_length, num_blocks, vocab_size)

def generate(transformer, embedding, max_seq_len=100):
    tokens = [27]

    for _ in range(max_seq_len):
        padded = np.zeros((1, max_seq_len), dtype=np.int32)
        padded[0, :len(tokens)] = tokens

        x = embedding.forward(padded)
        logits = transformer.forward(x, padded)

        last_logits = logits[0, len(tokens) - 1, :]

        probs = np.exp(last_logits - np.max(last_logits))
        probs /= probs.sum()
        next_tok = np.random.choice(len(probs), p=probs)

        if next_tok == 0:
            break

        tokens.append(next_tok)
    return ''.join(' ' if t == 27 else chr(t + ord('a') - 1) for t in tokens[1:])

print(generate(transformer, embedding))


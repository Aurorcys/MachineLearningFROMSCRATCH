import numpy as np
import pandas as pd 
from transformer import Transformer
from entropyloss import EntropyLoss
from embedding import Embedding

with open('Random English Sentences.txt', 'r') as f:
    sentences = [line.strip() for line in f if line.strip()]

def sentence_to_vec(sentence):
    vec = []
    for char in sentence.lower():
        if char.isalpha():
            vec.append(ord(char) - ord('a') + 1)
        elif char == ' ':
            vec.append(27) #space = 27
    return vec

vectors = [sentence_to_vec(s) for s in sentences]

max_length = 100

#padding
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
batch_size = token_ids.shape[0]


def update_params(transformer, embedding, lr):
    # Embedding
    embedding.word_embeddings.weights -= lr * embedding.word_embeddings.d_weights
    
    # Output projection
    transformer.output_proj -= lr * transformer.d_output_proj
    
    # Each block
    for block in transformer.blocks:
        # LayerNorm 1
        block.ln1.gamma -= lr * block.ln1.d_gamma
        block.ln1.beta -= lr * block.ln1.d_beta
        
        # LayerNorm 2
        block.ln2.gamma -= lr * block.ln2.d_gamma
        block.ln2.beta -= lr * block.ln2.d_beta
        
        # FeedForward
        block.ffn.W1 -= lr * block.ffn.d_W1
        block.ffn.b1 -= lr * block.ffn.d_b1
        block.ffn.W2 -= lr * block.ffn.d_W2
        block.ffn.b2 -= lr * block.ffn.d_b2
        
        # Attention
        block.attention.W_q -= lr * block.attention.d_W_q
        block.attention.W_k -= lr * block.attention.d_W_k
        block.attention.W_v -= lr * block.attention.d_W_v
        block.attention.W_o -= lr * block.attention.d_W_o

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


embedding = Embedding(d_model, vocab_size=28, max_seq_len=max_length)
transformer = Transformer(d_model, num_heads, d_ff, max_length, num_blocks, vocab_size=28)

loss_fn = EntropyLoss()
learning_rate = 0.01

targets = np.zeros_like(token_ids)
targets[:, :-1] = token_ids[:, 1:]
targets[:, -1] = 0 #nothing to predict

num_epochs = 4
batch_size = 32

for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0

    indices = np.random.permutation(len(token_ids))

    for start in range(0, len(token_ids), batch_size):
        batch_indices = indices[start:start+batch_size]
        x_batch = token_ids[batch_indices]
        target_batch = targets[batch_indices]

        x = embedding.forward(x_batch)
        logits = transformer.forward(x, x_batch)

        loss = loss_fn.forward(logits, target_batch)
        total_loss += loss
        num_batches += 1

        d_logits = loss_fn.backward()
        d_embed = transformer.backward(d_logits)
        embedding.backward(d_embed)

        update_params(transformer, embedding, learning_rate)
    avg_loss = total_loss / num_batches
    print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}')
    print("Sample:", generate(transformer, embedding))
    print("---")

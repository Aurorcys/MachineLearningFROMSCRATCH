import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd

X, y = make_blobs(n_samples=300, centers=2, random_state=42, cluster_std=1.8)
y = np.where(y == 0, -1, 1)

df = pd.DataFrame(X, columns=['x1', 'x2'])
df['y'] = y
df.to_csv('svm_data.csv', index=False)
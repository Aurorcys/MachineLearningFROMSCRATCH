import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('svm_data.csv')

X = df[['x1', 'x2']].values
y = df['y'].values

"""plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
plt.show()"""


class SVM:
    def __init__(self, lr=0.01, C=1.0, epochs=500):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n, d = X.shape #n = samples, d = features
        self.w = np.zeros(d)
        self.b = 0

        for epoch in range(self.epochs):
            for i in range(n):
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)
                if margin < 1:
                    self.w -= self.lr * (self.w - self.C * y[i] * X[i])
                    self.b += self.lr * self.C * y[i]
                else:
                    self.w -= self.lr * self.w

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b


def plot_svm(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
    
    # Decision boundary
    x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    xx, yy = np.meshgrid(x0, x1)
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'red'], 
                linestyles=['--', '-', '--'], alpha=0.5)
    plt.title(f'SVM with C={model.C}')
    plt.show()

for C in [0.1, 1.0, 10.0]:
    svm = SVM(C=C, lr=0.001, epochs=500)
    svm.fit(X, y)
    plot_svm(X, y, svm)
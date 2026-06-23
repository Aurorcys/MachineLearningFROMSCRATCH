import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
y = df['price']

# Normalize square_feet
sqft = df['square_feet'].values
sqft_mean = sqft.mean()
sqft_std = sqft.std()
sqft_norm = (sqft - sqft_mean) / sqft_std

X = np.c_[np.ones(len(df)), sqft_norm]
theta = np.zeros(2)
theta[0] = y.mean()

def predict(X, theta):
    return X.dot(theta)

def update_theta(X, y, theta, lr=0.01):
    gradient = X.T.dot(X.dot(theta) - y) / len(y)
    theta -= lr * gradient
    return theta

def plot(theta, X, y):
    # Predict on the ACTUAL data points
    y_pred = predict(X, theta)
    
    plt.clf()
    plt.scatter(df['square_feet'], df['price'], color='blue', label='Actual')
    plt.plot(df['square_feet'], y_pred, 'r-', linewidth=2, label='Predicted')
    plt.xlabel('Square Feet')
    plt.ylabel('Price')
    plt.xlim(0, df['square_feet'].max() + 500)
    plt.ylim(0, y.max() + 100000)
    plt.legend()
    plt.show()

for i in range(500):
    if i % 50 == 0:
        plot(theta, X, y)
    theta = update_theta(X, y, theta)
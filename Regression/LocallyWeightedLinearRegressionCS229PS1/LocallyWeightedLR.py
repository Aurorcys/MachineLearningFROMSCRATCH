import pandas as pd
import numpy as np

df = pd.read_csv("ds5_train.csv")
features = [col for col in df.columns if col != "y"]

X_train = df[features].values
y_train = df["y"].values

df_valid = pd.read_csv("ds5_valid.csv")
X_valid = df_valid[features].values
y_valid = df_valid['y'].values

tau = 0.5  # bandwidth (tune this)
y_pred = []

for x_query in X_valid:
    # Compute weights
    dist_sq = np.sum((X_train - x_query) ** 2, axis=1)
    w = np.exp(-dist_sq / (2 * tau ** 2))
    
    # Diagonal weight matrix
    W = np.diag(w)
    
    # Weighted normal equation
    XtWX = X_train.T @ W @ X_train
    XtWy = X_train.T @ W @ y_train
    
    # Solve for theta
    theta = np.linalg.solve(XtWX, XtWy)
    
    # Predict
    y_pred.append(theta @ x_query)

y_pred = np.array(y_pred)

# Evaluate
mse = np.mean((y_valid - y_pred) ** 2)
print(f"MSE (tau = {tau}): {mse:.4f}")

import matplotlib.pyplot as plt

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], y_train, c='blue', marker='x', label='Training set')
plt.scatter(X_valid[:, 0], y_valid, c='red', marker='o', label='Validation set (true)')
plt.scatter(X_valid[:, 0], y_pred, c='green', marker='^', label='Validation set (predicted)')
plt.xlabel('Feature x1')
plt.ylabel('Target y')
plt.legend()
plt.title(f'Locally Weighted Linear Regression (τ = {tau})')
plt.show()


tau_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
mse_list = []

for tau in tau_values:
    y_pred = []
    for x_query in X_valid:
        dist_sq = np.sum((X_train - x_query) ** 2, axis=1)
        w = np.exp(-dist_sq / (2 * tau ** 2))
        W = np.diag(w)
        theta = np.linalg.solve(X_train.T @ W @ X_train, X_train.T @ W @ y_train)
        y_pred.append(theta @ x_query)
    mse = np.mean((y_valid - np.array(y_pred)) ** 2)
    mse_list.append(mse)
    print(f"τ = {tau:.2f}, VALID MSE = {mse:.4f}")

best_tau = tau_values[np.argmin(mse_list)]
print(f"Best τ: {best_tau}")

y_pred = []
for x_query in X_valid:
        dist_sq = np.sum((X_train - x_query) ** 2, axis=1)
        w = np.exp(-dist_sq / (2 * best_tau ** 2))
        W = np.diag(w)
        theta = np.linalg.solve(X_train.T @ W @ X_train, X_train.T @ W @ y_train)
        y_pred.append(theta @ x_query)

plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], y_train, c='blue', marker='x', label='Training set')
plt.scatter(X_valid[:, 0], y_valid, c='red', marker='o', label='Validation set (true)')
plt.scatter(X_valid[:, 0], y_pred, c='green', marker='^', label='Validation set (predicted)')
plt.xlabel('Feature x1')
plt.ylabel('Target y')
plt.legend()
plt.title(f'Locally Weighted Linear Regression (τ = {best_tau})')
plt.show()

df_test = pd.read_csv("ds5_test.csv")
y_test = df_test["y"].values
X_test = df_test[features].values


y_pred = []
for x_query in X_test:
        dist_sq = np.sum((X_train - x_query) ** 2, axis=1)
        w = np.exp(-dist_sq / (2 * best_tau ** 2))
        W = np.diag(w)
        theta = np.linalg.solve(X_train.T @ W @ X_train, X_train.T @ W @ y_train)
        y_pred.append(theta @ x_query)

    
mse = np.mean((y_test - np.array(y_pred)) ** 2)
mse_list.append(mse)
print(f"τ = {best_tau:.2f}, TEST MSE = {mse:.4f}")
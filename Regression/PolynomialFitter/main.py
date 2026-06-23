import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.linspace(0, 10, 50)
y = 3 + 0.5*X**2 + np.random.randn(50) * 5

degrees = [1, 2, 3, 4, 5]
errors = []


for degree in degrees:
    X_poly = np.zeros((len(X), degree + 1))
    for i in range(degree + 1):
        X_poly[:, i] = X ** i
    # Normal equation: theta = (X^T X)^(-1) X^T y
    theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
    y_train_pred = X_poly @ theta
    avg_squared_loss = np.mean((y_train_pred - y) ** 2)
    #plot
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, y_train_pred, color='red', label=f'Polynomial Degree {degree}')
    plt.title(f'Polynomial Regression (Training), Avg. Squared Loss: {avg_squared_loss:.2f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()


    X_plot = np.linspace(0, 10, 100)
    X_plot_poly = np.zeros((100, degree + 1))
    for i in range(degree + 1):
        X_plot_poly[:, i] = X_plot ** i
    y_plot_pred = X_plot_poly @ theta
    y_test = 3 + 0.5*X_plot**2 + np.random.randn(100) * 5
    avg_test_squared_loss = np.mean((y_plot_pred - y_test) ** 2)
    errors.append((degree, avg_test_squared_loss))
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X_plot, y_plot_pred, color='red', label=f'Polynomial Degree {degree}')
    plt.title(f'Polynomial Regression (Test), Avg. Squared Loss: {avg_test_squared_loss:.2f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

plt.plot([d for d, e in errors], [e for d, e in errors], marker='o')
plt.title('Test Error vs. Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('Average Test Squared Loss')
plt.show()


print(errors)
best = min(errors, key=lambda x: x[1])
print(f"Best degree: {best[0]}, Loss: {best[1]:.4f}")
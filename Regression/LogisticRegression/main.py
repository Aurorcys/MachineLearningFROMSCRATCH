import pandas as pd
import numpy as np

data_path = 'data.csv'
df = pd.read_csv(data_path)
answers = []

# Collect training labels
for idx, row in df[:9][['salary_hkd', 'hours_per_day']].iterrows():
    salary = row['salary_hkd']
    hours = row['hours_per_day']
    print(f"Salary (HKD): {salary}, Hours per Day: {hours}")
    ans = input('Would you choose this job? Type y/n for yes or no resp: ')
    answers.append(ans)

y = [1 if ans == 'y' else 0 for ans in answers]
y = np.array(y)

# Define sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define logistic regression
def linear_regression(X, y, lr=0.01, epochs=5000):
    theta = np.zeros(X.shape[1])
    for _ in range(epochs):
        z = X @ theta
        h = sigmoid(z)
        gradient = X.T @ (h - y) / len(y)
        theta -= lr * gradient
    return theta

# Prepare training data
X_train = df[:9][['salary_hkd', 'hours_per_day']].values

# Normalize
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_train = np.c_[np.ones(X_train.shape[0]), X_train]

# Train
theta = linear_regression(X_train, y, lr=0.01, epochs=5000)

score = 0

print('\nNOW TIME TO PREDICT')

# Evaluate on remaining data
for idx, row in df[9:].iterrows():
    salary = row['salary_hkd']
    hours = row['hours_per_day']
    print(f"Salary (HKD): {salary}, Hours per Day: {hours}")
    ans = input('Would you choose this job? Type y/n for yes or no resp: ')
    real = 1 if ans == 'y' else 0
    
    # Normalize the test sample using TRAINING statistics
    x_test = np.array([salary, hours])
    x_test = (x_test - X_mean) / X_std
    x_test = np.insert(x_test, 0, 1)  # Add bias
    
    z = np.dot(theta, x_test)
    prob = sigmoid(z)
    predicted = 1 if prob > 0.5 else 0
    print(f"Prob: {prob:.3f}, Predicted: {predicted}, Real: {real}")
    if predicted == real:
        print('CORRECT')
        score += 1
    else:
        print('FALSE')

print(f"Final Score: {score}/{len(df[9:])}")
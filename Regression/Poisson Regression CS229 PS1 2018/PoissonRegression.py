import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ds4_train.csv")

features = [col for col in df.columns if col != 'y']

# Log transform to compress
X = np.log1p(df[features])

# Normalize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Add intercept
X = np.c_[np.ones(X.shape[0]), X]
y = df['y'].values / 1000000

alpha = 1e-6
theta = np.zeros(X.shape[1])


# ANIMATED PLOT
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

losses = []

for epoch in range(3000):
    # Clip to prevent overflow
    z = np.clip(X @ theta, -20, 20)
    y_pred = np.exp(z)
    grad = X.T @ (y - y_pred)
    theta += alpha * grad
    
    if epoch % 50 == 0:
        loss = -np.mean(y * (X @ theta) - np.exp(np.clip(X @ theta, -20, 20)))
        losses.append(loss)
        
        ax.clear()
        ax.plot(range(0, epoch + 1, 50), losses, color='#00d4ff', linewidth=2.5)
        ax.fill_between(range(0, epoch + 1, 50), losses, alpha=0.1, color='#00d4ff')
        ax.set_xlabel('Epoch', fontsize=14, color='white', fontweight='bold')
        ax.set_ylabel('Negative Log-Likelihood', fontsize=14, color='white', fontweight='bold')
        ax.set_title(f'Poisson Regression — Training Loss (Epoch {epoch})', 
                     fontsize=16, color='white', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.15, color='white')
        ax.tick_params(colors='white', labelsize=11)
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)
        
        plt.draw()
        plt.pause(0.01)
        
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

plt.ioff()
plt.show()

print("Final theta:", theta)


df_valid = pd.read_csv("ds4_valid.csv")


# Log transform to compress
X = np.log1p(df_valid[features])

# Normalize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Add intercept
X_valid = np.c_[np.ones(X.shape[0]), X]


z_pred = np.clip(X_valid @ theta, -20, 20)
y_pred_valid = np.exp(z_pred) * 1_000_000   # scale back to original

# Compare first few
print("True:", df_valid['y'].values[:10])
print("Pred:", y_pred_valid[:10])

y_valid = df_valid['y'].values
deviance = 2 * np.sum(y_valid * np.log(y_valid / y_pred_valid) - (y_valid - y_pred_valid))
print(f"Validation Deviance: {deviance:.2f}")



# Null model: predict just the mean
y_mean = np.mean(y_valid)
null_deviance = 2 * np.sum(y_valid * np.log(y_valid / y_mean) - (y_valid - y_mean))
print(f"Null Deviance: {null_deviance:.2f}")
print(f"Explained Deviance: {(null_deviance - deviance) / null_deviance * 100:.2f}%")
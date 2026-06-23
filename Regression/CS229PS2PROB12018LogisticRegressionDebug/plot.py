import util
import matplotlib.pyplot as plt

Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


ax1.scatter(Xa[:, 1], Xa[:, 2], c=Ya, cmap='bwr', edgecolors='k', alpha=0.7)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_title('Dataset A')

ax2.scatter(Xb[:, 1], Xb[:, 2], c=Yb, cmap='bwr', edgecolors='k', alpha=0.7)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Dataset B')

plt.tight_layout()
plt.show()



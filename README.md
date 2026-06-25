# ML From Scratch

A complete machine learning library built from scratch in Python and NumPy. No PyTorch. No TensorFlow. No scikit-learn. Just the mathematics, derived and coded.

Every algorithm includes full mathematical derivations, clean implementations, and visualizations. This repository is the public record of my journey from zero to practitioner in six months.

---

## 📂 What's Inside

### Regression
| Algorithm | Description |
|-----------|-------------|
| Linear Regression | Ordinary least squares with normal equation |
| Logistic Regression | Binary classification with gradient descent |
| Logistic Regression (Newton-Raphson) | Second-order optimization with convergence animation |
| Poisson Regression | Count data regression with log link, 99.45% explained deviance |
| Locally Weighted Linear Regression | Nonparametric regression with Gaussian kernels and bandwidth tuning |
| Polynomial Fitter | High-degree polynomial regression with regularization |

### Classification
| Algorithm | Description |
|-----------|-------------|
| Gaussian Discriminant Analysis | Generative classifier with MLE derivations and Gaussian contours |
| Naive Bayes Spam Classifier | Multinomial Naive Bayes for text classification |
| SVM | Support Vector Machine with hinge loss and full gradient derivations |
| Linear Spam Classifier | Binary text classification with feature engineering |
| Positive-Only Labels | Learning from positive and unlabeled data (CS229) |

### Tree Algorithms
| Algorithm | Description |
|-----------|-------------|
| Totally Random Forest | Pure random splits, no optimization — demonstrating the power of voting |
| Extra Trees | Random splits with Gini impurity selection |
| Extra Trees with Stochastic Lookahead | One-step future split evaluation for better node selection |
| Random Forest | Full Breiman algorithm with optimal split search, bootstrap sampling |

### Deep Learning
| Algorithm | Description |
|-----------|-------------|
| Neural Network | Multi-layer perceptron with backpropagation from scratch |
| Transformer (×3) | Complete encoder with multi-head self-attention, positional encodings, causal masking, LayerNorm, FFN. Built three times including blind rebuild |
| Transformer Backpropagation | Full backward pass derived and implemented |

### Reinforcement Learning
| Algorithm | Description |
|-----------|-------------|
| Q-Learning Maze Runner | Tabular Q-learning agent navigating obstacles with BFS path validation |
| Tic-Tac-Toe MCTS | Monte Carlo Tree Search with UCB1, loss penalty, self-play, live visualization |
| Wordle DQN | Deep Q-Network with custom state encoding, 96.8% win rate, 4.2 average guesses |

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** NumPy, pandas, matplotlib
- **No:** scikit-learn, PyTorch, TensorFlow, Keras, XGBoost, LightGBM

---

## Related Content

- [YouTube: Transformer Forward Pass — The Math (Part 1)][(https://youtube.com)](https://youtu.be/ZDbGt0XLwRM?si=5S-SypmO5Y-dhz8V)
- [YouTube: Transformer Forward Pass — The Code (Part 2)][(https://youtube.com)](https://youtu.be/8R9iNiD2C-4?si=c4-Sv8FvhBUenMaM)
- [LinkedIn: Daily builds and ML content][(https://linkedin.com)](https://www.linkedin.com/in/cyrus-hong-86b10b3a5/)

---

## Quick Start

```bash
git clone https://github.com/yourusername/ML-From-Scratch
cd ML-From-Scratch
pip install numpy pandas matplotlib
python RL/WordleDQN/train.py
```

---

## Stats

- **18 implementations**
- **Built in 17 days** (alongside exams, quitting a job, and reading a PhD thesis)
- **100% from scratch** — no ML libraries
- **96.8% Wordle win rate**
- **99.45% Poisson explained deviance**

---

## Why This Exists

I started learning machine learning in January 2026. I couldn't finish a single problem on Stanford's CS229 without AI help. Six months later, I've built the core algorithms of modern ML from scratch, read a PhD thesis on random forests, completed Stanford's graduate ML problem set, and launched a YouTube channel teaching transformers.

This repository is the evidence. Every algorithm here was derived mathematically, implemented in clean code, validated on real data, and documented for others to learn from.

If I can go from zero to this in six months, you can too. The only secret is daily output.

---

## Connect

- **LinkedIn:** [[Your LinkedIn URL]](https://www.linkedin.com/in/cyrus-hong-86b10b3a5/)
- **YouTube:** [[Your YouTube URL]](https://www.youtube.com/@AurorcysQKT)


---

**In progress:** More RL (DQN variants, policy gradients, ARC-AGI-3 agent), XGBoost from scratch, prediction market models, and whatever else the grind demands.


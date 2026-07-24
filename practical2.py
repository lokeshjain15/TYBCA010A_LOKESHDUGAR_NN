# ==========================================================
# Practical No. 2
# Comparison of Activation Functions
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt

# Generate input values
x = np.linspace(-10, 10, 400)

# ----------------------------------------------------------
# Activation Functions
# ----------------------------------------------------------

def binary_step(x):
    return np.where(x >= 0, 1, 0)

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)

# ----------------------------------------------------------
# Compute Outputs
# ----------------------------------------------------------

y_step = binary_step(x)
y_linear = linear(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky = leaky_relu(x)
y_softmax = softmax(x)

# ==========================================================
# Binary Step
# ==========================================================

plt.figure(figsize=(6,4))
plt.plot(x, y_step, linewidth=2)
plt.title("Binary Step Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)

# ==========================================================
# Linear
# ==========================================================

plt.figure(figsize=(6,4))
plt.plot(x, y_linear, linewidth=2)
plt.title("Linear Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)

# ==========================================================
# Sigmoid
# ==========================================================

plt.figure(figsize=(6,4))
plt.plot(x, y_sigmoid, linewidth=2)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)

# ==========================================================
# Tanh
# ==========================================================

plt.figure(figsize=(6,4))
plt.plot(x, y_tanh, linewidth=2)
plt.title("Tanh Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)

# ==========================================================
# ReLU
# ==========================================================

plt.figure(figsize=(6,4))
plt.plot(x, y_relu, linewidth=2)
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)

# ==========================================================
# Leaky ReLU
# ==========================================================

plt.figure(figsize=(6,4))
plt.plot(x, y_leaky, linewidth=2)
plt.title("Leaky ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)

# ==========================================================
# Softmax
# ==========================================================

plt.figure(figsize=(6,4))
plt.plot(x, y_softmax, linewidth=2)
plt.title("Softmax Activation Function")
plt.xlabel("Input")
plt.ylabel("Probability")
plt.grid(True)

# ==========================================================
# Comparison Graph
# ==========================================================

plt.figure(figsize=(10,6))

plt.plot(x, y_linear, label="Linear")
plt.plot(x, y_sigmoid, label="Sigmoid")
plt.plot(x, y_tanh, label="Tanh")
plt.plot(x, y_relu, label="ReLU")
plt.plot(x, y_leaky, label="Leaky ReLU")

plt.title("Comparison of Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)

plt.show()

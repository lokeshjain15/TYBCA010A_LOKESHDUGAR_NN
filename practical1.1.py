import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step Activation Function
# -----------------------------
def activation(z):
    return 1 if z >= 0 else 0

# -----------------------------
# AND Gate Dataset
# -----------------------------
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([0,0,0,1])

# -----------------------------
# Parameters
# -----------------------------
learning_rate = 0.1
epochs = 10

weights = np.zeros(2)
bias = 0

error_history = []
w1_history = []
w2_history = []

print("="*50)
print("Perceptron Training")
print("="*50)

# -----------------------------
# Training
# -----------------------------
for epoch in range(epochs):

    total_error = 0

    for i in range(len(X)):

        x = X[i]
        target = y[i]

        z = np.dot(weights, x) + bias
        prediction = activation(z)

        error = target - prediction
        total_error += abs(error)

        weights = weights + learning_rate * error * x
        bias = bias + learning_rate * error

    error_history.append(total_error)
    w1_history.append(weights[0])
    w2_history.append(weights[1])

print("\nTraining Completed")

print("Weights =",weights)
print("Bias =",bias)

# -----------------------------
# Predictions
# -----------------------------
print("\nPredictions")

for x in X:

    z = np.dot(weights,x)+bias
    prediction = activation(z)

    print(x," --> ",prediction)

# ======================================================
# GRAPH 1 : Dataset
# ======================================================

plt.figure(figsize=(5,5))

for i in range(len(X)):

    if y[i]==0:
        plt.scatter(X[i][0],X[i][1],
                    color='red',
                    s=120,
                    label='Class 0' if i==0 else "")

    else:
        plt.scatter(X[i][0],X[i][1],
                    color='blue',
                    s=120,
                    label='Class 1')

plt.title("AND Gate Dataset")
plt.xlabel("Input x1")
plt.ylabel("Input x2")
plt.xticks([0,1])
plt.yticks([0,1])
plt.grid(True)
plt.legend()

# ======================================================
# GRAPH 2 : Decision Boundary
# ======================================================

x_values = np.linspace(-0.5,1.5,100)

if weights[1] != 0:

    y_values = -(weights[0]*x_values+bias)/weights[1]

    plt.figure(figsize=(5,5))

    plt.scatter(0,0,color='red',s=120)
    plt.scatter(0,1,color='red',s=120)
    plt.scatter(1,0,color='red',s=120)
    plt.scatter(1,1,color='blue',s=120)

    plt.plot(x_values,
             y_values,
             linewidth=2,
             label="Decision Boundary")

    plt.xlim(-0.5,1.5)
    plt.ylim(-0.5,1.5)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Perceptron Decision Boundary")
    plt.grid(True)
    plt.legend()

# ======================================================
# GRAPH 3 : Error vs Epoch
# ======================================================

plt.figure(figsize=(6,4))

plt.plot(range(1,epochs+1),
         error_history,
         marker='o',
         linewidth=2)

plt.title("Training Error")
plt.xlabel("Epoch")
plt.ylabel("Total Error")
plt.grid(True)

# ======================================================
# GRAPH 4 : Weight Convergence
# ======================================================

plt.figure(figsize=(6,4))

plt.plot(range(1,epochs+1),
         w1_history,
         marker='o',
         label='Weight 1')

plt.plot(range(1,epochs+1),
         w2_history,
         marker='s',
         label='Weight 2')

plt.xlabel("Epoch")
plt.ylabel("Weight Value")
plt.title("Weight Convergence")
plt.grid(True)
plt.legend()

plt.show()
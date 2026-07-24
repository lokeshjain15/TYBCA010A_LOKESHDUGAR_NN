# ---------------------------------------------
# Practical 1: Perceptron Model for AND Gate
# ---------------------------------------------

import numpy as np

# -----------------------------
# Step Activation Function
# -----------------------------
def activation(z):
    if z >= 0:
        return 1
    else:
        return 0

# -----------------------------
# AND Gate Dataset
# -----------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# -----------------------------
# Initialize Parameters
# -----------------------------
learning_rate = 0.1
epochs = 100

weights = np.zeros(2)
bias = 0

print("=" * 60)
print("PERCEPTRON TRAINING FOR AND GATE")
print("=" * 60)

# -----------------------------
# Training
# -----------------------------
for epoch in range(epochs):

    print(f"\nEpoch {epoch + 1}")

    for i in range(len(X)):

        x = X[i]
        target = y[i]

        # Weighted Sum
        z = np.dot(x, weights) + bias

        # Prediction
        prediction = activation(z)


        # Error
        error = target - prediction

        # Weight Update
        weights = weights + learning_rate * error * x

        # Bias Update
        bias = bias + learning_rate * error

        print("--------------------------------")
        print("Input :", x)
        print("Target :", target)
        print("Prediction :", prediction)
        print("Error :", error)
        print("Weights :", weights)
        print("Bias :", bias)

print("\nTraining Completed")
print("=" * 60)

print("Final Weights :", weights)
print("Final Bias :", bias)

# -----------------------------
# Testing
# -----------------------------
print("\nTesting the Perceptron")
print("=" * 60)

for x in X:

    z = np.dot(x, weights) + bias
    prediction = activation(z)

    print(f"Input = {x}  --> Output = {prediction}")
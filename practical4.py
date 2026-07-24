############################################################
# PRACTICAL 4
# Compute Loss Functions
############################################################

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("LOSS FUNCTION PRACTICAL")
print("="*60)

############################################################
# Mean Squared Error
############################################################

print("\nMean Squared Error")

actual=np.array([5,6,7,8,9])

predicted=np.array([4,5,8,7,10])

print("\nActual Values")

print(actual)

print("\nPredicted Values")

print(predicted)

difference=actual-predicted

print("\nDifference")

print(difference)

square=difference**2

print("\nSquared Error")

print(square)

mse=np.mean(square)

print("\nMean Squared Error")

print(mse)

############################################################
# Binary Cross Entropy
############################################################

print("\n"+"="*60)

print("Binary Cross Entropy")

actual=np.array([1,0,1,0,1])

prediction=np.array([0.9,0.2,0.8,0.1,0.95])

print("\nActual")

print(actual)

print("\nPrediction")

print(prediction)

epsilon=1e-10

prediction=np.clip(prediction,epsilon,1-epsilon)

bce=-(actual*np.log(prediction)+(1-actual)*np.log(1-prediction))

print("\nLoss for Every Sample")

print(bce)

print("\nAverage BCE")

print(np.mean(bce))

############################################################
# Categorical Cross Entropy
############################################################

print("\n"+"="*60)

print("Categorical Cross Entropy")

actual=np.array([1,0,0])

prediction=np.array([0.80,0.15,0.05])

prediction=np.clip(prediction,epsilon,1)

loss=-np.sum(actual*np.log(prediction))

print("\nActual")

print(actual)

print("\nPrediction")

print(prediction)

print("\nCCE")

print(loss)

############################################################
# GRAPH 1
############################################################

plt.figure(figsize=(6,4))

plt.plot(actual,label="Actual",marker="o")

plt.plot(predicted,label="Prediction",marker="s")

plt.title("Actual vs Predicted")

plt.xlabel("Sample")

plt.ylabel("Value")

plt.legend()

plt.grid(True)

############################################################
# GRAPH 2
############################################################

plt.figure(figsize=(6,4))

plt.bar(range(len(square)),square)

plt.title("Squared Error")

plt.xlabel("Sample")

plt.ylabel("Squared Error")

plt.grid(True)

############################################################
# GRAPH 3
############################################################

plt.figure(figsize=(6,4))

plt.bar(["MSE"],[mse])

plt.title("Mean Squared Error")

plt.ylim(0,max(square)+1)

############################################################
# GRAPH 4
############################################################

plt.figure(figsize=(6,4))

plt.bar(range(len(bce)),bce)

plt.title("Binary Cross Entropy")

plt.xlabel("Sample")

plt.ylabel("Loss")

plt.grid(True)

############################################################
# GRAPH 5
############################################################

plt.figure(figsize=(6,4))

plt.bar(["CCE"],[loss])

plt.title("Categorical Cross Entropy")

plt.grid(True)

plt.show()
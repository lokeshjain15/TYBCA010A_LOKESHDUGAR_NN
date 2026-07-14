# ==========================================================
# PRACTICAL 3
# Demonstration of Forward Propagation in Neural Network
#
# Developed From Scratch using NumPy
#
# Network Architecture:
#
#       Input Layer (2 Neurons)
#              ↓
#       Hidden Layer (2 Neurons)
#              ↓
#       Output Layer (1 Neuron)
#
# Activation Function : Sigmoid
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# STEP 1 : Define Sigmoid Activation Function
# ==========================================================

def sigmoid(x):
    """
    Sigmoid Activation Function

                1
    -------------------------
       (1 + e^(-x))

    Output Range : 0 to 1
    """

    return 1 / (1 + np.exp(-x))


# ==========================================================
# STEP 2 : Input Layer
# ==========================================================

print("="*60)
print("STEP 1 : INPUT LAYER")
print("="*60)

# Two input features

x = np.array([1,0])

print("Input Vector")
print(x)

print("\nNumber of Inputs :",len(x))

# ==========================================================
# STEP 3 : Hidden Layer Weights
# ==========================================================

print("\n"+"="*60)
print("STEP 2 : INITIALIZE HIDDEN LAYER WEIGHTS")
print("="*60)

"""
           H1      H2

x1        0.5     0.2

x2        0.3     0.8
"""

W_hidden = np.array([
                    [0.5,0.2],
                    [0.3,0.8]
                    ])

print("Hidden Layer Weight Matrix")

print(W_hidden)

# ==========================================================
# STEP 4 : Hidden Layer Bias
# ==========================================================

bias_hidden = np.array([0.1,0.2])

print("\nHidden Layer Bias")

print(bias_hidden)

# ==========================================================
# STEP 5 : Weighted Sum of Hidden Layer
# ==========================================================

print("\n"+"="*60)
print("STEP 3 : HIDDEN LAYER CALCULATION")
print("="*60)

"""
Formula

Z = XW + b
"""

hidden_input = np.dot(x,W_hidden)+bias_hidden

print("Weighted Sum of Hidden Layer")

print(hidden_input)

# ==========================================================
# STEP 6 : Hidden Layer Activation
# ==========================================================

hidden_output = sigmoid(hidden_input)

print("\nAfter Applying Sigmoid")

print(hidden_output)

print("\nIndividual Hidden Neurons")

print("H1 =",hidden_output[0])

print("H2 =",hidden_output[1])

# ==========================================================
# STEP 7 : Output Layer Weights
# ==========================================================

print("\n"+"="*60)
print("STEP 4 : OUTPUT LAYER WEIGHTS")
print("="*60)

"""
Hidden Layer → Output Layer

          Output

H1         0.7

H2         0.9
"""

W_output = np.array([
                    [0.7],
                    [0.9]
                    ])

print(W_output)

# ==========================================================
# STEP 8 : Output Bias
# ==========================================================

bias_output = np.array([0.3])

print("\nOutput Bias")

print(bias_output)

# ==========================================================
# STEP 9 : Output Layer Calculation
# ==========================================================

print("\n"+"="*60)
print("STEP 5 : OUTPUT LAYER")
print("="*60)

output_input = np.dot(hidden_output,W_output)+bias_output

print("Weighted Sum")

print(output_input)

# ==========================================================
# STEP 10 : Final Prediction
# ==========================================================

output = sigmoid(output_input)

print("\nFinal Network Output")

print(output)

# ==========================================================
# STEP 11 : Manual Verification
# ==========================================================

print("\n"+"="*60)
print("STEP 6 : COMPLETE CALCULATIONS")
print("="*60)

print("\nHidden Layer")

print("-------------------------------------")

print("Neuron H1")

print("=(1×0.5)+(0×0.3)+0.1")

print("=0.6")

print("Sigmoid =",sigmoid(0.6))

print()

print("Neuron H2")

print("=(1×0.2)+(0×0.8)+0.2")

print("=0.4")

print("Sigmoid =",sigmoid(0.4))

print()

print("-------------------------------------")

print("Output Neuron")

value = hidden_output[0]*0.7 + hidden_output[1]*0.9 + 0.3

print("=(%.4f×0.7)+(%.4f×0.9)+0.3"%(hidden_output[0],hidden_output[1]))

print("=",value)

print("Sigmoid =",sigmoid(value))

# ==========================================================
# GRAPH 1
# ==========================================================

plt.figure(figsize=(6,4))

plt.bar(["Input x1","Input x2"],x)

plt.title("Input Layer")

plt.ylabel("Input Value")

plt.grid(True)

# ==========================================================
# GRAPH 2
# ==========================================================

plt.figure(figsize=(6,4))

plt.bar(["Hidden H1","Hidden H2"],hidden_output)

plt.title("Hidden Layer Activations")

plt.ylabel("Activation")

plt.ylim(0,1)

plt.grid(True)

# ==========================================================
# GRAPH 3
# ==========================================================

plt.figure(figsize=(5,4))

plt.bar(["Output"],output)

plt.title("Final Output")

plt.ylim(0,1)

plt.grid(True)

# ==========================================================
# GRAPH 4
# Sigmoid Function
# ==========================================================

x_axis=np.linspace(-10,10,200)

y_axis=sigmoid(x_axis)

plt.figure(figsize=(6,4))

plt.plot(x_axis,y_axis,linewidth=3)

plt.title("Sigmoid Activation Function")

plt.xlabel("Input")

plt.ylabel("Output")

plt.grid(True)

# ==========================================================
# GRAPH 5
# Neural Network Architecture
# ==========================================================

plt.figure(figsize=(8,5))

# Input Layer

plt.scatter([1,1],[3,1],s=800)

# Hidden Layer

plt.scatter([3,3],[3,1],s=800)

# Output Layer

plt.scatter([5],[2],s=800)

# Connections

plt.plot([1,3],[3,3])

plt.plot([1,3],[3,1])

plt.plot([1,3],[1,3])

plt.plot([1,3],[1,1])

plt.plot([3,5],[3,2])

plt.plot([3,5],[1,2])

# Labels

plt.text(0.8,3.2,"x1")

plt.text(0.8,1.2,"x2")

plt.text(2.8,3.2,"H1")

plt.text(2.8,1.2,"H2")

plt.text(4.9,2.2,"Output")

plt.title("Neural Network Architecture")

plt.axis("off")

# ==========================================================
# GRAPH 6
# Signal Flow
# ==========================================================

plt.figure(figsize=(10,2))

plt.scatter([1],[1],s=700)

plt.scatter([4],[1],s=700)

plt.scatter([7],[1],s=700)

plt.text(0.7,1.2,"Input")

plt.text(3.6,1.2,"Hidden")

plt.text(6.6,1.2,"Output")

plt.arrow(1.3,1,2.2,0,width=0.02)

plt.arrow(4.3,1,2.2,0,width=0.02)

plt.axis("off")

plt.title("Forward Propagation Flow")

# ==========================================================
# SHOW ALL GRAPHS
# ==========================================================

plt.show()

print("\n"+"="*60)
print("FORWARD PROPAGATION COMPLETED SUCCESSFULLY")
print("="*60)
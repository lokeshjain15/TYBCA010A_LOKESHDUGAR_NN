# ==========================================================
# PRACTICAL 5
# Binary Classification Using Neural Network
# ==========================================================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==========================================================
# STEP 1 : Generate Dataset
# ==========================================================

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_classes=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

print("Shape of X :", X.shape)
print("Shape of y :", y.shape)

# ==========================================================
# STEP 2 : Visualize Dataset
# ==========================================================

plt.figure(figsize=(6,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=y,
    cmap='bwr'
)

plt.title("Binary Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)

# ==========================================================
# STEP 3 : Split Dataset
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# STEP 4 : Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# STEP 5 : Build Neural Network
# ==========================================================

model = Sequential()

# Hidden Layer 1
model.add(Dense(
    units=8,
    activation='relu',
    input_shape=(2,)
))

# Hidden Layer 2
model.add(Dense(
    units=6,
    activation='relu'
))

# Output Layer
model.add(Dense(
    units=1,
    activation='sigmoid'
))

# ==========================================================
# STEP 6 : Compile Model
# ==========================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display Model Summary
print(model.summary())

# ==========================================================
# STEP 7 : Train Model
# ==========================================================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ==========================================================
# STEP 8 : Prediction
# ==========================================================

y_probability = model.predict(X_test)

y_prediction = (y_probability > 0.5)

# ==========================================================
# STEP 9 : Accuracy
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)

print("\nAccuracy :", accuracy)

# ==========================================================
# STEP 10 : Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    y_test,
    y_prediction
)

print("\nConfusion Matrix")

print(cm)

# ==========================================================
# STEP 11 : Classification Report
# ==========================================================

print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_prediction
    )
)

# ==========================================================
# STEP 12 : Plot Loss Graph
# ==========================================================

plt.figure(figsize=(6,4))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("Loss vs Epoch")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

# ==========================================================
# STEP 13 : Plot Accuracy Graph
# ==========================================================

plt.figure(figsize=(6,4))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Accuracy vs Epoch")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

# ==========================================================
# STEP 14 : Visualize Predictions
# ==========================================================

plt.figure(figsize=(6,5))

plt.scatter(
    X_test[:,0],
    X_test[:,1],
    c=y_prediction.flatten(),
    cmap='bwr'
)

plt.title("Predicted Classes")

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

plt.grid(True)

plt.show()

# ==========================================================
# STEP 15 : Predict New Sample
# ==========================================================

sample = np.array([[0.5, -0.2]])

sample = scaler.transform(sample)

prediction = model.predict(sample)

print("\nProbability :", prediction[0][0])

if prediction > 0.5:
    print("Predicted Class : 1")
else:
    print("Predicted Class : 0")
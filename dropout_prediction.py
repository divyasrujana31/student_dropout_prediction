import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
data = pd.read_csv('student-mat.csv', sep=';')


# Show first few rows
print(data.head())

# Target Variable - let's assume 'G3' (final grade)
# We'll create a binary target: Dropout (1) if G3 < 10 else No Dropout (0)
data['dropout'] = data['G3'].apply(lambda x: 1 if x < 10 else 0)

# Features selection - let's use some simple numeric features
features = data[['G1', 'G2', 'studytime', 'failures', 'absences']]
target = data['dropout']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:\n", accuracy_score(y_test, y_pred))

# Save the trained model
import pickle

with open('dropout_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved successfully as dropout_model.pkl")

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Convert to DataFrame for better readability
data = pd.DataFrame(X, columns=feature_names)
data['Target'] = y

# Display dataset information
print("First 5 rows of the dataset:")
print(data.head())
print("\nSummary statistics:")
print(data.describe())
# Check for missing values
print("\nChecking for missing values:")
print(data.isnull().sum())

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("\nDataset split: ")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "SVM": SVC(kernel='linear', random_state=42)
}

# Train and evaluate each model
results = {}
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Classification report and confusion matrix
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print(f"\nConfusion Matrix for {model_name}:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # Store results
    results[model_name] = {
        "accuracy": model.score(X_test, y_test),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
# Compare accuracy scores
print("\nModel Performance Summary:")
for model_name, metrics in results.items():
    print(f"{model_name} - Accuracy: {metrics['accuracy']:.2f}")

# Suggest improvements
print("\nInsights and Suggestions:")
print("1. Evaluate using different hyperparameters for the models.")
print("2. Test with other datasets (e.g., Breast Cancer or Wine Quality).")
print("3. Use advanced techniques such as GridSearchCV for hyperparameter tuning.")

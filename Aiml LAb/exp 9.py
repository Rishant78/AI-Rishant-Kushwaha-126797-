import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import os

# 1. Dataset Import and Exploration
# Load your dataset (replace 'your_dataset.csv' with the actual path to your dataset)
df = pd.read_csv('imbalanced_dataset.csv')  # Replace with your actual dataset path

# Show basic details about the dataset
print("Dataset Info:")
print(df.info())  # Data types and missing values
print("\nClass Distribution:")
print(df['target_column'].value_counts())  # Replace 'target_column' with your actual target column name

# Save original dataset to CSV in the working directory
df.to_csv('original_dataset.csv', index=False)

# Split into features (X) and target (y)
X = df.drop(columns=['target_column'])  # Replace 'target_column' with your actual target column name
y = df['target_column']  # Replace 'target_column' with your actual target column name

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Techniques for Handling Imbalanced Data

# Random Oversampling (Increase instances of the minority class by duplicating samples)
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Random Under Sampling (Reduce instances of the majority class by randomly removing samples)
rus = RandomUnderSampler(random_state=42)
X_train_resampled_undersample, y_train_resampled_undersample = rus.fit_resample(X_train, y_train)

# SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_resampled_smote, y_train_resampled_smote = smote.fit_resample(X_train, y_train)

# Save the resampled datasets to CSV in the working directory
pd.DataFrame(X_train_resampled).to_csv('oversampled_data.csv', index=False)
pd.DataFrame(X_train_resampled_undersample).to_csv('undersampled_data.csv', index=False)
pd.DataFrame(X_train_resampled_smote).to_csv('smote_data.csv', index=False)

# 3. Scaling the Data (StandardScaler)
scaler = StandardScaler()

# Scale the training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
X_test_resampled_scaled = scaler.transform(X_test)
X_train_resampled_undersample_scaled = scaler.fit_transform(X_train_resampled_undersample)
X_test_resampled_undersample_scaled = scaler.transform(X_test)
X_train_resampled_smote_scaled = scaler.fit_transform(X_train_resampled_smote)
X_test_resampled_smote_scaled = scaler.transform(X_test)

# 4. Classifier Evaluation - Logistic Regression with Increased max_iter and Solver = 'liblinear'
# Initialize Logistic Regression with solver and max_iter
model = LogisticRegression(solver='liblinear', max_iter=500)  # You can adjust max_iter if needed

# Train the model on the original imbalanced data
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluate performance on the original dataset
print("\nOriginal Dataset Performance:")
print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))

# Train and evaluate on resampled data - Random Oversampling
model.fit(X_train_resampled_scaled, y_train_resampled)
y_pred_resampled = model.predict(X_test_resampled_scaled)

print("\nRandom Oversampling Performance:")
print(classification_report(y_test, y_pred_resampled))
print("AUC-ROC:", roc_auc_score(y_test, model.predict_proba(X_test_resampled_scaled)[:, 1]))

# Train and evaluate on resampled data - Random Under Sampling
model.fit(X_train_resampled_undersample_scaled, y_train_resampled_undersample)
y_pred_resampled_undersample = model.predict(X_test_resampled_undersample_scaled)

print("\nRandom Under Sampling Performance:")
print(classification_report(y_test, y_pred_resampled_undersample))
print("AUC-ROC:", roc_auc_score(y_test, model.predict_proba(X_test_resampled_undersample_scaled)[:, 1]))

# Train and evaluate on resampled data - SMOTE
model.fit(X_train_resampled_smote_scaled, y_train_resampled_smote)
y_pred_resampled_smote = model.predict(X_test_resampled_smote_scaled)

print("\nSMOTE Performance:")
print(classification_report(y_test, y_pred_resampled_smote))
print("AUC-ROC:", roc_auc_score(y_test, model.predict_proba(X_test_resampled_smote_scaled)[:, 1]))

# 5. Class Weighting - Modify the model to assign higher importance to the minority class
model_with_weights = LogisticRegression(solver='liblinear', max_iter=500, class_weight='balanced')
model_with_weights.fit(X_train_scaled, y_train)
y_pred_with_weights = model_with_weights.predict(X_test_scaled)

print("\nClass Weighting Performance:")
print(classification_report(y_test, y_pred_with_weights))
print("AUC-ROC:", roc_auc_score(y_test, model_with_weights.predict_proba(X_test_scaled)[:, 1]))

# 6. Summary of Results
# You can summarize the results in a table or any other format as needed

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from a CSV file (replace 'data.csv' with your file path)
df = pd.read_csv('data.csv')

# 1. General Information about the dataset
print("\nGeneral Information about the dataset:")
df.info()

# 2. Statistical Summary for numerical and categorical features
print("\nStatistical Summary for Numerical Columns:")
print(df.describe())

print("\nStatistical Summary for Categorical Columns:")
print(df.describe(include=['object']))

# 3. Checking for missing values
print("\nMissing Values per Column:")
missing_values = df.isnull().sum()
print(missing_values)

# Visualize missing values with a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Heatmap of Missing Values")
plt.show()

# 4. Correlation between numerical columns
print("\nCorrelation Matrix for Numerical Columns:")
correlation_matrix = df.corr()
print(correlation_matrix)

# Visualize correlation using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 5. Distribution of numerical columns
df.hist(bins=20, figsize=(14, 10), color='blue')
plt.suptitle("Distribution of Numerical Columns", fontsize=16)
plt.show()

# 6. Boxplot to detect outliers in numerical columns
plt.figure(figsize=(14, 8))
df.select_dtypes(include=['number']).plot(kind='box', subplots=True, layout=(2, 3), sharex=False, sharey=False, figsize=(14, 8))
plt.suptitle("Boxplot of Numerical Columns")
plt.show()

# 7. Checking unique values for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}':", df[col].nunique())
    print(df[col].value_counts())

# 8. Visualizing the count of each category in categorical columns
plt.figure(figsize=(14, 8))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=col, data=df)
    plt.title(f"Count Plot for {col}")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. Scatterplot for numerical relationships
numerical_cols = df.select_dtypes(include=['number']).columns
if len(numerical_cols) >= 2:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=numerical_cols[0], y=numerical_cols[1], data=df)
    plt.title(f"Scatterplot between {numerical_cols[0]} and {numerical_cols[1]}")
    plt.show()

# 10. Exporting cleaned and analyzed data (optional)
df.to_csv('cleaned_data.csv', index=False)
print("\nCleaned and analyzed data exported successfully to 'cleaned_data.csv'.")

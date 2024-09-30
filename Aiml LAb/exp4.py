# Import required libraries
import pandas as pd

# Load dataset from a CSV file (replace 'data.csv' with your file path)
# Example: df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv')

# Display number of rows and columns
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

# Display the first five rows
print("\nFirst Five Rows:\n", df.head())

# Display the size of the dataset (total elements)
print("\nSize of Dataset (Total Elements):", df.size)

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values per Column:\n", missing_values)

# Describe the numerical columns (Sum, Average, Min, Max, etc.)
print("\nSummary Statistics for Numerical Columns:")
print(df.describe())  # Shows count, mean, std, min, 25%, 50%, 75%, max

# If you want to explicitly show sum, mean, min, and max for numerical columns:
print("\nSum of Numerical Columns:\n", df.select_dtypes(include=['number']).sum())
print("\nAverage of Numerical Columns:\n", df.select_dtypes(include=['number']).mean())
print("\nMinimum Values of Numerical Columns:\n", df.select_dtypes(include=['number']).min())
print("\nMaximum Values of Numerical Columns:\n", df.select_dtypes(include=['number']).max())

# Export the dataset to a new CSV file after analysis (optional)
df.to_csv('exported_data.csv', index=False)

print("\nData exported successfully to 'exported_data.csv'.")

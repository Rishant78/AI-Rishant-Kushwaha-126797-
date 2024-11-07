import pandas as pd
from scipy.stats import zscore

# List of file paths for the datasets you want to analyze
file_paths = ['exported_data.csv']  # Add more paths as needed

# Define a function to detect outliers using the IQR method for numerical columns
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

# Define a function to detect outliers using the Z-score method
def detect_outliers_zscore(df, column, threshold=3):
    z_scores = zscore(df[column].dropna())
    return (z_scores > threshold) | (z_scores < -threshold)

# Setting: Choose to remove or replace outliers
remove_outliers = True  # Set to False to replace outliers instead of removing them

# Loop over each dataset
for file_path in file_paths:
    print(f"Processing dataset: {file_path}")
    
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Identify numerical columns in the dataset
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    # Apply both methods to handle outliers for each numerical column
    for col in numerical_columns:
        # Detect outliers using IQR and Z-score methods
        outliers_iqr_mask = detect_outliers_iqr(data, col)
        outliers_zscore_mask = detect_outliers_zscore(data, col)
        
        # Combine both outlier masks
        combined_outliers_mask = outliers_iqr_mask | outliers_zscore_mask
        
        if combined_outliers_mask.any():  # If there are outliers
            if remove_outliers:
                # Remove rows with outliers
                data = data[~combined_outliers_mask]
                print(f"Removed outliers in {col} using IQR and Z-score methods.")
            else:
                # Replace outliers with the median value
                median_value = data[col].median()
                data.loc[combined_outliers_mask, col] = median_value
                print(f"Replaced outliers in {col} with median value {median_value} using IQR and Z-score methods.")
    
    # Display or save the cleaned dataset if needed
    print("Cleaned dataset preview:")
    print(data.head())
    print("\n" + "="*40 + "\n")  # Separator between datasets

    # Optional: Save the cleaned data
    cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    data.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved to {cleaned_file_path}")

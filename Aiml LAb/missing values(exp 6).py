import pandas as pd

# List of file names for multiple datasets
file_names = ['exported_data.csv']

# Loop through each file, load the dataset, and process it
for file_name in file_names:
    # Load each dataset
    df = pd.read_csv(file_name)
    
    print(f"Processing file: {file_name}")
    
    # Check and display missing values before filling
    print("Missing values per column before filling:")
    print(df.isnull().sum())
    
    # Fill missing values with the mean of numeric columns
    df_filled = df.fillna(df.select_dtypes(include='number').mean())

    # Check for missing values after filling
    print("Missing values per column after filling:")
    print(df_filled.isnull().sum())
    
    # Optionally drop any rows with remaining missing values
    df_cleaned = df_filled.dropna()

    # Show the cleaned dataset
    print("Cleaned DataFrame (first few rows):")
    print(df_cleaned.head())  # Displaying only the top few rows for each dataset
    print("\n")
    
    # Save the cleaned DataFrame to a new CSV file with "Cleaned_" prefix
    cleaned_file_name = f"Cleaned_{file_name}"
    df_cleaned.to_csv(cleaned_file_name, index=False)
    print(f"Saved cleaned data to {cleaned_file_name}")
    print("\n")

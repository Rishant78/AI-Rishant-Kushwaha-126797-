# Step 1: Import necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Step 2: Load the dataset
file_path = 'E:\\Aiml LAb\\exported_data.csv'  
data = pd.read_csv(file_path)

# Step 3: Define the categorical and numerical columns
categorical_columns = ['Supplier', 'Part', 'Location']
numerical_columns = ['Quantity', 'Price']

# Step 4: Handle missing values if any
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

# Step 5: Encode categorical features
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Corrected argument
encoded_data = encoder.fit_transform(data[categorical_columns])

# Create a DataFrame for the encoded features
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

# Step 6: Scale numerical features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numerical_columns])

# Create a DataFrame for the scaled features
scaled_df = pd.DataFrame(scaled_data, columns=numerical_columns)

# Step 7: Combine processed features into a final DataFrame
final_data = pd.concat([scaled_df, encoded_df], axis=1)

# Step 8: Save or display the final DataFrame
final_data.to_csv('E:\\Aiml LAb\\processed_data.csv', index=False)
print(final_data.head())

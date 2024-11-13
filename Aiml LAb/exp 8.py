
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load the dataset
file_path = 'exported_data.csv'  # Replace with the path to your dataset
data = pd.read_csv(file_path)

# Step 3: Extract numerical columns for PCA
numerical_columns = ['Quantity', 'Price']  # Replace with your dataset's numerical columns
numerical_data = data[numerical_columns]

# Step 4: Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Step 5: Apply PCA
pca = PCA(n_components=2)  # Number of principal components
principal_components = pca.fit_transform(scaled_data)

# Step 6: Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Combine the PCA results with the original dataset (optional)
final_data = pd.concat([data, pca_df], axis=1)

# Step 7: Display the explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Save the final dataset with PCA results (optional)
final_data.to_csv('pca_transformed_dataset.csv', index=False)

# Step 8: Display the result
print(final_data.head())

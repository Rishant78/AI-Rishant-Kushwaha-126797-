# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load the Dataset
# Replace 'dataset.csv' with the actual dataset filename from the previous lab.
df = pd.read_csv('Iris_Dataset.csv')  

# Displaying the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Step 2: Data Preprocessing
# Selecting only numerical columns for clustering
numerical_data = df.select_dtypes(include=[np.number])

# Normalizing/Scaling the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Splitting the data into training and testing subsets
X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=42)

# Step 3: Perform Clustering using K-Means
kmeans = KMeans(n_clusters=3, random_state=42)  # Replace 3 with the desired number of clusters
kmeans.fit(X_train)

# Predicting cluster labels for training and testing data
train_labels = kmeans.predict(X_train)
test_labels = kmeans.predict(X_test)

# Visualizing the clusters for training data
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=train_labels, cmap='viridis', label='Train Clusters')
plt.title('K-Means Clustering (Training Data)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()

# Step 4: Evaluate Performance
# Calculating evaluation metrics
train_silhouette = silhouette_score(X_train, train_labels)
test_silhouette = silhouette_score(X_test, test_labels)

train_db_index = davies_bouldin_score(X_train, train_labels)
test_db_index = davies_bouldin_score(X_test, test_labels)

print(f"K-Means Silhouette Score (Train): {train_silhouette:.3f}")
print(f"K-Means Silhouette Score (Test): {test_silhouette:.3f}")
print(f"K-Means Davies-Bouldin Index (Train): {train_db_index:.3f}")
print(f"K-Means Davies-Bouldin Index (Test): {test_db_index:.3f}")

# Step 5: Compare Algorithms (Optional)
# DBSCAN Implementation
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
dbscan_labels_train = dbscan.fit_predict(X_train)
dbscan_labels_test = dbscan.fit_predict(X_test)

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels_train = agglo.fit_predict(X_train)
agglo_labels_test = agglo.fit_predict(X_test)

# Evaluation for DBSCAN
dbscan_silhouette_train = silhouette_score(X_train, dbscan_labels_train)
dbscan_silhouette_test = silhouette_score(X_test, dbscan_labels_test)

# Evaluation for Agglomerative Clustering
agglo_silhouette_train = silhouette_score(X_train, agglo_labels_train)
agglo_silhouette_test = silhouette_score(X_test, agglo_labels_test)

print("\nComparison of Algorithms:")
print(f"DBSCAN Silhouette Score (Train): {dbscan_silhouette_train:.3f}")
print(f"DBSCAN Silhouette Score (Test): {dbscan_silhouette_test:.3f}")
print(f"Agglomerative Clustering Silhouette Score (Train): {agglo_silhouette_train:.3f}")
print(f"Agglomerative Clustering Silhouette Score (Test): {agglo_silhouette_test:.3f}")

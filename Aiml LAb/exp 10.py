import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display the first few rows of the dataset
print(df.head())

# --- 1. Histogram of Numerical Features ---
# Plot histograms for numerical features to analyze their distributions
plt.figure(figsize=(12, 8))  # Set figure size
df.iloc[:, :-1].hist(bins=15, edgecolor='black', figsize=(12, 8))
plt.suptitle('Histograms of Iris Features', fontsize=16)
plt.tight_layout()
plt.show()

# --- 2. Bar Plot of Categorical Data ---
plt.figure(figsize=(8, 6))
sns.countplot(x='species', data=df, palette='Set2')
plt.title('Species Distribution in the Iris Dataset', fontsize=14)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# --- 3. Pie Chart for Species Proportion ---
species_counts = df['species'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set2'),
        startangle=90, wedgeprops={'edgecolor': 'black'})
plt.title('Proportion of Each Iris Species', fontsize=14)
plt.show()

# --- 4. Box Plot to Analyze Spread of Petal Length ---
plt.figure(figsize=(8, 6))
sns.boxplot(x='species', y='petal length (cm)', data=df, palette='Set1')
plt.title('Box Plot of Petal Length by Species', fontsize=14)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Petal Length (cm)', fontsize=12)
plt.show()

# --- 5. Violin Plot for Petal Length by Species ---
plt.figure(figsize=(8, 6))
sns.violinplot(x='species', y='petal length (cm)', data=df, palette='Set2')
plt.title('Violin Plot of Petal Length by Species', fontsize=14)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Petal Length (cm)', fontsize=12)
plt.show()

# --- 6. Pair Plot to Show Pairwise Relationships ---
sns.pairplot(df, hue='species', palette='Set1')
plt.suptitle('Pair Plot of Iris Features', fontsize=16, y=1.02)
plt.show()

# --- 7. Heatmap for Correlation Matrix ---
plt.figure(figsize=(8, 6))
corr_matrix = df.iloc[:, :-1].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=1, linecolor='black')
plt.title('Heatmap of Feature Correlations', fontsize=14)
plt.show()

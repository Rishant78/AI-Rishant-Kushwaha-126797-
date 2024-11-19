# Load the dataset (if saved and need to read it back)
import pandas as pd  # Import pandas for data manipulation

data = pd.read_csv("synthetic_regression_dataset.csv")

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())
print(data.describe())
import matplotlib.pyplot as plt

# Scatter plot
plt.scatter(data["Feature"], data["Target"], alpha=0.7)
plt.title("Feature vs. Target")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()
print(data.corr())
from sklearn.model_selection import train_test_split

X = data[["Feature"]]
y = data["Target"]

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display model coefficients
print("Coefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)
plt.scatter(y_test, y_pred, alpha=0.7)
plt.title("Actual vs. Predicted")
plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.show()
# Regression line on training data
plt.scatter(X_train, y_train, color="blue", alpha=0.5, label="Training Data")
plt.plot(X_train, model.predict(X_train), color="red", label="Regression Line")
plt.title("Regression Line on Training Data")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()

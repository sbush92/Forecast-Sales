import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Preprocessing and feature engineering (you will need to customize this part)
# For simplicity, let's consider only numeric features and no missing values
X = train_data.select_dtypes(include="number").drop("Weekly_Sales", axis=1)
y = train_data["Weekly_Sales"]

# Split into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
predictions = model.predict(X_valid)
mae = mean_absolute_error(y_valid, predictions)
print(f"Mean Absolute Error: {mae}")
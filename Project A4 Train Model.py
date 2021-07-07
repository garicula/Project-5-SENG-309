# Garrick Morley
# SENG 309 Project A4
# This program trains the data model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load our data set
df = pd.read_csv("insurance.csv")

# Create the X and y arrays (X is features, Y is labels)
X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["cost_amount"]

# Split the data set in a training set (70%) and a test set (3%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it to make predictions later
joblib.dump(model, 'insurance_charges.pkl')

# Report how well the model is performing
print("Model training results: ")

# Report an error rate on the training set
mse_train = mean_absolute_error(y_train, model.predict(X_train))
print(f" - Training Set Error: {mse_train}")

# Report an error rate on the test set
mse_test = mean_absolute_error(y_test, model.predict(X_test))
print(f" - Test Set Error: {mse_test}")


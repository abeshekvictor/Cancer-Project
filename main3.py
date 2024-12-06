import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = pd.read_csv("breastcancer.csv")

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Check data types
print("\nData types:\n", data.dtypes)

# Print the column names to identify the target variable
print("\nColumn names:\n", data.columns)

# Ensure that the target column is correctly specified
# Replace 'Outcome' with the correct target variable from your dataset
# Example: If the target column is 'diagnosis', change it below
target_column = 'diagnosis'  # Update this based on your dataset
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a Random Forest classifier
model = RandomForestClassifier(random_state=42)

# Flag to track if the model is trained
is_trained = False

try:
    # Train the model
    model.fit(X_train, y_train)
    is_trained = True
    print("\n The model has been successfully trained.")
except Exception as e:
    print(f"Training failed: {e}")

# Test the model if trained
if is_trained:
    try:
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model's accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Test Case: Predictions made successfully.")
        print("Accuracy:", accuracy*100)
    except Exception as e:
        print(f"Prediction or evaluation failed: {e}")
else:
    print("The model was not trained, so predictions can't be made.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ask the user for the file path and target column name
file_path = input("Enter the path to the breast cancer dataset CSV file: ")
target_column = input("Enter the name of the target variable (e.g., 'diagnosis'): ")

# Load the dataset
try:
    data = pd.read_csv(file_path)
    print("\nDataset loaded successfully.")
except FileNotFoundError:
    print("Error: File not found. Please check the file path and try again.")
    exit()

# Check for missing values
print("\nMissing values:\n", data.isnull().sum())

# Check data types
print("\nData types:\n", data.dtypes)

# Print the column names to identify the target variable
print("\nColumn names:\n", data.columns)

# Ensure the target column is correctly specified by the user
if target_column not in data.columns:
    print(f"Error: The specified target column '{target_column}' does not exist.")
    exit()

# Define features and target variable
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
    print("\nThe model has been successfully trained.")
except Exception as e:
    print(f"Training failed: {e}")

# Test the model if trained
if is_trained:
    try:
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model's accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("\nPredictions made successfully.")
        print("Accuracy%:", accuracy * 100)
    except Exception as e:
        print(f"Prediction or evaluation failed: {e}")
else:
    print("The model was not trained, so predictions can't be made.")

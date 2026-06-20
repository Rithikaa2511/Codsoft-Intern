# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = pd.read_csv("Iris.csv", sep="\t")

# Display the first 5 rows of the dataset
print("\nFirst 5 Rows of Dataset:\n")
print(data.head())

# Display column names
print("\nColumn Names:")
print(data.columns)

# Select input features
X = data[[
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]]

# Select target column
y = data["species"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict the test data
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy
print("\nModel Accuracy :", round(accuracy * 100, 2), "%")

# Take flower measurements from the user
print("\nEnter Flower Measurements")

sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))

# Create a DataFrame for the new flower
new_flower = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
)

# Predict the flower species
prediction = model.predict(new_flower)

# Display the prediction
print("\nPredicted Iris Flower Species :", prediction[0])

print("\nProject Completed Successfully!")
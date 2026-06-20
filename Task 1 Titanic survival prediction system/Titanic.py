import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv", sep="\t")

print(df.columns)
print(df.head())

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop unnecessary columns
df.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis=1, inplace=True)

# Convert categorical values to numbers
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Titanic Survival Prediction System")
print("Model Accuracy:", round(accuracy * 100, 2), "%")
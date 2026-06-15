import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("movie_dataset.csv", sep="\t", encoding="latin1")

# Select required columns
df = df[[
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3",
    "Duration",
    "Votes",
    "Rating"
]]

# Remove missing values
df = df.dropna()

# Convert Duration from "140 min" to 140
df["Duration"] = df["Duration"].astype(str)
df["Duration"] = df["Duration"].str.replace(" min", "", regex=False)
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

# Convert Votes from "1,234" to 1234
df["Votes"] = df["Votes"].astype(str)
df["Votes"] = df["Votes"].str.replace(",", "", regex=False)
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

# Remove rows that became NaN after conversion
df = df.dropna()

# Encode categorical columns
for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("Rating", axis=1)
y = df["Rating"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Show predictions
print("\nActual Rating vs Predicted Rating")
for actual, predicted in zip(y_test.head(10), y_pred[:10]):
    print(f"Actual: {actual:.1f}  Predicted: {predicted:.1f}")

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMovie Rating Prediction")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))
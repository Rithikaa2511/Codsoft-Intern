# SALES PREDICTION USING PYTHON

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------
# LOAD DATASET
# ----------------------------------------

df = pd.read_csv("advertising.csv", sep="\t")

print("=" * 60)
print("SALES PREDICTION USING PYTHON")
print("=" * 60)

# ----------------------------------------
# DATA EXPLORATION
# ----------------------------------------

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ----------------------------------------
# FEATURES AND TARGET
# ----------------------------------------

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

print("\nInput Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

# ----------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ----------------------------------------
# MODEL TRAINING
# ----------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# ----------------------------------------
# MODEL COEFFICIENTS
# ----------------------------------------

print("\nModel Coefficients:")

for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", round(coef, 4))

print("\nIntercept:", round(model.intercept_, 4))

# ----------------------------------------
# MODEL TESTING
# ----------------------------------------

y_pred = model.predict(X_test)

# ----------------------------------------
# ACTUAL VS PREDICTED
# ----------------------------------------

comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print("\nFirst 10 Actual vs Predicted Sales:")
print(comparison.head(10))

# ----------------------------------------
# MODEL EVALUATION
# ----------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("Mean Absolute Error :", round(mae, 2))
print("Mean Squared Error  :", round(mse, 2))
print("R2 Score            :", round(r2, 4))

# ----------------------------------------
# USER INPUT FOR SALES PREDICTION
# ----------------------------------------

print("\n" + "=" * 60)
print("PREDICT SALES FOR NEW ADVERTISEMENT BUDGET")
print("=" * 60)

tv = float(input("Enter TV Budget: "))
radio = float(input("Enter Radio Budget: "))
newspaper = float(input("Enter Newspaper Budget: "))

new_data = pd.DataFrame({
    'TV': [tv],
    'Radio': [radio],
    'Newspaper': [newspaper]
})

predicted_sales = model.predict(new_data)

print("\nAdvertisement Budget")
print("TV Budget        :", tv)
print("Radio Budget     :", radio)
print("Newspaper Budget :", newspaper)

print("\nPredicted Sales:", round(predicted_sales[0], 2))

print("\nProject Completed Successfully!")
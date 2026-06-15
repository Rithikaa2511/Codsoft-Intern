# 🎬 Movie Rating Prediction using Machine Learning

## 📖 Project Overview

Movie Rating Prediction is a Machine Learning project that predicts the rating of a movie based on its features such as Genre, Director, Actors, Duration, and Votes. The project demonstrates data preprocessing, feature encoding, model training, prediction, and evaluation using the Random Forest Regression algorithm.

---

## 🎯 Objective

The objective of this project is to build a machine learning model that can accurately predict movie ratings based on various movie attributes.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regressor

---

## 📂 Dataset Features

The dataset contains the following attributes:

- Genre
- Director
- Actor 1
- Actor 2
- Actor 3
- Duration
- Votes
- Rating (Target Variable)

---

## 🔄 Project Workflow

### 1. Data Collection
The movie dataset is loaded using Pandas.

### 2. Data Preprocessing
- Handling missing values
- Converting duration and votes into numeric format
- Encoding categorical features

### 3. Feature Selection
Selected important features for prediction.

### 4. Model Training
The dataset is divided into training and testing sets and trained using the Random Forest Regressor algorithm.

### 5. Prediction
The trained model predicts movie ratings on unseen data.

### 6. Evaluation
The model performance is evaluated using:
- Mean Absolute Error (MAE)
- R² Score

---

## 🤖 Machine Learning Algorithm

### Random Forest Regressor

Random Forest Regressor is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## 📊 Results

### Sample Predictions

| Actual Rating | Predicted Rating |
|--------------|------------------|
| 6.0 | 5.7 |
| 2.4 | 6.7 |
| 3.8 | 6.1 |
| 7.2 | 6.1 |
| 6.6 | 5.8 |

### Performance Metrics

| Metric | Value |
|---------|---------|
| Mean Absolute Error (MAE) | 0.98 |
| R² Score | 0.17 |

---

## ✅ Conclusion

The Movie Rating Prediction model was successfully developed using Machine Learning techniques. The Random Forest Regressor was able to predict movie ratings with reasonable accuracy. The project demonstrates the complete machine learning workflow from preprocessing to model evaluation.

---

## 🚀 Future Enhancements

- Add more movie-related features
- Perform hyperparameter tuning
- Compare multiple regression algorithms
- Improve prediction accuracy
- Deploy the model as a web application


# 🚢 Titanic Survival Prediction using Machine Learning

## 📌 Project Overview

This project uses Machine Learning to predict whether a passenger survived the Titanic disaster based on passenger information such as age, gender, ticket class, fare, and embarkation point.

The model is trained using the Titanic dataset and uses Logistic Regression for classification.

---

## 🎯 Objective

The objective of this project is to build a predictive model that determines whether a passenger survived the Titanic disaster.

This project demonstrates data preprocessing, feature engineering, model training, and evaluation using Machine Learning techniques.

---

## 📊 Dataset

The dataset contains information about Titanic passengers.

### Features Used

* Pclass (Passenger Class)
* Sex
* Age
* SibSp (Number of Siblings/Spouses Aboard)
* Parch (Number of Parents/Children Aboard)
* Fare
* Embarked (Port of Embarkation)

### Target Variable

* Survived

  * 0 = Did Not Survive
  * 1 = Survived

---

## 🛠 Technologies Used

* Python
* Pandas
* Scikit-learn

---

## 🤖 Machine Learning Algorithm

### Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for binary classification problems.

In this project, it predicts whether a passenger survived or not based on available passenger details.

---

## ⚙️ Project Workflow

1. Load the Titanic dataset
2. Explore dataset structure
3. Handle missing values
4. Remove unnecessary columns
5. Convert categorical features into numerical values
6. Split data into training and testing sets
7. Train the Logistic Regression model
8. Evaluate model performance
9. Predict passenger survival

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### Missing Values

* Missing values in Age were replaced with the mean age.
* Missing values in Embarked were replaced with the most frequent value.

### Columns Removed

The following columns were removed because they were not required for prediction:

* PassengerId
* Name
* Ticket
* Cabin

### Encoding

Categorical values were converted into numerical values:

#### Sex

* Male → 0
* Female → 1

#### Embarked

* S → 0
* C → 1
* Q → 2

---

## 📈 Model Performance

The model was evaluated using Accuracy Score.

Example Output:

```text
Titanic Survival Prediction System
Model Accuracy: 81.56%
```

*Accuracy may vary slightly depending on dataset version and train-test split.*

---

## 🚀 Future Enhancements

* Add a graphical user interface (GUI)
* Compare multiple machine learning algorithms
* Visualize passenger survival statistics
* Deploy the model as a web application
* Accept user input for real-time survival prediction

---

## 📚 Learning Outcomes

Through this project, you will learn:

* Data Cleaning
* Handling Missing Values
* Feature Engineering
* Data Encoding
* Logistic Regression
* Model Evaluation
* Machine Learning Workflow

---

## 📜 Conclusion

This project demonstrates how Machine Learning can be used to predict passenger survival on the Titanic. By applying Logistic Regression and preprocessing techniques, the model successfully learns patterns from historical passenger data and makes accurate survival predictions.

# Customer Churn Prediction System

## 📌 Project Overview
This project builds an end-to-end machine learning system to predict customer churn using real-world banking data. The objective is to identify customers at risk of leaving the bank and enable proactive retention strategies.

The project follows an industry-standard workflow, including data exploration, feature engineering, handling class imbalance, model training, and evaluation.

---

## 📊 Dataset
- Source: Bank Customer Churn Dataset
- Size: 10,000+ customer records
- Target Variable: `Attrition_Flag`
  - Existing Customer
  - Attrited Customer

The dataset includes demographic, financial, and behavioral attributes.

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights from EDA:
- Customer churn is strongly driven by **behavioral factors** rather than demographics
- Higher churn observed among customers with:
  - Lower transaction counts
  - Lower transaction amounts
  - Higher inactivity periods
- Transaction behavior and engagement metrics are the strongest churn indicators

---

## 🛠️ Feature Engineering & Preprocessing
- Dropped data leakage columns
- Encoded categorical variables
- Scaled numerical features
- Addressed class imbalance using **SMOTE**

---

## 🤖 Modeling Approach
The following models were trained and evaluated:
- Logistic Regression (baseline)
- Random Forest
- **XGBoost (final model)**

Evaluation metrics:
- ROC-AUC
- Precision & Recall
- Confusion Matrix

XGBoost combined with SMOTE achieved the best overall performance in identifying churned customers.

---

## 📈 Results
- Improved recall for churned customers
- Behavioral features such as transaction count and inactivity emerged as top predictors
- The final model balances predictive performance with business interpretability

---

## 📂 Project Structure

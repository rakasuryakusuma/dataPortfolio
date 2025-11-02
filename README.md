# 🕵️‍♂️ Fraud Detection Project

## 📌 Overview
This project focuses on analyzing financial transaction data to detect fraudulent activities. The dataset contains transaction records labeled as **fraudulent (1)** and **non-fraudulent (0)**.  
The goal is to explore transaction patterns, understand differences between fraud and non-fraud cases, and prepare the data for machine learning models.

---

## 📊 Exploratory Data Analysis (EDA)
Key steps in the analysis:
- **Boxplot Visualization**: Compare transaction amount distributions between fraud and non-fraud cases.  
  Initial results show that fraudulent transactions tend to have higher median values and greater variability.
- **Outlier Detection**: Boxplots also highlight extreme transactions that may influence model performance.

---

## 🔄 Data Transformation
During preprocessing, one important question was raised:  
> *Should we use `np.log(x)` or `np.log1p(x)` for transaction amount transformation?*

### ✨ Explanation
- `np.log(x)` → computes the natural logarithm of **x**.  
  ⚠️ Cannot handle zero values (`log(0)` is undefined).
- `np.log1p(x)` → computes the natural logarithm of **(1 + x)**.  
  ✅ Safe for zero values, since `log1p(0) = 0`.  
  ✅ More numerically stable for very small values.

### 📌 Why use it in Fraud Detection?
Transaction amounts are typically **highly skewed** (long right tail).  
Applying a log transformation:
- Compresses large transaction values.  
- Makes the distribution closer to normal.  
- Helps models learn patterns more effectively.  
- `np.log1p` is preferred here because the dataset may contain transactions with amount = 0.

---

## ⚙️ Workflow
1. **Data Cleaning** → filter transactions, remove duplicates, handle missing values.  
2. **EDA** → boxplots, histograms, correlation analysis.  
3. **Feature Engineering** → log transformation (`np.log1p`) on the `Amount` column.  
4. **Modeling** → train classification models (e.g., Logistic Regression, Random Forest, XGBoost).  
5. **Evaluation** → assess performance with Precision, Recall, F1-score, and ROC-AUC.  
6. **Deployment** → deploy the trained model with Streamlit for interactive use.

---

## 💾 Model Saving and Deployment
- The trained model is serialized using **joblib**:
  ```python
  import joblib

  # Save model
  joblib.dump(model, "fraud_model.pkl")

  # Load model
  model = joblib.load("fraud_model.pkl")
  import streamlit as st
  import joblib

- Deploy the app locally and use the model
  ```python
    import streamlit as st
  import joblib
  import numpy as np
  
  # Load model
  model = joblib.load("fraud_model.pkl")
  
  st.title("Fraud Detection App")
  
  amount = st.number_input("Transaction Amount", min_value=0.0)
  amount_log = np.log1p(amount)
  
  if st.button("Predict"):
      prediction = model.predict([[amount_log]])
      st.write("Fraudulent" if prediction[0] == 1 else "Non-Fraudulent")
  ```
And then run the app locally
```cmd
streamlit run fraud_detect.py
```
This launches an interactive web interface where users can input transaction amounts and get prediction.


# 🕵️‍♂️ Fraud Detection Project

## 📌 Overview
This project focuses on analyzing financial transaction data to detect fraudulent activities. The dataset contains transaction records labeled as **isFraud (1)** and **isFlaggedFraud (0)**.  
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
4. **Modeling** → train classification models with Logistic Regression.  
5. **Evaluation** → assess performance with Precision, Recall, F1-score, and ROC-AUC.  
6. **Deployment** → deploy the trained model with Streamlit for interactive use.

---

## 💾 Model Saving and Deployment
- The trained model is serialized using **joblib**:
  ```python
  import joblib

  # Save model
  joblib.dump(model, "fraud_detection_model.pkl")

  # Load model
  model = joblib.load("fraud_detection_model.pkl")
  import streamlit as st
  import joblib

- Deploy the app locally and use the model
  ```python
    import streamlit as st
    import pandas as pd
    import joblib
    
    model = joblib.load('fraud_detection_model.pkl')
    
    st.title("Fraud Detection Application")
    st.write("Enter the transaction details below to predict if it is fraudulent or legitimate.")
    
    st.divider()
    
    transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSTIT"])
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    oldbalanceOrg = st.number_input("Old Balance Sender", min_value=0.0, value=10000.0)
    newbalanceOrig = st.number_input("New Balance Sender", min_value=0.0, value=9000.0)
    oldbalanceDest = st.number_input("Old Balance Receiver", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("New Balance Receiver", min_value=0.0, value=0.0)
    
    if st.button("Predict Fraud"):
        input_data = pd.DataFrame({
            'type': [transaction_type],
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig],
            'oldbalanceDest': [oldbalanceDest],
            'newbalanceDest': [newbalanceDest]
        })

    prediction = model.predict(input_data)[0]
    
    st.subheader(f"Prediction Result : '{int(prediction)}'")

    if prediction == 1:
        st.error("The transaction is predicted to be FRAUDULENT.")
    else:
        st.success("The transaction is predicted to be LEGITIMATE.")
  ```
And then run the app locally
```cmd
streamlit run fraud_detect.py
```
This launches an interactive web interface where users can input transaction amounts and get prediction.


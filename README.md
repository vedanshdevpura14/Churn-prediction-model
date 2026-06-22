# Customer Churn Prediction

A machine learning system that predicts whether a telecom customer is likely to churn (cancel their subscription), built on the Telco Customer Churn dataset and deployed as an interactive web app.

**Live App:** [https://churn-prediction-model-vd.streamlit.app/](https://churn-prediction-model-vd.streamlit.app/)

## What This Project Does

Customer churn — when a paying customer stops using a service — is one of the most expensive problems for subscription-based businesses, since acquiring a new customer costs far more than retaining an existing one. This project builds a classifier that takes a customer's profile (contract type, services subscribed, tenure, billing details, etc.) and predicts:

1. **Whether** they are likely to churn (Yes/No)
2. **How likely** — a churn probability score, so the business can prioritize high-risk customers for retention efforts

The end result is a Streamlit web app where anyone can enter a customer's details and get an instant churn risk prediction.

## Dataset

- **Source:** Telco Customer Churn dataset (IBM sample dataset, ~7,000 customer records)
- **Features:** 19 customer attributes including demographics (gender, senior citizen status), account info (tenure, contract type, payment method), and subscribed services (internet, phone, streaming, tech support, etc.)
- **Target:** `Churn` (Yes/No)
- **Class imbalance:** ~73% No-churn vs ~27% Churn — a key challenge addressed during modeling

## Approach

### 1. Data Cleaning & Preprocessing
- Converted `TotalCharges` to numeric and dropped rows with missing values
- Binned `tenure` (in months) into grouped categories (`tenure_group`) instead of using raw tenure, to capture non-linear relationships between customer lifetime and churn risk
- Dropped non-predictive identifier columns (`customerID`)
- One-hot encoded all categorical features, producing a 50-feature matrix

### 2. Handling Class Imbalance
Since churners are a minority class, a model trained naively would be biased toward predicting "No Churn." To fix this, **SMOTEENN** (a combination of SMOTE oversampling and Edited Nearest Neighbors cleaning) was applied to the training data — this generates synthetic minority-class samples while also removing ambiguous/noisy points near the class boundary, producing a cleaner, balanced dataset.

### 3. Model Training
- **Algorithm:** Random Forest Classifier
- **Hyperparameters:** `n_estimators=100`, `max_depth=6`, `min_samples_leaf=8`, `criterion='gini'`
- Random Forest was selected after comparison with other classifiers (Logistic Regression, Decision Tree) for its balance of accuracy and robustness to overfitting on this dataset

### 4. Evaluation
On the held-out test set (post-SMOTEENN):

| Metric | Score |
|---|---|
| Accuracy | 93.4% |
| Precision (Churn class) | 0.91 |
| Recall (Churn class) | 0.97 |
| F1-score (Churn class) | 0.94 |

High recall on the churn class was prioritized — in a real business setting, missing an actual churner (false negative) is costlier than flagging a loyal customer for retention outreach (false positive).

### 5. Deployment
The trained model was serialized with `pickle` and wrapped in a **Streamlit** web app:
- The app reconstructs the same preprocessing pipeline (tenure binning → one-hot encoding → column alignment) used during training, so raw user input is transformed into the exact feature format the model expects
- Deployed on **Streamlit Community Cloud** for public access

**Key engineering challenge solved:** the model was originally trained on Windows and failed to load on the Linux-based deployment server due to a known scikit-learn issue where Random Forest/Decision Tree models pickled on Windows use a different internal integer width than on Linux. This was resolved by retraining and re-serializing the model directly in a Linux environment matching the deployment target, with the scikit-learn version pinned in `requirements.txt` to avoid future version drift.

## Tech Stack

- **Language:** Python
- **ML Libraries:** scikit-learn, imbalanced-learn (SMOTEENN)
- **Data Handling:** pandas, NumPy
- **Deployment:** Streamlit, Streamlit Community Cloud

## Project Structure

```
├── app.py                                      # Streamlit app (UI + inference pipeline)
├── model.sav                                   # Trained Random Forest model (pickled)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv         # Dataset (used to reconstruct training columns)
├── requirements.txt                            # Python dependencies (with pinned scikit-learn version)
└── README.md
```

## Running Locally

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How to Use the App

1. Fill in the customer's details — demographics, contract type, services subscribed, billing info
2. Click **Predict Churn**
3. View the prediction (Likely to Churn / Likely to Stay) along with the churn probability score

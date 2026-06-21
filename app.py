import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "model.sav"

# ---------- Cached loaders ----------

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_reference_columns():
    """
    Rebuilds the exact preprocessing pipeline used in training
    (drop customerID/tenure, bin tenure_group, get_dummies) so we
    know the exact column set/order the model expects.
    """
    df = pd.read_csv(CSV_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(how="any", inplace=True)

    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df["tenure_group"] = pd.cut(df["tenure"], range(1, 80, 12), right=False, labels=labels)
    df.drop(columns=["customerID", "tenure"], inplace=True)

    df["Churn"] = np.where(df["Churn"] == "Yes", 1, 0)
    dummies = pd.get_dummies(df)
    train_cols = dummies.drop(columns=["Churn"]).columns

    raw_options = {
        col: sorted(df[col].dropna().unique().tolist())
        for col in df.columns
        if df[col].dtype == object
    }
    return train_cols, raw_options, labels


def preprocess_input(raw: dict, train_cols, labels) -> pd.DataFrame:
    row = pd.DataFrame([raw])

    tenure_bins = pd.cut(row["tenure"], range(1, 80, 12), right=False, labels=labels)
    row["tenure_group"] = tenure_bins
    row.drop(columns=["tenure"], inplace=True)

    row_dummies = pd.get_dummies(row)
    row_dummies = row_dummies.reindex(columns=train_cols, fill_value=0)
    return row_dummies


# ---------- App ----------

st.title("📉 Customer Churn Predictor")
st.caption("Random Forest + SMOTEENN model — predicts whether a telecom customer is likely to churn.")

try:
    model = load_model()
    train_cols, raw_options, labels = load_reference_columns()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Make sure model.sav and the CSV are in the app folder.")
    st.stop()

st.subheader("Customer details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", raw_options.get("gender", ["Male", "Female"]))
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Partner", raw_options.get("Partner", ["Yes", "No"]))
    dependents = st.selectbox("Dependents", raw_options.get("Dependents", ["Yes", "No"]))
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    phone_service = st.selectbox("Phone Service", raw_options.get("PhoneService", ["Yes", "No"]))
    multiple_lines = st.selectbox("Multiple Lines", raw_options.get("MultipleLines", ["Yes", "No", "No phone service"]))
    internet_service = st.selectbox("Internet Service", raw_options.get("InternetService", ["DSL", "Fiber optic", "No"]))
    online_security = st.selectbox("Online Security", raw_options.get("OnlineSecurity", ["Yes", "No", "No internet service"]))
    online_backup = st.selectbox("Online Backup", raw_options.get("OnlineBackup", ["Yes", "No", "No internet service"]))

with col2:
    device_protection = st.selectbox("Device Protection", raw_options.get("DeviceProtection", ["Yes", "No", "No internet service"]))
    tech_support = st.selectbox("Tech Support", raw_options.get("TechSupport", ["Yes", "No", "No internet service"]))
    streaming_tv = st.selectbox("Streaming TV", raw_options.get("StreamingTV", ["Yes", "No", "No internet service"]))
    streaming_movies = st.selectbox("Streaming Movies", raw_options.get("StreamingMovies", ["Yes", "No", "No internet service"]))
    contract = st.selectbox("Contract", raw_options.get("Contract", ["Month-to-month", "One year", "Two year"]))
    paperless_billing = st.selectbox("Paperless Billing", raw_options.get("PaperlessBilling", ["Yes", "No"]))
    payment_method = st.selectbox("Payment Method", raw_options.get("PaymentMethod", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]))
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)

if st.button("Predict Churn", type="primary", use_container_width=True):
    raw_input = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    X = preprocess_input(raw_input, train_cols, labels)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ Likely to CHURN — probability: {proba:.1%}")
    else:
        st.success(f"✅ Likely to STAY — churn probability: {proba:.1%}")

    st.progress(min(max(proba, 0.0), 1.0))

st.divider()
st.caption("Model: RandomForestClassifier trained on SMOTEENN-resampled Telco churn data.")

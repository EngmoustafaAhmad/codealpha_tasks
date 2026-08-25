import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================
# Load Models
# =========================

lr_package = joblib.load(
    "models/logistic_regression.pkl"
)

rf_package = joblib.load(
    "models/random_forest.pkl"
)


lr_model = lr_package["model"]
rf_model = rf_package["model"]

encoder = rf_package["encoder"]
imputer = rf_package["imputer"]

cat_cols = rf_package["cat_cols"]
num_cols = rf_package["num_cols"]


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="wide"
)


# =========================
# Title
# =========================

st.title("💳 Credit Scoring System")

st.write(
    "Enter customer information to predict the quality of the loan."
)


# =========================
# Sidebar
# =========================

st.sidebar.header("Model")

model_name = st.sidebar.selectbox(
    "Choose Model",
    [
        "Random Forest",
        "Logistic Regression"
    ]
)


if model_name == "Random Forest":
    selected_model = rf_model
else:
    selected_model = lr_model


# =========================
# Customer Information
# =========================

st.header("Customer Information")


col1, col2 = st.columns(2)


with col1:

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced", "Widowed"]
    )

    client_type = st.selectbox(
        "Client Type",
        ["Individual", "Corporate"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    compensation = st.selectbox(
        "Compensation Charged",
        ["Yes", "No"]
    )

    repay_mode = st.selectbox(
        "Repayment Mode",
        ["Cash", "Bank", "Cheque", "Other"]
    )


with col2:

    investment_total = st.number_input(
        "Investment Total",
        min_value=0.0,
        value=10000.0
    )

    current_balance = st.number_input(
        "Account Current Balance",
        min_value=0.0,
        value=5000.0
    )

    install_size = st.number_input(
        "Installment Size",
        min_value=0.0,
        value=1000.0
    )

    due_payment = st.number_input(
        "Due Payment",
        min_value=0.0,
        value=500.0
    )


# =========================
# Prediction
# =========================

if st.button(
    "🔍 Predict Credit Risk",
    use_container_width=True
):

    # Create input DataFrame
    input_df = pd.DataFrame({

        "INF_MARITAL_STATUS": [
            marital_status
        ],

        "CLIENT_TYPE": [
            client_type
        ],

        "INF_GENDER": [
            gender
        ],

        "COMPENSATION_CHARGED": [
            compensation
        ],

        "REPAY_MODE": [
            repay_mode
        ],

        "INVESTMENT_TOTAL_log": [
            np.log1p(investment_total)
        ],

        "ACCCURRENTBALANCE_log": [
            np.log1p(current_balance)
        ],

        "INSTALL_SIZE_log": [
            np.log1p(install_size)
        ],

        "DUE_PAYMENT_log": [
            np.log1p(due_payment)
        ]
    })


    # =========================
    # Encode categorical data
    # =========================

    encoded = encoder.transform(
        input_df[cat_cols]
    )


    # =========================
    # Numerical features
    # =========================

    numerical = input_df[num_cols].values


    # =========================
    # Combine features
    # =========================

    X_input = np.hstack([
        numerical,
        encoded.toarray()
    ])


    # =========================
    # Impute missing values
    # =========================

    X_input = imputer.transform(
        X_input
    )


    # =========================
    # Prediction
    # =========================

    prediction = selected_model.predict(
        X_input
    )[0]

    probability = selected_model.predict_proba(
        X_input
    )[0]


    # =========================
    # Display Result
    # =========================

    st.divider()

    st.header("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ High Credit Risk"
        )

    else:

        st.success(
            "✅ Low Credit Risk"
        )


    # Probability of each class

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Prediction",
            str(prediction)
        )


    with col2:

        st.metric(
            "Default Probability",
            f"{probability[1] * 100:.2f}%"
        )


    # Progress bar

    st.write(
        "Default Probability"
    )

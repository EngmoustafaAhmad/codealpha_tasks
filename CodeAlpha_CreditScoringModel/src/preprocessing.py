import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_data(train_path="../data/train.csv", test_path="../data/test.csv"):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Drop non-predictive ID column
    if "ACC_NO" in df_train.columns:
        df_train.drop("ACC_NO", axis=1, inplace=True)
    if "ACC_NO" in df_test.columns:
        df_test.drop("ACC_NO", axis=1, inplace=True)

    return df_train, df_test


def preprocess_data(df_train, df_test):
    cat_cols = [
        "INF_MARITAL_STATUS",
        "CLIENT_TYPE",
        "INF_GENDER",
        "COMPENSATION_CHARGED",
        "REPAY_MODE",
    ]

    numeric_cols = [
        "INVESTMENT_TOTAL",
        "ACCCURRENTBALANCE",
        "INSTALL_SIZE",
        "DUE_PAYMENT",
    ]

    # 1. Impute Categorical NaNs with Mode
    for col in cat_cols:
        mode_val = df_train[col].mode()[0]
        df_train[col] = df_train[col].fillna(mode_val)
        df_test[col] = df_test[col].fillna(mode_val)

    # 2. Apply Log Transformations
    for col in numeric_cols:
        df_train[col + "_log"] = np.log1p(df_train[col])
        df_test[col + "_log"] = np.log1p(df_test[col])

    df_train.drop(columns=numeric_cols, inplace=True)
    df_test.drop(columns=numeric_cols, inplace=True)

    # 3. Separate Target
    X = df_train.drop("QUALITY_OF_LOAN", axis=1)
    y = df_train["QUALITY_OF_LOAN"]

    # 4. Stratified Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. One-Hot Encoding
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(X_train[cat_cols])

    X_train_encoded = encoder.transform(X_train[cat_cols])
    X_val_encoded = encoder.transform(X_val[cat_cols])
    X_test_encoded = encoder.transform(df_test[cat_cols])

    num_cols = [c for c in X.columns if c not in cat_cols]

    # 6. Combine Numerical & Encoded Features
    X_train_full = np.hstack([X_train[num_cols], X_train_encoded.toarray()])
    X_val_full = np.hstack([X_val[num_cols], X_val_encoded.toarray()])
    X_test_full = np.hstack([df_test[num_cols], X_test_encoded.toarray()])

    # 7. Numerical Imputation
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_full)
    X_val_imputed = imputer.transform(X_val_full)
    X_test_imputed = imputer.transform(X_test_full)

    # Construct feature names
    ohe_feature_names = encoder.get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(ohe_feature_names)

    return (
        X_train_imputed,
        X_val_imputed,
        X_test_imputed,
        y_train,
        y_val,
        all_feature_names,
    )
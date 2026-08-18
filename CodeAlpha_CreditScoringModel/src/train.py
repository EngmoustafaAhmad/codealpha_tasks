from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train):
    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_model.fit(X_train, y_train)
    return lr_model


def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_model.fit(X_train, y_train)
    return rf_model
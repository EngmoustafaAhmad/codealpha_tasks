import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)


def evaluate_model(model, X_val, y_val, model_name="Model"):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print(f"=== {model_name} Validation ===")
    print(classification_report(y_val, y_pred))
    print("ROC-AUC:", roc_auc_score(y_val, y_prob))
    return y_pred


def plot_feature_importance(
    model, feature_names, save_path="../images/feature_importance.png"
):
    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    importances.head(10).plot(kind="barh", color="skyblue").invert_yaxis()
    plt.title("Top 10 Important Features for Credit Scoring")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(
    y_val, y_pred, save_path="../images/confusion_matrix.png"
):
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred, cmap="Blues")
    plt.title("Confusion Matrix - Validation Set")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def export_predictions(
    model, X_test, df_test, output_path="../data/credit_scoring_predictions.csv"
):
    df_test["PREDICTION"] = model.predict(X_test)
    df_test["PROBABILITY_OF_DEFAULT"] = model.predict_proba(X_test)[:, 0]
    df_test.to_csv(output_path, index=False)
    print(f"Predictions successfully exported to {output_path}")
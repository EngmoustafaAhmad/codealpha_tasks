from src.evaluate import (
    evaluate_model,
    export_predictions,
    plot_confusion_matrix,
    plot_feature_importance,
)
from src.preprocessing import load_data, preprocess_data
from src.train import train_logistic_regression, train_random_forest


def main():
    # 1. Load Data
    df_train, df_test = load_data()

    # 2. Preprocess & Split Data
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        feature_names,
    ) = preprocess_data(df_train, df_test)

    # 3. Train & Evaluate Logistic Regression Baseline
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_val, y_val, model_name="Logistic Regression")

    # 4. Train & Evaluate Random Forest
    rf_model = train_random_forest(X_train, y_train)
    y_pred_rf = evaluate_model(
        rf_model, X_val, y_val, model_name="Random Forest"
    )

    # 5. Visualizations & Export
    plot_feature_importance(rf_model, feature_names)
    plot_confusion_matrix(y_val, y_pred_rf)
    export_predictions(rf_model, X_test, df_test)


if __name__ == "__main__":
    main()
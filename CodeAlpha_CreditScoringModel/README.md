# Credit Scoring Model

A Machine Learning project for predicting loan quality and credit risk based on customer financial and demographic information.

The project uses two classification models:

- Logistic Regression
- Random Forest Classifier

It includes data preprocessing, feature engineering, model evaluation, model saving, and a Streamlit interface for making credit-risk predictions.

---

## Dataset

The dataset contains customer financial and demographic information.

### Target Variable

`QUALITY_OF_LOAN`

Represents the quality/class of the customer's loan.

### Categorical Features

- `INF_MARITAL_STATUS`
- `CLIENT_TYPE`
- `INF_GENDER`
- `COMPENSATION_CHARGED`
- `REPAY_MODE`

### Numerical Features

- `INVESTMENT_TOTAL`
- `ACCCURRENTBALANCE`
- `INSTALL_SIZE`
- `DUE_PAYMENT`

### Removed Feature

`ACC_NO`

The account number is removed because it is an identifier and does not provide meaningful information for predicting loan quality.

---

## Data Preprocessing

The dataset goes through several preprocessing steps before model training.

### Missing Values

Missing categorical values are handled using the mode of each feature.

Missing numerical values are handled using median imputation.

### Skewness Analysis

The financial numerical features are analyzed for skewness because financial data can contain uneven distributions and extreme values.

### Log Transformation

The following financial features are transformed using a logarithmic transformation to reduce skewness and the effect of extreme values:

- `INVESTMENT_TOTAL`
- `ACCCURRENTBALANCE`
- `INSTALL_SIZE`
- `DUE_PAYMENT`

The transformed features are used instead of the original numerical features.

---

## Train and Validation Split

The dataset is divided into:

- 80% Training Data
- 20% Validation Data

A random state of 42 is used to make the split reproducible.

Stratification is applied to preserve the distribution of the target classes in both training and validation datasets.

---

## Categorical Encoding

Categorical features are converted into numerical features using One-Hot Encoding.

The encoder is fitted only on the training data and then applied to the validation and test data.

This prevents information from the validation or test datasets from being used during the learning process.

Unknown categories are ignored to prevent errors when new categories appear.

---

## Feature Preparation

After preprocessing, the numerical features and One-Hot Encoded categorical features are combined into a single feature matrix.

Missing numerical values are handled using median imputation based on the training data.

This approach helps prevent data leakage between the training, validation, and test datasets.

---

## Machine Learning Models

### Logistic Regression

Logistic Regression is used as a baseline classification model.

It is:

- Simple
- Fast
- Easy to interpret
- Suitable for classification
- Useful for comparing against more complex models

### Random Forest

Random Forest is used as the second classification model.

It can:

- Capture nonlinear relationships
- Handle multiple features
- Model feature interactions
- Provide feature importance
- Work effectively with encoded categorical features

Both models use class balancing to give additional importance to underrepresented classes.

---

## Model Evaluation

The models are evaluated using the validation dataset.

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Classification Report

The classification report provides:

- Precision
- Recall
- F1-Score
- Support

for each target class.

### Confusion Matrix

The confusion matrix is used to analyze:

- True Positives
- True Negatives
- False Positives
- False Negatives

This helps understand the types of prediction errors made by the models.

---

## Feature Importance

Random Forest provides feature importance values that show which features contribute most to its predictions.

The project visualizes the top 10 most important features to help understand the factors influencing the credit-scoring model.

---

## Model Saving

The trained models are saved as `.pkl` files using Joblib.

The saved model packages include the model and the preprocessing objects required to make predictions on new data.

Generated files:

- `logistic_regression.pkl`
- `random_forest.pkl`

---

## Streamlit Application

The project includes a Streamlit-based user interface for credit-risk prediction.

The user can enter customer information including:

- Marital Status
- Client Type
- Gender
- Compensation Charged
- Repayment Mode
- Investment Total
- Account Current Balance
- Installment Size
- Due Payment

The user can choose between:

- Random Forest
- Logistic Regression

---

## Prediction Process

The Streamlit application follows the same preprocessing process used during model training.

The prediction workflow is:

**User Input → Data Preparation → Log Transformation → One-Hot Encoding → Feature Combination → Imputation → Model Prediction → Probability**

This ensures that new customer data is processed consistently with the training data.

---

## Prediction Output

The application displays:

- Credit Risk Prediction
- Default Probability

Example:

**Prediction:** 1

**Default Probability:** 82.45%

The probability is also displayed visually using a progress bar.

The exact meaning of prediction classes `0` and `1` depends on the encoding of `QUALITY_OF_LOAN` in the dataset.

---

## Project Workflow

**Dataset → Data Cleaning → Missing Value Handling → Skewness Analysis → Log Transformation → Train/Validation Split → One-Hot Encoding → Imputation → Model Training → Model Evaluation → Model Saving → Streamlit UI → Credit Risk Prediction**

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## Project Structure

```text
CodeAlpha_CreditScoringModel/
│
├── data/
│   ├── test-FIN_ANA_DATA .xls
│   ├── test.csv
│   ├── train-FIN_ANA_DATA .xls
│   └── train.csv
│
├── models/
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── models/
│   ├── credit_scoring_model.ipynb
│   └── credit_scoring_prediction.csv
│
├── requirements/
│   └── requirements

├── src/
│   ├── convert_xls_to_csv.py
│   ├── evaluate.py
│   ├── main.py
│   ├── preprocessing.py
│   └── train.py
│
├── app.py
└── README.md
```

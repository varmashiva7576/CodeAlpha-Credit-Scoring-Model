import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("credit_data.csv")

print("=" * 60)
print("CREDIT SCORING DATASET")
print("=" * 60)

print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 2. BASIC DATA CLEANING
# ============================================================

# Remove duplicate rows
df = df.drop_duplicates()

# Make sure target column exists
if "target" not in df.columns:
    raise ValueError(
        "The CSV file must contain a 'target' column."
    )


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Debt-to-Income Ratio
if "debt" in df.columns and "income" in df.columns:

    df["debt_to_income"] = (
        df["debt"] / df["income"].replace(0, np.nan)
    )

# Payment Risk
if "late_payments" in df.columns:

    # Simple risk indicator
    df["payment_risk"] = (
        df["late_payments"] > 2
    ).astype(int)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]


# ============================================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 6. PREPROCESSING
# ============================================================

# Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical preprocessing
# One-hot encoding is performed using pandas before
# model training for simplicity.


# Convert categorical columns to numerical columns
X = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=True
)


# Handle missing values
X = X.fillna(X.median(numeric_only=True))

# Any remaining missing values
X = X.fillna(0)


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. CREATE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000
            )
        )
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
}


# ============================================================
# 9. TRAIN AND EVALUATE MODELS
# ============================================================

results = {}

for name, model in models.items():

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Probability prediction
    y_probability = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }

    # Display results
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results).T

print("\n")
print("=" * 80)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 80)

print(results_df)


# ============================================================
# 11. VISUALIZE MODEL PERFORMANCE
# ============================================================

results_df.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Credit Scoring Model Performance"
)

plt.ylabel("Score")

plt.xlabel("Machine Learning Model")

plt.ylim(0, 1.1)

plt.xticks(rotation=0)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.show()


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_model_name = results_df[
    "ROC-AUC"
].idxmax()

best_model = models[
    best_model_name
]

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best Model:", best_model_name)

print(
    "ROC-AUC:",
    round(
        results_df.loc[
            best_model_name,
            "ROC-AUC"
        ],
        4
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

best_predictions = best_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Bad Credit",
        "Good Credit"
    ],
    yticklabels=[
        "Bad Credit",
        "Good Credit"
    ]
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.show()


# ============================================================
# 14. ROC CURVES
# ============================================================

plt.figure(
    figsize=(8, 6)
)

for name, model in models.items():

    probability = model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random Guess"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Credit Scoring Models"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 15. FEATURE IMPORTANCE - RANDOM FOREST
# ============================================================

random_forest = models[
    "Random Forest"
]

importance = random_forest.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 60)
print("TOP IMPORTANT FEATURES")
print("=" * 60)

print(
    feature_importance.head(10)
)


# Plot feature importance
plt.figure(
    figsize=(10, 6)
)

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 10 Features - Random Forest"
)

plt.tight_layout()

plt.show()


# ============================================================
# 16. PREDICT CREDITWORTHINESS OF A NEW APPLICANT
# ============================================================

# Example applicant
# The values should match the columns in your CSV.

new_applicant = pd.DataFrame({
    "age": [35],
    "income": [65000],
    "debt": [12000],
    "loan_amount": [15000],
    "credit_history": [8],
    "late_payments": [1],
    "credit_utilization": [0.30]
})


# Feature engineering for new applicant

if "debt" in new_applicant.columns and \
   "income" in new_applicant.columns:

    new_applicant["debt_to_income"] = (
        new_applicant["debt"] /
        new_applicant["income"]
    )

if "late_payments" in new_applicant.columns:

    new_applicant["payment_risk"] = (
        new_applicant["late_payments"] > 2
    ).astype(int)


# Make columns identical to training data
new_applicant = pd.get_dummies(
    new_applicant
)

new_applicant = new_applicant.reindex(
    columns=X.columns,
    fill_value=0
)


# Predict
prediction = best_model.predict(
    new_applicant
)[0]

probability = best_model.predict_proba(
    new_applicant
)[0][1]


# Display prediction
print("\n")
print("=" * 60)
print("CREDIT SCORE PREDICTION")
print("=" * 60)

if prediction == 1:

    print(
        "Prediction: GOOD CREDIT"
    )

else:

    print(
        "Prediction: BAD CREDIT"
    )

print(
    f"Probability of Good Credit: "
    f"{probability:.2%}"
)

print("\nPrediction completed.")
print(
    "This machine-learning result is for "
    "educational purposes and should not be "
    "used as the sole basis for real lending decisions."
)
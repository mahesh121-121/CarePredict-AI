
import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    PROJECT_DIR, "dataset", "diabetic_data.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR, "readmission_model.pkl"
)

EVALUATION_PATH = os.path.join(
    PROJECT_DIR, "model_evaluation.json"
)

# --------------------------------------------------
# 2. INPUT FEATURES
# --------------------------------------------------

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

CATEGORICAL_FEATURES = [
    "age",
    "A1Cresult",
    "diabetesMed",
    "max_glu_serum"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "readmitted"
PATIENT_ID = "patient_nbr"

# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

print("\n========================================")
print("       CAREPREDICT AI")
print(" Hospital Readmission Prediction")
print("========================================")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}\n"
        "Check that diabetic_data.csv is inside the dataset folder."
    )

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)

required_columns = FEATURES + [TARGET, PATIENT_ID]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns are missing from the dataset: "
        + ", ".join(missing_columns)
    )

# --------------------------------------------------
# 4. PREPARE TARGET
# --------------------------------------------------

# Predict readmission within 30 days.
# "<30" = positive class.
# ">30" and "NO" = negative class.

df = df[df[TARGET].isin(["<30", ">30", "NO"])].copy()

df["target_readmission"] = (
    df[TARGET] == "<30"
).astype(int)

# Remove rows with missing patient IDs.
df = df.dropna(subset=[PATIENT_ID])

# --------------------------------------------------
# 5. PREPARE INPUT FEATURES
# --------------------------------------------------

X = df[FEATURES].copy()
y = df["target_readmission"].copy()
groups = df[PATIENT_ID].copy()

# Replace special missing-value markers.
X = X.replace(
    ["?", "Unknown/Invalid", "unknown", "UNKNOWN", ""],
    np.nan
)

# Convert numerical columns into numeric values.
for column in NUMERIC_FEATURES:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

print("\nFeatures used:")
for feature in FEATURES:
    print("-", feature)

print("\nGlucose feature: max_glu_serum")
print("This is a categorical dataset feature, not an exact")
print("glucometer reading in mg/dL.")

# --------------------------------------------------
# 6. SPLIT DATA WITHOUT PATIENT OVERLAP
# --------------------------------------------------

# Keep records from the same patient in the same split.

outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_val_indices, test_indices = next(
    outer_split.split(X, y, groups=groups)
)

X_train_val = X.iloc[train_val_indices]
y_train_val = y.iloc[train_val_indices]
groups_train_val = groups.iloc[train_val_indices]

X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]
groups_test = groups.iloc[test_indices]

inner_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_indices, val_indices = next(
    inner_split.split(
        X_train_val,
        y_train_val,
        groups=groups_train_val
    )
)

X_train = X_train_val.iloc[train_indices]
y_train = y_train_val.iloc[train_indices]

X_val = X_train_val.iloc[val_indices]
y_val = y_train_val.iloc[val_indices]

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))

if y_train.nunique() < 2 or y_val.nunique() < 2:
    raise ValueError(
        "Training or validation split contains only one class. "
        "Check the dataset and split."
    )

# --------------------------------------------------
# 7. PREPROCESSING
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ]
)

# --------------------------------------------------
# 8. DEFINE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "HistGradientBoosting": Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=100,
                    learning_rate=0.1,
                    max_leaf_nodes=15,
                    random_state=42
                )
            )
        ]
    )
}

# --------------------------------------------------
# 9. TRAIN AND COMPARE MODELS
# --------------------------------------------------

print("\nTraining models...")

trained_models = {}
model_scores = {}

for model_name, classifier in models.items():

    if model_name == "HistGradientBoosting":
        # This model uses its own preprocessing pipeline.
        model = clone(classifier)
    else:
        model = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(classifier))
            ]
        )

    model.fit(X_train, y_train)

    validation_probabilities = model.predict_proba(X_val)[:, 1]

    average_precision = average_precision_score(
        y_val,
        validation_probabilities
    )

    trained_models[model_name] = model
    model_scores[model_name] = average_precision

    print(
        f"{model_name}: "
        f"Validation Average Precision = "
        f"{average_precision:.4f}"
    )

# --------------------------------------------------
# 10. SELECT BEST MODEL
# --------------------------------------------------

best_model_name = max(
    model_scores,
    key=model_scores.get
)

best_model = trained_models[best_model_name]

print("\nSelected model:", best_model_name)

# --------------------------------------------------
# 11. SELECT A THRESHOLD USING VALIDATION DATA
# --------------------------------------------------

validation_probabilities = best_model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.01)

best_threshold = 0.50
best_f1 = -1.0

for threshold in thresholds:

    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_val,
        validation_predictions,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)

print("Selected threshold:", round(best_threshold, 3))

# --------------------------------------------------
# 12. EVALUATE ON UNSEEN TEST DATA
# --------------------------------------------------

test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(y_test, test_predictions)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

average_precision_test = average_precision_score(
    y_test,
    test_probabilities
)

if y_test.nunique() == 2:
    roc_auc = roc_auc_score(y_test, test_probabilities)
else:
    roc_auc = None

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

print("\n========================================")
print("          TEST RESULTS")
print("========================================")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("Average Precision:", round(average_precision_test, 4))
print("ROC-AUC:", None if roc_auc is None else round(roc_auc, 4))
print("\nConfusion Matrix:")
print(matrix)

# --------------------------------------------------
# 13. SAVE MODEL AND METADATA
# --------------------------------------------------

model_bundle = {
    "model": best_model,
    "features": FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "threshold": best_threshold,
    "target_definition": (
        "1 = readmitted within 30 days (<30); "
        "0 = readmitted after 30 days (>30) or not readmitted (NO)"
    ),
    "glucose_feature_note": (
        "max_glu_serum is a categorical dataset feature. "
        "It is not an exact blood glucose measurement in mg/dL."
    ),
    "best_model_name": best_model_name
}

joblib.dump(model_bundle, MODEL_PATH)

evaluation = {
    "best_model": best_model_name,
    "threshold": best_threshold,
    "features": FEATURES,
    "test_metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "average_precision": float(average_precision_test),
        "roc_auc": (
            None if roc_auc is None else float(roc_auc)
        ),
        "confusion_matrix": matrix.tolist()
    },
    "glucose_feature": (
        "Categorical max_glu_serum; not exact mg/dL."
    ),
    "clinical_warning": (
        "Educational prototype only. Not clinically validated "
        "and not for real medical decisions."
    )
}

with open(EVALUATION_PATH, "w", encoding="utf-8") as file:
    json.dump(evaluation, file, indent=4)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nEvaluation saved to:")
print(EVALUATION_PATH)

print("\nTraining completed successfully.")


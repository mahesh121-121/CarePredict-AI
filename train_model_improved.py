import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

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
    fbeta_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CAREPREDICT AI - IMPROVED TRAINING SCRIPT
# Original project: CarePredict_AI
#
# Target: Readmission within 30 days
# Positive class: readmitted == "<30"
#
# This script saves a NEW model and evaluation report.
# It does NOT overwrite the original model.
# ============================================================


PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    PROJECT_DIR, "dataset", "diabetic_data.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR, "readmission_model_improved.pkl"
)

EVALUATION_PATH = os.path.join(
    PROJECT_DIR, "model_evaluation_improved.json"
)


NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

CATEGORICAL_FEATURES = [
    "age",
    "A1Cresult",
    "diabetesMed",
    "max_glu_serum",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "readmitted"
PATIENT_COLUMN = "patient_nbr"


def make_preprocessor():
    """Create preprocessing independently for each model."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def calculate_metrics(y_true, probabilities, threshold):
    """Calculate classification metrics at a selected threshold."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f2_score": float(
            fbeta_score(
                y_true,
                predictions,
                beta=2,
                zero_division=0,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }


def choose_threshold(y_true, probabilities):
    """
    Choose a threshold on validation data by maximizing F2.

    F2 gives recall more weight than precision.
    The held-out test set is NOT used for threshold selection.
    """

    best_threshold = 0.5
    best_f2 = -1.0
    best_recall = -1.0
    best_precision = -1.0

    # Evaluate a range of thresholds.
    for threshold in np.arange(0.02, 0.81, 0.01):

        metrics = calculate_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        current_f2 = metrics["f2_score"]
        current_recall = metrics["recall"]
        current_precision = metrics["precision"]

        # Prefer F2; break ties using recall and precision.
        candidate = (
            current_f2,
            current_recall,
            current_precision,
        )

        best = (
            best_f2,
            best_recall,
            best_precision,
        )

        if candidate > best:
            best_f2 = current_f2
            best_recall = current_recall
            best_precision = current_precision
            best_threshold = float(threshold)

    return best_threshold


def main():

    print("\n" + "=" * 65)
    print("       CAREPREDICT AI - MODEL IMPROVEMENT")
    print("=" * 65)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print("\n[1/7] Loading dataset...")

    df = pd.read_csv(DATASET_PATH)

    required_columns = (
        FEATURES
        + [TARGET_COLUMN, PATIENT_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required dataset columns: {missing_columns}"
        )

    # Retain the three expected readmission categories.
    df = df[
        df[TARGET_COLUMN].isin(["<30", ">30", "NO"])
    ].copy()

    # Patient identifiers are used only for splitting.
    df = df.dropna(subset=[PATIENT_COLUMN])

    # Positive class means readmitted within 30 days.
    y = (
        df[TARGET_COLUMN] == "<30"
    ).astype(int)

    X = df[FEATURES].copy()

    groups = df[PATIENT_COLUMN].copy()

    # Normalize missing-value representations.
    X = X.replace(
        [
            "?",
            "Unknown/Invalid",
            "unknown",
            "UNKNOWN",
            "",
        ],
        np.nan,
    )

    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    print("Dataset rows:", len(df))
    print("Unique patients:", groups.nunique())
    print("Positive cases:", int(y.sum()))
    print("Positive class percentage:", round(y.mean() * 100, 2))

    print("\n[2/7] Separating patients into train, validation and test...")

    # First, hold out test patients.
    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_val_idx, test_idx = next(
        outer_split.split(X, y, groups=groups)
    )

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    groups_test = groups.iloc[test_idx]

    # Split the remaining patients into training and validation.
    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=43,
    )

    train_idx, val_idx = next(
        inner_split.split(
            X_train_val,
            y_train_val,
            groups=groups_train_val,
        )
    )

    X_train = X_train_val.iloc[train_idx]
    y_train = y_train_val.iloc[train_idx]

    X_val = X_train_val.iloc[val_idx]
    y_val = y_train_val.iloc[val_idx]

    # Verify that no patient appears in multiple partitions.
    train_patients = set(
        groups.iloc[train_val_idx].iloc[train_idx]
    )

    val_patients = set(
        groups.iloc[train_val_idx].iloc[val_idx]
    )

    test_patients = set(groups_test)

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Test rows:", len(X_test))
    print("Patient overlap check: PASSED")

    print("\n[3/7] Preparing candidate models...")

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=15,
            random_state=42,
        ),
    }

    results = {}
    fitted_models = {}

    print("\n[4/7] Training and comparing models...")
    print("This step may take several minutes.\n")

    for model_name, classifier in models.items():

        print("-" * 55)
        print("Training:", model_name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("model", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        val_probabilities = pipeline.predict_proba(
            X_val
        )[:, 1]

        threshold = choose_threshold(
            y_val,
            val_probabilities,
        )

        val_metrics = calculate_metrics(
            y_val,
            val_probabilities,
            threshold,
        )

        fitted_models[model_name] = pipeline

        results[model_name] = {
            "validation_threshold": threshold,
            "validation_metrics": val_metrics,
            "validation_average_precision": float(
                average_precision_score(
                    y_val,
                    val_probabilities,
                )
            ),
        }

        print(
            "Validation recall:",
            round(val_metrics["recall"], 4),
        )

        print(
            "Validation precision:",
            round(val_metrics["precision"], 4),
        )

        print(
            "Validation F2:",
            round(val_metrics["f2_score"], 4),
        )

        print(
            "Validation average precision:",
            round(val_metrics["average_precision"], 4),
        )

        print(
            "Chosen validation threshold:",
            round(threshold, 3),
        )

    print("\n[5/7] Selecting the model using validation F2...")

    best_model_name = max(
        results,
        key=lambda name: (
            results[name]["validation_metrics"]["f2_score"],
            results[name]["validation_metrics"]["recall"],
            results[name]["validation_metrics"]["precision"],
        ),
    )

    best_model = fitted_models[best_model_name]

    best_threshold = results[
        best_model_name
    ]["validation_threshold"]

    print("Selected model:", best_model_name)
    print("Selected threshold:", round(best_threshold, 3))

    print("\n[6/7] Evaluating the selected model on held-out test data...")

    # The test set is used only after model and threshold selection.
    test_probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        best_threshold,
    )

    print("\nTEST RESULTS")
    print("-" * 55)

    for metric_name in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "f2_score",
        "average_precision",
        "roc_auc",
    ]:
        print(
            f"{metric_name}: "
            f"{test_metrics[metric_name]:.4f}"
        )

    print("\nConfusion matrix:")
    print(test_metrics["confusion_matrix"])

    print("\n[7/7] Saving the new model and evaluation report...")

    # Keep the original bundle structure compatible with the backend.
    model_bundle = {
        "model": best_model,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "threshold": best_threshold,
        "target_definition": (
            "Positive class means readmitted within 30 days "
            "(readmitted == '<30')."
        ),
        "glucose_feature_note": (
            "Categorical max_glu_serum; not exact mg/dL."
        ),
        "best_model_name": best_model_name,
    }

    joblib.dump(model_bundle, MODEL_PATH)

    evaluation_report = {
        "project": "CarePredict AI",
        "model": best_model_name,
        "threshold": best_threshold,
        "target_definition": (
            "Readmission within 30 days (readmitted == '<30')"
        ),
        "features": FEATURES,
        "split_method": "Patient-separated GroupShuffleSplit",
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "patient_overlap_check": "PASSED",
        "validation_model_comparison": results,
        "test_metrics": test_metrics,
        "glucose_feature": (
            "Categorical max_glu_serum; not exact mg/dL."
        ),
        "clinical_warning": (
            "Educational prototype only. Not clinically validated "
            "and not for real medical decisions."
        ),
    }

    with open(
        EVALUATION_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation_report,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)

    print("New model:", MODEL_PATH)
    print("New evaluation:", EVALUATION_PATH)

    print("\nOriginal model was NOT overwritten.")
    print("Original evaluation was NOT overwritten.")

    print("\nImportant:")
    print(
        "This model is an educational prototype, "
        "not a clinically validated medical tool."
    )


if __name__ == "__main__":
    main()

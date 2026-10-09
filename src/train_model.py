"""
train_model.py
Fraud Detection Model Training

Dataset split:
    70% Train
    10% Validation
    20% Test

Important:
- train_raw.csv, val_raw.csv, and test_raw.csv are already chronologically split.
- Validation is used ONLY for threshold selection.
- Test is used ONLY for final evaluation.
- No scaling is performed here because preprocessing outputs raw numerical features.
- The saved model is compatible with utils.py through best_model_fixed.pkl.
"""

import pandas as pd
import numpy as np
import pickle
import os
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    fbeta_score
)
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "../data/processed/train_raw.csv"
VAL_PATH = "../data/processed/val_raw.csv"
TEST_PATH = "../data/processed/test_raw.csv"

MODEL_DIR = "../models"
REPORT_DIR = "../reports"

RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


print("=" * 75)
print("FRAUD DETECTION MODEL TRAINING")
print("70/10/20 Chronological Train/Validation/Test Split")
print("=" * 75)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading processed data...")

train_raw = pd.read_csv(TRAIN_PATH)
val_raw = pd.read_csv(VAL_PATH)
test_raw = pd.read_csv(TEST_PATH)

print(f"  Train      : {len(train_raw):,} transactions")
print(f"  Validation : {len(val_raw):,} transactions")
print(f"  Test       : {len(test_raw):,} transactions")

print(
    f"  Train fraud rate      : "
    f"{train_raw['fraud_label'].mean() * 100:.2f}%"
)

print(
    f"  Validation fraud rate : "
    f"{val_raw['fraud_label'].mean() * 100:.2f}%"
)

print(
    f"  Test fraud rate       : "
    f"{test_raw['fraud_label'].mean() * 100:.2f}%"
)


# ============================================================
# 2. VALIDATE DATA SCHEMA
# ============================================================

print("\n[2] Validating dataset schema...")

if "fraud_label" not in train_raw.columns:
    raise ValueError("fraud_label missing from training data.")

if "fraud_label" not in val_raw.columns:
    raise ValueError("fraud_label missing from validation data.")

if "fraud_label" not in test_raw.columns:
    raise ValueError("fraud_label missing from test data.")


# ------------------------------------------------------------
# Separate target
# ------------------------------------------------------------

X_train = train_raw.drop("fraud_label", axis=1)
y_train = train_raw["fraud_label"].astype(int)

X_val = val_raw.drop("fraud_label", axis=1)
y_val = val_raw["fraud_label"].astype(int)

X_test = test_raw.drop("fraud_label", axis=1)
y_test = test_raw["fraud_label"].astype(int)


# ------------------------------------------------------------
# Check feature columns
# ------------------------------------------------------------

train_features = list(X_train.columns)
val_features = list(X_val.columns)
test_features = list(X_test.columns)

if train_features != val_features:
    missing_val = set(train_features) - set(val_features)
    extra_val = set(val_features) - set(train_features)

    raise ValueError(
        "Train/Validation feature mismatch.\n"
        f"Missing in validation: {missing_val}\n"
        f"Extra in validation: {extra_val}"
    )

if train_features != test_features:
    missing_test = set(train_features) - set(test_features)
    extra_test = set(test_features) - set(train_features)

    raise ValueError(
        "Train/Test feature mismatch.\n"
        f"Missing in test: {missing_test}\n"
        f"Extra in test: {extra_test}"
    )


print(f"  Features: {len(train_features)}")
print("  Feature schema: OK")


# ------------------------------------------------------------
# Check missing values
# ------------------------------------------------------------

train_missing = int(X_train.isna().sum().sum())
val_missing = int(X_val.isna().sum().sum())
test_missing = int(X_test.isna().sum().sum())

print(f"  Train missing values: {train_missing:,}")
print(f"  Val missing values  : {val_missing:,}")
print(f"  Test missing values : {test_missing:,}")

if train_missing > 0 or val_missing > 0 or test_missing > 0:
    print("  WARNING: Missing values detected.")
else:
    print("  Missing-value check: OK")


# ============================================================
# 3. CLASS IMBALANCE
# ============================================================

print("\n[3] Calculating class imbalance...")

positive_count = int(y_train.sum())
negative_count = int(len(y_train) - positive_count)

if positive_count == 0:
    raise ValueError("Training set contains no fraud samples.")

imbalance_ratio = negative_count / positive_count

print(f"  Normal transactions: {negative_count:,}")
print(f"  Fraud transactions : {positive_count:,}")
print(f"  Imbalance ratio    : {imbalance_ratio:.2f}")


# ============================================================
# 4. TRAIN LIGHTGBM
# ============================================================

print("\n[4] Training LightGBM...")

lgb = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.02,
    max_depth=6,
    num_leaves=25,
    scale_pos_weight=imbalance_ratio,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=2.0,
    min_child_weight=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)

lgb.fit(X_train, y_train)

print("  LightGBM training complete.")


# ============================================================
# 5. TRAIN XGBOOST
# ============================================================

print("\n[5] Training XGBoost...")

xgb = XGBClassifier(
    scale_pos_weight=imbalance_ratio,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=10,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    n_jobs=-1
)

xgb.fit(X_train, y_train)

print("  XGBoost training complete.")


# ============================================================
# 6. CREATE ENSEMBLE
# ============================================================

print("\n[6] Creating soft-voting ensemble...")

ensemble = VotingClassifier(
    estimators=[
        ("lgb", lgb),
        ("xgb", xgb)
    ],
    voting="soft",
    weights=[1, 1]
)

ensemble.fit(X_train, y_train)

print("  Ensemble training complete.")


# ============================================================
# 7. FIND OPTIMAL THRESHOLD USING VALIDATION
# ============================================================

print("\n[7] Finding optimal thresholds using validation F2-score...")


def find_optimal_threshold_f2(model, X_val, y_val):
    """
    Select threshold using validation data only.

    F2 gives more importance to recall than precision,
    which is appropriate for fraud detection where
    missing fraud can be costly.
    F2 = 5 × (precision × recall) / (4 × precision + recall)
    """

    proba = model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.10, 0.81, 0.01)

    best_score = -1.0
    best_th = 0.50
    best_prec = 0.0
    best_rec = 0.0

    for threshold in thresholds:

        pred = (proba >= threshold).astype(int)

        f2 = fbeta_score(
            y_val,
            pred,
            beta=2,
            zero_division=0
        )

        if f2 > best_score:

            best_score = f2
            best_th = float(threshold)

            best_prec = precision_score(
                y_val,
                pred,
                zero_division=0
            )

            best_rec = recall_score(
                y_val,
                pred,
                zero_division=0
            )

    return (
        best_th,
        best_score,
        best_prec,
        best_rec
    )


lgb_th, lgb_f2, lgb_prec, lgb_rec = (
    find_optimal_threshold_f2(
        lgb,
        X_val,
        y_val
    )
)

xgb_th, xgb_f2, xgb_prec, xgb_rec = (
    find_optimal_threshold_f2(
        xgb,
        X_val,
        y_val
    )
)

ens_th, ens_f2, ens_prec, ens_rec = (
    find_optimal_threshold_f2(
        ensemble,
        X_val,
        y_val
    )
)


print("\n  VALIDATION THRESHOLD RESULTS")
print("  " + "-" * 70)

print(
    f"  LightGBM : "
    f"Threshold={lgb_th:.2f} | "
    f"F2={lgb_f2:.4f} | "
    f"Recall={lgb_rec:.4f} | "
    f"Precision={lgb_prec:.4f}"
)

print(
    f"  XGBoost  : "
    f"Threshold={xgb_th:.2f} | "
    f"F2={xgb_f2:.4f} | "
    f"Recall={xgb_rec:.4f} | "
    f"Precision={xgb_prec:.4f}"
)

print(
    f"  Ensemble : "
    f"Threshold={ens_th:.2f} | "
    f"F2={ens_f2:.4f} | "
    f"Recall={ens_rec:.4f} | "
    f"Precision={ens_prec:.4f}"
)


# ============================================================
# 8. TEST EVALUATION
# ============================================================

print("\n[8] Evaluating models on the locked test set...")


def evaluate_model(model, X_test, y_test, threshold, name):

    proba = model.predict_proba(X_test)[:, 1]

    pred = (proba >= threshold).astype(int)

    return {
        "name": name,
        "threshold": float(threshold),

        "f1": f1_score(
            y_test,
            pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            pred,
            zero_division=0
        ),

        "precision": precision_score(
            y_test,
            pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            proba
        ),

        "proba": proba,
        "pred": pred,
        "y_true": y_test.to_numpy()
    }


results = {

    "LightGBM": evaluate_model(
        lgb,
        X_test,
        y_test,
        lgb_th,
        "LightGBM"
    ),

    "XGBoost": evaluate_model(
        xgb,
        X_test,
        y_test,
        xgb_th,
        "XGBoost"
    ),

    "Ensemble": evaluate_model(
        ensemble,
        X_test,
        y_test,
        ens_th,
        "Ensemble (LGB+XGB)"
    )
}


# ============================================================
# 9. DISPLAY TEST PERFORMANCE
# ============================================================

print("\n  TEST SET PERFORMANCE")
print("  " + "=" * 80)

print(
    f"  {'Model':<20}"
    f"{'Threshold':<12}"
    f"{'F1':<10}"
    f"{'Recall':<10}"
    f"{'Precision':<12}"
    f"{'ROC-AUC':<10}"
)

print("  " + "-" * 80)

for name, result in results.items():

    print(
        f"  {name:<20}"
        f"{result['threshold']:<12.2f}"
        f"{result['f1']:<10.4f}"
        f"{result['recall']:<10.4f}"
        f"{result['precision']:<12.4f}"
        f"{result['roc_auc']:<10.4f}"
    )

print("  " + "=" * 80)


# ============================================================
# 10. SELECT BEST MODEL
# ============================================================

best_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

best_results = results[best_name]

if best_name == "LightGBM":
    best_model = lgb
elif best_name == "XGBoost":
    best_model = xgb
else:
    best_model = ensemble


print(
    f"\n  BEST MODEL: {best_name}"
)

print(
    f"  Test F1        : {best_results['f1']:.4f}"
)

print(
    f"  Test Recall    : {best_results['recall']:.4f}"
)

print(
    f"  Test Precision : {best_results['precision']:.4f}"
)

print(
    f"  Test ROC-AUC   : {best_results['roc_auc']:.4f}"
)

print(
    f"  Threshold      : {best_results['threshold']:.2f}"
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\n[9] Best-model confusion matrix...")

cm = confusion_matrix(
    y_test,
    best_results["pred"]
)

print("\n  Confusion Matrix:")
print("  " + "-" * 35)
print(f"  TN: {cm[0, 0]:,}    FP: {cm[0, 1]:,}")
print(f"  FN: {cm[1, 0]:,}    TP: {cm[1, 1]:,}")
print("  " + "-" * 35)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n  Classification Report:\n")

print(
    classification_report(
        y_test,
        best_results["pred"],
        target_names=["Normal", "Fraud"],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 13. SAVE INDIVIDUAL MODELS
# ============================================================

print("\n[10] Saving models...")

model_paths = {
    "lightgbm.pkl": lgb,
    "xgboost.pkl": xgb,
    "ensemble.pkl": ensemble
}

for filename, model_obj in model_paths.items():

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    with open(path, "wb") as f:
        pickle.dump(model_obj, f)

    print(f"  Saved: {path}")


# ============================================================
# 14. SAVE BEST MODEL PACKAGE
# ============================================================

print("\n[11] Saving best model package...")

best_model_data = {

    # Model
    "model": best_model,

    # Identification
    "name": best_name,
    "version": "final_70_10_20",

    # Decision threshold
    "threshold": float(
        best_results["threshold"]
    ),

    # Exact feature schema
    "features": train_features,

    # Dataset information
    "train_samples": int(len(train_raw)),
    "val_samples": int(len(val_raw)),
    "test_samples": int(len(test_raw)),

    "train_fraud_rate": float(
        y_train.mean()
    ),

    "val_fraud_rate": float(
        y_val.mean()
    ),

    "test_fraud_rate": float(
        y_test.mean()
    ),

    # Validation metrics
    "validation_f2": float(
        lgb_f2 if best_name == "LightGBM"
        else xgb_f2 if best_name == "XGBoost"
        else ens_f2
    ),

    # Test metrics
    "test_f1": float(
        best_results["f1"]
    ),

    "test_recall": float(
        best_results["recall"]
    ),

    "test_precision": float(
        best_results["precision"]
    ),

    "test_roc_auc": float(
        best_results["roc_auc"]
    ),

    # Training configuration
    "random_state": RANDOM_STATE,

    "imbalance_ratio": float(
        imbalance_ratio
    ),

    "feature_count": int(
        len(train_features)
    )
}


best_model_path = os.path.join(
    MODEL_DIR,
    "best_model_fixed.pkl"
)

with open(best_model_path, "wb") as f:
    pickle.dump(
        best_model_data,
        f
    )

print(
    f"  Saved: {best_model_path}"
)


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

print("\n[12] Calculating feature importance...")


if best_name == "LightGBM":

    importance = np.asarray(
        lgb.feature_importances_
    )

elif best_name == "XGBoost":

    importance = np.asarray(
        xgb.feature_importances_
    )

else:

    lgb_importance = np.asarray(
        lgb.feature_importances_
    )

    xgb_importance = np.asarray(
        xgb.feature_importances_
    )

    importance = (
        lgb_importance +
        xgb_importance
    ) / 2.0


feature_importance = pd.DataFrame({

    "feature": train_features,

    "importance": importance

}).sort_values(
    "importance",
    ascending=False
)


importance_path = os.path.join(
    REPORT_DIR,
    "feature_importance_final.csv"
)

feature_importance.to_csv(
    importance_path,
    index=False
)


print(
    f"\n  Saved: {importance_path}"
)


print("\n  TOP 15 FEATURES")
print("  " + "=" * 60)

for rank, (_, row) in enumerate(
    feature_importance.head(15).iterrows(),
    start=1
):

    print(
        f"  {rank:2d}. "
        f"{row['feature']:<40} "
        f"{row['importance']:.4f}"
    )


# ============================================================
# 16. SAVE TEST PREDICTIONS
# ============================================================

print("\n[13] Saving test predictions...")

test_predictions = test_raw.copy()

test_predictions["fraud_probability"] = (
    best_results["proba"]
)

test_predictions["predicted_fraud"] = (
    best_results["pred"]
)

predictions_path = os.path.join(
    REPORT_DIR,
    "test_predictions_final.csv"
)

test_predictions.to_csv(
    predictions_path,
    index=False
)

print(
    f"  Saved: {predictions_path}"
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("TRAINING COMPLETE")
print("=" * 75)

print(f"Best model       : {best_name}")
print(f"Threshold        : {best_results['threshold']:.2f}")
print(f"Test F1          : {best_results['f1']:.4f}")
print(f"Test Recall      : {best_results['recall']:.4f}")
print(f"Test Precision   : {best_results['precision']:.4f}")
print(f"Test ROC-AUC     : {best_results['roc_auc']:.4f}")
print(f"Features         : {len(train_features)}")
print(f"Model file       : {best_model_path}")
print("=" * 75)
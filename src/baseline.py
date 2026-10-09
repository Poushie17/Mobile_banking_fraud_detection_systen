

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, fbeta_score,
)

RANDOM_STATE = 42


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"
MODELS       = PROJECT_ROOT / "models"
REPORTS      = PROJECT_ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROCESSED / "train_raw.csv"
VAL_PATH   = PROCESSED / "val_raw.csv"
TEST_PATH  = PROCESSED / "test_raw.csv"

print("=" * 78)
print("FULL COMPARISON  (baselines + trained models)")
print("=" * 78)

# ------------------------------------------------------------------
# 1. Load data (read-only)
# ------------------------------------------------------------------

train = pd.read_csv(TRAIN_PATH)
val   = pd.read_csv(VAL_PATH)
test  = pd.read_csv(TEST_PATH)

X_tr, y_tr = train.drop(columns="fraud_label"), train["fraud_label"].astype(int)
X_va, y_va = val.drop(columns="fraud_label"),   val["fraud_label"].astype(int)
X_te, y_te = test.drop(columns="fraud_label"),  test["fraud_label"].astype(int)

print(f"Train: {len(X_tr):,}  Val: {len(X_va):,}  Test: {len(X_te):,}")
print(f"Features: {X_tr.shape[1]}")

# ------------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------------

def pick_threshold_f2(model, X_val, y_val):
    """Pick threshold maximizing F2 on validation. Returns 0.5 if no proba."""
    if not hasattr(model, "predict_proba"):
        return 0.5
    proba = model.predict_proba(X_val)[:, 1]
    best_th, best_f2 = 0.5, -1.0
    for th in np.arange(0.10, 0.81, 0.01):
        pred = (proba >= th).astype(int)
        f2 = fbeta_score(y_val, pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_th = f2, th
    return float(best_th)


def score(name, model, X_va, y_va, X_te, y_te, is_unsupervised=False):
    """Evaluate a model on the test set. Returns a dict row."""
    if is_unsupervised:
        raw = model.predict(X_te)                    
        pred = (raw == -1).astype(int)
        score_raw = model.decision_function(X_te)
        proba = 1.0 / (1.0 + np.exp(score_raw))       # higher = more anomalous
        threshold = 0.5
    else:
        threshold = pick_threshold_f2(model, X_va, y_va)
        proba = model.predict_proba(X_te)[:, 1]
        pred  = (proba >= threshold).astype(int)

    return {
        "model":     name,
        "threshold": round(threshold, 2),
        "f1":        round(f1_score(y_te, pred, zero_division=0), 4),
        "recall":    round(recall_score(y_te, pred, zero_division=0), 4),
        "precision": round(precision_score(y_te, pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_te, proba), 4),
    }


def score_with_fixed_threshold(name, model, X_te, y_te, threshold):
    """Evaluate with an explicit (already-tuned) threshold."""
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "model":     name,
        "threshold": round(float(threshold), 2),
        "f1":        round(f1_score(y_te, pred, zero_division=0), 4),
        "recall":    round(recall_score(y_te, pred, zero_division=0), 4),
        "precision": round(precision_score(y_te, pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_te, proba), 4),
    }


results = []

# ------------------------------------------------------------------
# 3. Trivial baselines
# ------------------------------------------------------------------

maj = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
results.append(score("Majority class", maj, X_va, y_va, X_te, y_te))

rnd = DummyClassifier(strategy="stratified",
                      random_state=RANDOM_STATE).fit(X_tr, y_tr)
results.append(score("Random (stratified)", rnd, X_va, y_va, X_te, y_te))

# ------------------------------------------------------------------
# 4. Rule-based baseline (no training)
# ------------------------------------------------------------------

rule_pred = (
    (X_te["amount_to_avg_ratio"] > 3) |
    (X_te["new_device_flag"] == 1)
).astype(int)

results.append({
    "model":     "Rule-based (amount/device)",
    "threshold": "n/a",
    "f1":        round(f1_score(y_te, rule_pred, zero_division=0), 4),
    "recall":    round(recall_score(y_te, rule_pred, zero_division=0), 4),
    "precision": round(precision_score(y_te, rule_pred, zero_division=0), 4),
    "roc_auc":   round(roc_auc_score(y_te, rule_pred), 4),
})

# ------------------------------------------------------------------
# 5. Logistic Regression (scaled pipeline would be better, but
#    we keep the plain version here to match the original baseline)
# ------------------------------------------------------------------

lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
).fit(X_tr, y_tr)
results.append(score("Logistic Regression", lr, X_va, y_va, X_te, y_te))

# ------------------------------------------------------------------
# 6. Decision Tree
# ------------------------------------------------------------------

dt = DecisionTreeClassifier(
    max_depth=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
).fit(X_tr, y_tr)
results.append(score("Decision Tree (depth=5)", dt, X_va, y_va, X_te, y_te))

# ------------------------------------------------------------------
# 7. Random Forest
# ------------------------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
).fit(X_tr, y_tr)
results.append(score("Random Forest", rf, X_va, y_va, X_te, y_te))

# ------------------------------------------------------------------
# 8. Isolation Forest (unsupervised)
# ------------------------------------------------------------------

iso = IsolationForest(
    n_estimators=200,
    contamination=float(y_tr.mean()),
    random_state=RANDOM_STATE,
    n_jobs=-1,
).fit(X_tr)
results.append(score("Isolation Forest (unsup.)",
                     iso, X_va, y_va, X_te, y_te,
                     is_unsupervised=True))

# ------------------------------------------------------------------
# 9. Load pre-trained models from disk (READ-ONLY)
#    - lightgbm.pkl / xgboost.pkl / ensemble.pkl are raw model objects
#    - best_model_fixed.pkl is a dict with metadata + the deployed model
#      and carries the authoritative threshold for the deployed model.
# ------------------------------------------------------------------

print("\n[Loading pre-trained models from models/ ...]")

def _load(path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

lgb_model   = _load(MODELS / "lightgbm.pkl")
xgb_model   = _load(MODELS / "xgboost.pkl")
ens_model   = _load(MODELS / "ensemble.pkl")
best_pkg    = _load(MODELS / "best_model_fixed.pkl")

# Deployed threshold (from best_model_fixed.pkl) — used for XGBoost
deployed_threshold = 0.55
deployed_name = "XGBoost"
if isinstance(best_pkg, dict):
    deployed_threshold = float(best_pkg.get("threshold", 0.55))
    deployed_name = best_pkg.get("name", "XGBoost")

print(f"  Deployed model name      : {deployed_name}")
print(f"  Deployed threshold       : {deployed_threshold:.2f}")

# LightGBM — use F2-optimal threshold on validation
if lgb_model is not None:
    results.append(score("LightGBM", lgb_model, X_va, y_va, X_te, y_te))
    print("  LightGBM loaded")
else:
    print("  ⚠️  lightgbm.pkl not found — skipping")

# XGBoost — use the SAME threshold stored in best_model_fixed.pkl
if xgb_model is not None:
    results.append(score_with_fixed_threshold(
        "XGBoost (deployed)", xgb_model, X_te, y_te, deployed_threshold))
    print("  XGBoost loaded")
else:
    print("  ⚠️  xgboost.pkl not found — skipping")

# Ensemble — use F2-optimal threshold on validation
if ens_model is not None:
    results.append(score("Ensemble (LGB+XGB)",
                         ens_model, X_va, y_va, X_te, y_te))
    print("  Ensemble loaded")
else:
    print("  ⚠️  ensemble.pkl not found — skipping")

# ------------------------------------------------------------------
# 10. Print unified table
# ------------------------------------------------------------------

df = pd.DataFrame(results)

print("\n" + "=" * 78)
print("FULL COMPARISON TABLE  (test set)")
print("=" * 78)
print(f"{'Model':<32}{'Thr':>6}{'F1':>9}{'Recall':>10}{'Precision':>12}{'ROC-AUC':>10}")
print("-" * 78)
for _, r in df.iterrows():
    auc = "   n/a" if pd.isna(r["roc_auc"]) else f"{r['roc_auc']:>9.4f}"
    thr = "   n/a" if isinstance(r["threshold"], str) else f"{r['threshold']:>6.2f}"
    print(f"{r['model']:<32}{thr}"
          f"{r['f1']:>9.4f}{r['recall']:>10.4f}"
          f"{r['precision']:>12.4f}{auc}")
print("=" * 78)

print("\nNote: LightGBM / XGBoost / Ensemble were loaded from saved .pkl files.")
print("      models/ and best_model_fixed.pkl were NOT touched.")

# ------------------------------------------------------------------
# 11. Save CSV for the report
# ------------------------------------------------------------------

out_csv = REPORTS / "baseline_comparison.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv.resolve()}")
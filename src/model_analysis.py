
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    confusion_matrix,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_theme(style="darkgrid")
sns.set_palette("husl")

print("="*70)
print("MODEL ERROR ANALYSIS")
print("Fraud Detection - Bangladesh Mobile Banking")
print("="*70)



print("\n[1] Loading data and model...")

# Load original data (direct path)
original_df = pd.read_csv('E:\\Fraud_Project2\\data\\transactions.csv')
original_df['timestamp'] = pd.to_datetime(original_df['timestamp'])

# Sort by timestamp to get time-based split
sorted_df = original_df.sort_values('timestamp')
split_idx = int(len(sorted_df) * 0.8)
test_df = sorted_df.iloc[split_idx:].reset_index(drop=True)

# Load test data
test_raw = pd.read_csv('E:\\Fraud_Project2\\data\\processed\\test_raw.csv')
X_test = test_raw.drop(columns=['fraud_label'])
y_test = test_raw['fraud_label']

# Get fraud_type for test data
fraud_type_series = test_df['fraud_type']

print(f"  Test size: {len(X_test):,}")
print(f"  Test fraud rate: {y_test.mean()*100:.2f}%")

# Load model
try:
    with open('../models/random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    print("  Model loaded: Random Forest")
except FileNotFoundError:
    try:
        with open('../models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("  Model loaded: Best Model")
    except FileNotFoundError:
        print("  ERROR: No model found. Run train_baseline.py first.")
        exit(1)



print("\n[2] Making predictions...")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"  Predicted fraud: {y_pred.sum():,}")
print(f"  Actual fraud: {y_test.sum():,}")



print("\n[3] Confusion Matrix Details...")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"  True Negatives: {tn:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Positives: {tp:,}")

print(f"\n  False Positive Rate: {fp/(fp+tn)*100:.2f}%")
print(f"  False Negative Rate: {fn/(fn+tp)*100:.2f}%")


# 4. EVALUATE BY FRAUD TYPE


print("\n[4] Evaluating by Fraud Type...")

fraud_mask = y_test == 1
fraud_types = fraud_type_series[fraud_mask].reset_index(drop=True)

if len(fraud_types) > 0:
    fraud_pred = y_pred[fraud_mask]
    fraud_proba = y_proba[fraud_mask]
    fraud_type_results = []

    for fraud_type in fraud_types.unique():
        type_mask = fraud_types == fraud_type
        type_total = type_mask.sum()
        type_correct = (fraud_pred[type_mask] == 1).sum()
        type_recall = type_correct / type_total if type_total > 0 else 0
        avg_prob = fraud_proba[type_mask].mean()

        fraud_type_results.append({
            'fraud_type': fraud_type,
            'total': type_total,
            'detected': type_correct,
            'recall': type_recall,
            'avg_probability': avg_prob
        })

    fraud_type_df = pd.DataFrame(fraud_type_results).sort_values('recall')

    print("\n  Fraud Type Performance:")
    print("  " + "="*60)
    print(f"  {'Fraud Type':<25} {'Total':<8} {'Detected':<10} {'Recall':<10} {'Avg Prob':<10}")
    print("  " + "-"*60)

    for _, row in fraud_type_df.iterrows():
        print(f"  {row['fraud_type']:<25} {row['total']:<8} {row['detected']:<10} {row['recall']*100:<10.1f}% {row['avg_probability']:<10.3f}")


# 5. FEATURE DISTRIBUTION ANALYSIS


print("\n[5] Feature Distribution Analysis...")

key_features = ['amount', 'merchant_risk', 'anomaly_score', 
                'velocity_5min', 'compromise_score', 'device_change_flag']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(key_features):
    if feature in X_test.columns:
        normal_vals = X_test.loc[y_test == 0, feature]
        fraud_vals = X_test.loc[y_test == 1, feature]

        data = [normal_vals, fraud_vals]
        bp = axes[i].boxplot(data, patch_artist=True)
        axes[i].set_xticklabels(['Normal', 'Fraud'])
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        axes[i].set_title(f'{feature}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Value')

plt.tight_layout()
plt.savefig('../reports/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: reports/feature_distributions.png")


# 6. PROBABILITY DISTRIBUTION


print("\n[6] Probability Distribution Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

normal_proba = y_proba[y_test == 0]
fraud_proba = y_proba[y_test == 1]

axes[0].hist(normal_proba, bins=50, alpha=0.7, color='#2ecc71', label='Normal', density=True)
axes[0].hist(fraud_proba, bins=50, alpha=0.7, color='#e74c3c', label='Fraud', density=True)
axes[0].axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
axes[0].set_xlabel('Fraud Probability')
axes[0].set_ylabel('Density')
axes[0].set_title('Probability Distribution by Class', fontsize=14, fontweight='bold')
axes[0].legend()

# Boxplot without 'labels' parameter
data = [normal_proba, fraud_proba]
bp = axes[1].boxplot(data, patch_artist=True)
axes[1].set_xticklabels(['Normal', 'Fraud'])
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
axes[1].set_title('Fraud Probability Box Plot', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Probability')
axes[1].axhline(y=0.5, color='red', linestyle='--', label='Threshold 0.5')

plt.tight_layout()
plt.savefig('../reports/probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: reports/probability_distribution.png")


# 7. OPTIMAL THRESHOLD


print("\n[7] Finding Optimal Threshold...")

thresholds = np.arange(0.1, 0.9, 0.05)
threshold_results = []

for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_thresh).ravel()
    
    precision = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    recall = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    threshold_results.append({
        'threshold': thresh,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

threshold_df = pd.DataFrame(threshold_results)

best_f1_row = threshold_df.loc[threshold_df['f1'].idxmax()]
best_threshold = best_f1_row['threshold']
best_f1 = best_f1_row['f1']

print(f"\n  Best Threshold: {best_threshold:.2f}")
print(f"  Best F1 Score: {best_f1:.4f}")
print(f"  Precision at threshold: {best_f1_row['precision']:.4f}")
print(f"  Recall at threshold: {best_f1_row['recall']:.4f}")

# Threshold plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(threshold_df['threshold'], threshold_df['precision'], 
        label='Precision', linewidth=2)
ax.plot(threshold_df['threshold'], threshold_df['recall'], 
        label='Recall', linewidth=2)
ax.plot(threshold_df['threshold'], threshold_df['f1'], 
        label='F1 Score', linewidth=2)
ax.axvline(x=best_threshold, color='red', linestyle='--', 
           label=f'Best F1 ({best_threshold:.2f})')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Threshold vs Performance Metrics', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('../reports/threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: reports/threshold_analysis.png")


# 8. SUMMARY


print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

print(f"\nSummary of Findings:")
print(f"  - Total Test Samples: {len(y_test):,}")
print(f"  - Fraud Rate: {y_test.mean()*100:.2f}%")
print(f"  - Default F1 (Threshold 0.5): {f1_score(y_test, y_pred):.4f}")
print(f"  - Best F1 (Threshold {best_threshold:.2f}): {best_f1:.4f}")

if 'fraud_type_df' in locals() and len(fraud_type_df) > 0:
    print(f"\n  Hardest Fraud Types:")
    for _, row in fraud_type_df.head(3).iterrows():
        print(f"    - {row['fraud_type']}: {row['recall']*100:.1f}% recall")

print("\n" + "="*70)
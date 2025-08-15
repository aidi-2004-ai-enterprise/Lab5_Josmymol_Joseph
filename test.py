# Lab 4 – Binary Classification (Improved)
# Josmymol Joseph

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
import xgboost as xgb

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data.csv")  # Change filename if needed
target_col = "Bankrupt?"      # Updated target column
print("Dataset Shape:", df.shape)
print(df[target_col].value_counts())

# -----------------------------
# 2. EDA
# -----------------------------
plt.figure(figsize=(5,4))
sns.countplot(x=target_col, data=df)
plt.title("Class Distribution")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr().iloc[:20, :20], cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (partial features)")
plt.show()

# -----------------------------
# 3. Features and Target
# -----------------------------
X = df.drop(target_col, axis=1)
y = df[target_col]

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5. Feature Scaling for LR
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 6. Logistic Regression
# -----------------------------
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:,1]

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
print("PR-AUC:", average_precision_score(y_test, y_prob_lr))

# -----------------------------
# 7. Random Forest
# -----------------------------
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
print("PR-AUC:", average_precision_score(y_test, y_prob_rf))

# -----------------------------
# 8. XGBoost
# -----------------------------
xgb_model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:,1]

print("\n--- XGBoost ---")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))
print("PR-AUC:", average_precision_score(y_test, y_prob_xgb))

# -----------------------------
# 9. Confusion Matrices
# -----------------------------
models_preds = {"Logistic Regression": y_pred_lr,
                "Random Forest": y_pred_rf,
                "XGBoost": y_pred_xgb}

fig, axes = plt.subplots(1,3, figsize=(15,4))
for ax, (name, preds) in zip(axes, models_preds.items()):
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"{name} CM")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------
# 10. ROC & PR Curves
# -----------------------------
plt.figure(figsize=(8,6))
for name, prob in zip(["Logistic Regression","Random Forest","XGBoost"],
                      [y_prob_lr,y_prob_rf,y_prob_xgb]):
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
for name, prob in zip(["Logistic Regression","Random Forest","XGBoost"],
                      [y_prob_lr,y_prob_rf,y_prob_xgb]):
    precision, recall, _ = precision_recall_curve(y_test, prob)
    plt.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_test, prob):.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.show()

# -----------------------------
# 11. Cross-Validation Scores
# -----------------------------
models = [log_reg, rf, xgb_model]
names = ["Logistic Regression", "Random Forest", "XGBoost"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- 5-Fold CV Scores ---")
for name, model in zip(names, models):
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"{name} ROC-AUC CV: {scores.mean():.4f} ± {scores.std():.4f}")

"""
Tasks 4 & 6: Threshold Optimization & Power Averaging
Evaluates direct accuracy gain from probability calibration and power averaging on OOF.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from pathlib import Path
COMP_DIR = str(Path(__file__).resolve().parent / "comp_output")

train_hand = np.nan_to_num(np.load(f"{COMP_DIR}/train_features.npz")["X"])
train_res = normalize(np.load(f"{COMP_DIR}/train_resnet18.npz")["feats"])
train_eff = normalize(np.load(f"{COMP_DIR}/train_efficientnet_b0.npz")["feats"])
train_res34 = normalize(np.load(f"{COMP_DIR}/train_resnet34.npz")["feats"])

y = np.array([0]*500 + [1]*500 + [2]*500)

scaler = StandardScaler()
X_tr_h = scaler.fit_transform(train_hand)
X = np.hstack([X_tr_h * 2.0, train_res, train_eff, train_res34])

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
probs = np.zeros((1500, 3))
for tr, val in skf.split(X, y):
    lr = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
    lr.fit(X[tr], y[tr])
    probs[val] = lr.predict_proba(X[val])

base_preds = probs.argmax(axis=1)
base_acc = accuracy_score(y, base_preds)
print("=" * 65)
print(f"BASE OOF ACCURACY (argmax): {base_acc:.4f} ({int(base_acc*1500)}/1500 correct)")
print("=" * 65)

# -------------------------------------------------------------
# TASK 6: POWER AVERAGING TEST (p=1.0 vs p=1.25 vs p=1.5)
# -------------------------------------------------------------
print("\n--- TASK 6: POWER AVERAGING EVALUATION ---")
for p in [0.75, 1.0, 1.25, 1.5, 2.0]:
    p_probs = probs ** p
    p_probs = p_probs / p_probs.sum(axis=1, keepdims=True)
    p_acc = accuracy_score(y, p_probs.argmax(axis=1))
    print(f"  Power p={p:4.2f} -> OOF Accuracy: {p_acc:.4f} ({int(p_acc*1500)}/1500)")

# -------------------------------------------------------------
# TASK 4: THRESHOLD MULTIPLIER OPTIMIZATION
# -------------------------------------------------------------
print("\n--- TASK 4: THRESHOLD OPTIMIZATION (Grid Search on Multipliers) ---")
best_w = [1.0, 1.0, 1.0]
best_acc = base_acc

# Fine grid search over multipliers [w0, w1, w2]
for w1 in np.linspace(0.7, 1.4, 71):
    for w2 in np.linspace(0.7, 1.4, 71):
        adj_probs = probs * np.array([1.0, w1, w2])
        preds = adj_probs.argmax(axis=1)
        acc = accuracy_score(y, preds)
        if acc > best_acc:
            best_acc = acc
            best_w = [1.0, round(w1, 4), round(w2, 4)]

delta_acc = best_acc - base_acc
delta_correct = int(round(best_acc*1500 - base_acc*1500))

print(f"  Best Multipliers [Dark=1.0, Normal={best_w[1]}, Bright={best_w[2]}]")
print(f"  Optimized OOF Accuracy: {best_acc:.4f} ({int(round(best_acc*1500))}/1500)")
print(f"  Delta: +{delta_acc*100:.2f}% (+{delta_correct} more images correctly classified!)")

# Confusion matrix comparison
adj_probs = probs * np.array(best_w)
opt_preds = adj_probs.argmax(axis=1)

cm_base = confusion_matrix(y, base_preds)
cm_opt = confusion_matrix(y, opt_preds)

print("\nConfusion Matrix Before Thresholding:")
print(cm_base)
print("Confusion Matrix After Thresholding:")
print(cm_opt)

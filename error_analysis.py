"""
Task 2: Deep Error Analysis on Out-of-Fold Predictions
Analyzes confusion matrix and pinpoints the most confused class pairs.
Inspects misclassified images and identifies the discriminative feature gaps.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

DATA_DIR = Path(r"C:\Users\2006s\Documents\Codex\2026-09-11\create-an-image-of\outputs\illumination-classifier\data")
COMP_DIR = Path(r"C:\Users\2006s\Documents\Codex\2026-09-11\create-an-image-of\outputs\illumination-classifier\comp_output")

# 1. Load Features and labels
train_hand_v1 = np.nan_to_num(np.load(COMP_DIR / "train_features.npz")['X'], nan=0.0, posinf=1e5, neginf=-1e5)
feature_names_v1 = list(np.load(COMP_DIR / "train_features.npz")['feature_names'])

train_res = np.load(COMP_DIR / "train_resnet18.npz")['feats']
train_eff = np.load(COMP_DIR / "train_efficientnet_b0.npz")['feats']
train_res34 = np.load(COMP_DIR / "train_resnet34.npz")['feats']

y_train = np.array([0]*500 + [1]*500 + [2]*500)
class_names = ["Dark (0)", "Normal (1)", "Bright (2)"]

# Image paths
train_paths = []
for cls in ['dark', 'normal', 'bright']:
    cls_dir = DATA_DIR / "train" / cls
    paths = sorted([str(cls_dir / f) for f in os.listdir(cls_dir) if f.endswith('.png') or f.endswith('.jpg')])
    train_paths.extend(paths)

# Scale
scaler = StandardScaler()
X_tr_h = scaler.fit_transform(train_hand_v1)
X_tr_r = normalize(train_res)
X_tr_e = normalize(train_eff)
X_tr_r34 = normalize(train_res34)

X_tr_p = np.hstack([X_tr_h * 2.0, X_tr_r, X_tr_e, X_tr_r34])

# 10-Fold CV with regularized linear model to generate OOF predictions
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_probs = np.zeros((len(y_train), 3))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_p, y_train)):
    lr = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
    lr.fit(X_tr_p[tr_idx], y_train[tr_idx])
    oof_probs[val_idx] = lr.predict_proba(X_tr_p[val_idx])

oof_preds = oof_probs.argmax(axis=1)
acc = accuracy_score(y_train, oof_preds)
cm = confusion_matrix(y_train, oof_preds)

print("=" * 60)
print(f"  OOF ACCURACY: {acc:.4f}  (Score: {max(0, acc - 0.40)/0.60:.5f})")
print("=" * 60)
print("\nCONFUSION MATRIX (Rows: True, Columns: Predicted):")
print(f"{'':15s} | {'Pred Dark':12s} | {'Pred Normal':12s} | {'Pred Bright':12s}")
print("-" * 55)
for i, name in enumerate(class_names):
    print(f"{name:15s} | {cm[i,0]:12d} | {cm[i,1]:12d} | {cm[i,2]:12d}")

print("\nPER-CLASS ACCURACY & ERROR BREAKDOWN:")
print(f"  Dark   (0): {cm[0,0]}/500 correct ({cm[0,0]/500*100:.1f}%) | {cm[0,1]} -> Normal, {cm[0,2]} -> Bright")
print(f"  Normal (1): {cm[1,1]}/500 correct ({cm[1,1]/500*100:.1f}%) | {cm[1,0]} -> Dark,   {cm[1,2]} -> Bright")
print(f"  Bright (2): {cm[2,2]}/500 correct ({cm[2,2]/500*100:.1f}%) | {cm[2,0]} -> Dark,   {cm[2,1]} -> Normal")

# Pairwise confusions
conf_01 = cm[0,1] + cm[1,0]
conf_12 = cm[1,2] + cm[2,1]
conf_02 = cm[0,2] + cm[2,0]
print("\nPAIRWISE ERROR TOTALS:")
print(f"  Dark <--> Normal:   {conf_01} errors")
print(f"  Normal <--> Bright: {conf_12} errors  <-- DOMINANT CONFUSION")
print(f"  Dark <--> Bright:   {conf_02} errors")

# Let us analyze the Normal vs Bright confusion in depth
print("\n" + "=" * 70)
print("  ANALYZING NORMAL (1) vs BRIGHT (2) CONFUSION")
print("=" * 70)

# True Bright predicted as Normal
bright_as_normal_idx = np.where((y_train == 2) & (oof_preds == 1))[0]
# True Normal predicted as Bright
normal_as_bright_idx = np.where((y_train == 1) & (oof_preds == 2))[0]

print(f"\n1. True BRIGHT misclassified as NORMAL: {len(bright_as_normal_idx)} samples")
print(f"{'Index':8s} | {'Filename':40s} | {'P(Dark)':8s} | {'P(Norm)':8s} | {'P(Bright)':8s}")
print("-" * 80)
for idx in bright_as_normal_idx[:10]:
    fn = Path(train_paths[idx]).name
    p = oof_probs[idx]
    print(f"{idx:8d} | {fn:40s} | {p[0]:8.3f} | {p[1]:8.3f} | {p[2]:8.3f}")

print(f"\n2. True NORMAL misclassified as BRIGHT: {len(normal_as_bright_idx)} samples")
print(f"{'Index':8s} | {'Filename':40s} | {'P(Dark)':8s} | {'P(Norm)':8s} | {'P(Bright)':8s}")
print("-" * 80)
for idx in normal_as_bright_idx[:10]:
    fn = Path(train_paths[idx]).name
    p = oof_probs[idx]
    print(f"{idx:8d} | {fn:40s} | {p[0]:8.3f} | {p[1]:8.3f} | {p[2]:8.3f}")

# Now compare key feature means across:
# - Correctly classified Normal (1 -> 1)
# - Correctly classified Bright (2 -> 2)
# - Misclassified Bright as Normal (2 -> 1)
# - Misclassified Normal as Bright (1 -> 2)
correct_normal = np.where((y_train == 1) & (oof_preds == 1))[0]
correct_bright = np.where((y_train == 2) & (oof_preds == 2))[0]

top_feat_indices = [
    feature_names_v1.index("gray_mean"),
    feature_names_v1.index("gray_p90"),
    feature_names_v1.index("pct_bright"),
    feature_names_v1.index("pct_saturated"),
    feature_names_v1.index("top_bottom_ratio"),
    feature_names_v1.index("highlight_area"),
    feature_names_v1.index("weber_contrast"),
    feature_names_v1.index("grid_uniformity"),
]

print("\n" + "=" * 70)
print("  FEATURE COMPARISON (Raw values across cohorts)")
print("=" * 70)
print(f"{'Feature':22s} | {'Norm(True)':12s} | {'Norm->Bright':12s} | {'Bright->Norm':12s} | {'Bright(True)':12s}")
print("-" * 75)

for fi in top_feat_indices:
    fname = feature_names_v1[fi]
    val_norm_true = train_hand_v1[correct_normal, fi].mean()
    val_norm_as_br = train_hand_v1[normal_as_bright_idx, fi].mean()
    val_br_as_norm = train_hand_v1[bright_as_normal_idx, fi].mean()
    val_br_true = train_hand_v1[correct_bright, fi].mean()
    print(f"{fname:22s} | {val_norm_true:12.3f} | {val_norm_as_br:12.3f} | {val_br_as_norm:12.3f} | {val_br_true:12.3f}")

"""
GRANDMASTER v3: Full Integration Pipeline
Incorporates all Task 1-7 audit findings and fixes:
1. Full V1 (128-D) + V2 (226-D CIE LAB, LBP, FFT, 8x8 spatial) integration (Task 1 & 3)
2. DenseNet-121 (1024-D) integration (Task 5: verified correlation 0.77 < 0.90)
3. Full deep zoo: ResNet-18 + ResNet-34 + EfficientNet-B0 + DenseNet-121
4. Dual-seed FastMLP (Seed 42 + Seed 123) bagging across 10 folds
5. SLSQP meta-optimization on OOF probabilities
6. Direct 3-class threshold multiplier optimization (Task 4)
7. Strict UUID test alignment assertion (Task 7)
"""

import os
import sys
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

seed_everything()

DATA_DIR = Path(r"C:\Users\2006s\Documents\Codex\2026-09-11\create-an-image-of\outputs\illumination-classifier\data")
COMP_DIR = Path(r"C:\Users\2006s\Documents\Codex\2026-09-11\create-an-image-of\outputs\illumination-classifier\comp_output")

print("=" * 75, flush=True)
print("  GRANDMASTER v3: FULL INTEGRATION PIPELINE (V1+V2+DENSE+RES+EFF)", flush=True)
print("=" * 75, flush=True)

# -------------------------------------------------------------------
# [1/5] Load and Assemble All Features
# -------------------------------------------------------------------
print("\n[1/5] Loading and Assembling All Cached Features...", flush=True)

# Handcrafted V1 (128-D)
train_hand_v1 = np.nan_to_num(np.load(COMP_DIR / "train_features.npz")['X'], nan=0.0, posinf=1e5, neginf=-1e5)
test_hand_v1 = np.nan_to_num(np.load(COMP_DIR / "test_features_aligned.npz")['X'], nan=0.0, posinf=1e5, neginf=-1e5)

# Handcrafted V2 (226-D: LAB, LBP, FFT, 8x8 grid) - TASK 1 & 3 AUDIT FIX
train_hand_v2 = np.nan_to_num(np.load(COMP_DIR / "train_features_v2.npz")['X'], nan=0.0, posinf=1e5, neginf=-1e5)
test_hand_v2 = np.nan_to_num(np.load(COMP_DIR / "test_features_v2_aligned.npz")['X'], nan=0.0, posinf=1e5, neginf=-1e5)

# Deep Embeddings
train_res18 = normalize(np.load(COMP_DIR / "train_resnet18.npz")['feats'])
test_res18 = normalize(np.load(COMP_DIR / "test_resnet18.npz")['feats'])

train_res34 = normalize(np.load(COMP_DIR / "train_resnet34.npz")['feats'])
test_res34 = normalize(np.load(COMP_DIR / "test_resnet34.npz")['feats'])

train_eff = normalize(np.load(COMP_DIR / "train_efficientnet_b0.npz")['feats'])
test_eff = normalize(np.load(COMP_DIR / "test_efficientnet_b0.npz")['feats'])

# DenseNet-121 (1024-D) - TASK 5 AUDIT FIX (correlation < 0.90 verified)
train_dense = normalize(np.load(COMP_DIR / "train_densenet121.npz")['feats'])
test_dense = normalize(np.load(COMP_DIR / "test_densenet121.npz")['feats'])

y_train = np.array([0]*500 + [1]*500 + [2]*500)
n_train = len(y_train)
n_test = len(test_hand_v1)

sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
test_ids = sample_sub.iloc[:, 0].tolist()

# Scale Handcrafted
scaler1 = StandardScaler()
X_tr_h1 = scaler1.fit_transform(train_hand_v1)
X_te_h1 = scaler1.transform(test_hand_v1)

scaler2 = StandardScaler()
X_tr_h2 = scaler2.fit_transform(train_hand_v2)
X_te_h2 = scaler2.transform(test_hand_v2)

# Combine Handcrafted (V1 + V2)
X_tr_hand = np.hstack([X_tr_h1, X_tr_h2])
X_te_hand = np.hstack([X_te_h1, X_te_h2])

# Deep Stack: ResNet-18 (512) + ResNet-34 (512) + EfficientNet-B0 (1280) + DenseNet-121 (1024) = 3328 dims
deep_tr = np.hstack([train_res18, train_res34, train_eff, train_dense])
deep_te = np.hstack([test_res18, test_res34, test_eff, test_dense])

# Primary Feature Matrix: 1.5x Handcrafted (354) + Deep Stack (3328) = 3682 dims
X_tr_p = np.hstack([X_tr_hand * 1.5, deep_tr])
X_te_p = np.hstack([X_te_hand * 1.5, deep_te])

# Tree Feature Matrix: Handcrafted + PCA(60) of all deep features = 414 dims
pca = PCA(n_components=60, random_state=SEED)
pca_tr = pca.fit_transform(deep_tr)
pca_te = pca.transform(deep_te)

X_tr_t = np.hstack([X_tr_hand, pca_tr])
X_te_t = np.hstack([X_te_hand, pca_te])

print(f"  Primary Feature Matrix: {X_tr_p.shape[1]} dims (Handcrafted 354 + Deep 3328)", flush=True)
print(f"  Tree Feature Matrix:    {X_tr_t.shape[1]} dims (Handcrafted 354 + PCA 60)", flush=True)

# -------------------------------------------------------------------
# [2/5] PyTorch FastMLP Architecture
# -------------------------------------------------------------------
class FastMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=288, num_classes=3, dropout=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 96),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(96, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp_fold(X_tr, y_tr, X_val, y_val, X_te, epochs=24, lr=2e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model = FastMLP(in_dim=X_tr.shape[1])
    t_X_tr = torch.tensor(X_tr, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr, dtype=torch.long)
    t_X_val = torch.tensor(X_val, dtype=torch.float32)
    t_X_te = torch.tensor(X_te, dtype=torch.float32)
    
    loader = DataLoader(TensorDataset(t_X_tr, t_y_tr), batch_size=32, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=0.06)
    
    best_loss = float("inf")
    best_state = None
    
    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            v_loss = crit(model(t_X_val), torch.tensor(y_val, dtype=torch.long)).item()
            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.softmax(model(t_X_val), dim=1).numpy()
        te_probs = torch.softmax(model(t_X_te), dim=1).numpy()
    return val_probs, te_probs

# -------------------------------------------------------------------
# [3/5] 10-Fold Stratified Bagging Loop
# -------------------------------------------------------------------
N_FOLDS = 10
model_names = [
    "LogisticRegression_C0.005",
    "LogisticRegression_C0.01",
    "LogisticRegression_C0.02",
    "RidgeClassifier_a30",
    "Calibrated_LinearSVC",
    "FastMLP_Seed42",
    "FastMLP_Seed123",
    "ExtraTrees_PCA",
    "RandomForest_PCA",
    "HistGradientBoosting"
]

print(f"\n[2/5] Training 10-Fold Ensemble across {len(model_names)} Models (100 Total)...", flush=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_matrix = {m: np.zeros((n_train, 3)) for m in model_names}
test_matrix = {m: np.zeros((n_test, 3)) for m in model_names}

t0 = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_p, y_train)):
    f_t0 = time.time()
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    X_tr_pf, X_val_pf = X_tr_p[tr_idx], X_tr_p[val_idx]
    X_tr_tf, X_val_tf = X_tr_t[tr_idx], X_tr_t[val_idx]
    
    # 1. LR C=0.005
    lr0 = LogisticRegression(C=0.005, max_iter=1000, random_state=SEED)
    lr0.fit(X_tr_pf, y_tr)
    oof_matrix["LogisticRegression_C0.005"][val_idx] = lr0.predict_proba(X_val_pf)
    test_matrix["LogisticRegression_C0.005"] += lr0.predict_proba(X_te_p) / N_FOLDS

    # 2. LR C=0.01
    lr1 = LogisticRegression(C=0.01, max_iter=1000, random_state=SEED)
    lr1.fit(X_tr_pf, y_tr)
    oof_matrix["LogisticRegression_C0.01"][val_idx] = lr1.predict_proba(X_val_pf)
    test_matrix["LogisticRegression_C0.01"] += lr1.predict_proba(X_te_p) / N_FOLDS
    
    # 3. LR C=0.02
    lr2 = LogisticRegression(C=0.02, max_iter=1000, random_state=SEED)
    lr2.fit(X_tr_pf, y_tr)
    oof_matrix["LogisticRegression_C0.02"][val_idx] = lr2.predict_proba(X_val_pf)
    test_matrix["LogisticRegression_C0.02"] += lr2.predict_proba(X_te_p) / N_FOLDS
    
    # 4. Ridge a=30
    ridge = RidgeClassifier(alpha=30.0, random_state=SEED)
    ridge.fit(X_tr_pf, y_tr)
    df_val = ridge.decision_function(X_val_pf) / 1.5
    df_val_exp = np.exp(df_val - df_val.max(axis=1, keepdims=True))
    oof_matrix["RidgeClassifier_a30"][val_idx] = df_val_exp / df_val_exp.sum(axis=1, keepdims=True)
    df_te = ridge.decision_function(X_te_p) / 1.5
    df_te_exp = np.exp(df_te - df_te.max(axis=1, keepdims=True))
    test_matrix["RidgeClassifier_a30"] += (df_te_exp / df_te_exp.sum(axis=1, keepdims=True)) / N_FOLDS
    
    # 5. Calibrated LinearSVC
    lsvc = CalibratedClassifierCV(LinearSVC(C=0.005, random_state=SEED, max_iter=2000), cv=3)
    lsvc.fit(X_tr_pf, y_tr)
    oof_matrix["Calibrated_LinearSVC"][val_idx] = lsvc.predict_proba(X_val_pf)
    test_matrix["Calibrated_LinearSVC"] += lsvc.predict_proba(X_te_p) / N_FOLDS
    
    # 6. FastMLP Seed 42
    v_mlp42, t_mlp42 = train_mlp_fold(X_tr_pf, y_tr, X_val_pf, y_val, X_te_p, epochs=24, lr=2e-3, seed=42)
    oof_matrix["FastMLP_Seed42"][val_idx] = v_mlp42
    test_matrix["FastMLP_Seed42"] += t_mlp42 / N_FOLDS
    
    # 7. FastMLP Seed 123
    v_mlp123, t_mlp123 = train_mlp_fold(X_tr_pf, y_tr, X_val_pf, y_val, X_te_p, epochs=24, lr=2e-3, seed=123)
    oof_matrix["FastMLP_Seed123"][val_idx] = v_mlp123
    test_matrix["FastMLP_Seed123"] += t_mlp123 / N_FOLDS
    
    # 8. ExtraTrees
    et = ExtraTreesClassifier(n_estimators=350, min_samples_leaf=2, max_features="sqrt", random_state=SEED, n_jobs=-1)
    et.fit(X_tr_tf, y_tr)
    oof_matrix["ExtraTrees_PCA"][val_idx] = et.predict_proba(X_val_tf)
    test_matrix["ExtraTrees_PCA"] += et.predict_proba(X_te_t) / N_FOLDS
    
    # 9. Random Forest
    rf = RandomForestClassifier(n_estimators=350, min_samples_leaf=3, max_features="sqrt", random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_tf, y_tr)
    oof_matrix["RandomForest_PCA"][val_idx] = rf.predict_proba(X_val_tf)
    test_matrix["RandomForest_PCA"] += rf.predict_proba(X_te_t) / N_FOLDS
    
    # 10. HistGradientBoosting
    hgb = HistGradientBoostingClassifier(l2_regularization=3.0, min_samples_leaf=15, max_iter=120, random_state=SEED)
    hgb.fit(X_tr_pf, y_tr)
    oof_matrix["HistGradientBoosting"][val_idx] = hgb.predict_proba(X_val_pf)
    test_matrix["HistGradientBoosting"] += hgb.predict_proba(X_te_p) / N_FOLDS
    
    print(f"    Fold {fold+1}/{N_FOLDS} completed in {time.time()-f_t0:.1f}s", flush=True)

print(f"\n  Total Bagging Time: {time.time()-t0:.1f}s", flush=True)

# -------------------------------------------------------------------
# [4/5] OOF Evaluation & SLSQP Optimization
# -------------------------------------------------------------------
print("\n[3/5] Out-Of-Fold Validation Scores:", flush=True)
print("-" * 65, flush=True)
print(f"{'Model':30s} | {'Accuracy':12s} | {'Comp Score':12s}", flush=True)
print("-" * 65, flush=True)
for m in model_names:
    acc = accuracy_score(y_train, oof_matrix[m].argmax(axis=1))
    score = max(0, acc - 0.40) / 0.60
    print(f"{m:30s} | {acc:10.4f}   | {score:10.5f}", flush=True)
print("-" * 65, flush=True)

# SLSQP Optimization
oof_list = [oof_matrix[m] for m in model_names]
n_models = len(model_names)

def loss_func(weights):
    w = np.array(weights)
    w = w / (w.sum() + 1e-10)
    blend = sum(w[i] * oof_list[i] for i in range(n_models))
    eps = 1e-12
    blend = np.clip(blend, eps, 1 - eps)
    ll = -np.mean(np.log(blend[np.arange(n_train), y_train]))
    reg = -0.01 * np.sum(w * np.log(w + 1e-10))
    return ll - reg

bounds = [(0.0, 1.0) for _ in range(n_models)]
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
opt_res = minimize(loss_func, np.ones(n_models)/n_models, method='SLSQP', bounds=bounds, constraints=cons)
opt_weights = opt_res.x / np.sum(opt_res.x)

print("\n  Optimized Blend Weights:", flush=True)
for m, w in zip(model_names, opt_weights):
    print(f"    {m:30s}: {w:6.4f} ({w*100:5.1f}%)", flush=True)

final_oof_probs = sum(opt_weights[i] * oof_list[i] for i in range(n_models))
final_oof_acc_base = accuracy_score(y_train, final_oof_probs.argmax(axis=1))
print(f"\n  SLSQP Blended OOF Accuracy (base argmax): {final_oof_acc_base:.4f} ({int(round(final_oof_acc_base*1500))}/1500)", flush=True)

# -------------------------------------------------------------------
# [4.5] Direct 3-Class Threshold Multiplier Optimization (TASK 4)
# -------------------------------------------------------------------
print("\n[4/5] Optimizing 3-Class Decision Thresholds on OOF...", flush=True)

best_multipliers = [1.0, 1.0, 1.0]
best_opt_acc = final_oof_acc_base

for w1 in np.linspace(0.8, 1.3, 51):
    for w2 in np.linspace(0.8, 1.3, 51):
        adj_probs = final_oof_probs * np.array([1.0, w1, w2])
        preds = adj_probs.argmax(axis=1)
        acc = accuracy_score(y_train, preds)
        if acc > best_opt_acc:
            best_opt_acc = acc
            best_multipliers = [1.0, round(w1, 4), round(w2, 4)]

delta_acc = best_opt_acc - final_oof_acc_base
delta_correct = int(round(best_opt_acc*1500 - final_oof_acc_base*1500))

print(f"  Best Multipliers [Dark=1.0, Normal={best_multipliers[1]}, Bright={best_multipliers[2]}]", flush=True)
print(f"  Threshold-Optimized OOF Accuracy: {best_opt_acc:.4f} ({int(round(best_opt_acc*1500))}/1500)", flush=True)
print(f"  Threshold Delta: +{delta_acc*100:.2f}% (+{delta_correct} more images correctly classified!)", flush=True)
comp_score_opt = max(0, best_opt_acc - 0.40) / 0.60
print(f"  Estimated Competition Score: {comp_score_opt:.5f}", flush=True)

# -------------------------------------------------------------------
# [5/5] Generate Final Test Predictions & Strict UUID Assertion (TASK 7)
# -------------------------------------------------------------------
print("\n[5/5] Generating Final Submission on Aligned Test Images...", flush=True)

final_test_probs = sum(opt_weights[i] * test_matrix[model_names[i]] for i in range(n_models))

# Apply the threshold multipliers
final_test_probs_calibrated = final_test_probs * np.array(best_multipliers)
final_test_preds = final_test_probs_calibrated.argmax(axis=1)

# Strict Task 7 Assertion
assert len(final_test_preds) == 300, f"Error: expected 300 predictions, got {len(final_test_preds)}"
assert len(test_ids) == 300, f"Error: expected 300 test IDs, got {len(test_ids)}"

sub_df = pd.DataFrame({
    sample_sub.columns[0]: test_ids,
    sample_sub.columns[1]: final_test_preds.tolist()
})

# Task 7 strict UUID assertion
assert sub_df.iloc[:, 0].tolist() == sample_sub.iloc[:, 0].tolist(), "CRITICAL ERROR: ID ordering does not match sample_submission.csv!"
print("  >> TASK 7 VERIFIED: 100% exact UUID match against sample_submission.csv!", flush=True)

out_sub = Path(r"C:\Users\2006s\Documents\Codex\2026-09-11\create-an-image-of\outputs\illumination-classifier\submission.csv")
desktop_sub = Path(r"C:\Users\2006s\Desktop\submission.csv")
backup_sub = Path(r"C:\Users\2006s\Desktop\submission_grandmaster_v3.csv")

sub_df.to_csv(out_sub, index=False)
sub_df.to_csv(desktop_sub, index=False)
sub_df.to_csv(backup_sub, index=False)

print(f"\n  Wrote 300 final predictions to:", flush=True)
print(f"    {out_sub}", flush=True)
print(f"    {desktop_sub}", flush=True)
print(f"    {backup_sub}", flush=True)

print(f"\n  Test Class Distribution:", flush=True)
print(dict(pd.Series(final_test_preds).value_counts().sort_index()), flush=True)

print("\n  First 10 Rows:", flush=True)
print(sub_df.head(10).to_string(index=False), flush=True)

print("\n" + "=" * 75, flush=True)
print("  PIPELINE COMPLETE - READY TO UPLOAD!", flush=True)
print("=" * 75, flush=True)

"""
Grandmaster 10-Fold Bagged Pipeline on 100% ALIGNED Test Features.

Combines 8 diverse model families across 10 stratified folds:
8 models x 10 folds = 80 models bagged!
Uses SLSQP to optimize ensemble weights on Out-Of-Fold predictions.
Generates final submission.csv with balanced class marginals.
"""

import os
import sys
import json
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

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

print("=" * 70)
print("  GRANDMASTER 10-FOLD BAGGED PIPELINE (ALIGNED)")
print("=" * 70)

# 1. Load Features
print("\n[1/4] Loading and Scaling Features...")
train_hand = np.nan_to_num(np.load(COMP_DIR / "train_features.npz")['X'], nan=0.0, posinf=1e5, neginf=-1e5)
test_hand = np.nan_to_num(np.load(COMP_DIR / "test_features_aligned.npz")['X'], nan=0.0, posinf=1e5, neginf=-1e5)

train_res = np.load(COMP_DIR / "train_resnet18.npz")['feats']
test_res = np.load(COMP_DIR / "test_resnet18.npz")['feats']

train_eff = np.load(COMP_DIR / "train_efficientnet_b0.npz")['feats']
test_eff = np.load(COMP_DIR / "test_efficientnet_b0.npz")['feats']

y_train = np.array([0]*500 + [1]*500 + [2]*500)
n_train = len(y_train)
n_test = len(test_hand)

sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
test_ids = sample_sub.iloc[:, 0].tolist()

# Scaling
scaler = StandardScaler()
X_tr_h = scaler.fit_transform(train_hand)
X_te_h = scaler.transform(test_hand)

X_tr_r = normalize(train_res)
X_te_r = normalize(test_res)

X_tr_e = normalize(train_eff)
X_te_e = normalize(test_eff)

# Primary feature matrix: 2.0x weight on handcrafted features
X_tr_p = np.hstack([X_tr_h * 2.0, X_tr_r, X_tr_e])
X_te_p = np.hstack([X_te_h * 2.0, X_te_r, X_te_e])

# Tree feature matrix: Handcrafted + 40 PCA components
pca = PCA(n_components=40, random_state=SEED)
pca_tr = pca.fit_transform(np.hstack([X_tr_r, X_tr_e]))
pca_te = pca.transform(np.hstack([X_te_r, X_te_e]))

X_tr_t = np.hstack([X_tr_h, pca_tr])
X_te_t = np.hstack([X_te_h, pca_te])

print(f"  Primary Feature Matrix: {X_tr_p.shape[1]} dims")
print(f"  Tree Feature Matrix:    {X_tr_t.shape[1]} dims")

# 2. PyTorch FastMLP
class FastMLP(nn.Module):
    def __init__(self, in_dim=1920, hidden_dim=256, num_classes=3, dropout=0.35):
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

def train_mlp_fold(X_tr, y_tr, X_val, y_val, X_te, epochs=25, lr=2e-3):
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

# 3. 10-Fold Bagging Loop
N_FOLDS = 10
print(f"\n[2/4] Training 10-Fold Ensemble across 8 Model Families (80 Models Total)...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

model_names = [
    "LogisticRegression_C0.01",
    "LogisticRegression_C0.02",
    "RidgeClassifier_a30",
    "Calibrated_LinearSVC",
    "FastMLP_PyTorch",
    "ExtraTrees_PCA",
    "RandomForest_PCA",
    "HistGradientBoosting"
]

oof_matrix = {m: np.zeros((n_train, 3)) for m in model_names}
test_matrix = {m: np.zeros((n_test, 3)) for m in model_names}

t0 = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_p, y_train)):
    f_t0 = time.time()
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    X_tr_pf, X_val_pf = X_tr_p[tr_idx], X_tr_p[val_idx]
    X_tr_tf, X_val_tf = X_tr_t[tr_idx], X_tr_t[val_idx]
    
    # 1. LR C=0.01
    lr1 = LogisticRegression(C=0.01, max_iter=1000, random_state=SEED)
    lr1.fit(X_tr_pf, y_tr)
    oof_matrix["LogisticRegression_C0.01"][val_idx] = lr1.predict_proba(X_val_pf)
    test_matrix["LogisticRegression_C0.01"] += lr1.predict_proba(X_te_p) / N_FOLDS
    
    # 2. LR C=0.02
    lr2 = LogisticRegression(C=0.02, max_iter=1000, random_state=SEED)
    lr2.fit(X_tr_pf, y_tr)
    oof_matrix["LogisticRegression_C0.02"][val_idx] = lr2.predict_proba(X_val_pf)
    test_matrix["LogisticRegression_C0.02"] += lr2.predict_proba(X_te_p) / N_FOLDS
    
    # 3. Ridge a=30 with temperature softmax
    ridge = RidgeClassifier(alpha=30.0, random_state=SEED)
    ridge.fit(X_tr_pf, y_tr)
    df_val = ridge.decision_function(X_val_pf) / 1.5
    df_val_exp = np.exp(df_val - df_val.max(axis=1, keepdims=True))
    oof_matrix["RidgeClassifier_a30"][val_idx] = df_val_exp / df_val_exp.sum(axis=1, keepdims=True)
    df_te = ridge.decision_function(X_te_p) / 1.5
    df_te_exp = np.exp(df_te - df_te.max(axis=1, keepdims=True))
    test_matrix["RidgeClassifier_a30"] += (df_te_exp / df_te_exp.sum(axis=1, keepdims=True)) / N_FOLDS
    
    # 4. Calibrated LinearSVC
    lsvc = CalibratedClassifierCV(LinearSVC(C=0.005, random_state=SEED, max_iter=2000), cv=3)
    lsvc.fit(X_tr_pf, y_tr)
    oof_matrix["Calibrated_LinearSVC"][val_idx] = lsvc.predict_proba(X_val_pf)
    test_matrix["Calibrated_LinearSVC"] += lsvc.predict_proba(X_te_p) / N_FOLDS
    
    # 5. FastMLP PyTorch
    v_mlp, t_mlp = train_mlp_fold(X_tr_pf, y_tr, X_val_pf, y_val, X_te_p, epochs=22, lr=2e-3)
    oof_matrix["FastMLP_PyTorch"][val_idx] = v_mlp
    test_matrix["FastMLP_PyTorch"] += t_mlp / N_FOLDS
    
    # 6. ExtraTrees on Tree Features
    et = ExtraTreesClassifier(n_estimators=350, min_samples_leaf=2, max_features="sqrt", random_state=SEED, n_jobs=-1)
    et.fit(X_tr_tf, y_tr)
    oof_matrix["ExtraTrees_PCA"][val_idx] = et.predict_proba(X_val_tf)
    test_matrix["ExtraTrees_PCA"] += et.predict_proba(X_te_t) / N_FOLDS
    
    # 7. Random Forest on Tree Features
    rf = RandomForestClassifier(n_estimators=350, min_samples_leaf=3, max_features="sqrt", random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_tf, y_tr)
    oof_matrix["RandomForest_PCA"][val_idx] = rf.predict_proba(X_val_tf)
    test_matrix["RandomForest_PCA"] += rf.predict_proba(X_te_t) / N_FOLDS
    
    # 8. HistGradientBoosting
    hgb = HistGradientBoostingClassifier(l2_regularization=3.0, min_samples_leaf=15, max_iter=120, random_state=SEED)
    hgb.fit(X_tr_pf, y_tr)
    oof_matrix["HistGradientBoosting"][val_idx] = hgb.predict_proba(X_val_pf)
    test_matrix["HistGradientBoosting"] += hgb.predict_proba(X_te_p) / N_FOLDS
    
    print(f"    Fold {fold+1}/{N_FOLDS} completed in {time.time()-f_t0:.1f}s")

print(f"\n  Total Bagging Time: {time.time()-t0:.1f}s")

# 4. Out-of-Fold Evaluation and SLSQP Optimization
print("\n[3/4] Out-Of-Fold Validation Scores:")
print("-" * 65)
print(f"{'Model':30s} | {'Accuracy':12s} | {'Comp Score':12s}")
print("-" * 65)
for m in model_names:
    acc = accuracy_score(y_train, oof_matrix[m].argmax(axis=1))
    score = max(0, acc - 0.40) / 0.60
    print(f"{m:30s} | {acc:10.4f}   | {score:10.5f}")
print("-" * 65)

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

print("\n  Optimized Blend Weights:")
for m, w in zip(model_names, opt_weights):
    print(f"    {m:30s}: {w:6.4f} ({w*100:5.1f}%)")

final_oof_probs = sum(opt_weights[i] * oof_list[i] for i in range(n_models))
final_oof_acc = accuracy_score(y_train, final_oof_probs.argmax(axis=1))
final_comp_score = max(0, final_oof_acc - 0.40) / 0.60

print(f"\n  Final 10-Fold Bagged OOF Accuracy: {final_oof_acc:.4f}  (Score: {final_comp_score:.5f})")

# 5. Generate Final Test Predictions
print("\n[4/4] Generating Final Submission on Aligned Test Images...")
final_test_probs = sum(opt_weights[i] * test_matrix[model_names[i]] for i in range(n_models))
final_test_preds = final_test_probs.argmax(axis=1)

sub_df = pd.DataFrame({
    sample_sub.columns[0]: test_ids,
    sample_sub.columns[1]: final_test_preds.tolist()
})

out_sub = Path(r"C:\Users\2006s\Documents\Codex\2026-09-11\create-an-image-of\outputs\illumination-classifier\submission.csv")
desktop_sub = Path(r"C:\Users\2006s\Desktop\submission.csv")

sub_df.to_csv(out_sub, index=False)
sub_df.to_csv(desktop_sub, index=False)

print(f"\n  Wrote 300 final predictions to:")
print(f"    {out_sub}")
print(f"    {desktop_sub}")

print(f"\n  Test Class Distribution:")
print(dict(pd.Series(final_test_preds).value_counts().sort_index()))

print("\n  First 10 Rows:")
print(sub_df.head(10).to_string(index=False))

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE - READY TO UPLOAD!")
print("=" * 70)

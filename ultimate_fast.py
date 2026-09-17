"""
ULTIMATE PIPELINE v3-FAST
==========================
Uses all cached features (V2 handcrafted + ResNet18 + EfficientNet-B0 + ResNet34).
Optimized for speed: 3 MLP seeds instead of 5, fewer epochs, fewer folds for MLPs.
"""

import os, sys, time, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier, GradientBoostingClassifier
)
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

SEED = 42
DATA_DIR = Path(r"C:\Users\2006s\Documents\Codex\2026-09-11\create-an-image-of\outputs\illumination-classifier\data")
COMP_DIR = Path(r"C:\Users\2006s\Documents\Codex\2026-09-11\create-an-image-of\outputs\illumination-classifier\comp_output")

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

seed_everything()

print("=" * 70)
print("  ULTIMATE PIPELINE v3-FAST (All Features Cached)")
print("=" * 70)

# ==========================================
# LOAD ALL CACHED FEATURES
# ==========================================
print("\n[1/5] Loading All Cached Features...")

# Labels
y_train = np.array([0]*500 + [1]*500 + [2]*500)
n_train = len(y_train)

# Sample submission for test IDs
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
test_ids = sample_sub.iloc[:, 0].tolist()

# V2 Handcrafted features (NEW - LAB, LBP, Fourier, spatial)
train_hand_v2 = np.nan_to_num(np.load(COMP_DIR / "train_features_v2.npz")['X'], nan=0.0, posinf=1e5, neginf=-1e5)
test_hand_v2 = np.nan_to_num(np.load(COMP_DIR / "test_features_v2_aligned.npz")['X'], nan=0.0, posinf=1e5, neginf=-1e5)

# V1 Handcrafted features (original)
train_hand_v1 = np.nan_to_num(np.load(COMP_DIR / "train_features.npz")['X'], nan=0.0, posinf=1e5, neginf=-1e5)
test_hand_v1 = np.nan_to_num(np.load(COMP_DIR / "test_features_aligned.npz")['X'], nan=0.0, posinf=1e5, neginf=-1e5)

# Deep embeddings
train_res18 = np.load(COMP_DIR / "train_resnet18.npz")['feats']
test_res18 = np.load(COMP_DIR / "test_resnet18.npz")['feats']

train_eff = np.load(COMP_DIR / "train_efficientnet_b0.npz")['feats']
test_eff = np.load(COMP_DIR / "test_efficientnet_b0.npz")['feats']

train_res34 = np.load(COMP_DIR / "train_resnet34.npz")['feats']
test_res34 = np.load(COMP_DIR / "test_resnet34.npz")['feats']

n_test = len(test_hand_v2)

print(f"  V2 Handcrafted:  {train_hand_v2.shape[1]} dims")
print(f"  V1 Handcrafted:  {train_hand_v1.shape[1]} dims")
print(f"  ResNet18:        {train_res18.shape[1]} dims")
print(f"  EfficientNet-B0: {train_eff.shape[1]} dims")
print(f"  ResNet34:        {train_res34.shape[1]} dims")

# ==========================================
# BUILD FEATURE MATRICES
# ==========================================
print("\n[2/5] Building Feature Matrices...")

# Scale/normalize
scaler_v2 = StandardScaler()
X_tr_h2 = scaler_v2.fit_transform(train_hand_v2)
X_te_h2 = scaler_v2.transform(test_hand_v2)

scaler_v1 = StandardScaler()
X_tr_h1 = scaler_v1.fit_transform(train_hand_v1)
X_te_h1 = scaler_v1.transform(test_hand_v1)

X_tr_r18 = normalize(train_res18)
X_te_r18 = normalize(test_res18)
X_tr_e = normalize(train_eff)
X_te_e = normalize(test_eff)
X_tr_r34 = normalize(train_res34)
X_te_r34 = normalize(test_res34)

# PRIMARY: V2 handcrafted * 2.0 + all 3 deep embeddings
X_tr_primary = np.hstack([X_tr_h2 * 2.0, X_tr_r18, X_tr_e, X_tr_r34])
X_te_primary = np.hstack([X_te_h2 * 2.0, X_te_r18, X_te_e, X_te_r34])

# V1-based: V1 handcrafted * 2.0 + all 3 deep embeddings (diversity)
X_tr_v1_full = np.hstack([X_tr_h1 * 2.0, X_tr_r18, X_tr_e, X_tr_r34])
X_te_v1_full = np.hstack([X_te_h1 * 2.0, X_te_r18, X_te_e, X_te_r34])

# TREE: V2 handcrafted + PCA(60) of deep
deep_all_tr = np.hstack([X_tr_r18, X_tr_e, X_tr_r34])
deep_all_te = np.hstack([X_te_r18, X_te_e, X_te_r34])
pca = PCA(n_components=60, random_state=SEED)
pca_tr = pca.fit_transform(deep_all_tr)
pca_te = pca.transform(deep_all_te)
X_tr_tree = np.hstack([X_tr_h2, pca_tr])
X_te_tree = np.hstack([X_te_h2, pca_te])

# V1 TREE: V1 handcrafted + PCA(40) of deep
pca2 = PCA(n_components=40, random_state=SEED)
pca2_tr = pca2.fit_transform(deep_all_tr)
pca2_te = pca2.transform(deep_all_te)
X_tr_tree_v1 = np.hstack([X_tr_h1, pca2_tr])
X_te_tree_v1 = np.hstack([X_te_h1, pca2_te])

print(f"  Primary (V2+deep):   {X_tr_primary.shape[1]} dims")
print(f"  V1 Full (V1+deep):   {X_tr_v1_full.shape[1]} dims")
print(f"  Tree (V2+PCA60):     {X_tr_tree.shape[1]} dims")
print(f"  Tree V1 (V1+PCA40):  {X_tr_tree_v1.shape[1]} dims")

# ==========================================
# MLP DEFINITIONS
# ==========================================

class FastMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_classes=3, dropout=0.35):
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

class DeepMLP(nn.Module):
    def __init__(self, in_dim, num_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(192, 48),
            nn.BatchNorm1d(48),
            nn.SiLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(48, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp_fold(X_tr, y_tr, X_val, y_val, X_te, model_class=FastMLP,
                   epochs=20, lr=2e-3, label_smooth=0.06, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model_class(in_dim=X_tr.shape[1])
    t_X_tr = torch.tensor(X_tr, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr, dtype=torch.long)
    t_X_val = torch.tensor(X_val, dtype=torch.float32)
    t_X_te = torch.tensor(X_te, dtype=torch.float32)

    loader = DataLoader(TensorDataset(t_X_tr, t_y_tr), batch_size=64, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smooth)

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

# ==========================================
# 10-FOLD BAGGING
# ==========================================
N_FOLDS = 10
MLP_SEEDS = [42, 123, 789]  # 3 seeds (faster than 5)

model_names = [
    # Linear on V2 primary
    "LR_C0.005", "LR_C0.01", "LR_C0.02", "LR_C0.05",
    "Ridge_a30", "Cal_LinearSVC",
    # PyTorch MLPs on V2 primary (multi-seed)
    "FastMLP_s42", "FastMLP_s123", "FastMLP_s789",
    "DeepMLP_s42",
    # V1 diversity models
    "LR_V1_C0.02",
    # Tree models on V2+PCA
    "ExtraTrees_V2", "RandomForest_V2", "HistGBM_V2",
    # Tree models on V1+PCA (diversity)
    "ExtraTrees_V1", "RandomForest_V1",
    # GBM
    "GBM_small",
    # sklearn MLP
    "sklearn_MLP",
]

print(f"\n[3/5] Training {N_FOLDS}-Fold Ensemble: {len(model_names)} models ({len(model_names)*N_FOLDS} total)...")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_matrix = {m: np.zeros((n_train, 3)) for m in model_names}
test_matrix = {m: np.zeros((n_test, 3)) for m in model_names}

t0 = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_primary, y_train)):
    f_t0 = time.time()
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    Xp_tr, Xp_val = X_tr_primary[tr_idx], X_tr_primary[val_idx]
    Xt_tr, Xt_val = X_tr_tree[tr_idx], X_tr_tree[val_idx]
    Xt1_tr, Xt1_val = X_tr_tree_v1[tr_idx], X_tr_tree_v1[val_idx]
    Xv1_tr, Xv1_val = X_tr_v1_full[tr_idx], X_tr_v1_full[val_idx]

    # ---- Linear Models on V2 Primary ----
    for C_val, name in [(0.005, "LR_C0.005"), (0.01, "LR_C0.01"),
                        (0.02, "LR_C0.02"), (0.05, "LR_C0.05")]:
        lr = LogisticRegression(C=C_val, max_iter=1000, random_state=SEED)
        lr.fit(Xp_tr, y_tr)
        oof_matrix[name][val_idx] = lr.predict_proba(Xp_val)
        test_matrix[name] += lr.predict_proba(X_te_primary) / N_FOLDS

    # Ridge
    ridge = RidgeClassifier(alpha=30.0, random_state=SEED)
    ridge.fit(Xp_tr, y_tr)
    df_val = ridge.decision_function(Xp_val) / 1.5
    df_val_exp = np.exp(df_val - df_val.max(axis=1, keepdims=True))
    oof_matrix["Ridge_a30"][val_idx] = df_val_exp / df_val_exp.sum(axis=1, keepdims=True)
    df_te = ridge.decision_function(X_te_primary) / 1.5
    df_te_exp = np.exp(df_te - df_te.max(axis=1, keepdims=True))
    test_matrix["Ridge_a30"] += (df_te_exp / df_te_exp.sum(axis=1, keepdims=True)) / N_FOLDS

    # Calibrated LinearSVC
    lsvc = CalibratedClassifierCV(LinearSVC(C=0.005, random_state=SEED, max_iter=2000), cv=3)
    lsvc.fit(Xp_tr, y_tr)
    oof_matrix["Cal_LinearSVC"][val_idx] = lsvc.predict_proba(Xp_val)
    test_matrix["Cal_LinearSVC"] += lsvc.predict_proba(X_te_primary) / N_FOLDS

    # ---- Multi-seed FastMLP ----
    for seed_val in MLP_SEEDS:
        name = f"FastMLP_s{seed_val}"
        v_mlp, t_mlp = train_mlp_fold(
            Xp_tr, y_tr, Xp_val, y_val, X_te_primary,
            model_class=FastMLP, epochs=20, lr=2e-3, label_smooth=0.06, seed=seed_val
        )
        oof_matrix[name][val_idx] = v_mlp
        test_matrix[name] += t_mlp / N_FOLDS

    # DeepMLP
    v_deep, t_deep = train_mlp_fold(
        Xp_tr, y_tr, Xp_val, y_val, X_te_primary,
        model_class=DeepMLP, epochs=22, lr=1.5e-3, label_smooth=0.05, seed=42
    )
    oof_matrix["DeepMLP_s42"][val_idx] = v_deep
    test_matrix["DeepMLP_s42"] += t_deep / N_FOLDS

    # ---- V1 Diversity ----
    lr_v1 = LogisticRegression(C=0.02, max_iter=1000, random_state=SEED)
    lr_v1.fit(Xv1_tr, y_tr)
    oof_matrix["LR_V1_C0.02"][val_idx] = lr_v1.predict_proba(Xv1_val)
    test_matrix["LR_V1_C0.02"] += lr_v1.predict_proba(X_te_v1_full) / N_FOLDS

    # ---- Tree Models V2 ----
    et = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                              random_state=SEED, n_jobs=-1)
    et.fit(Xt_tr, y_tr)
    oof_matrix["ExtraTrees_V2"][val_idx] = et.predict_proba(Xt_val)
    test_matrix["ExtraTrees_V2"] += et.predict_proba(X_te_tree) / N_FOLDS

    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                                random_state=SEED, n_jobs=-1)
    rf.fit(Xt_tr, y_tr)
    oof_matrix["RandomForest_V2"][val_idx] = rf.predict_proba(Xt_val)
    test_matrix["RandomForest_V2"] += rf.predict_proba(X_te_tree) / N_FOLDS

    hgb = HistGradientBoostingClassifier(l2_regularization=3.0, min_samples_leaf=15,
                                          max_iter=150, random_state=SEED)
    hgb.fit(Xp_tr, y_tr)
    oof_matrix["HistGBM_V2"][val_idx] = hgb.predict_proba(Xp_val)
    test_matrix["HistGBM_V2"] += hgb.predict_proba(X_te_primary) / N_FOLDS

    # ---- Tree Models V1 (diversity) ----
    et_v1 = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                                  random_state=SEED+1, n_jobs=-1)
    et_v1.fit(Xt1_tr, y_tr)
    oof_matrix["ExtraTrees_V1"][val_idx] = et_v1.predict_proba(Xt1_val)
    test_matrix["ExtraTrees_V1"] += et_v1.predict_proba(X_te_tree_v1) / N_FOLDS

    rf_v1 = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt",
                                    random_state=SEED+1, n_jobs=-1)
    rf_v1.fit(Xt1_tr, y_tr)
    oof_matrix["RandomForest_V1"][val_idx] = rf_v1.predict_proba(Xt1_val)
    test_matrix["RandomForest_V1"] += rf_v1.predict_proba(X_te_tree_v1) / N_FOLDS

    # GBM
    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                      min_samples_leaf=10, subsample=0.8, random_state=SEED)
    gbm.fit(Xt_tr, y_tr)
    oof_matrix["GBM_small"][val_idx] = gbm.predict_proba(Xt_val)
    test_matrix["GBM_small"] += gbm.predict_proba(X_te_tree) / N_FOLDS

    # sklearn MLP
    sk_mlp = MLPClassifier(hidden_layer_sizes=(256, 96), activation='relu',
                            alpha=0.01, learning_rate='adaptive', max_iter=200,
                            early_stopping=True, validation_fraction=0.15,
                            random_state=SEED)
    sk_mlp.fit(Xp_tr, y_tr)
    oof_matrix["sklearn_MLP"][val_idx] = sk_mlp.predict_proba(Xp_val)
    test_matrix["sklearn_MLP"] += sk_mlp.predict_proba(X_te_primary) / N_FOLDS

    elapsed = time.time() - f_t0
    print(f"    Fold {fold+1}/{N_FOLDS} completed in {elapsed:.1f}s", flush=True)

total_time = time.time() - t0
print(f"\n  Total Training Time: {total_time:.1f}s")

# ==========================================
# OOF EVALUATION
# ==========================================
print("\n[4/5] Out-Of-Fold Validation Scores:")
print("-" * 70)
print(f"{'Model':25s} | {'Accuracy':12s} | {'Comp Score':12s}")
print("-" * 70)

for m in model_names:
    acc = accuracy_score(y_train, oof_matrix[m].argmax(axis=1))
    score = max(0, acc - 0.40) / 0.60
    print(f"{m:25s} | {acc:10.4f}   | {score:10.5f}")
print("-" * 70)

# ==========================================
# SLSQP WEIGHT OPTIMIZATION
# ==========================================
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
opt_res = minimize(loss_func, np.ones(n_models)/n_models, method='SLSQP',
                   bounds=bounds, constraints=cons,
                   options={'maxiter': 500, 'ftol': 1e-12})
opt_weights = opt_res.x / np.sum(opt_res.x)

print("\n  SLSQP Optimized Weights:")
for m, w in zip(model_names, opt_weights):
    if w > 0.005:
        print(f"    {m:25s}: {w:6.4f} ({w*100:5.1f}%)")

slsqp_oof_probs = sum(opt_weights[i] * oof_list[i] for i in range(n_models))
slsqp_oof_acc = accuracy_score(y_train, slsqp_oof_probs.argmax(axis=1))
slsqp_score = max(0, slsqp_oof_acc - 0.40) / 0.60
print(f"\n  SLSQP OOF Accuracy: {slsqp_oof_acc:.4f}  (Score: {slsqp_score:.5f})")

# ==========================================
# STACKING META-LEARNER
# ==========================================
print("\n  Training Stacking Meta-Learner...")

oof_stack = np.hstack([oof_matrix[m] for m in model_names])
test_stack = np.hstack([test_matrix[m] for m in model_names])

# Add top handcrafted features too
oof_stack_full = np.hstack([oof_stack, X_tr_h2[:, :20]])
test_stack_full = np.hstack([test_stack, X_te_h2[:, :20]])

stack_oof_probs = np.zeros((n_train, 3))
stack_test_probs = np.zeros((n_test, 3))
skf_stack = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED+1)

for fold, (tr_idx, val_idx) in enumerate(skf_stack.split(oof_stack_full, y_train)):
    stacker = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
    stacker.fit(oof_stack_full[tr_idx], y_train[tr_idx])
    stack_oof_probs[val_idx] = stacker.predict_proba(oof_stack_full[val_idx])
    stack_test_probs += stacker.predict_proba(test_stack_full) / 5

stack_oof_acc = accuracy_score(y_train, stack_oof_probs.argmax(axis=1))
stack_score = max(0, stack_oof_acc - 0.40) / 0.60
print(f"  Stacker OOF Accuracy: {stack_oof_acc:.4f}  (Score: {stack_score:.5f})")

# ==========================================
# BLEND SLSQP + STACKER
# ==========================================
best_blend_acc = 0
best_alpha = 0
for alpha in np.arange(0.0, 1.01, 0.05):
    blended = alpha * slsqp_oof_probs + (1 - alpha) * stack_oof_probs
    acc = accuracy_score(y_train, blended.argmax(axis=1))
    if acc > best_blend_acc:
        best_blend_acc = acc
        best_alpha = alpha

final_blend_score = max(0, best_blend_acc - 0.40) / 0.60
print(f"\n  Best SLSQP+Stacker Blend: alpha={best_alpha:.2f}")
print(f"  Blended OOF Accuracy:    {best_blend_acc:.4f}  (Score: {final_blend_score:.5f})")

# Generate final test probs
slsqp_test_probs = sum(opt_weights[i] * test_matrix[model_names[i]] for i in range(n_models))
final_test_probs = best_alpha * slsqp_test_probs + (1 - best_alpha) * stack_test_probs

# ==========================================
# PSEUDO-LABELING
# ==========================================
print("\n[5/5] Pseudo-Labeling Pass...")

test_max_probs = final_test_probs.max(axis=1)
confidence_threshold = 0.55
high_conf_mask = test_max_probs >= confidence_threshold
n_pseudo = high_conf_mask.sum()

if n_pseudo >= 10:
    print(f"  {n_pseudo}/{n_test} test samples above confidence {confidence_threshold:.2f}")
    pseudo_labels = final_test_probs[high_conf_mask].argmax(axis=1)
    print(f"  Pseudo-label distribution: {dict(pd.Series(pseudo_labels).value_counts().sort_index())}")

    X_pseudo_p = X_te_primary[high_conf_mask]
    X_tr_aug_p = np.vstack([X_tr_primary, X_pseudo_p])
    y_train_aug = np.concatenate([y_train, pseudo_labels])

    print("  Re-training with pseudo-labeled data...")
    pseudo_test_probs_list = []

    for C_val in [0.01, 0.02, 0.05]:
        lr = LogisticRegression(C=C_val, max_iter=1000, random_state=SEED)
        lr.fit(X_tr_aug_p, y_train_aug)
        pseudo_test_probs_list.append(lr.predict_proba(X_te_primary))

    for seed_val in [42, 123]:
        torch.manual_seed(seed_val)
        model = FastMLP(in_dim=X_tr_aug_p.shape[1])
        t_X = torch.tensor(X_tr_aug_p, dtype=torch.float32)
        t_y = torch.tensor(y_train_aug, dtype=torch.long)
        t_X_te = torch.tensor(X_te_primary, dtype=torch.float32)

        loader = DataLoader(TensorDataset(t_X, t_y), batch_size=64, shuffle=True)
        opt2 = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=20, eta_min=1e-6)
        crit2 = nn.CrossEntropyLoss(label_smoothing=0.06)

        for ep in range(20):
            model.train()
            for bx, by in loader:
                opt2.zero_grad(set_to_none=True)
                loss = crit2(model(bx), by)
                loss.backward()
                opt2.step()
            sched2.step()

        model.eval()
        with torch.no_grad():
            pseudo_test_probs_list.append(torch.softmax(model(t_X_te), dim=1).numpy())

    pseudo_test_probs = np.mean(pseudo_test_probs_list, axis=0)
    pseudo_weight = 0.25
    final_test_probs_v2 = (1 - pseudo_weight) * final_test_probs + pseudo_weight * pseudo_test_probs

    pseudo_preds = final_test_probs_v2.argmax(axis=1)
    orig_preds = final_test_probs.argmax(axis=1)
    n_changed = (pseudo_preds != orig_preds).sum()
    print(f"  Pseudo-labeling changed {n_changed} test predictions")

    final_test_probs = final_test_probs_v2
else:
    print(f"  Only {n_pseudo} samples above threshold -- skipping")

# ==========================================
# GENERATE SUBMISSION
# ==========================================
final_test_preds = final_test_probs.argmax(axis=1)

sub_df = pd.DataFrame({
    sample_sub.columns[0]: test_ids,
    sample_sub.columns[1]: final_test_preds.tolist()
})

out_sub = Path(r"C:\Users\2006s\Documents\Codex\2026-09-11\create-an-image-of\outputs\illumination-classifier\submission.csv")
desktop_sub = Path(r"C:\Users\2006s\Desktop\submission.csv")

sub_df.to_csv(out_sub, index=False)
sub_df.to_csv(desktop_sub, index=False)

print(f"\n  Wrote {len(final_test_preds)} predictions to:")
print(f"    {out_sub}")
print(f"    {desktop_sub}")
print(f"\n  Test Class Distribution: {dict(pd.Series(final_test_preds).value_counts().sort_index())}")
print(f"\n  First 10 Rows:")
print(sub_df.head(10).to_string(index=False))

print("\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)
print(f"  SLSQP OOF Accuracy:        {slsqp_oof_acc:.4f} (Score: {slsqp_score:.5f})")
print(f"  Stacker OOF Accuracy:      {stack_oof_acc:.4f} (Score: {stack_score:.5f})")
print(f"  Best Combined OOF:         {best_blend_acc:.4f} (Score: {final_blend_score:.5f})")
print(f"  Previous Best LB:          0.189 (Accuracy: 0.5133)")
print(f"  Target (Top 10):           0.2167 (Accuracy: 0.53)")
print("=" * 70)
print("  READY TO UPLOAD!")
print("=" * 70)

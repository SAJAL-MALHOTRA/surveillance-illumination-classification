"""
Extract Swin Transformer (swin_t) embeddings for train and test images.
Direct 224x224 resize ensures full-frame illumination visibility (ceiling fixtures + floor).
Evaluates standalone 10-fold CV performance.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torchvision import models, transforms

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
COMP_DIR = PROJECT_DIR / "comp_output"

train_cache = COMP_DIR / "train_swin_t.npz"
test_cache = COMP_DIR / "test_swin_t.npz"

sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
test_ids = sample_sub.iloc[:, 0].tolist()
test_dir = DATA_DIR / "test"
test_paths = [str(list(test_dir.glob(f"{tid}.*"))[0]) for tid in test_ids]

train_paths = []
for cls in ['dark', 'normal', 'bright']:
    cls_dir = DATA_DIR / "train" / cls
    paths = sorted([str(cls_dir / f) for f in os.listdir(cls_dir) if f.endswith('.png') or f.endswith('.jpg')])
    train_paths.extend(paths)

y_train = np.array([0]*500 + [1]*500 + [2]*500)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

if not (train_cache.exists() and test_cache.exists()):
    print("Extracting Swin-T embeddings (full-frame 224x224)...", flush=True)
    swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    swin.head = nn.Identity()
    swin.eval()

    def get_feats(paths, desc=""):
        feats = []
        batch_tensors = []
        t0 = time.time()
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            batch_tensors.append(transform(img))
            if len(batch_tensors) == 32 or i == len(paths) - 1:
                batch = torch.stack(batch_tensors)
                with torch.no_grad():
                    f = swin(batch).numpy()
                feats.append(f)
                batch_tensors = []
                if (i + 1) % 200 == 0 or i == len(paths) - 1:
                    print(f"  {desc} {i+1}/{len(paths)} ({time.time()-t0:.1f}s)", flush=True)
        return np.vstack(feats)

    train_feats = get_feats(train_paths, "Train")
    test_feats = get_feats(test_paths, "Test")

    np.savez_compressed(train_cache, feats=train_feats)
    np.savez_compressed(test_cache, feats=test_feats)
    print("Saved Swin-T features to cache.", flush=True)
else:
    print("Loading Swin-T embeddings from cache...", flush=True)
    train_feats = np.load(train_cache)['feats']
    test_feats = np.load(test_cache)['feats']

print(f"Swin-T Feature shape: train={train_feats.shape}, test={test_feats.shape}", flush=True)

# 10-Fold CV evaluation on Swin-T alone
X_swin = normalize(train_feats)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for c_val in [0.005, 0.01, 0.02, 0.05, 0.1]:
    oof = np.zeros((1500, 3))
    for tr, val in skf.split(X_swin, y_train):
        lr = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
        lr.fit(X_swin[tr], y_train[tr])
        oof[val] = lr.predict_proba(X_swin[val])
    acc = accuracy_score(y_train, oof.argmax(axis=1))
    score = max(0, acc - 0.40) / 0.60
    print(f"Swin-T alone LR C={c_val:5.3f} -> OOF Acc: {acc:.4f} ({int(acc*1500)}/1500) Score: {score:.5f}", flush=True)

# Check correlation with ResNet-18
train_res18 = normalize(np.load(COMP_DIR / "train_resnet18.npz")['feats'])
from scipy.stats import pearsonr
# Linear model on ResNet18
oof_r18 = np.zeros((1500, 3))
for tr, val in skf.split(train_res18, y_train):
    lr = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
    lr.fit(train_res18[tr], y_train[tr])
    oof_r18[val] = lr.predict_proba(train_res18[val])

oof_swin = np.zeros((1500, 3))
for tr, val in skf.split(X_swin, y_train):
    lr = LogisticRegression(C=0.02, max_iter=1000, random_state=42)
    lr.fit(X_swin[tr], y_train[tr])
    oof_swin[val] = lr.predict_proba(X_swin[val])

r, _ = pearsonr(oof_r18.flatten(), oof_swin.flatten())
print(f"\nPearson correlation between ResNet-18 and Swin-T probabilities: {r:.4f}", flush=True)

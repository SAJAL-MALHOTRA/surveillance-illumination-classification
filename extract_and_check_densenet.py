"""
Task 5: Extract DenseNet-121 & Verify Prediction Correlation with ResNet
Only adds DenseNet if correlation with ResNet is < 0.90 to ensure true diversity.
"""

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torchvision import models, transforms

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
COMP_DIR = PROJECT_DIR / "comp_output"

train_cache = COMP_DIR / "train_densenet121.npz"
test_cache = COMP_DIR / "test_densenet121.npz"

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
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

if not (train_cache.exists() and test_cache.exists()):
    print("Extracting DenseNet-121 embeddings...")
    densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    densenet.classifier = nn.Identity()
    densenet.eval()

    def get_feats(paths):
        feats = []
        batch_tensors = []
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            batch_tensors.append(transform(img))
            if len(batch_tensors) == 32 or i == len(paths) - 1:
                batch = torch.stack(batch_tensors)
                with torch.no_grad():
                    f = densenet(batch).numpy()
                feats.append(f)
                batch_tensors = []
            if (i+1) % 300 == 0:
                print(f"  {i+1}/{len(paths)}")
        return np.vstack(feats)

    t0 = time.time()
    train_feats = get_feats(train_paths)
    print(f"Train DenseNet extracted in {time.time()-t0:.1f}s, shape: {train_feats.shape}")
    np.savez(train_cache, feats=train_feats)

    t0 = time.time()
    test_feats = get_feats(test_paths)
    print(f"Test DenseNet extracted in {time.time()-t0:.1f}s, shape: {test_feats.shape}")
    np.savez(test_cache, feats=test_feats)
else:
    print("Loading cached DenseNet-121 embeddings...")
    train_feats = np.load(train_cache)['feats']
    test_feats = np.load(test_cache)['feats']

# 2. Compute correlation with ResNet-34 OOF predictions
print("\nComparing OOF Predictions: DenseNet-121 vs ResNet-34 vs ResNet-18...")
train_r34 = normalize(np.load(COMP_DIR / "train_resnet34.npz")['feats'])
train_r18 = normalize(np.load(COMP_DIR / "train_resnet18.npz")['feats'])
train_dense = normalize(train_feats)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_dense = np.zeros((1500, 3))
oof_r34 = np.zeros((1500, 3))
oof_r18 = np.zeros((1500, 3))

for tr, val in skf.split(train_dense, y_train):
    lr_d = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    lr_d.fit(train_dense[tr], y_train[tr])
    oof_dense[val] = lr_d.predict_proba(train_dense[val])

    lr_34 = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    lr_34.fit(train_r34[tr], y_train[tr])
    oof_r34[val] = lr_34.predict_proba(train_r34[val])

    lr_18 = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    lr_18.fit(train_r18[tr], y_train[tr])
    oof_r18[val] = lr_18.predict_proba(train_r18[val])

acc_dense = accuracy_score(y_train, oof_dense.argmax(axis=1))
acc_r34 = accuracy_score(y_train, oof_r34.argmax(axis=1))
acc_r18 = accuracy_score(y_train, oof_r18.argmax(axis=1))

print(f"  DenseNet-121 Standalone Accuracy: {acc_dense:.4f}")
print(f"  ResNet-34    Standalone Accuracy: {acc_r34:.4f}")
print(f"  ResNet-18    Standalone Accuracy: {acc_r18:.4f}")

# Compute Pearson correlation across flattened probability vectors
corr_dense_r34, _ = pearsonr(oof_dense.ravel(), oof_r34.ravel())
corr_dense_r18, _ = pearsonr(oof_dense.ravel(), oof_r18.ravel())
corr_r34_r18, _ = pearsonr(oof_r34.ravel(), oof_r18.ravel())

print("\n" + "=" * 60)
print("  PEARSON PREDICTION CORRELATION MATRIX:")
print("=" * 60)
print(f"  DenseNet-121 <--> ResNet-34: {corr_dense_r34:.4f}")
print(f"  DenseNet-121 <--> ResNet-18: {corr_dense_r18:.4f}")
print(f"  ResNet-34    <--> ResNet-18: {corr_r34_r18:.4f}")

if corr_dense_r34 < 0.90:
    print(f"\n>> VERIFIED: DenseNet correlation ({corr_dense_r34:.4f}) < 0.90 threshold! High ensemble diversity confirmed!")
else:
    print(f"\n>> WARNING: DenseNet correlation ({corr_dense_r34:.4f}) >= 0.90.")

# Blended accuracy: ResNet-34 + DenseNet-121
blend_probs = 0.5 * oof_r34 + 0.5 * oof_dense
blend_acc = accuracy_score(y_train, blend_probs.argmax(axis=1))
print(f"  Blended ResNet-34 + DenseNet-121 OOF Accuracy: {blend_acc:.4f} (Gain: +{blend_acc - acc_r34:.4f})")

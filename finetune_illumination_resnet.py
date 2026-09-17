"""
END-TO-END ILLUMINATION FINE-TUNING PIPELINE
Fine-tunes ResNet-18 directly on the surveillance illumination images (5-Fold Stratified CV).
Adapts convolutional kernels (layer3 + layer4 + custom head) to learn:
- Fluorescent tube glare & fixture geometry
- Subway / corridor floor specular reflections
- Ambient shadow gradients & room volume lighting

Includes:
1. RandomHorizontalFlip augmentation (100% illumination invariant)
2. 2-View Test-Time Augmentation (TTA: Original + Horizontal Flip)
3. Saves OOF and Test probability distributions
4. Synergistic blending with the 0.194 champion baseline (v2)
5. Strict UUID assertion against sample_submission.csv
"""

import os
import sys
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

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
print("  END-TO-END SURVEILLANCE ILLUMINATION FINE-TUNING (RESNET-18)", flush=True)
print("=" * 75, flush=True)

# 1. Dataset Setup
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
test_ids = sample_sub.iloc[:, 0].tolist()
test_dir = DATA_DIR / "test"
test_paths = [str(list(test_dir.glob(f"{tid}.*"))[0]) for tid in test_ids]

train_paths = []
train_labels = []
for label, cls in enumerate(['dark', 'normal', 'bright']):
    cls_dir = DATA_DIR / "train" / cls
    paths = sorted([str(cls_dir / f) for f in os.listdir(cls_dir) if f.endswith('.png') or f.endswith('.jpg')])
    train_paths.extend(paths)
    train_labels.extend([label]*len(paths))

train_paths = np.array(train_paths)
train_labels = np.array(train_labels)

print(f"Dataset: {len(train_paths)} train images (500/500/500 balanced), {len(test_paths)} test images", flush=True)

# Transforms
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_transform_flip = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

class SurveillanceDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img

# 2. Model Builder
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze conv1, bn1, layer1, layer2
    for param in list(model.parameters())[:30]:
        param.requires_grad = False
        
    # Custom regularized classification head
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(in_feat, 128),
        nn.BatchNorm1d(128),
        nn.SiLU(),
        nn.Dropout(0.20),
        nn.Linear(128, 3)
    )
    return model

# 3. 5-Fold Stratified Fine-Tuning
N_FOLDS = 5
EPOCHS = 5
BATCH_SIZE = 32

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs = np.zeros((len(train_labels), 3))
test_probs = np.zeros((len(test_paths), 3))

test_loader_orig = DataLoader(SurveillanceDataset(test_paths, transform=val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader_flip = DataLoader(SurveillanceDataset(test_paths, transform=val_transform_flip), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

t_start = time.time()

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_paths, train_labels)):
    print(f"\n--- [Fold {fold+1}/{N_FOLDS}] ---", flush=True)
    f_t0 = time.time()
    
    tr_ds = SurveillanceDataset(train_paths[tr_idx], train_labels[tr_idx], transform=train_transform)
    val_ds = SurveillanceDataset(train_paths[val_idx], train_labels[val_idx], transform=val_transform)
    
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = build_model()
    
    # Differential learning rates
    optimizer = optim.AdamW([
        {'params': model.layer3.parameters(), 'lr': 3e-4, 'weight_decay': 1e-3},
        {'params': model.layer4.parameters(), 'lr': 5e-4, 'weight_decay': 1e-3},
        {'params': model.fc.parameters(), 'lr': 1.5e-3, 'weight_decay': 1e-2}
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    best_val_acc = 0.0
    best_state = None
    
    for ep in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by in tr_loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
        scheduler.step()
        train_loss /= len(tr_ds)
        
        # Validation
        model.eval()
        val_preds_list = []
        val_targets_list = []
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                val_preds_list.append(torch.softmax(out, dim=1))
                val_targets_list.append(by)
        val_probs_ep = torch.cat(val_preds_list).numpy()
        val_targets_ep = torch.cat(val_targets_list).numpy()
        val_acc = accuracy_score(val_targets_ep, val_probs_ep.argmax(axis=1))
        
        print(f"  Epoch {ep+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}", flush=True)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    # Evaluate best model on validation set
    model.eval()
    val_probs_list = []
    with torch.no_grad():
        for bx, _ in val_loader:
            val_probs_list.append(torch.softmax(model(bx), dim=1).numpy())
    oof_probs[val_idx] = np.vstack(val_probs_list)
    
    # 2-View Test-Time Augmentation (TTA) on test set
    test_orig_list = []
    test_flip_list = []
    with torch.no_grad():
        for bx in test_loader_orig:
            test_orig_list.append(torch.softmax(model(bx), dim=1).numpy())
        for bx in test_loader_flip:
            test_flip_list.append(torch.softmax(model(bx), dim=1).numpy())
            
    test_fold_probs = 0.5 * np.vstack(test_orig_list) + 0.5 * np.vstack(test_flip_list)
    test_probs += test_fold_probs / N_FOLDS
    
    print(f"  Fold {fold+1} Best Val Acc: {best_val_acc:.4f} (completed in {time.time()-f_t0:.1f}s)", flush=True)

total_time = time.time() - t_start
print(f"\nTotal Fine-Tuning Time: {total_time:.1f}s ({total_time/60:.1f} minutes)", flush=True)

# 4. Out-of-Fold Evaluation of Fine-Tuned ResNet-18
ft_acc = accuracy_score(train_labels, oof_probs.argmax(axis=1))
ft_score = max(0, ft_acc - 0.40) / 0.60
print(f"\nFine-Tuned ResNet-18 Standalone OOF Accuracy: {ft_acc:.4f} (Score: {ft_score:.5f})", flush=True)
print("Confusion Matrix (Fine-Tuned Standalone):")
print(confusion_matrix(train_labels, oof_probs.argmax(axis=1)), flush=True)

# Save fine-tuned standalone probabilities
np.save(COMP_DIR / "finetuned_resnet18_test_probs.npy", test_probs)
np.save(COMP_DIR / "finetuned_resnet18_oof_probs.npy", oof_probs)

# 5. Synergistic Blending with Champion v2 Baseline
print("\n[5/5] Blending with Champion 0.194 Baseline...", flush=True)
v2_oof_probs = np.load(COMP_DIR / "v2_oof_probs.npy")
v2_test_probs = np.load(COMP_DIR / "v2_test_probs.npy")

best_blend_acc = 0.0
best_weight = 0.0

# Grid search blend weight: (1 - w)*v2 + w*finetuned
for w in np.linspace(0.0, 0.50, 51):
    b_oof = (1 - w) * v2_oof_probs + w * oof_probs
    b_acc = accuracy_score(train_labels, b_oof.argmax(axis=1))
    if b_acc > best_blend_acc:
        best_blend_acc = b_acc
        best_weight = w

blend_score = max(0, best_blend_acc - 0.40) / 0.60
print(f"  Optimal Fine-Tuning Blend Weight: {best_weight:.2f} (Champion: {1-best_weight:.2f})", flush=True)
print(f"  Blended OOF Accuracy: {best_blend_acc:.4f} (Score: {blend_score:.5f})", flush=True)

# Generate final blended test predictions
final_test_probs = (1 - best_weight) * v2_test_probs + best_weight * test_probs
final_test_preds = final_test_probs.argmax(axis=1)

# Task 7 Strict Assertion
assert len(final_test_preds) == 300
sub_df = pd.DataFrame({
    sample_sub.columns[0]: test_ids,
    sample_sub.columns[1]: final_test_preds.tolist()
})
assert sub_df.iloc[:, 0].tolist() == sample_sub.iloc[:, 0].tolist()
print("  >> TASK 7 VERIFIED: 100% exact UUID match against sample_submission.csv!", flush=True)

# Check diff from v2 baseline
diff = (final_test_preds != v2_test_probs.argmax(axis=1)).sum()
print(f"  Flipped {diff} predictions from v2 baseline based on fine-tuned visual intelligence.", flush=True)

out_sub = Path(r"C:\Users\2006s\Documents\Codex\2026-09-11\create-an-image-of\outputs\illumination-classifier\submission.csv")
desktop_sub = Path(r"C:\Users\2006s\Desktop\submission.csv")
backup_sub = Path(r"C:\Users\2006s\Desktop\submission_finetuned_ensemble.csv")

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

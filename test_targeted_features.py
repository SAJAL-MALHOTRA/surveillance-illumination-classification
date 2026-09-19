"""
Task 3: Test Targeted Normal vs Bright Features
Evaluates features specifically designed to resolve the Normal (1) vs Bright (2) confusion.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import maximum_filter
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

train_paths = []
train_labels = []
for label, cls in enumerate(['dark', 'normal', 'bright']):
    cls_dir = DATA_DIR / "train" / cls
    paths = sorted([str(cls_dir / f) for f in os.listdir(cls_dir) if f.endswith('.png') or f.endswith('.jpg')])
    train_paths.extend(paths)
    train_labels.extend([label]*len(paths))

train_labels = np.array(train_labels)

# We only care about separating Normal (1) vs Bright (2)
nb_mask = (train_labels == 1) | (train_labels == 2)
nb_paths = [train_paths[i] for i in range(len(train_paths)) if nb_mask[i]]
nb_y = train_labels[nb_mask] - 1  # 0 for Normal, 1 for Bright

print(f"Extracting targeted features on {len(nb_paths)} Normal vs Bright images...")

def extract_targeted_features(p):
    img = Image.open(p).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    v = hsv[:,:,2]
    total_pixels = gray.size

    feats = {}
    
    # 1. Overexposed pixels in V channel
    feats["v_gt_240"] = (v > 240).sum() / total_pixels
    feats["v_gt_220"] = (v > 220).sum() / total_pixels
    feats["v_gt_200"] = (v > 200).sum() / total_pixels
    feats["glare_ratio"] = feats["v_gt_240"] / (feats["v_gt_200"] + 1e-6)
    
    # 2. Histogram equalization delta
    img_gray = Image.fromarray(gray.astype(np.uint8))
    img_eq = ImageOps.equalize(img_gray)
    arr_eq = np.array(img_eq, dtype=np.float32)
    feats["hist_eq_mean_diff"] = np.abs(arr_eq - gray).mean()
    feats["hist_eq_std_diff"] = np.abs(arr_eq - gray).std()
    
    # 3. Local contrast in bright regions specifically
    bright_mask = gray > np.percentile(gray, 80)
    if bright_mask.sum() > 10:
        feats["bright_region_std"] = gray[bright_mask].std()
        feats["bright_region_mean"] = gray[bright_mask].mean()
        feats["bright_region_contrast"] = feats["bright_region_std"] / (feats["bright_region_mean"] + 1e-6)
    else:
        feats["bright_region_std"] = 0
        feats["bright_region_mean"] = 0
        feats["bright_region_contrast"] = 0

    # 4. Upper row ceiling glare vs floor
    h, w = gray.shape
    upper_20 = gray[:h//5, :]
    lower_20 = gray[4*h//5:, :]
    feats["upper_max_ratio"] = upper_20.max() / (upper_20.mean() + 1e-6)
    feats["upper_lower_max_ratio"] = (upper_20.max() + 1e-6) / (lower_20.max() + 1e-6)
    
    # 5. Hotspot count (peaks with V > 230)
    local_max = maximum_filter(v, size=15) == v
    hotspots = local_max & (v > 230)
    feats["hotspot_count"] = hotspots.sum() / (total_pixels / 1000)

    # 6. Dynamic range in upper half
    upper_half = gray[:h//2, :]
    feats["upper_p95_minus_p50"] = np.percentile(upper_half, 95) - np.percentile(upper_half, 50)
    feats["upper_p99_minus_mean"] = np.percentile(upper_half, 99) - upper_half.mean()

    return feats

all_feats = []
for i, p in enumerate(nb_paths):
    all_feats.append(extract_targeted_features(p))
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(nb_paths)}")

feat_names = sorted(all_feats[0].keys())
X_targeted = np.array([[f[k] for k in feat_names] for f in all_feats])

print(f"\nExtracted {X_targeted.shape[1]} targeted features across {len(nb_paths)} Normal vs Bright images.")

# Feature statistics
print("\n" + "=" * 70)
print(f"{'Feature Name':25s} | {'Normal Mean':12s} | {'Bright Mean':12s} | {'Mutual Info':12s}")
print("-" * 70)
mi = mutual_info_classif(X_targeted, nb_y, random_state=42)

for i, fn in enumerate(feat_names):
    norm_mean = X_targeted[nb_y == 0, i].mean()
    br_mean = X_targeted[nb_y == 1, i].mean()
    print(f"{fn:25s} | {norm_mean:12.4f} | {br_mean:12.4f} | {mi[i]:12.4f}")

# Cross validation on Normal vs Bright
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
Xs = StandardScaler().fit_transform(X_targeted)
lr = LogisticRegression(C=0.1, random_state=42)
scores = cross_val_score(lr, Xs, nb_y, cv=cv, scoring='accuracy')
print(f"\nTargeted Features Alone (Binary Normal vs Bright): CV Accuracy = {scores.mean():.4f} +/- {scores.std():.4f}")

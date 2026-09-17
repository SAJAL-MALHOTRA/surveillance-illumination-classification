"""
Competition-grade illumination classifier.
Combines handcrafted illumination features + deep CNN features in an ensemble.

Strategy:
1. Extract 100+ handcrafted features (brightness, contrast, histogram, spatial, color)
2. Train GradientBoosting/RandomForest on handcrafted features
3. Train ResNet18 with PROPER augmentations (no brightness jitter!)
4. Train EfficientNet-B0 as alternative backbone
5. 5-fold stratified CV for robust evaluation
6. Final ensemble: weighted average of all models
7. TTA on CNN predictions

Usage:
    python train_competitive.py --data-dir data --output-dir comp_output
"""

import argparse
import os
import json
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.transforms import functional as TF

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════

CLASS_MAP = {"dark": 0, "normal": 1, "bright": 2}
LABEL_NAMES = ["dark", "normal", "bright"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════
# PART 1: HANDCRAFTED FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════


def extract_features(img_path):
    """Extract 100+ illumination-related features from an image."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Grayscale (luminance)
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    features = {}

    # ── 1. Basic brightness statistics ──
    features["gray_mean"] = gray.mean()
    features["gray_std"] = gray.std()
    features["gray_median"] = np.median(gray)
    features["gray_min"] = gray.min()
    features["gray_max"] = gray.max()
    features["gray_range"] = gray.max() - gray.min()
    features["gray_skew"] = _skewness(gray.ravel())
    features["gray_kurtosis"] = _kurtosis(gray.ravel())

    # Percentiles
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        features[f"gray_p{p}"] = np.percentile(gray, p)

    # ── 2. Per-channel statistics ──
    for name, ch in [("r", r), ("g", g), ("b", b)]:
        features[f"{name}_mean"] = ch.mean()
        features[f"{name}_std"] = ch.std()
        features[f"{name}_median"] = np.median(ch)
        features[f"{name}_skew"] = _skewness(ch.ravel())

    # ── 3. HSV features ──
    hsv = img.convert("HSV")
    hsv_arr = np.array(hsv, dtype=np.float32)
    h, s, v = hsv_arr[:, :, 0], hsv_arr[:, :, 1], hsv_arr[:, :, 2]

    features["v_mean"] = v.mean()
    features["v_std"] = v.std()
    features["v_median"] = np.median(v)
    features["v_skew"] = _skewness(v.ravel())
    features["v_kurtosis"] = _kurtosis(v.ravel())
    features["s_mean"] = s.mean()
    features["s_std"] = s.std()
    features["h_mean"] = h.mean()
    features["h_std"] = h.std()

    for p in [5, 10, 25, 50, 75, 90, 95]:
        features[f"v_p{p}"] = np.percentile(v, p)
        features[f"s_p{p}"] = np.percentile(s, p)

    # ── 4. Contrast metrics ──
    features["rms_contrast"] = gray.std() / (gray.mean() + 1e-8)
    if gray.max() + gray.min() > 0:
        features["michelson_contrast"] = (gray.max() - gray.min()) / (
            gray.max() + gray.min()
        )
    else:
        features["michelson_contrast"] = 0

    # Weber contrast (local)
    features["weber_contrast"] = gray.std() / (gray.mean() + 1e-8)

    # ── 5. Histogram features ──
    hist_gray, _ = np.histogram(gray.ravel(), bins=32, range=(0, 256))
    hist_gray_norm = hist_gray / hist_gray.sum()
    for i in range(32):
        features[f"hist_gray_{i}"] = hist_gray_norm[i]

    # Histogram entropy
    features["hist_entropy"] = -np.sum(
        hist_gray_norm * np.log2(hist_gray_norm + 1e-10)
    )

    # Histogram peaks and valleys
    features["hist_peak_bin"] = np.argmax(hist_gray_norm)
    features["hist_peak_value"] = hist_gray_norm.max()

    # Dark/bright pixel ratios
    total_pixels = gray.size
    features["pct_very_dark"] = (gray < 25).sum() / total_pixels
    features["pct_dark"] = (gray < 50).sum() / total_pixels
    features["pct_dim"] = (gray < 80).sum() / total_pixels
    features["pct_mid"] = ((gray >= 80) & (gray <= 180)).sum() / total_pixels
    features["pct_bright"] = (gray > 180).sum() / total_pixels
    features["pct_very_bright"] = (gray > 220).sum() / total_pixels
    features["pct_saturated"] = (gray > 245).sum() / total_pixels

    # ── 6. Spatial features ──
    h_img, w_img = gray.shape
    ch, cw = h_img // 2, w_img // 2
    margin_h, margin_w = h_img // 4, w_img // 4

    center = gray[margin_h : h_img - margin_h, margin_w : w_img - margin_w]
    features["center_mean"] = center.mean()
    features["center_std"] = center.std()

    # Edge regions
    top = gray[:margin_h, :]
    bottom = gray[h_img - margin_h :, :]
    left = gray[:, :margin_w]
    right = gray[:, w_img - margin_w :]
    edge_mean = np.mean([top.mean(), bottom.mean(), left.mean(), right.mean()])
    features["edge_mean"] = edge_mean
    features["center_edge_ratio"] = center.mean() / (edge_mean + 1e-8)

    # Quadrant brightness
    q1 = gray[: h_img // 2, : w_img // 2].mean()
    q2 = gray[: h_img // 2, w_img // 2 :].mean()
    q3 = gray[h_img // 2 :, : w_img // 2].mean()
    q4 = gray[h_img // 2 :, w_img // 2 :].mean()
    features["q1_mean"] = q1
    features["q2_mean"] = q2
    features["q3_mean"] = q3
    features["q4_mean"] = q4
    features["quadrant_std"] = np.std([q1, q2, q3, q4])

    # Top vs bottom (ceiling lights vs floor)
    top_half = gray[: h_img // 2, :].mean()
    bot_half = gray[h_img // 2 :, :].mean()
    features["top_mean"] = top_half
    features["bottom_mean"] = bot_half
    features["top_bottom_ratio"] = top_half / (bot_half + 1e-8)

    # ── 7. Color temperature ──
    features["rb_ratio"] = r.mean() / (b.mean() + 1e-8)
    features["rg_ratio"] = r.mean() / (g.mean() + 1e-8)
    features["gb_ratio"] = g.mean() / (b.mean() + 1e-8)

    # Color dominance
    rgb_max = np.argmax(arr, axis=2)
    features["pct_r_dominant"] = (rgb_max == 0).sum() / total_pixels
    features["pct_g_dominant"] = (rgb_max == 1).sum() / total_pixels
    features["pct_b_dominant"] = (rgb_max == 2).sum() / total_pixels

    # ── 8. Edge/Gradient features (illumination affects edge visibility) ──
    # Simple gradient magnitude using Sobel-like finite differences
    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    grad_mag_y = np.abs(gy)
    grad_mag_x = np.abs(gx)
    features["grad_mean"] = (grad_mag_y.mean() + grad_mag_x.mean()) / 2
    features["grad_std"] = (grad_mag_y.std() + grad_mag_x.std()) / 2
    features["grad_max"] = max(grad_mag_y.max(), grad_mag_x.max())

    # ── 9. Texture roughness (variance of local patches) ──
    patch_size = 16
    local_vars = []
    for i in range(0, h_img - patch_size, patch_size):
        for j in range(0, w_img - patch_size, patch_size):
            patch = gray[i : i + patch_size, j : j + patch_size]
            local_vars.append(patch.var())
    local_vars = np.array(local_vars)
    features["local_var_mean"] = local_vars.mean()
    features["local_var_std"] = local_vars.std()
    features["local_var_median"] = np.median(local_vars)

    # ── 10. Dynamic range in different zones ──
    # Split image into a grid and measure brightness uniformity
    grid_means = []
    gs = 4  # 4x4 grid
    ph, pw = h_img // gs, w_img // gs
    for gi in range(gs):
        for gj in range(gs):
            cell = gray[gi * ph : (gi + 1) * ph, gj * pw : (gj + 1) * pw]
            grid_means.append(cell.mean())
    grid_means = np.array(grid_means)
    features["grid_uniformity"] = 1.0 - (grid_means.std() / (grid_means.mean() + 1e-8))
    features["grid_range"] = grid_means.max() - grid_means.min()
    features["grid_entropy"] = _entropy(grid_means)

    # ── 11. Highlight/shadow analysis ──
    features["highlight_area"] = (v > 200).sum() / total_pixels
    features["shadow_area"] = (v < 50).sum() / total_pixels
    features["midtone_area"] = ((v >= 50) & (v <= 200)).sum() / total_pixels
    features["highlight_shadow_ratio"] = (
        features["highlight_area"] / (features["shadow_area"] + 1e-8)
    )

    return features


def _skewness(x):
    m = x.mean()
    s = x.std()
    if s < 1e-8:
        return 0.0
    return ((x - m) ** 3).mean() / (s**3)


def _kurtosis(x):
    m = x.mean()
    s = x.std()
    if s < 1e-8:
        return 0.0
    return ((x - m) ** 4).mean() / (s**4) - 3.0


def _entropy(x):
    x = x / (x.sum() + 1e-10)
    return -np.sum(x * np.log2(x + 1e-10))


def build_feature_matrix(image_paths, cache_path=None):
    """Extract features for all images, with optional caching."""
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["feature_names"]

    print(f"  Extracting features from {len(image_paths)} images...")
    all_features = []
    for i, path in enumerate(image_paths):
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(image_paths)}")
        feat = extract_features(path)
        all_features.append(feat)

    feature_names = sorted(all_features[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in all_features])

    if cache_path:
        np.savez(cache_path, X=X, feature_names=feature_names)
        print(f"  Cached features to {cache_path}")

    return X, feature_names


# ═══════════════════════════════════════════════════════
# PART 2: CNN MODEL & DATASET
# ═══════════════════════════════════════════════════════


class ImageDataset(Dataset):
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
        return img, str(self.paths[idx])


def get_train_transforms():
    """Augmentations for illumination classification.
    CRITICAL: No brightness/contrast jitter — that would corrupt the signal!
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(10),
        # Subtle color jitter WITHOUT brightness
        transforms.ColorJitter(brightness=0, contrast=0.05, saturation=0.05, hue=0.02),
        transforms.RandomGrayscale(p=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_tta_transforms():
    """5 TTA views — spatial only, no brightness manipulation."""
    base = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return [
        transforms.Compose(base),  # Standard
        transforms.Compose([  # HFlip
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([  # Larger view
            transforms.Resize(240),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([  # Tighter crop
            transforms.Resize(288),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([  # Five crop center
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    ]


def build_cnn(arch="resnet18", num_classes=3, dropout=0.4):
    """Build CNN with dropout regularization."""
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
    elif arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
    elif arch == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
    return model


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = torch.nn.functional.log_softmax(pred, dim=1)
        nll = -log_preds.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -log_preds.mean(dim=1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()


def train_cnn_fold(
    model, train_paths, train_labels, val_paths, val_labels,
    device, epochs=25, batch_size=32, lr=1e-3, lr_backbone=5e-5,
    patience=8, label_smoothing=0.1,
):
    """Train CNN for one fold with proper 2-phase training."""
    train_ds = ImageDataset(train_paths, train_labels, get_train_transforms())
    val_ds = ImageDataset(val_paths, val_labels, get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, pin_memory=device.type == "cuda")

    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    model = model.to(device)

    # Phase 1: Head only
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze classifier
    if hasattr(model, 'fc'):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif hasattr(model, 'classifier'):
        for p in model.classifier.parameters():
            p.requires_grad = True

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)

    print("    Phase 1: Head warmup (3 epochs)")
    for ep in range(3):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()

    # Phase 2: Full fine-tune
    for p in model.parameters():
        p.requires_grad = True

    # Differential LR
    if hasattr(model, 'fc'):
        head_params = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
    else:
        head_params = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]

    optimizer = AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr * 0.3, "weight_decay": 1e-3},
    ])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-7)

    best_acc = 0
    best_state = None
    stale = 0

    print(f"    Phase 2: Fine-tuning ({epochs} epochs, patience={patience})")
    for ep in range(epochs):
        model.train()
        train_correct = train_total = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_correct += (out.argmax(1) == lbls).sum().item()
            train_total += len(lbls)
        scheduler.step()

        # Validate
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device, dtype=torch.long)
                out = model(imgs)
                val_correct += (out.argmax(1) == lbls).sum().item()
                val_total += len(lbls)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        lr_now = optimizer.param_groups[0]["lr"]

        if (ep + 1) % 3 == 0 or val_acc > best_acc:
            print(f"      Epoch {ep+1}: train={train_acc:.4f} val={val_acc:.4f} lr={lr_now:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"      Early stopping at epoch {ep+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


@torch.no_grad()
def predict_cnn_tta(model, image_paths, device, batch_size=32):
    """Predict with TTA, return averaged probabilities."""
    model.eval()
    tta_transforms = get_tta_transforms()
    all_probs = np.zeros((len(image_paths), 3))

    for t_idx, t in enumerate(tta_transforms):
        ds = ImageDataset(image_paths, labels=[0] * len(image_paths), transform=t)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=0)
        probs_list = []
        for imgs, _ in loader:
            imgs = imgs.to(device)
            out = model(imgs).softmax(1).cpu().numpy()
            probs_list.append(out)
        all_probs += np.concatenate(probs_list, axis=0)

    all_probs /= len(tta_transforms)
    return all_probs


# ═══════════════════════════════════════════════════════
# PART 3: MAIN PIPELINE
# ═══════════════════════════════════════════════════════


def load_data(data_dir):
    """Load training and test data."""
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"

    train_paths = []
    train_labels = []
    for cls_name, cls_label in CLASS_MAP.items():
        cls_dir = train_dir / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing: {cls_dir}")
        for img_file in sorted(cls_dir.iterdir()):
            if img_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                train_paths.append(str(img_file))
                train_labels.append(cls_label)

    print(f"Training data: {len(train_paths)} images")
    for name, label in CLASS_MAP.items():
        count = sum(1 for l in train_labels if l == label)
        print(f"  {label} ({name}): {count}")

    # Test data
    test_dir = data_dir / "test"
    test_paths = []
    test_ids = []

    # Check for test.csv
    test_csv = data_dir / "test.csv"
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        id_col = df.columns[0]
        for img_id in df[id_col]:
            img_id = str(img_id)
            for ext in [".png", ".jpg", ".jpeg"]:
                p = test_dir / f"{img_id}{ext}"
                if p.exists():
                    test_paths.append(str(p))
                    test_ids.append(img_id)
                    break
    else:
        for f in sorted(test_dir.iterdir()):
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                test_paths.append(str(f))
                test_ids.append(f.stem)

    print(f"Test data: {len(test_paths)} images")

    return (
        np.array(train_paths),
        np.array(train_labels),
        np.array(test_paths),
        test_ids,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="comp_output")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cnn-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-cnn", action="store_true", help="Skip CNN, use only tabular models")
    parser.add_argument("--archs", nargs="+", default=["resnet18", "resnet34"], help="CNN architectures")
    args = parser.parse_args()

    seed_everything()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    train_paths, train_labels, test_paths, test_ids = load_data(args.data_dir)
    n_train = len(train_paths)
    n_test = len(test_paths)

    # ═══════════════════════════════════════════
    # STAGE 1: Handcrafted features + ML models
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 1: Handcrafted features + ML ensemble")
    print("=" * 60)

    X_train, feat_names = build_feature_matrix(
        train_paths, cache_path=str(output / "train_features.npz")
    )
    X_test, _ = build_feature_matrix(
        test_paths, cache_path=str(output / "test_features.npz")
    )
    y_train = train_labels

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Try multiple models with CV
    print("\n  Cross-validation results:")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)

    ml_models = {
        "GBM_200": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=SEED
        ),
        "GBM_500": GradientBoostingClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=SEED
        ),
        "RF_500": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3,
            random_state=SEED, n_jobs=-1
        ),
        "RF_1000": RandomForestClassifier(
            n_estimators=1000, max_depth=20, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1
        ),
        "SVM_rbf": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=SEED),
        "SVM_poly": SVC(kernel="poly", degree=3, C=1, probability=True, random_state=SEED),
    }

    cv_scores = {}
    for name, model in ml_models.items():
        X_input = X_train_scaled if "SVM" in name else X_train
        scores = cross_val_score(model, X_input, y_train, cv=skf, scoring="accuracy")
        cv_scores[name] = scores.mean()
        print(f"    {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Train best models on full data + get OOF predictions for stacking
    print("\n  Training models and generating OOF predictions...")
    oof_preds_ml = {}  # {model_name: oof_probs}
    test_preds_ml = {}  # {model_name: test_probs}

    # Select top models
    top_models = sorted(cv_scores.items(), key=lambda x: -x[1])[:4]
    print(f"  Top models: {[m[0] + f' ({m[1]:.4f})' for m in top_models]}")

    for model_name, _ in top_models:
        model = ml_models[model_name]
        X_input_train = X_train_scaled if "SVM" in model_name else X_train
        X_input_test = X_test_scaled if "SVM" in model_name else X_test

        # OOF predictions
        oof_probs = np.zeros((n_train, 3))
        test_probs = np.zeros((n_test, 3))

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_input_train, y_train)):
            model_clone = _clone_model(ml_models[model_name])
            model_clone.fit(X_input_train[tr_idx], y_train[tr_idx])
            oof_probs[val_idx] = model_clone.predict_proba(X_input_train[val_idx])
            test_probs += model_clone.predict_proba(X_input_test) / args.n_folds

        oof_preds_ml[model_name] = oof_probs
        test_preds_ml[model_name] = test_probs

        oof_acc = accuracy_score(y_train, oof_probs.argmax(1))
        print(f"    {model_name} OOF accuracy: {oof_acc:.4f}")

    # ═══════════════════════════════════════════
    # STAGE 2: CNN models (if not skipped)
    # ═══════════════════════════════════════════
    oof_preds_cnn = {}
    test_preds_cnn = {}

    if not args.skip_cnn:
        print("\n" + "=" * 60)
        print("STAGE 2: CNN models with 5-fold CV + TTA")
        print("=" * 60)

        for arch in args.archs:
            print(f"\n  Architecture: {arch}")
            oof_probs = np.zeros((n_train, 3))
            test_probs = np.zeros((n_test, 3))

            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(train_paths, y_train)):
                print(f"\n  Fold {fold_idx + 1}/{args.n_folds}")
                model = build_cnn(arch=arch, dropout=0.4)
                model, val_acc = train_cnn_fold(
                    model,
                    train_paths[tr_idx], y_train[tr_idx],
                    train_paths[val_idx], y_train[val_idx],
                    device=device,
                    epochs=args.cnn_epochs,
                    batch_size=args.batch_size,
                    patience=8,
                    label_smoothing=0.1,
                )
                print(f"      Fold {fold_idx+1} best val acc: {val_acc:.4f}")

                # OOF predictions with TTA
                oof_probs[val_idx] = predict_cnn_tta(model, train_paths[val_idx], device, args.batch_size)
                # Test predictions with TTA
                test_probs += predict_cnn_tta(model, test_paths, device, args.batch_size) / args.n_folds

                # Save fold model
                torch.save(model.state_dict(), output / f"{arch}_fold{fold_idx}.pt")

            oof_preds_cnn[arch] = oof_probs
            test_preds_cnn[arch] = test_probs

            oof_acc = accuracy_score(y_train, oof_probs.argmax(1))
            print(f"\n  {arch} OOF accuracy: {oof_acc:.4f}")

    # ═══════════════════════════════════════════
    # STAGE 3: Ensemble with stacking
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 3: Ensemble")
    print("=" * 60)

    # Collect all OOF predictions
    all_oof = {}
    all_test = {}
    all_oof.update(oof_preds_ml)
    all_oof.update(oof_preds_cnn)
    all_test.update(test_preds_ml)
    all_test.update(test_preds_cnn)

    print(f"  Ensembling {len(all_oof)} models: {list(all_oof.keys())}")

    # Method 1: Simple average
    avg_oof = np.mean(list(all_oof.values()), axis=0)
    avg_test = np.mean(list(all_test.values()), axis=0)
    avg_acc = accuracy_score(y_train, avg_oof.argmax(1))
    print(f"  Simple average OOF accuracy: {avg_acc:.4f}")

    # Method 2: Weighted average (weights from OOF performance)
    weights = {}
    for name, probs in all_oof.items():
        acc = accuracy_score(y_train, probs.argmax(1))
        # Weight = (acc - 0.33)^2 to emphasize better models
        weights[name] = max(0, (acc - 0.33)) ** 2
    total_w = sum(weights.values())
    if total_w > 0:
        for k in weights:
            weights[k] /= total_w

    weighted_oof = np.zeros((n_train, 3))
    weighted_test = np.zeros((n_test, 3))
    for name in all_oof:
        weighted_oof += weights[name] * all_oof[name]
        weighted_test += weights[name] * all_test[name]
    weighted_acc = accuracy_score(y_train, weighted_oof.argmax(1))
    print(f"  Weighted average OOF accuracy: {weighted_acc:.4f}")
    print(f"  Weights: { {k: f'{v:.3f}' for k, v in weights.items()} }")

    # Method 3: Stacking with LogisticRegression
    stack_X_train = np.hstack([all_oof[name] for name in sorted(all_oof.keys())])
    stack_X_test = np.hstack([all_test[name] for name in sorted(all_test.keys())])

    stacker = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    stack_scores = cross_val_score(stacker, stack_X_train, y_train, cv=skf, scoring="accuracy")
    print(f"  Stacking (LR) CV accuracy: {stack_scores.mean():.4f} (+/- {stack_scores.std():.4f})")

    stacker.fit(stack_X_train, y_train)
    stacked_test = stacker.predict(stack_X_test)
    stacked_oof = stacker.predict(stack_X_train)
    stacked_acc = accuracy_score(y_train, stacked_oof)
    print(f"  Stacking train accuracy: {stacked_acc:.4f}")

    # Pick best ensemble method
    best_method = max(
        [("avg", avg_acc, avg_test), ("weighted", weighted_acc, weighted_test), ("stacking", stack_scores.mean(), None)],
        key=lambda x: x[1],
    )
    print(f"\n  Best method: {best_method[0]} (OOF acc={best_method[1]:.4f})")

    # Generate final predictions
    if best_method[0] == "stacking":
        final_preds = stacker.predict(stack_X_test)
    elif best_method[0] == "weighted":
        final_preds = weighted_test.argmax(1)
    else:
        final_preds = avg_test.argmax(1)

    # Also save individual model predictions for manual ensembling
    best_score = best_method[1]
    score = max(0, best_score - 0.40) / 0.60
    print(f"\n  Estimated competition score: {score:.4f}")
    if best_score >= 0.90:
        print("  Rating: ★★★ EXCELLENT")
    elif best_score >= 0.70:
        print("  Rating: ★★ GOOD")
    elif best_score >= 0.53:
        print("  Rating: ★ TOP 10 POTENTIAL")
    else:
        print("  Rating: NEEDS IMPROVEMENT")

    # ── Save submission ──
    sample_sub = Path(args.data_dir) / "sample_submission.csv"
    if sample_sub.exists():
        cols = pd.read_csv(sample_sub, nrows=0).columns.tolist()
        sub = pd.DataFrame({cols[0]: test_ids, cols[1]: final_preds.tolist()})
    else:
        sub = pd.DataFrame({"id": test_ids, "label": final_preds.tolist()})

    sub_path = output / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\n  Submission saved: {sub_path} ({len(sub)} predictions)")

    # Distribution check
    print("  Predicted distribution:")
    for label in [0, 1, 2]:
        count = (final_preds == label).sum()
        print(f"    {label} ({LABEL_NAMES[label]}): {count} ({count / len(final_preds) * 100:.1f}%)")

    # Save all individual submissions too
    for name in sorted(all_test.keys()):
        preds = all_test[name].argmax(1)
        individual_sub = pd.DataFrame({"id": test_ids, "label": preds.tolist()})
        individual_sub.to_csv(output / f"submission_{name}.csv", index=False)

    # Save validation report
    report = {
        "cv_scores": {k: float(v) for k, v in cv_scores.items()},
        "ensemble_methods": {
            "simple_avg": float(avg_acc),
            "weighted_avg": float(weighted_acc),
            "stacking_cv": float(stack_scores.mean()),
        },
        "best_method": best_method[0],
        "best_oof_accuracy": float(best_method[1]),
        "estimated_score": float(score),
        "weights": {k: float(v) for k, v in weights.items()},
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {output / 'validation_report.json'}")

    # Confusion matrix
    if best_method[0] == "stacking":
        cm_preds = stacked_oof
    elif best_method[0] == "weighted":
        cm_preds = weighted_oof.argmax(1)
    else:
        cm_preds = avg_oof.argmax(1)

    print(f"\n  OOF Classification Report ({best_method[0]}):")
    print(classification_report(y_train, cm_preds, target_names=LABEL_NAMES))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_train, cm_preds))

    print("\n" + "=" * 60)
    print("DONE! Check comp_output/ for submissions.")
    print("=" * 60)


def _clone_model(model):
    """Clone a sklearn model."""
    from sklearn.base import clone
    return clone(model)


if __name__ == "__main__":
    main()

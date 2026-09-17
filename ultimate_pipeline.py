"""
ULTIMATE COMPETITION PIPELINE v3
=================================
Improvements over the best scoring run_aligned_10fold_grandmaster.py (0.189 LB):

1. Extract ResNet34 embeddings (different arch, more capacity than ResNet18)
2. Extract ENHANCED handcrafted features v2 (LAB, LBP-like, Fourier, more spatial)
3. Multi-seed FastMLP (5 seeds) to reduce variance of dominant model
4. Stacking meta-learner on top of OOF predictions
5. More diverse model families (SGD, Nearest Centroid, wider LR sweep)
6. Pseudo-labeling on high-confidence test predictions
7. Careful hyperparameter tuning based on OOF feedback
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
from scipy.fft import fft2
from PIL import Image, ImageFilter
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier, 
    HistGradientBoostingClassifier, GradientBoostingClassifier,
    BaggingClassifier
)
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms

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

# ==========================================
# STEP 0: HELPER FUNCTIONS
# ==========================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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
    x = np.asarray(x, dtype=np.float64)
    x = x / (x.sum() + 1e-10)
    return -np.sum(x * np.log2(x + 1e-10))

# ==========================================
# STEP 1: EXTRACT RESNET34 EMBEDDINGS
# ==========================================

def extract_deep_embeddings(model_name, train_dir, test_paths, train_cache, test_cache):
    """Extract embeddings from a pretrained torchvision model."""
    if os.path.exists(train_cache) and os.path.exists(test_cache):
        print(f"  [CACHED] {model_name} embeddings already exist")
        tr = np.load(train_cache)['feats']
        te = np.load(test_cache)['feats']
        return tr, te
    
    print(f"  Extracting {model_name} embeddings...")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    if model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Identity()
        feat_dim = 512
    elif model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
        feat_dim = 512
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Identity()
        feat_dim = 1280
    elif model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Identity()
        feat_dim = 1024
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.eval()
    
    def get_feats(image_paths):
        all_feats = []
        for i, p in enumerate(image_paths):
            img = Image.open(p).convert('RGB')
            t = transform(img).unsqueeze(0)
            with torch.no_grad():
                f = model(t).squeeze().numpy()
            all_feats.append(f)
            if (i+1) % 100 == 0:
                print(f"    {i+1}/{len(image_paths)}")
        return np.array(all_feats)
    
    # Train: dark/normal/bright order
    train_paths = []
    for cls in ['dark', 'normal', 'bright']:
        cls_dir = Path(train_dir) / cls
        paths = sorted([cls_dir / f for f in os.listdir(cls_dir) if f.endswith('.png') or f.endswith('.jpg')])
        train_paths.extend(paths)
    
    print(f"    Extracting {len(train_paths)} train embeddings...")
    train_feats = get_feats(train_paths)
    np.savez(train_cache, feats=train_feats)
    
    print(f"    Extracting {len(test_paths)} test embeddings...")
    test_feats = get_feats(test_paths)
    np.savez(test_cache, feats=test_feats)
    
    print(f"    {model_name}: train={train_feats.shape}, test={test_feats.shape}")
    return train_feats, test_feats


# ==========================================
# STEP 2: ENHANCED HANDCRAFTED FEATURES V2
# ==========================================

def extract_features_v2(img_path):
    """Enhanced illumination features with LAB, LBP-like, Fourier, extended spatial."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    h_img, w_img = gray.shape
    total_pixels = gray.size
    
    features = {}
    
    # ---- ORIGINAL FEATURES (kept) ----
    
    # 1. Basic brightness
    features["gray_mean"] = gray.mean()
    features["gray_std"] = gray.std()
    features["gray_median"] = np.median(gray)
    features["gray_min"] = gray.min()
    features["gray_max"] = gray.max()
    features["gray_range"] = gray.max() - gray.min()
    features["gray_skew"] = _skewness(gray.ravel())
    features["gray_kurtosis"] = _kurtosis(gray.ravel())
    
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        features[f"gray_p{p}"] = np.percentile(gray, p)
    
    # 2. Per-channel
    for name, ch in [("r", r), ("g", g), ("b", b)]:
        features[f"{name}_mean"] = ch.mean()
        features[f"{name}_std"] = ch.std()
        features[f"{name}_median"] = np.median(ch)
        features[f"{name}_skew"] = _skewness(ch.ravel())
    
    # 3. HSV
    hsv = img.convert("HSV")
    hsv_arr = np.array(hsv, dtype=np.float32)
    h_ch, s_ch, v_ch = hsv_arr[:, :, 0], hsv_arr[:, :, 1], hsv_arr[:, :, 2]
    
    features["v_mean"] = v_ch.mean()
    features["v_std"] = v_ch.std()
    features["v_median"] = np.median(v_ch)
    features["v_skew"] = _skewness(v_ch.ravel())
    features["v_kurtosis"] = _kurtosis(v_ch.ravel())
    features["s_mean"] = s_ch.mean()
    features["s_std"] = s_ch.std()
    features["h_mean"] = h_ch.mean()
    features["h_std"] = h_ch.std()
    
    for p in [5, 10, 25, 50, 75, 90, 95]:
        features[f"v_p{p}"] = np.percentile(v_ch, p)
        features[f"s_p{p}"] = np.percentile(s_ch, p)
    
    # 4. Contrast
    features["rms_contrast"] = gray.std() / (gray.mean() + 1e-8)
    if gray.max() + gray.min() > 0:
        features["michelson_contrast"] = (gray.max() - gray.min()) / (gray.max() + gray.min())
    else:
        features["michelson_contrast"] = 0
    features["weber_contrast"] = gray.std() / (gray.mean() + 1e-8)
    
    # 5. Histogram
    hist_gray, _ = np.histogram(gray.ravel(), bins=32, range=(0, 256))
    hist_gray_norm = hist_gray / hist_gray.sum()
    for i in range(32):
        features[f"hist_gray_{i}"] = hist_gray_norm[i]
    features["hist_entropy"] = -np.sum(hist_gray_norm * np.log2(hist_gray_norm + 1e-10))
    features["hist_peak_bin"] = np.argmax(hist_gray_norm)
    features["hist_peak_value"] = hist_gray_norm.max()
    
    features["pct_very_dark"] = (gray < 25).sum() / total_pixels
    features["pct_dark"] = (gray < 50).sum() / total_pixels
    features["pct_dim"] = (gray < 80).sum() / total_pixels
    features["pct_mid"] = ((gray >= 80) & (gray <= 180)).sum() / total_pixels
    features["pct_bright"] = (gray > 180).sum() / total_pixels
    features["pct_very_bright"] = (gray > 220).sum() / total_pixels
    features["pct_saturated"] = (gray > 245).sum() / total_pixels
    
    # 6. Spatial
    ch_h, cw = h_img // 2, w_img // 2
    margin_h, margin_w = h_img // 4, w_img // 4
    center = gray[margin_h:h_img-margin_h, margin_w:w_img-margin_w]
    features["center_mean"] = center.mean()
    features["center_std"] = center.std()
    
    top = gray[:margin_h, :]
    bottom = gray[h_img-margin_h:, :]
    left = gray[:, :margin_w]
    right = gray[:, w_img-margin_w:]
    edge_mean = np.mean([top.mean(), bottom.mean(), left.mean(), right.mean()])
    features["edge_mean"] = edge_mean
    features["center_edge_ratio"] = center.mean() / (edge_mean + 1e-8)
    
    q1 = gray[:h_img//2, :w_img//2].mean()
    q2 = gray[:h_img//2, w_img//2:].mean()
    q3 = gray[h_img//2:, :w_img//2].mean()
    q4 = gray[h_img//2:, w_img//2:].mean()
    features["q1_mean"] = q1
    features["q2_mean"] = q2
    features["q3_mean"] = q3
    features["q4_mean"] = q4
    features["quadrant_std"] = np.std([q1, q2, q3, q4])
    
    top_half = gray[:h_img//2, :].mean()
    bot_half = gray[h_img//2:, :].mean()
    features["top_mean"] = top_half
    features["bottom_mean"] = bot_half
    features["top_bottom_ratio"] = top_half / (bot_half + 1e-8)
    
    # 7. Color temperature
    features["rb_ratio"] = r.mean() / (b.mean() + 1e-8)
    features["rg_ratio"] = r.mean() / (g.mean() + 1e-8)
    features["gb_ratio"] = g.mean() / (b.mean() + 1e-8)
    
    rgb_max = np.argmax(arr, axis=2)
    features["pct_r_dominant"] = (rgb_max == 0).sum() / total_pixels
    features["pct_g_dominant"] = (rgb_max == 1).sum() / total_pixels
    features["pct_b_dominant"] = (rgb_max == 2).sum() / total_pixels
    
    # 8. Gradients
    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    grad_mag_y = np.abs(gy)
    grad_mag_x = np.abs(gx)
    features["grad_mean"] = (grad_mag_y.mean() + grad_mag_x.mean()) / 2
    features["grad_std"] = (grad_mag_y.std() + grad_mag_x.std()) / 2
    features["grad_max"] = max(grad_mag_y.max(), grad_mag_x.max())
    
    # 9. Local variance (texture)
    patch_size = 16
    local_vars = []
    for i in range(0, h_img - patch_size, patch_size):
        for j in range(0, w_img - patch_size, patch_size):
            patch = gray[i:i+patch_size, j:j+patch_size]
            local_vars.append(patch.var())
    local_vars = np.array(local_vars)
    features["local_var_mean"] = local_vars.mean()
    features["local_var_std"] = local_vars.std()
    features["local_var_median"] = np.median(local_vars)
    
    # 10. Grid uniformity
    grid_means = []
    gs = 4
    ph, pw = h_img // gs, w_img // gs
    for gi in range(gs):
        for gj in range(gs):
            cell = gray[gi*ph:(gi+1)*ph, gj*pw:(gj+1)*pw]
            grid_means.append(cell.mean())
    grid_means = np.array(grid_means)
    features["grid_uniformity"] = 1.0 - (grid_means.std() / (grid_means.mean() + 1e-8))
    features["grid_range"] = grid_means.max() - grid_means.min()
    features["grid_entropy"] = _entropy(grid_means)
    
    # 11. Highlight/shadow
    features["highlight_area"] = (v_ch > 200).sum() / total_pixels
    features["shadow_area"] = (v_ch < 50).sum() / total_pixels
    features["midtone_area"] = ((v_ch >= 50) & (v_ch <= 200)).sum() / total_pixels
    features["highlight_shadow_ratio"] = features["highlight_area"] / (features["shadow_area"] + 1e-8)
    
    # ---- NEW V2 FEATURES ----
    
    # 12. LAB color space features (perceptually uniform - key for lighting)
    # Manual RGB->LAB approximation (avoid dependency on cv2)
    # Use linear RGB -> XYZ -> Lab
    rgb_lin = arr / 255.0
    # sRGB linearization
    rgb_lin = np.where(rgb_lin > 0.04045, ((rgb_lin + 0.055) / 1.055) ** 2.4, rgb_lin / 12.92)
    
    X = 0.4124564 * rgb_lin[:,:,0] + 0.3575761 * rgb_lin[:,:,1] + 0.1804375 * rgb_lin[:,:,2]
    Y = 0.2126729 * rgb_lin[:,:,0] + 0.7151522 * rgb_lin[:,:,1] + 0.0721750 * rgb_lin[:,:,2]
    Z = 0.0193339 * rgb_lin[:,:,0] + 0.1191920 * rgb_lin[:,:,1] + 0.9503041 * rgb_lin[:,:,2]
    
    # D65 reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    def f_lab(t):
        delta = 6/29
        return np.where(t > delta**3, t**(1/3), t/(3*delta**2) + 4/29)
    
    fx = f_lab(X / Xn)
    fy = f_lab(Y / Yn)
    fz = f_lab(Z / Zn)
    
    L = 116 * fy - 16  # Lightness
    a_lab = 500 * (fx - fy)  # green-red
    b_lab = 200 * (fy - fz)  # blue-yellow
    
    features["lab_L_mean"] = L.mean()
    features["lab_L_std"] = L.std()
    features["lab_L_skew"] = _skewness(L.ravel())
    features["lab_L_kurtosis"] = _kurtosis(L.ravel())
    features["lab_a_mean"] = a_lab.mean()
    features["lab_a_std"] = a_lab.std()
    features["lab_b_mean"] = b_lab.mean()
    features["lab_b_std"] = b_lab.std()
    
    for p in [5, 25, 50, 75, 95]:
        features[f"lab_L_p{p}"] = np.percentile(L, p)
    
    # LAB histogram
    hist_L, _ = np.histogram(L.ravel(), bins=16, range=(0, 100))
    hist_L_norm = hist_L / (hist_L.sum() + 1e-10)
    for i in range(16):
        features[f"hist_L_{i}"] = hist_L_norm[i]
    features["hist_L_entropy"] = -np.sum(hist_L_norm * np.log2(hist_L_norm + 1e-10))
    
    # Chroma (colorfulness under different illumination)
    chroma = np.sqrt(a_lab**2 + b_lab**2)
    features["chroma_mean"] = chroma.mean()
    features["chroma_std"] = chroma.std()
    
    # 13. LBP-like texture features (simplified - no cv2 needed)
    # Compare center pixel to 4-neighbors to get a texture code
    gray_small = gray[::4, ::4]  # Downsample for speed
    h_s, w_s = gray_small.shape
    if h_s > 2 and w_s > 2:
        center_px = gray_small[1:-1, 1:-1]
        # 4-neighbor comparisons
        up = (gray_small[:-2, 1:-1] >= center_px).astype(np.float32)
        down = (gray_small[2:, 1:-1] >= center_px).astype(np.float32)
        left_px = (gray_small[1:-1, :-2] >= center_px).astype(np.float32)
        right_px = (gray_small[1:-1, 2:] >= center_px).astype(np.float32)
        
        lbp_code = up * 1 + down * 2 + left_px * 4 + right_px * 8  # 0-15
        lbp_hist, _ = np.histogram(lbp_code.ravel(), bins=16, range=(0, 16))
        lbp_hist_norm = lbp_hist / (lbp_hist.sum() + 1e-10)
        for i in range(16):
            features[f"lbp_{i}"] = lbp_hist_norm[i]
        features["lbp_entropy"] = -np.sum(lbp_hist_norm * np.log2(lbp_hist_norm + 1e-10))
        features["lbp_uniformity"] = lbp_hist_norm.max()
    else:
        for i in range(16):
            features[f"lbp_{i}"] = 0.0
        features["lbp_entropy"] = 0.0
        features["lbp_uniformity"] = 0.0
    
    # 14. Fourier domain features (frequency content differs with lighting)
    # Low-frequency dominant = uniform lighting, high-frequency = textured/varied
    gray_resized = np.array(Image.fromarray(gray.astype(np.uint8)).resize((64, 64)), dtype=np.float32)
    fft_mag = np.abs(fft2(gray_resized))
    fft_mag = np.fft.fftshift(fft_mag)
    
    center_y, center_x = 32, 32
    Y_grid, X_grid = np.ogrid[:64, :64]
    dist = np.sqrt((Y_grid - center_y)**2 + (X_grid - center_x)**2)
    
    # Radial frequency bands
    low_freq = fft_mag[dist < 8].sum()
    mid_freq = fft_mag[(dist >= 8) & (dist < 20)].sum()
    high_freq = fft_mag[dist >= 20].sum()
    total_freq = low_freq + mid_freq + high_freq + 1e-10
    
    features["fft_low_ratio"] = low_freq / total_freq
    features["fft_mid_ratio"] = mid_freq / total_freq
    features["fft_high_ratio"] = high_freq / total_freq
    features["fft_low_mid_ratio"] = low_freq / (mid_freq + 1e-10)
    features["fft_total_energy"] = np.log1p(total_freq)
    
    # 15. Extended spatial features - 8x8 grid for finer spatial resolution
    grid_means_8 = []
    gs8 = 8
    ph8, pw8 = h_img // gs8, w_img // gs8
    for gi in range(gs8):
        for gj in range(gs8):
            cell = gray[gi*ph8:(gi+1)*ph8, gj*pw8:(gj+1)*pw8]
            grid_means_8.append(cell.mean())
    grid_means_8 = np.array(grid_means_8)
    
    # Upper row brightness (where ceiling lights typically are)
    features["upper_row_mean"] = grid_means_8[:8].mean()
    features["lower_row_mean"] = grid_means_8[-8:].mean()
    features["upper_lower_diff"] = features["upper_row_mean"] - features["lower_row_mean"]
    
    # Brightness of the brightest cell vs average (light source detection)
    features["max_cell_ratio"] = grid_means_8.max() / (grid_means_8.mean() + 1e-8)
    features["grid8_std"] = grid_means_8.std()
    features["grid8_range"] = grid_means_8.max() - grid_means_8.min()
    
    # Count of bright cells (potential light sources)
    features["n_bright_cells"] = (grid_means_8 > grid_means_8.mean() + grid_means_8.std()).sum()
    features["n_dark_cells"] = (grid_means_8 < grid_means_8.mean() - grid_means_8.std()).sum()
    
    # 16. Color consistency (how uniform is color across the image)
    for name, ch in [("r", r), ("g", g), ("b", b)]:
        ch_grid = []
        for gi in range(4):
            for gj in range(4):
                cell = ch[gi*ph:(gi+1)*ph, gj*pw:(gj+1)*pw]
                ch_grid.append(cell.mean())
        features[f"{name}_grid_std"] = np.std(ch_grid)
    
    # 17. Brightness derivative features (how quickly brightness changes spatially)
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    features["row_brightness_std"] = row_means.std()
    features["col_brightness_std"] = col_means.std()
    features["row_brightness_range"] = row_means.max() - row_means.min()
    features["col_brightness_range"] = col_means.max() - col_means.min()
    
    # Gradient of row/col means (brightness transition smoothness)
    row_grad = np.abs(np.diff(row_means))
    col_grad = np.abs(np.diff(col_means))
    features["row_grad_mean"] = row_grad.mean()
    features["col_grad_mean"] = col_grad.mean()
    features["row_grad_max"] = row_grad.max()
    features["col_grad_max"] = col_grad.max()
    
    # 18. Bimodality coefficient (helps detect mixed lighting)
    n = gray.size
    sk = _skewness(gray.ravel())
    ku = _kurtosis(gray.ravel())
    features["bimodality_coeff"] = (sk**2 + 1) / (ku + 3 * (n-1)**2 / ((n-2)*(n-3)) + 1e-8)
    
    # 19. V-channel histogram features (more granular)
    hist_v, _ = np.histogram(v_ch.ravel(), bins=16, range=(0, 256))
    hist_v_norm = hist_v / (hist_v.sum() + 1e-10)
    for i in range(16):
        features[f"hist_v_{i}"] = hist_v_norm[i]
    
    # 20. Inter-quartile range features
    features["gray_iqr"] = np.percentile(gray, 75) - np.percentile(gray, 25)
    features["v_iqr"] = np.percentile(v_ch, 75) - np.percentile(v_ch, 25)
    features["L_iqr"] = np.percentile(L, 75) - np.percentile(L, 25)
    
    # 21. Ratio features that might help Normal vs Bright
    features["mean_over_p95"] = gray.mean() / (np.percentile(gray, 95) + 1e-8)
    features["std_over_mean"] = gray.std() / (gray.mean() + 1e-8)
    features["p95_minus_p5"] = np.percentile(gray, 95) - np.percentile(gray, 5)
    features["center_over_global"] = center.mean() / (gray.mean() + 1e-8)
    
    return features


def build_feature_matrix_v2(image_paths, cache_path=None):
    """Extract V2 features for all images, with caching."""
    if cache_path and os.path.exists(cache_path):
        print(f"  [CACHED] V2 features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['X'], list(data['feature_names'])
    
    print(f"  Extracting V2 features from {len(image_paths)} images...")
    all_features = []
    for i, path in enumerate(image_paths):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(image_paths)}")
        feat = extract_features_v2(path)
        all_features.append(feat)
    
    feature_names = sorted(all_features[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in all_features])
    
    if cache_path:
        np.savez(cache_path, X=X, feature_names=np.array(feature_names))
        print(f"  Cached {X.shape[1]} V2 features to {cache_path}")
    
    return X, feature_names


# ==========================================
# STEP 3: PYTORCH MLP MODELS
# ==========================================

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

class DeepMLP(nn.Module):
    """Deeper MLP with 3 hidden layers."""
    def __init__(self, in_dim=1920, num_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class WideMLP(nn.Module):
    """Wide MLP with larger hidden layer."""
    def __init__(self, in_dim=1920, num_classes=3, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 384),
            nn.BatchNorm1d(384),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp_fold(X_tr, y_tr, X_val, y_val, X_te, model_class=FastMLP, 
                   epochs=25, lr=2e-3, label_smooth=0.06, seed=42):
    """Train a single MLP fold with a specific seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model_class(in_dim=X_tr.shape[1])
    t_X_tr = torch.tensor(X_tr, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr, dtype=torch.long)
    t_X_val = torch.tensor(X_val, dtype=torch.float32)
    t_X_te = torch.tensor(X_te, dtype=torch.float32)
    
    loader = DataLoader(TensorDataset(t_X_tr, t_y_tr), batch_size=32, shuffle=True)
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
# MAIN PIPELINE
# ==========================================

if __name__ == '__main__':
    print("=" * 70)
    print("  ULTIMATE COMPETITION PIPELINE v3")
    print("=" * 70)
    
    # ---- Load sample submission for test ID order ----
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    test_ids = sample_sub.iloc[:, 0].tolist()
    
    # Build test image paths in sample_submission order
    test_dir = DATA_DIR / "test"
    test_paths = []
    for tid in test_ids:
        # Find the image file
        candidates = list(test_dir.glob(f"{tid}.*"))
        if candidates:
            test_paths.append(str(candidates[0]))
        else:
            print(f"  WARNING: Test image {tid} not found!")
    
    print(f"  Test images: {len(test_paths)} (matched to sample_submission order)")
    
    y_train = np.array([0]*500 + [1]*500 + [2]*500)
    n_train = len(y_train)
    n_test = len(test_paths)
    
    # ---- Step 1: Extract/load all deep embeddings ----
    print("\n[1/6] Loading Deep Embeddings...")
    
    train_dir = str(DATA_DIR / "train")
    
    # ResNet18 (cached)
    train_res18 = np.load(COMP_DIR / "train_resnet18.npz")['feats']
    test_res18 = np.load(COMP_DIR / "test_resnet18.npz")['feats']
    print(f"  ResNet18:       train={train_res18.shape}, test={test_res18.shape}")
    
    # EfficientNet-B0 (cached)
    train_eff = np.load(COMP_DIR / "train_efficientnet_b0.npz")['feats']
    test_eff = np.load(COMP_DIR / "test_efficientnet_b0.npz")['feats']
    print(f"  EfficientNet-B0: train={train_eff.shape}, test={test_eff.shape}")
    
    # ResNet34 (NEW - extract if not cached)
    try:
        train_res34, test_res34 = extract_deep_embeddings(
            'resnet34', train_dir, test_paths,
            str(COMP_DIR / "train_resnet34.npz"),
            str(COMP_DIR / "test_resnet34.npz")
        )
        has_resnet34 = True
        print(f"  ResNet34:       train={train_res34.shape}, test={test_res34.shape}")
    except Exception as e:
        print(f"  ResNet34 FAILED: {e} -- continuing without it")
        has_resnet34 = False
    
    # ---- Step 2: Extract/load V2 handcrafted features ----
    print("\n[2/6] Loading Handcrafted Features...")
    
    # Build train image paths in dark/normal/bright order
    train_paths = []
    for cls in ['dark', 'normal', 'bright']:
        cls_dir = DATA_DIR / "train" / cls
        paths = sorted([str(cls_dir / f) for f in os.listdir(cls_dir) if f.endswith('.png') or f.endswith('.jpg')])
        train_paths.extend(paths)
    
    train_hand_v2, feat_names_v2 = build_feature_matrix_v2(
        train_paths, str(COMP_DIR / "train_features_v2.npz"))
    test_hand_v2, _ = build_feature_matrix_v2(
        test_paths, str(COMP_DIR / "test_features_v2_aligned.npz"))
    
    print(f"  V2 Handcrafted: train={train_hand_v2.shape}, test={test_hand_v2.shape}")
    print(f"  Feature count:  {len(feat_names_v2)}")
    
    # Also load original features
    train_hand_v1 = np.nan_to_num(np.load(COMP_DIR / "train_features.npz")['X'], nan=0.0, posinf=1e5, neginf=-1e5)
    test_hand_v1 = np.nan_to_num(np.load(COMP_DIR / "test_features_aligned.npz")['X'], nan=0.0, posinf=1e5, neginf=-1e5)
    
    # ---- Step 3: Build feature matrices ----
    print("\n[3/6] Building Feature Matrices...")
    
    # Clean V2 features
    train_hand_v2 = np.nan_to_num(train_hand_v2, nan=0.0, posinf=1e5, neginf=-1e5)
    test_hand_v2 = np.nan_to_num(test_hand_v2, nan=0.0, posinf=1e5, neginf=-1e5)
    
    # Scale handcrafted features
    scaler_v2 = StandardScaler()
    X_tr_h2 = scaler_v2.fit_transform(train_hand_v2)
    X_te_h2 = scaler_v2.transform(test_hand_v2)
    
    scaler_v1 = StandardScaler()
    X_tr_h1 = scaler_v1.fit_transform(train_hand_v1)
    X_te_h1 = scaler_v1.transform(test_hand_v1)
    
    # Normalize deep embeddings
    X_tr_r18 = normalize(train_res18)
    X_te_r18 = normalize(test_res18)
    X_tr_e = normalize(train_eff)
    X_te_e = normalize(test_eff)
    
    if has_resnet34:
        X_tr_r34 = normalize(train_res34)
        X_te_r34 = normalize(test_res34)
    
    # PRIMARY FEATURE MATRIX: V2 handcrafted * 2.0 + all deep embeddings
    if has_resnet34:
        X_tr_primary = np.hstack([X_tr_h2 * 2.0, X_tr_r18, X_tr_e, X_tr_r34])
        X_te_primary = np.hstack([X_te_h2 * 2.0, X_te_r18, X_te_e, X_te_r34])
    else:
        X_tr_primary = np.hstack([X_tr_h2 * 2.0, X_tr_r18, X_tr_e])
        X_te_primary = np.hstack([X_te_h2 * 2.0, X_te_r18, X_te_e])
    
    # TREE FEATURE MATRIX: V2 handcrafted + PCA of deep features
    if has_resnet34:
        deep_all_tr = np.hstack([X_tr_r18, X_tr_e, X_tr_r34])
        deep_all_te = np.hstack([X_te_r18, X_te_e, X_te_r34])
    else:
        deep_all_tr = np.hstack([X_tr_r18, X_tr_e])
        deep_all_te = np.hstack([X_te_r18, X_te_e])
    
    pca = PCA(n_components=50, random_state=SEED)
    pca_tr = pca.fit_transform(deep_all_tr)
    pca_te = pca.transform(deep_all_te)
    
    X_tr_tree = np.hstack([X_tr_h2, pca_tr])
    X_te_tree = np.hstack([X_te_h2, pca_te])
    
    # V1 FEATURE MATRIX (for diversity): V1 handcrafted + deep
    if has_resnet34:
        X_tr_v1_full = np.hstack([X_tr_h1 * 2.0, X_tr_r18, X_tr_e, X_tr_r34])
        X_te_v1_full = np.hstack([X_te_h1 * 2.0, X_te_r18, X_te_e, X_te_r34])
    else:
        X_tr_v1_full = np.hstack([X_tr_h1 * 2.0, X_tr_r18, X_tr_e])
        X_te_v1_full = np.hstack([X_te_h1 * 2.0, X_te_r18, X_te_e])
    
    print(f"  Primary Feature Matrix (V2+deep):  {X_tr_primary.shape[1]} dims")
    print(f"  Tree Feature Matrix (V2+PCA):      {X_tr_tree.shape[1]} dims")
    print(f"  V1 Full Feature Matrix:            {X_tr_v1_full.shape[1]} dims")
    
    # ---- Step 4: 10-Fold Bagging with Expanded Model Zoo ----
    N_FOLDS = 10
    MLP_SEEDS = [42, 123, 456, 789, 2024]  # Multi-seed for MLPs
    
    model_names = [
        # Linear models on primary features
        "LR_C0.005",
        "LR_C0.01",
        "LR_C0.02",
        "LR_C0.05",
        "Ridge_a30",
        "Cal_LinearSVC",
        # MLP models (multi-seed FastMLP + variants)
        "FastMLP_s42",
        "FastMLP_s123",
        "FastMLP_s456",
        "FastMLP_s789",
        "FastMLP_s2024",
        "DeepMLP_s42",
        "WideMLP_s42",
        # V1 features linear model (diversity)
        "LR_V1_C0.02",
        # Tree models on tree features
        "ExtraTrees",
        "RandomForest",
        "HistGBM",
        "GBM_small",
        # sklearn MLP on primary features
        "sklearn_MLP",
    ]
    
    print(f"\n[4/6] Training {N_FOLDS}-Fold Ensemble across {len(model_names)} Model Families ({len(model_names)*N_FOLDS} Models Total)...")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_matrix = {m: np.zeros((n_train, 3)) for m in model_names}
    test_matrix = {m: np.zeros((n_test, 3)) for m in model_names}
    
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_primary, y_train)):
        f_t0 = time.time()
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        
        # Primary features split
        Xp_tr, Xp_val = X_tr_primary[tr_idx], X_tr_primary[val_idx]
        # Tree features split
        Xt_tr, Xt_val = X_tr_tree[tr_idx], X_tr_tree[val_idx]
        # V1 features split
        Xv1_tr, Xv1_val = X_tr_v1_full[tr_idx], X_tr_v1_full[val_idx]
        
        # ---- Linear Models ----
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
                model_class=FastMLP, epochs=25, lr=2e-3, label_smooth=0.06, seed=seed_val
            )
            oof_matrix[name][val_idx] = v_mlp
            test_matrix[name] += t_mlp / N_FOLDS
        
        # DeepMLP
        v_deep, t_deep = train_mlp_fold(
            Xp_tr, y_tr, Xp_val, y_val, X_te_primary,
            model_class=DeepMLP, epochs=28, lr=1.5e-3, label_smooth=0.05, seed=42
        )
        oof_matrix["DeepMLP_s42"][val_idx] = v_deep
        test_matrix["DeepMLP_s42"] += t_deep / N_FOLDS
        
        # WideMLP
        v_wide, t_wide = train_mlp_fold(
            Xp_tr, y_tr, Xp_val, y_val, X_te_primary,
            model_class=WideMLP, epochs=25, lr=2e-3, label_smooth=0.06, seed=42
        )
        oof_matrix["WideMLP_s42"][val_idx] = v_wide
        test_matrix["WideMLP_s42"] += t_wide / N_FOLDS
        
        # ---- V1 LR (diversity through different features) ----
        lr_v1 = LogisticRegression(C=0.02, max_iter=1000, random_state=SEED)
        lr_v1.fit(Xv1_tr, y_tr)
        oof_matrix["LR_V1_C0.02"][val_idx] = lr_v1.predict_proba(Xv1_val)
        test_matrix["LR_V1_C0.02"] += lr_v1.predict_proba(X_te_v1_full) / N_FOLDS
        
        # ---- Tree Models ----
        et = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt", 
                                  random_state=SEED, n_jobs=-1)
        et.fit(Xt_tr, y_tr)
        oof_matrix["ExtraTrees"][val_idx] = et.predict_proba(Xt_val)
        test_matrix["ExtraTrees"] += et.predict_proba(X_te_tree) / N_FOLDS
        
        rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt", 
                                    random_state=SEED, n_jobs=-1)
        rf.fit(Xt_tr, y_tr)
        oof_matrix["RandomForest"][val_idx] = rf.predict_proba(Xt_val)
        test_matrix["RandomForest"] += rf.predict_proba(X_te_tree) / N_FOLDS
        
        hgb = HistGradientBoostingClassifier(l2_regularization=3.0, min_samples_leaf=15, 
                                              max_iter=150, random_state=SEED)
        hgb.fit(Xp_tr, y_tr)
        oof_matrix["HistGBM"][val_idx] = hgb.predict_proba(Xp_val)
        test_matrix["HistGBM"] += hgb.predict_proba(X_te_primary) / N_FOLDS
        
        gbm = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                          min_samples_leaf=10, subsample=0.8, random_state=SEED)
        gbm.fit(Xt_tr, y_tr)
        oof_matrix["GBM_small"][val_idx] = gbm.predict_proba(Xt_val)
        test_matrix["GBM_small"] += gbm.predict_proba(X_te_tree) / N_FOLDS
        
        # sklearn MLP (different optimizer/regularization than PyTorch)
        sk_mlp = MLPClassifier(hidden_layer_sizes=(256, 96), activation='relu', 
                                alpha=0.01, learning_rate='adaptive', max_iter=300,
                                early_stopping=True, validation_fraction=0.15,
                                random_state=SEED)
        sk_mlp.fit(Xp_tr, y_tr)
        oof_matrix["sklearn_MLP"][val_idx] = sk_mlp.predict_proba(Xp_val)
        test_matrix["sklearn_MLP"] += sk_mlp.predict_proba(X_te_primary) / N_FOLDS
        
        elapsed = time.time() - f_t0
        print(f"    Fold {fold+1}/{N_FOLDS} completed in {elapsed:.1f}s")
    
    total_time = time.time() - t0
    print(f"\n  Total Training Time: {total_time:.1f}s")
    
    # ---- Step 5: OOF Evaluation ----
    print("\n[5/6] Out-Of-Fold Validation Scores:")
    print("-" * 70)
    print(f"{'Model':25s} | {'Accuracy':12s} | {'Comp Score':12s}")
    print("-" * 70)
    
    model_accs = {}
    for m in model_names:
        acc = accuracy_score(y_train, oof_matrix[m].argmax(axis=1))
        score = max(0, acc - 0.40) / 0.60
        model_accs[m] = acc
        print(f"{m:25s} | {acc:10.4f}   | {score:10.5f}")
    print("-" * 70)
    
    # ---- SLSQP Weight Optimization ----
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
    
    print("\n  SLSQP Optimized Blend Weights:")
    for m, w in zip(model_names, opt_weights):
        if w > 0.005:
            print(f"    {m:25s}: {w:6.4f} ({w*100:5.1f}%)")
    
    slsqp_oof_probs = sum(opt_weights[i] * oof_list[i] for i in range(n_models))
    slsqp_oof_acc = accuracy_score(y_train, slsqp_oof_probs.argmax(axis=1))
    slsqp_score = max(0, slsqp_oof_acc - 0.40) / 0.60
    print(f"\n  SLSQP OOF Accuracy: {slsqp_oof_acc:.4f}  (Score: {slsqp_score:.5f})")
    
    # ---- Stacking Meta-Learner ----
    print("\n  Training Stacking Meta-Learner...")
    
    # Build stacking features: OOF probabilities from all models
    oof_stack = np.hstack([oof_matrix[m] for m in model_names])  # n_train x (n_models * 3)
    test_stack = np.hstack([test_matrix[m] for m in model_names])
    
    # Also add the V2 handcrafted features for stacking (top-k by importance)
    oof_stack_full = np.hstack([oof_stack, X_tr_h2[:, :30]])  # top 30 handcrafted
    test_stack_full = np.hstack([test_stack, X_te_h2[:, :30]])
    
    # Train stacking LR with nested CV
    stacker = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
    
    stack_oof_probs = np.zeros((n_train, 3))
    stack_test_probs = np.zeros((n_test, 3))
    skf_stack = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED+1)
    
    for fold, (tr_idx, val_idx) in enumerate(skf_stack.split(oof_stack_full, y_train)):
        stacker_fold = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
        stacker_fold.fit(oof_stack_full[tr_idx], y_train[tr_idx])
        stack_oof_probs[val_idx] = stacker_fold.predict_proba(oof_stack_full[val_idx])
        stack_test_probs += stacker_fold.predict_proba(test_stack_full) / 5
    
    stack_oof_acc = accuracy_score(y_train, stack_oof_probs.argmax(axis=1))
    stack_score = max(0, stack_oof_acc - 0.40) / 0.60
    print(f"  Stacker OOF Accuracy: {stack_oof_acc:.4f}  (Score: {stack_score:.5f})")
    
    # ---- Blend SLSQP + Stacker ----
    # Try different blend ratios
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
    
    # Generate test predictions with optimal blend
    slsqp_test_probs = sum(opt_weights[i] * test_matrix[model_names[i]] for i in range(n_models))
    final_test_probs = best_alpha * slsqp_test_probs + (1 - best_alpha) * stack_test_probs
    
    # ---- Step 6: Pseudo-labeling (optional boost) ----
    print("\n[6/6] Pseudo-Labeling Pass...")
    
    test_max_probs = final_test_probs.max(axis=1)
    confidence_threshold = 0.55  # Conservative threshold
    high_conf_mask = test_max_probs >= confidence_threshold
    n_pseudo = high_conf_mask.sum()
    
    if n_pseudo >= 10:
        print(f"  {n_pseudo}/{n_test} test samples above confidence {confidence_threshold:.2f}")
        pseudo_labels = final_test_probs[high_conf_mask].argmax(axis=1)
        print(f"  Pseudo-label distribution: {dict(pd.Series(pseudo_labels).value_counts().sort_index())}")
        
        # Augment training set with pseudo-labeled samples
        X_pseudo_p = X_te_primary[high_conf_mask]
        X_pseudo_t = X_te_tree[high_conf_mask]
        
        X_tr_aug_p = np.vstack([X_tr_primary, X_pseudo_p])
        X_tr_aug_t = np.vstack([X_tr_tree, X_pseudo_t])
        y_train_aug = np.concatenate([y_train, pseudo_labels])
        
        # Re-train top models on augmented data and re-predict test
        print("  Re-training top models with pseudo-labeled data...")
        
        pseudo_test_probs_list = []
        
        # Top LR
        for C_val in [0.01, 0.02, 0.05]:
            lr = LogisticRegression(C=C_val, max_iter=1000, random_state=SEED)
            lr.fit(X_tr_aug_p, y_train_aug)
            pseudo_test_probs_list.append(lr.predict_proba(X_te_primary))
        
        # Multi-seed FastMLP on full augmented data
        for seed_val in MLP_SEEDS:
            torch.manual_seed(seed_val)
            model = FastMLP(in_dim=X_tr_aug_p.shape[1])
            t_X = torch.tensor(X_tr_aug_p, dtype=torch.float32)
            t_y = torch.tensor(y_train_aug, dtype=torch.long)
            t_X_te = torch.tensor(X_te_primary, dtype=torch.float32)
            
            loader = DataLoader(TensorDataset(t_X, t_y), batch_size=32, shuffle=True)
            opt2 = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
            sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=25, eta_min=1e-6)
            crit2 = nn.CrossEntropyLoss(label_smoothing=0.06)
            
            for ep in range(25):
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
        
        # Average pseudo-labeled predictions
        pseudo_test_probs = np.mean(pseudo_test_probs_list, axis=0)
        
        # Blend with existing predictions (conservative weight for pseudo)
        pseudo_weight = 0.3
        final_test_probs_v2 = (1 - pseudo_weight) * final_test_probs + pseudo_weight * pseudo_test_probs
        
        # Check if pseudo-labeling helps on OOF (it shouldn't change OOF but check test distribution)
        pseudo_preds = final_test_probs_v2.argmax(axis=1)
        orig_preds = final_test_probs.argmax(axis=1)
        n_changed = (pseudo_preds != orig_preds).sum()
        print(f"  Pseudo-labeling changed {n_changed} test predictions")
        print(f"  Final test class distribution: {dict(pd.Series(pseudo_preds).value_counts().sort_index())}")
        
        # Use pseudo-labeled version
        final_test_probs = final_test_probs_v2
    else:
        print(f"  Only {n_pseudo} samples above threshold -- skipping pseudo-labeling")
    
    # ---- Generate Final Submission ----
    final_test_preds = final_test_probs.argmax(axis=1)
    
    sub_df = pd.DataFrame({
        sample_sub.columns[0]: test_ids,
        sample_sub.columns[1]: final_test_preds.tolist()
    })
    
    out_sub = Path(r"C:\Users\2006s\Documents\Codex\2026-09-11\create-an-image-of\outputs\illumination-classifier\submission.csv")
    desktop_sub = Path(r"C:\Users\2006s\Desktop\submission.csv")
    
    sub_df.to_csv(out_sub, index=False)
    sub_df.to_csv(desktop_sub, index=False)
    
    print(f"\n  Wrote {len(final_test_preds)} final predictions to:")
    print(f"    {out_sub}")
    print(f"    {desktop_sub}")
    
    print(f"\n  Final Test Class Distribution:")
    print(f"    {dict(pd.Series(final_test_preds).value_counts().sort_index())}")
    
    print(f"\n  First 10 Rows:")
    print(sub_df.head(10).to_string(index=False))
    
    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  SLSQP Blend OOF Accuracy:          {slsqp_oof_acc:.4f} (Score: {slsqp_score:.5f})")
    print(f"  Stacker OOF Accuracy:              {stack_oof_acc:.4f} (Score: {stack_score:.5f})")
    print(f"  Best Combined OOF Accuracy:        {best_blend_acc:.4f} (Score: {final_blend_score:.5f})")
    print(f"  Previous Best LB Score:            0.189 (Accuracy: 0.5133)")
    print(f"  Target (Top 10):                   0.2167 (Accuracy: 0.53)")
    print("=" * 70)
    print("  PIPELINE COMPLETE - READY TO UPLOAD!")
    print("=" * 70)

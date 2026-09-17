import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

COMP_DIR = r"C:\Users\2006s\Documents\Codex\2026-09-11\create-an-image-of\outputs\illumination-classifier\comp_output"

v1 = np.nan_to_num(np.load(f"{COMP_DIR}/train_features.npz")["X"])
v2 = np.nan_to_num(np.load(f"{COMP_DIR}/train_features_v2.npz")["X"])
r18 = normalize(np.load(f"{COMP_DIR}/train_resnet18.npz")["feats"])
eff = normalize(np.load(f"{COMP_DIR}/train_efficientnet_b0.npz")["feats"])
r34 = normalize(np.load(f"{COMP_DIR}/train_resnet34.npz")["feats"])

y = np.array([0]*500 + [1]*500 + [2]*500)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Compare feature combinations with LR C=0.01
configs = [
    ("V1 alone", StandardScaler().fit_transform(v1)),
    ("V2 alone", StandardScaler().fit_transform(v2)),
    ("V1*2 + R18 + Eff (Previous Best)", np.hstack([StandardScaler().fit_transform(v1)*2.0, r18, eff])),
    ("V1*2 + R18 + Eff + R34", np.hstack([StandardScaler().fit_transform(v1)*2.0, r18, eff, r34])),
    ("V2*2 + R18 + Eff + R34", np.hstack([StandardScaler().fit_transform(v2)*2.0, r18, eff, r34])),
    ("V1*1.5 + V2*1.5 + R18 + Eff + R34", np.hstack([StandardScaler().fit_transform(v1)*1.5, StandardScaler().fit_transform(v2)*1.5, r18, eff, r34])),
]

print(f"{'Config':36s} | {'CV Acc':8s} | {'Score':8s}")
print("-" * 60)
for name, X in configs:
    lr = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
    scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')
    acc = scores.mean()
    sc = max(0, acc - 0.40) / 0.60
    print(f"{name:36s} | {acc:8.4f} | {sc:8.4f}")

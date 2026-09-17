"""Generate a submission with five-view test-time augmentation."""
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as F
from dataset import ImageRecordsDataset, load_test_records

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", default="data"); p.add_argument("--checkpoint", default="artifacts/best_model.pt")
    p.add_argument("--output", default="submission.csv"); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--workers", type=int, default=2); args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = models.resnet18(weights=None); model.fc = torch.nn.Linear(model.fc.in_features, 3); model.load_state_dict(checkpoint["model_state"]); model.to(device).eval()
    records, _ = load_test_records(args.data_dir); base = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    loader = DataLoader(ImageRecordsDataset(records, base), batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    normalise = transforms.Normalize(MEAN, STD); ids, encoded = [], []
    with torch.no_grad():
        for images, batch_ids in loader:
            images = images.to(device)
            views = (images, F.hflip(images), F.adjust_brightness(images, .80), F.adjust_brightness(images, .90), F.adjust_brightness(images, 1.15))
            probabilities = torch.stack([model(normalise(view)).softmax(1) for view in views]).mean(0)
            encoded += probabilities.argmax(1).cpu().tolist(); ids += list(batch_ids)
    labels = encoded  # Numeric labels (0=dark, 1=normal, 2=bright) matching submission format
    sample = Path(args.data_dir) / "sample_submission.csv"
    if sample.exists():
        columns = pd.read_csv(sample, nrows=0).columns.tolist()
        if len(columns) < 2: raise ValueError("sample_submission.csv needs id and label columns.")
        result = pd.DataFrame({columns[0]: ids, columns[1]: labels})
    else: result = pd.DataFrame({"id": ids, "label": labels})
    result.to_csv(args.output, index=False); print(f"Wrote {len(result)} predictions to {args.output}")

if __name__ == "__main__": main()

"""Train a three-class ResNet18 illumination classifier."""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import ImageRecordsDataset, load_train_records

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def evaluate(model, loader, criterion, device):
    model.eval(); loss = correct = count = 0; actual = []; predicted = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            logits = model(images); loss += criterion(logits, labels).item() * len(labels)
            guess = logits.argmax(1); correct += (guess == labels).sum().item(); count += len(labels)
            actual += labels.cpu().tolist(); predicted += guess.cpu().tolist()
    return loss / count, correct / count, actual, predicted

def train_epoch(model, loader, criterion, optimizer, device):
    model.train(); loss = correct = count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True); logits = model(images); batch_loss = criterion(logits, labels)
        batch_loss.backward(); optimizer.step(); loss += batch_loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item(); count += len(labels)
    return loss / count, correct / count

def main():
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", default="data"); p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--warmup-epochs", type=int, default=5); p.add_argument("--finetune-epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5); p.add_argument("--workers", type=int, default=2); p.add_argument("--seed", type=int, default=42); args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    records, label_values = load_train_records(args.data_dir); labels = [label for _, label in records]
    train_records, val_records = train_test_split(records, test_size=.20, stratify=labels, random_state=args.seed)
    balance = {str(label_values[i]): {"train": sum(y == i for _, y in train_records), "val": sum(y == i for _, y in val_records)} for i in range(3)}
    print("Split balance:", json.dumps(balance))
    train_tf = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=.25, contrast=.2, saturation=.15), transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    val_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    pin = device.type == "cuda"; train_loader = DataLoader(ImageRecordsDataset(train_records, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(ImageRecordsDataset(val_records, val_tf), batch_size=args.batch_size, num_workers=args.workers, pin_memory=pin)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); model.fc = nn.Linear(model.fc.in_features, 3); model.to(device); criterion = nn.CrossEntropyLoss()
    best, stale, epoch = -1., 0, 0
    with (output / "metrics.csv").open("w", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=("epoch", "phase", "train_loss", "train_accuracy", "val_loss", "val_accuracy")); writer.writeheader()
        for phase, num_epochs in (("warmup", args.warmup_epochs), ("finetune", args.finetune_epochs)):
            for parameter in model.parameters(): parameter.requires_grad = phase == "finetune"
            for parameter in model.fc.parameters(): parameter.requires_grad = True
            groups = [{"params": model.fc.parameters(), "lr": 1e-3}]
            if phase == "finetune": groups.append({"params": [v for n, v in model.named_parameters() if not n.startswith("fc.")], "lr": 1e-4})
            optimizer = AdamW(groups, weight_decay=1e-4); scheduler = CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1))
            for _ in range(num_epochs):
                epoch += 1; train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device); val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device); scheduler.step()
                row = {"epoch": epoch, "phase": phase, "train_loss": f"{train_loss:.5f}", "train_accuracy": f"{train_acc:.5f}", "val_loss": f"{val_loss:.5f}", "val_accuracy": f"{val_acc:.5f}"}; writer.writerow(row); log.flush(); print(json.dumps(row))
                if val_acc > best:
                    best, stale = val_acc, 0; torch.save({"model_state": model.state_dict(), "label_values": label_values, "val_accuracy": val_acc, "epoch": epoch}, output / "best_model.pt")
                else: stale += 1
                if phase == "finetune" and stale >= args.patience: print("Early stopping."); break
            if phase == "finetune" and stale >= args.patience: break
    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model_state"])
    _, score, actual, predicted = evaluate(model, val_loader, criterion, device); matrix = confusion_matrix(actual, predicted, labels=[0, 1, 2]).tolist()
    (output / "validation_report.json").write_text(json.dumps({"best_val_accuracy": score, "labels": label_values, "confusion_matrix": matrix}, indent=2))
    print(f"Best validation accuracy: {score:.4f}"); print(np.array(matrix))

if __name__ == "__main__": main()

"""Flexible image/CSV dataset loading helpers.

Label mapping (matches task spec):
    0 = dark   (low illumination)
    1 = normal (average illumination)
    2 = bright (high illumination)
"""
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Canonical mapping: folder name -> numeric label (per task spec)
CLASS_MAP = {"dark": 0, "normal": 1, "bright": 2}
# Reverse: numeric label -> folder name (for submission decoding)
LABEL_VALUES = ["dark", "normal", "bright"]  # index = label

def _images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS) if Path(folder).exists() else []

def _column(frame, choices):
    names = {str(c).lower(): c for c in frame.columns}
    return next((names[x] for x in choices if x in names), None)

def _lookup(data_dir, split):
    result = {}
    for path in _images(Path(data_dir) / split):
        result.setdefault(path.name, path); result.setdefault(path.stem, path)
    return result

def _resolve(value, data_dir, split, lookup):
    value, data_dir = Path(str(value)), Path(data_dir)
    for candidate in (value, data_dir / value, data_dir / split / value):
        if candidate.exists(): return candidate
    found = lookup.get(value.name) or lookup.get(value.stem)
    if found: return found
    raise FileNotFoundError(f"Cannot find '{value}' in {data_dir / split}")

def load_train_records(data_dir, csv_name="train.csv"):
    """Return [(Path, encoded_label)] and the label_values list for submission.

    label_values is always ["dark", "normal", "bright"] so that
    label_values[encoded_label] gives the original class name.
    """
    data_dir, csv_path = Path(data_dir), Path(data_dir) / csv_name
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        label_col = _column(frame, ("label", "class", "target", "y"))
        image_col = _column(frame, ("image_path", "filepath", "file", "filename", "image", "id"))
        if label_col is None or image_col is None:
            raise ValueError("train.csv needs label and image/id columns.")
        lookup = _lookup(data_dir, "train")
        records = []
        for _, row in frame.iterrows():
            raw_label = row[label_col]
            # Support both numeric (0/1/2) and string labels
            if isinstance(raw_label, str) and raw_label in CLASS_MAP:
                encoded = CLASS_MAP[raw_label]
            else:
                encoded = int(raw_label)
            records.append((_resolve(row[image_col], data_dir, "train", lookup), encoded))
        return records, LABEL_VALUES

    # Folder-based loading: use CLASS_MAP for correct encoding
    train_dir = data_dir / "train"
    folders = [p for p in train_dir.iterdir() if p.is_dir()]
    if len(folders) != 3:
        raise ValueError(f"data/train must have exactly three class directories, found: {[f.name for f in folders]}")
    records = []
    for folder in folders:
        name = folder.name.lower()
        if name not in CLASS_MAP:
            raise ValueError(f"Unexpected folder name '{folder.name}'. Expected: {list(CLASS_MAP.keys())}")
        label = CLASS_MAP[name]
        for image in _images(folder):
            records.append((image, label))
    if not records:
        raise ValueError("No training images found.")
    print(f"Loaded {len(records)} training images:")
    for name, label in CLASS_MAP.items():
        count = sum(1 for _, l in records if l == label)
        print(f"  {label} ({name}): {count}")
    return records, LABEL_VALUES

def load_test_records(data_dir, csv_name="test.csv"):
    data_dir, csv_path, lookup = Path(data_dir), Path(data_dir) / csv_name, _lookup(data_dir, "test")
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        image_col = _column(frame, ("id", "image_path", "filepath", "file", "filename", "image"))
        if image_col is None:
            raise ValueError("test.csv needs an id or image-path column.")
        return [(_resolve(value, data_dir, "test", lookup), value) for value in frame[image_col]], image_col
    images = _images(data_dir / "test")
    if not images: raise ValueError("No test images found.")
    return [(image, image.stem) for image in images], "id"

class ImageRecordsDataset(Dataset):
    def __init__(self, records, transform=None): self.records, self.transform = records, transform
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        path, value = self.records[index]
        with Image.open(path) as loaded: image = loaded.convert("RGB")
        return (self.transform(image) if self.transform else image), value

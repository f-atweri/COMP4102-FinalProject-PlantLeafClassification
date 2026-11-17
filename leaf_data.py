import csv
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 128


def read_manifest(filename):
    with (PROJECT_DIR / filename).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"image_path", "label"} <= set(reader.fieldnames):
            raise ValueError(f"{filename} must contain image_path and label columns")
        rows = list(reader)
    if not rows or any(not row["image_path"] or not row["label"] for row in rows):
        raise ValueError(f"{filename} must contain image paths and labels")
    return rows


class LeafDataset(Dataset):
    def __init__(self, rows, class_to_index):
        self.rows = rows
        self.class_to_index = class_to_index
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        for row in rows:
            if row["label"] not in class_to_index:
                raise ValueError(f"Unknown class: {row['label']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = PROJECT_DIR / row["image_path"]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        label = self.class_to_index[row["label"]]
        return image_tensor, label


def main():
    train_rows = read_manifest("model_train.csv")
    validation_rows = read_manifest("validation.csv")
    class_names = sorted({row["label"] for row in train_rows})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    print("Class mapping:")
    for name, index in class_to_index.items():
        print(f"  {index}: {name}")

    for name, rows in (("Training", train_rows), ("Validation", validation_rows)):
        dataset = LeafDataset(rows, class_to_index)
        generator = torch.Generator().manual_seed(42)
        loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=generator)
        images, labels = next(iter(loader))
        print(f"\n{name}: {len(dataset)} images")
        print(f"  Batch image shape: {tuple(images.shape)}")
        print(f"  Batch labels: {labels.tolist()}")
        print(f"  Pixel range in batch: {images.min().item():.3f} to {images.max().item():.3f}")


if __name__ == "__main__":
    main()

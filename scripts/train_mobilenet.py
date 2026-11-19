import argparse
import csv
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from leaf_data import LeafDataset, PROJECT_DIR, read_manifest


SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 3
CHECKPOINT = PROJECT_DIR / "models" / "mobilenet_best.pt"


def main():
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 leaf classifier.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--history", type=Path, help="New CSV file for per-epoch training metrics.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing best-model checkpoint.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be a positive finite number")
    if args.checkpoint.exists() and not args.overwrite:
        raise FileExistsError(f"{args.checkpoint} exists; use --overwrite to replace it.")
    if args.history is not None:
        if args.history.resolve() == args.checkpoint.resolve():
            parser.error("history and checkpoint must be different files")
        if args.history.exists():
            raise FileExistsError(f"{args.history} exists; choose a new history file.")

    torch.manual_seed(args.seed)
    train_rows = read_manifest("model_train.csv")
    validation_rows = read_manifest("validation.csv")
    class_names = sorted({row["label"] for row in train_rows})
    class_to_index = {name: index for index, name in enumerate(class_names)}

    weights = MobileNet_V2_Weights.DEFAULT
    transform = weights.transforms()
    train_dataset = LeafDataset(train_rows, class_to_index, transform)
    validation_dataset = LeafDataset(validation_rows, class_to_index, transform)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    model = mobilenet_v2(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    history = []
    print(f"Device: {device}; pretrained weights: {weights.name}", flush=True)
    print(f"Classes: {class_names}", flush=True)
    print("Training the final layer; validation images are used to select the best epoch.", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.eval()
        model.classifier.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            scores = model(images)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        model.eval()
        correct = 0
        class_correct = [0] * len(class_names)
        class_total = [0] * len(class_names)
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                for index in range(len(class_names)):
                    mask = labels == index
                    class_total[index] += mask.sum().item()
                    class_correct[index] += ((predictions == labels) & mask).sum().item()

        accuracy = correct / len(validation_dataset)
        history.append((epoch, total_loss / len(train_dataset), accuracy))
        print(
            f"Epoch {epoch}/{args.epochs}: training loss={total_loss / len(train_dataset):.4f}, "
            f"validation accuracy={accuracy:.1%}",
            flush=True,
        )
        for index, name in enumerate(class_names):
            if class_total[index] == 0:
                raise ValueError(f"No validation images for class: {name}")
            print(
                f"  {name} recall: {class_correct[index]}/{class_total[index]} "
                f"= {class_correct[index] / class_total[index]:.1%}",
                flush=True,
            )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "weights": weights.name,
                "epoch": epoch,
                "validation_accuracy": accuracy,
                "training_settings": {
                    "epochs": args.epochs, "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate, "seed": args.seed,
                },
            }, args.checkpoint)
            print(f"  Saved best model to {args.checkpoint}", flush=True)
    if args.history is not None:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        with args.history.open("x", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("epoch", "training_loss", "validation_accuracy"))
            writer.writerows(history)
        print(f"Saved training history to {args.history}", flush=True)


if __name__ == "__main__":
    main()

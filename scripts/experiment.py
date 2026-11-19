import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from leaf_data import LeafDataset, PROJECT_DIR, read_manifest


class ImageCondition:
    def __init__(self, transform, brightness=1.0):
        self.transform = transform
        self.brightness = brightness

    def __call__(self, image):
        if self.brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(self.brightness)
        return self.transform(image)


def build_conditions(brightness):
    for value in brightness:
        if not math.isfinite(value) or value < 0:
            raise ValueError("Brightness values must be finite and non-negative.")
    if len(set(brightness)) != len(brightness):
        raise ValueError("Duplicate brightness values are not allowed.")
    return (
        [("Original", 1.0)]
        + [(f"Brightness {value}", value) for value in brightness if value != 1.0]
    )


def evaluate(model, dataset, class_names, batch_size):
    matrix = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64)
    all_predictions = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            all_predictions.extend(predictions.tolist())
            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[actual, predicted] += 1
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Evaluation split is empty.")
    recall = {}
    precision = {}
    for i, name in enumerate(class_names):
        actual = int(matrix[i, :].sum())
        predicted = int(matrix[:, i].sum())
        recall[name] = int(matrix[i, i]) / actual if actual else None
        precision[name] = int(matrix[i, i]) / predicted if predicted else None
    return {
        "accuracy": int(matrix.diag().sum()) / total,
        "image_count": total,
        "recall": recall,
        "precision": precision,
        "confusion_matrix": matrix.tolist(),
        "predictions": all_predictions,
    }


def compare_predictions(rows, class_names, original, changed):
    if len(rows) != len(original) or len(rows) != len(changed):
        raise ValueError("Predictions must match the evaluation images.")
    changed_count = newly_incorrect = newly_correct = 0
    for row, before, after in zip(rows, original, changed):
        if before != after:
            changed_count += 1
            actual = row["label"]
            if class_names[before] == actual and class_names[after] != actual:
                newly_incorrect += 1
            elif class_names[before] != actual and class_names[after] == actual:
                newly_correct += 1
    return {
        "predictions_changed": changed_count,
        "correct_to_incorrect": newly_incorrect,
        "incorrect_to_correct": newly_correct,
    }


def save_plots(output, conditions, results, class_names, example, baseline, rows, split):
    names = ["Majority baseline"] + [name for name, _ in conditions]
    accuracies = [baseline["accuracy"] * 100] + [
        results[name]["accuracy"] * 100 for name, _ in conditions
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(range(len(names)), accuracies)
    axes[0].set_xticks(range(len(names)), names, rotation=30, ha="right")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel(f"{split.capitalize()} accuracy (%)")
    axes[0].set_title(f"{split.capitalize()} accuracy by brightness")
    matrix = results["Original"]["confusion_matrix"]
    display_names = [name.replace("Apple___", "").replace("_", " ") for name in class_names]
    plot = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_xticks(range(len(display_names)), display_names, rotation=30, ha="right")
    axes[1].set_yticks(range(len(display_names)), display_names)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Original images: confusion matrix")
    maximum = max(max(row) for row in matrix)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axes[1].text(j, i, str(value), ha="center", va="center",
                         color="white" if value > maximum / 2 else "black")
    fig.colorbar(plot, ax=axes[1], label="Images")
    fig.tight_layout()
    fig.savefig(output / "summary.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(conditions), figsize=(3 * len(conditions), 3), squeeze=False)
    for ax, (name, brightness) in zip(axes.flat, conditions):
        image = example.copy()
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        ax.imshow(image)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"Same {split} image at each brightness (before model preprocessing)")
    fig.tight_layout()
    fig.savefig(output / "conditions.png", dpi=120)
    plt.close(fig)

    original_predictions = results["Original"]["predictions"]
    correct = [i for i, row in enumerate(rows) if row["label"] == class_names[original_predictions[i]]][:4]
    errors = [i for i, row in enumerate(rows) if row["label"] != class_names[original_predictions[i]]][:4]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), squeeze=False)
    for row_axes, indices, title in zip(axes, (correct, errors), ("Correct", "Incorrect")):
        for ax, index in zip(row_axes, indices):
            with Image.open(PROJECT_DIR / rows[index]["image_path"]) as image:
                ax.imshow(image.convert("RGB"))
            actual = rows[index]["label"].replace("Apple___", "").replace("_", " ")
            predicted = class_names[original_predictions[index]].replace("Apple___", "").replace("_", " ")
            ax.set_title(f"{title}\nActual: {actual}\nPredicted: {predicted}")
        for ax in row_axes:
            ax.axis("off")
    fig.suptitle(f"Original {split} images: correct and incorrect predictions")
    fig.tight_layout()
    fig.savefig(output / "predictions.png", dpi=120)
    plt.close(fig)

    changed_examples = [
        (index, name, brightness)
        for name, brightness in conditions if brightness != 1.0
        for index, prediction in enumerate(results[name]["predictions"])
        if prediction != original_predictions[index]
    ][:4]
    fig, axes = plt.subplots(max(1, len(changed_examples)), 2, figsize=(8, 3 * max(1, len(changed_examples))), squeeze=False)
    if not changed_examples:
        axes[0, 0].text(0.5, 0.5, "No changed predictions found.", ha="center", va="center")
        axes[0, 0].axis("off")
        axes[0, 1].axis("off")
    for (index, name, brightness), row_axes in zip(changed_examples, axes):
        with Image.open(PROJECT_DIR / rows[index]["image_path"]) as image:
            image = image.convert("RGB")
        actual = rows[index]["label"].replace("Apple___", "").replace("_", " ")
        for ax, shown, prediction, condition in (
            (row_axes[0], image, original_predictions[index], "Original"),
            (row_axes[1], ImageEnhance.Brightness(image).enhance(brightness),
             results[name]["predictions"][index], name),
        ):
            predicted = class_names[prediction].replace("Apple___", "").replace("_", " ")
            ax.imshow(shown)
            ax.set_title(f"{condition}\nActual: {actual}\nPredicted: {predicted}")
            ax.axis("off")
    fig.suptitle(f"Examples where brightness changed the prediction ({split})")
    fig.tight_layout()
    fig.savefig(output / "changes.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate brightness on validation or held-out test images.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "models" / "mobilenet_best.pt")
    parser.add_argument("--brightness", type=float, nargs="*", default=[0.5, 0.7, 1.3, 1.5])
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True, help="New directory for results.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    try:
        conditions = build_conditions(args.brightness)
    except ValueError as exc:
        parser.error(str(exc))

    rows = read_manifest(f"{args.split}.csv")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    class_names = checkpoint["class_names"]
    if sorted({row["label"] for row in rows}) != sorted(class_names):
        raise ValueError(f"{args.split.capitalize()} labels do not match the checkpoint classes.")
    weights = MobileNet_V2_Weights[checkpoint["weights"]]
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    class_to_index = {name: index for index, name in enumerate(class_names)}
    args.output.mkdir(parents=True, exist_ok=False)

    training_counts = Counter(row["label"] for row in read_manifest("model_train.csv"))
    majority = max(sorted(training_counts), key=training_counts.__getitem__)
    baseline = {
        "predicted_class": majority,
        "accuracy": sum(row["label"] == majority for row in rows) / len(rows),
    }
    print(f"Evaluating the same {len(rows)} {args.split} images for each brightness.", flush=True)
    print("Original images are the reference; brightness 1.0 is unchanged.", flush=True)
    if args.split == "validation":
        print("Validation is for model development, not final test performance.", flush=True)
    else:
        print("Test results are final only if the model and settings were chosen before viewing them.", flush=True)
    print(f"Majority baseline (always {majority}): {baseline['accuracy']:.1%}", flush=True)
    results = {}
    for name, brightness in conditions:
        transform = ImageCondition(weights.transforms(), brightness)
        dataset = LeafDataset(rows, class_to_index, transform)
        result = evaluate(model, dataset, class_names, args.batch_size)
        result["accuracy_change_percentage_points"] = (
            (result["accuracy"] - results["Original"]["accuracy"]) * 100 if name != "Original" else 0.0
        )
        result["comparison_to_original"] = compare_predictions(
            rows, class_names,
            results["Original"]["predictions"] if name != "Original" else result["predictions"],
            result["predictions"],
        )
        results[name] = result
        print(f"\n{name}: accuracy {result['accuracy']:.1%} "
              f"({result['accuracy_change_percentage_points']:+.1f} percentage points vs original)", flush=True)
        comparison = result["comparison_to_original"]
        print(f"  Changed predictions: {comparison['predictions_changed']}; "
              f"correct to incorrect: {comparison['correct_to_incorrect']}; "
              f"incorrect to correct: {comparison['incorrect_to_correct']}", flush=True)
        for label in class_names:
            recall = result["recall"][label]
            print(f"  {label} recall: {recall:.1%}" if recall is not None
                  else f"  {label} recall: undefined (no examples)", flush=True)

    report = {
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "pretrained_weights": weights.name,
        "checkpoint_epoch": checkpoint["epoch"],
        "class_names": class_names,
        "majority_baseline": baseline,
        "conditions": [
            {"name": name, "brightness": brightness}
            for name, brightness in conditions
        ],
        "results": results,
    }
    with (args.output / "results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    with (args.output / "accuracy.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["condition", "accuracy", "image_count", "change_percentage_points",
                         "predictions_changed", "correct_to_incorrect", "incorrect_to_correct"])
        writer.writerow(["Majority baseline", baseline["accuracy"], len(rows), "", "", "", ""])
        for name, _ in conditions:
            result = results[name]
            comparison = result["comparison_to_original"]
            writer.writerow([name, result["accuracy"], result["image_count"],
                             result["accuracy_change_percentage_points"], comparison["predictions_changed"],
                             comparison["correct_to_incorrect"], comparison["incorrect_to_correct"]])
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["condition", "image_path", "actual", "predicted"])
        for name, _ in conditions:
            for row, prediction in zip(rows, results[name]["predictions"]):
                writer.writerow([name, row["image_path"], row["label"], class_names[prediction]])
    with Image.open(PROJECT_DIR / rows[0]["image_path"]) as image:
        save_plots(args.output, conditions, results, class_names, image.convert("RGB"), baseline, rows, args.split)
    print(f"\nSaved results and plots in {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

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
from inspect_data import SPLIT_DIR


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


def select_examples(groups, rows, limit=4):
    selected = []
    seen_leaves = set()
    seen_images = set()
    seen_labels = set()
    seen_conditions = set()
    groups = [list(group) for group in groups]
    while len(selected) < limit and any(groups):
        for group in groups:
            if not group or len(selected) == limit:
                continue
            def preference(candidate):
                row = rows[candidate[0]]
                leaf = row.get("leaf_id") or row["image_path"]
                return (leaf in seen_leaves, row["image_path"] in seen_images,
                        row["label"] in seen_labels,
                        len(candidate) > 1 and candidate[1] in seen_conditions)

            index = min(range(len(group)), key=lambda i: preference(group[i]))
            candidate = group.pop(index)
            row = rows[candidate[0]]
            seen_leaves.add(row.get("leaf_id") or row["image_path"])
            seen_images.add(row["image_path"])
            seen_labels.add(row["label"])
            if len(candidate) > 1:
                seen_conditions.add(candidate[1])
            selected.append(candidate)
    return selected


def select_original_examples(rows, class_names, predictions, correct):
    groups = [
        [(i,) for i, row in enumerate(rows)
         if row["label"] == name
         and (row["label"] == class_names[predictions[i]]) == correct]
        for name in class_names
    ]
    return [candidate[0] for candidate in select_examples(groups, rows)]


def select_changed_examples(rows, class_names, conditions, results):
    original = results["Original"]["predictions"]
    groups = [[], [], []]
    for name, brightness in conditions:
        if brightness == 1.0:
            continue
        for index, prediction in enumerate(results[name]["predictions"]):
            before = original[index]
            if prediction == before:
                continue
            actual = rows[index]["label"]
            if class_names[before] == actual:
                group = 0
            elif class_names[prediction] == actual:
                group = 1
            else:
                group = 2
            groups[group].append((index, name, brightness))
    return select_examples(groups, rows)


def save_plots(output, conditions, results, class_names, example, baseline, rows, split):
    ordered = sorted(conditions, key=lambda item: item[1])
    factors = [brightness for _, brightness in ordered]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    axes[0].plot(factors, [results[name]["accuracy"] * 100 for name, _ in ordered],
                 marker="o", label="MobileNetV2")
    axes[0].axhline(baseline["accuracy"] * 100, color="gray", linestyle="--",
                    label="Majority-class baseline")
    axes[0].axvline(1.0, color="black", alpha=0.3, linestyle=":")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Overall accuracy")
    axes[0].legend()
    display_names = [name.replace("Apple___", "").replace("_", " ") for name in class_names]
    for label, display in zip(class_names, display_names):
        axes[1].plot(factors, [
            results[name]["recall"][label] * 100
            if results[name]["recall"][label] is not None else float("nan")
            for name, _ in ordered
        ], marker="o", label=display)
    axes[1].axvline(1.0, color="black", alpha=0.3, linestyle=":")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Recall (%)")
    axes[1].set_title("Recall by actual class")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Brightness factor (1.0 = original)")
        ax.set_xticks(factors)
        ax.grid(alpha=0.25)
    fig.suptitle(f"{split.capitalize()} images: performance under brightness changes (n={len(rows)})")
    fig.tight_layout()
    fig.savefig(output / "summary.png", dpi=200)
    plt.close(fig)

    selected = [ordered[0], ("Original", 1.0), ordered[-1]]
    selected = list(dict.fromkeys(selected))
    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 5),
                             squeeze=False, layout="constrained")
    for index, (ax, (name, _)) in enumerate(zip(axes.flat, selected)):
        matrix = results[name]["confusion_matrix"]
        normalized = [
            [100 * value / sum(row) if sum(row) else 0 for value in row]
            for row in matrix
        ]
        plot = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(display_names)), display_names, rotation=35, ha="right")
        ax.set_yticks(range(len(display_names)), display_names if index == 0 else [])
        ax.set_xlabel("Predicted")
        if index == 0:
            ax.set_ylabel("Actual")
        ax.set_title(name)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, f"{normalized[i][j]:.0f}%\n({value})",
                        ha="center", va="center", fontsize=8,
                        color="white" if normalized[i][j] > 50 else "black")
    fig.colorbar(plot, ax=list(axes.flat), label="Percentage of actual class", shrink=0.75)
    fig.suptitle(f"{split.capitalize()} confusion matrices: row percentages (image counts)")
    fig.savefig(output / "confusion.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    positions = range(len(ordered))
    ax.bar([i - 0.2 for i in positions], [
        results[name]["comparison_to_original"]["correct_to_incorrect"] for name, _ in ordered
    ], width=0.4, label="Correct to incorrect")
    ax.bar([i + 0.2 for i in positions], [
        results[name]["comparison_to_original"]["incorrect_to_correct"] for name, _ in ordered
    ], width=0.4, label="Incorrect to correct")
    ax.set_xticks(list(positions), [str(value) for _, value in ordered])
    ax.set_xlabel("Brightness factor (1.0 = original)")
    ax.set_ylabel("Number of images")
    ax.set_title(f"{split.capitalize()}: paired changes relative to original")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "paired_changes.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(ordered), figsize=(3 * len(ordered), 3), squeeze=False)
    for ax, (name, brightness) in zip(axes.flat, ordered):
        image = example.copy()
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        ax.imshow(image)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"Same {split} image at each brightness (before model preprocessing)")
    fig.tight_layout()
    fig.savefig(output / "conditions.png", dpi=200)
    plt.close(fig)

    original_predictions = results["Original"]["predictions"]
    correct = select_original_examples(rows, class_names, original_predictions, True)
    errors = select_original_examples(rows, class_names, original_predictions, False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), squeeze=False)
    for row_axes, indices, title in zip(axes, (correct, errors), ("Correct", "Incorrect")):
        for ax, index in zip(row_axes, indices):
            with Image.open(PROJECT_DIR / rows[index]["image_path"]) as image:
                ax.imshow(image.convert("RGB"))
            actual = rows[index]["label"].replace("Apple___", "").replace("_", " ")
            predicted = class_names[original_predictions[index]].replace("Apple___", "").replace("_", " ")
            ax.set_title(f"{title}\nActual: {actual}\nPredicted: {predicted}")
        for ax in row_axes[len(indices):]:
            ax.text(0.5, 0.5, f"No more {title.lower()}\nexamples", ha="center", va="center")
        for ax in row_axes:
            ax.axis("off")
    fig.suptitle(f"Original {split} images: correct and incorrect predictions")
    fig.tight_layout()
    fig.savefig(output / "predictions.png", dpi=200)
    plt.close(fig)

    changed_examples = select_changed_examples(rows, class_names, ordered, results)
    fig, axes = plt.subplots(max(1, len(changed_examples)), 2, figsize=(8, 3 * max(1, len(changed_examples))), squeeze=False)
    if not changed_examples:
        axes[0, 0].text(0.5, 0.5, "No changed predictions found.", ha="center", va="center")
        axes[0, 0].axis("off")
        axes[0, 1].axis("off")
    for (index, name, brightness), row_axes in zip(changed_examples, axes):
        with Image.open(PROJECT_DIR / rows[index]["image_path"]) as image:
            image = image.convert("RGB")
        actual = rows[index]["label"].replace("Apple___", "").replace("_", " ")
        before_correct = rows[index]["label"] == class_names[original_predictions[index]]
        after_correct = rows[index]["label"] == class_names[results[name]["predictions"][index]]
        transition = ("Correct" if before_correct else "Incorrect") + " to " + (
            "correct" if after_correct else "incorrect"
        )
        for ax, shown, prediction, condition in (
            (row_axes[0], image, original_predictions[index], "Original"),
            (row_axes[1], ImageEnhance.Brightness(image).enhance(brightness),
             results[name]["predictions"][index], name),
        ):
            predicted = class_names[prediction].replace("Apple___", "").replace("_", " ")
            ax.imshow(shown)
            ax.set_title(f"{condition} ({transition})\nActual: {actual}\nPredicted: {predicted}")
            ax.axis("off")
    fig.suptitle(f"Examples where brightness changed the prediction ({split})")
    fig.tight_layout()
    fig.savefig(output / "changes.png", dpi=200)
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
        "evaluation_manifest_sha256": hashlib.sha256(
            (SPLIT_DIR / f"{args.split}.csv").read_bytes()
        ).hexdigest(),
        "training_manifest_sha256": hashlib.sha256(
            (SPLIT_DIR / "model_train.csv").read_bytes()
        ).hexdigest(),
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

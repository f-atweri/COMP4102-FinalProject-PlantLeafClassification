"""Reproducible color/edge baseline for leaf-condition classification."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FEATURES = ("color", "color_edge")
CONDITIONS = {"original": 1.0, "dark": 0.6, "bright": 1.4}


def image_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_splits(data_dir, classes, seed):
    """Split unique file hashes within each class, keeping duplicates together."""
    rng = random.Random(seed)
    records = []
    hash_labels = {}
    for label in classes:
        directory = data_dir / label
        if not directory.is_dir():
            raise ValueError(f"Class directory does not exist: {directory}")
        groups = {}
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            digest = image_hash(path)
            if digest in hash_labels and hash_labels[digest] != label:
                raise ValueError(f"Identical image files have different labels: {path}")
            hash_labels[digest] = label
            groups.setdefault(digest, []).append(path)
        if len(groups) < 5:
            raise ValueError(f"Class {label!r} needs at least 5 distinct image files.")

        hashes = sorted(groups)
        rng.shuffle(hashes)
        val_count = max(1, round(len(hashes) * 0.15))
        test_count = max(1, round(len(hashes) * 0.15))
        train_count = len(hashes) - val_count - test_count
        for index, digest in enumerate(hashes):
            split = "train" if index < train_count else (
                "validation" if index < train_count + val_count else "test"
            )
            for path in groups[digest]:
                records.append({
                    "path": str(path.relative_to(data_dir)),
                    "label": label,
                    "sha256": digest,
                    "split": split,
                })
    return records


def extract_features(path, size, brightness=1.0):
    try:
        with Image.open(path) as source:
            image = source.convert("RGB").resize((size, size))
            image = ImageEnhance.Brightness(image).enhance(brightness)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not read image {path}: {error}") from error

    color = np.concatenate([
        np.histogram(pixels[:, :, channel], bins=16, range=(0, 1))[0]
        for channel in range(3)
    ]).astype(np.float64) / (size * size)

    gray = pixels @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    dy, dx = np.gradient(gray)
    magnitude = np.hypot(dx, dy)
    orientation = np.mod(np.arctan2(dy, dx), np.pi)
    edges = np.histogram(
        orientation, bins=9, range=(0, np.pi), weights=magnitude
    )[0].astype(np.float64)
    total = edges.sum()
    if total > 0:
        edges /= total
    return color, np.concatenate((color, edges))


def fit_centroids(features, labels, classes):
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (features - mean) / scale
    centers = np.stack([
        standardized[np.asarray(labels) == label].mean(axis=0)
        for label in classes
    ])
    return mean, scale, centers


def predict(features, model, classes):
    mean, scale, centers = model
    distances = ((features[:, None, :] - mean) / scale - centers[None, :, :]) ** 2
    return [classes[index] for index in distances.sum(axis=2).argmin(axis=1)]


def metrics(actual, predicted, classes):
    matrix = [[0 for _ in classes] for _ in classes]
    indices = {label: index for index, label in enumerate(classes)}
    for truth, guess in zip(actual, predicted):
        matrix[indices[truth]][indices[guess]] += 1
    per_class = {}
    for i, label in enumerate(classes):
        tp = matrix[i][i]
        actual_count = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / actual_count if actual_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": actual_count,
        }
    return {
        "accuracy": sum(matrix[i][i] for i in range(len(classes))) / len(actual),
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def run(data_dir, classes, seed, size):
    records = make_splits(data_dir, classes, seed)
    subsets = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "validation", "test")
    }

    def build_features(rows, brightness=1.0):
        pairs = [
            extract_features(data_dir / row["path"], size, brightness)
            for row in rows
        ]
        return {
            name: np.stack([pair[index] for pair in pairs])
            for index, name in enumerate(FEATURES)
        }

    train_features = build_features(subsets["train"])
    validation_features = build_features(subsets["validation"])
    train_labels = [row["label"] for row in subsets["train"]]
    validation_labels = [row["label"] for row in subsets["validation"]]
    models = {
        name: fit_centroids(train_features[name], train_labels, classes)
        for name in FEATURES
    }
    validation = {
        name: metrics(
            validation_labels, predict(validation_features[name], models[name], classes), classes
        )
        for name in FEATURES
    }
    selected = max(FEATURES, key=lambda name: validation[name]["accuracy"])
    test_labels = [row["label"] for row in subsets["test"]]
    test = {}
    for condition, brightness in CONDITIONS.items():
        features = build_features(subsets["test"], brightness)[selected]
        predictions = predict(features, models[selected], classes)
        test[condition] = {
            **metrics(test_labels, predictions, classes),
            "predictions": [
                {"path": row["path"], "actual": row["label"], "predicted": prediction}
                for row, prediction in zip(subsets["test"], predictions)
            ],
        }
    return {
        "settings": {
            "classes": classes, "seed": seed, "image_size": size,
            "brightness_factors": CONDITIONS,
        },
        "split_records": records,
        "validation": validation,
        "selected_feature_set": selected,
        "test": test,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing class folders")
    parser.add_argument("--classes", nargs="+", required=True, help="Class folder names")
    parser.add_argument("--output", type=Path, default=Path("results/experiment.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    if len(args.classes) < 2 or len(set(args.classes)) != len(args.classes):
        parser.error("--classes must contain at least two distinct class names")
    if any(Path(name).name != name or name in (".", "..") for name in args.classes):
        parser.error("--classes must be immediate folder names, not paths")
    if args.size < 2:
        parser.error("--size must be at least 2")
    try:
        result = run(args.data, args.classes, args.seed, args.size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Selected: {result['selected_feature_set']}")
    for condition, values in result["test"].items():
        print(f"Test accuracy ({condition}): {values['accuracy']:.3f}")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()

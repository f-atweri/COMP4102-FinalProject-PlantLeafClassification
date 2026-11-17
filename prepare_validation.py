import csv
from pathlib import Path
from random import Random


PROJECT_DIR = Path(__file__).resolve().parent
FIELDS = ("image_path", "label", "leaf_id")
VALIDATION_FRACTION = 0.2
SEED = 42


def main():
    with (PROJECT_DIR / "train.csv").open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("train.csv must contain image_path, label, and leaf_id")
        rows = list(reader)

    if not rows:
        raise ValueError("train.csv has no images")

    groups_by_class = {}
    leaf_labels = {}
    image_paths = set()
    for row in rows:
        if any(not row[field] for field in FIELDS):
            raise ValueError(f"Incomplete row in train.csv: {row}")
        if row["image_path"] in image_paths:
            raise ValueError(f"Duplicate image in train.csv: {row['image_path']}")
        image_paths.add(row["image_path"])

        leaf_id = row["leaf_id"]
        label = row["label"]
        if leaf_id in leaf_labels and leaf_labels[leaf_id] != label:
            raise ValueError(f"Leaf group has multiple labels: {leaf_id}")
        leaf_labels[leaf_id] = label
        groups_by_class.setdefault(label, {}).setdefault(leaf_id, []).append(row)

    random = Random(SEED)
    model_train_rows = []
    validation_rows = []
    for label, groups in sorted(groups_by_class.items()):
        leaf_ids = sorted(groups)
        if len(leaf_ids) < 2:
            raise ValueError(f"Need at least two leaf groups in class: {label}")
        random.shuffle(leaf_ids)
        validation_count = max(1, int(len(leaf_ids) * VALIDATION_FRACTION))
        validation_ids = leaf_ids[:validation_count]
        model_train_ids = leaf_ids[validation_count:]

        class_train = [row for leaf_id in model_train_ids for row in groups[leaf_id]]
        class_validation = [row for leaf_id in validation_ids for row in groups[leaf_id]]
        model_train_rows.extend(class_train)
        validation_rows.extend(class_validation)
        print(f"{label}: train={len(class_train)}, validation={len(class_validation)}")

    for filename, split_rows in (
        ("model_train.csv", model_train_rows),
        ("validation.csv", validation_rows),
    ):
        with (PROJECT_DIR / filename).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(split_rows)

    print(f"\nWrote {len(model_train_rows)} training and {len(validation_rows)} validation rows.")


if __name__ == "__main__":
    main()

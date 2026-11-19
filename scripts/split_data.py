import csv
import json
from random import Random

from inspect_data import DATASET, IMAGE_EXTENSIONS, PROJECT_DIR, SPLIT_DIR


LEAF_MAP = DATASET.parent.parent / "leaf-map.json"
TRAIN_FRACTION = 0.8
SEED = 42

# black rot images use a different class name in the leaf map.
MAP_LABELS = {"Apple___Black_rot": "Apple_Frogeye Spot"}


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET}")

    with LEAF_MAP.open(encoding="utf-8") as file:
        leaf_map = json.load(file)

    class_dirs = sorted(path for path in DATASET.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found under: {DATASET}")

    random = Random(SEED)
    train_rows = []
    test_rows = []

    for class_dir in class_dirs:
        groups = {}
        map_label = MAP_LABELS.get(class_dir.name, class_dir.name)
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No images found in: {class_dir}")

        for image in images:
            image_key = image.stem.replace("_final_masked", "")
            image_key = image_key.split("___")[-1].split("copy")[0].strip().lower()
            candidates = leaf_map.get(image_key, [])
            matches = [
                leaf_id for leaf_id in candidates
                if leaf_id.startswith(map_label + ":::")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one leaf group for {image}; found {matches}"
                )

            leaf_id = matches[0]
            row = (image.relative_to(PROJECT_DIR).as_posix(), class_dir.name, leaf_id)
            groups.setdefault(leaf_id, []).append(row)

        leaf_ids = sorted(groups)
        if len(leaf_ids) < 2:
            raise ValueError(f"Need at least two leaf groups in: {class_dir}")
        random.shuffle(leaf_ids)
        train_count = min(max(int(len(leaf_ids) * TRAIN_FRACTION), 1), len(leaf_ids) - 1)

        class_train = [row for leaf_id in leaf_ids[:train_count] for row in groups[leaf_id]]
        class_test = [row for leaf_id in leaf_ids[train_count:] for row in groups[leaf_id]]
        train_rows.extend(class_train)
        test_rows.extend(class_test)
        print(f"{class_dir.name}: train={len(class_train)}, test={len(class_test)}")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("train.csv", train_rows), ("test.csv", test_rows)):
        with (SPLIT_DIR / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("image_path", "label", "leaf_id"))
            writer.writerows(rows)

    print(f"\nWrote {len(train_rows)} training and {len(test_rows)} test rows.")


if __name__ == "__main__":
    main()

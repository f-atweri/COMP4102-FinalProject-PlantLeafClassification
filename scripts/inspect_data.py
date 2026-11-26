""" 
Dataset: PlantVillage apple images
Mohanty, Hughes, and Salathé (2016), Using Deep Learning for
Image-Based Plant Disease Detection.
https://doi.org/10.3389/fpls.2016.01419
Dataset source: https://github.com/spMohanty/PlantVillage-Dataset
"""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_DIR / "data" / "splits"
DATASET = (
    PROJECT_DIR
    / "data"
    / "PlantVillage-apple"
    / "raw"
    / "color"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET}")

    # count images per folder so we can check the dataset's class balance
    counts = {}
    for class_dir in sorted(DATASET.iterdir()):
        if class_dir.is_dir():
            images = [
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            counts[class_dir.name] = len(images)

    if not counts or sum(counts.values()) == 0:
        raise ValueError(f"No images found in class directories under: {DATASET}")

    total = sum(counts.values())
    print(f"Dataset: {DATASET}")
    print(f"Total images: {total}")
    print()
    print(f"{'Class':30} {'Images':>8} {'Share':>8}")
    print("-" * 48)
    # report both counts and shares to make class imbalance easy to spot
    for label, count in counts.items():
        print(f"{label:30} {count:8} {count / total:7.1%}")


if __name__ == "__main__":
    main()

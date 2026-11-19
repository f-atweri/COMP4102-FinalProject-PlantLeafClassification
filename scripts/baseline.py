import csv
from collections import Counter
from inspect_data import SPLIT_DIR


def read_labels(filename):
    with (SPLIT_DIR / filename).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "label" not in reader.fieldnames:
            raise ValueError(f"{filename} must contain a label column")
        labels = [row["label"] for row in reader]
    if not labels or any(not label for label in labels):
        raise ValueError(f"{filename} must contain non-empty labels")
    return labels


def main():
    training_labels = read_labels("model_train.csv")
    validation_labels = read_labels("validation.csv")

    counts = Counter(training_labels)
    prediction = max(sorted(counts), key=counts.__getitem__)
    validation_counts = Counter(validation_labels)
    correct = validation_counts[prediction]

    print(f"Most common training class: {prediction}")
    print(f"Validation accuracy: {correct}/{len(validation_labels)} = {correct / len(validation_labels):.1%}")
    print("\nRecall by class:")
    for label, count in sorted(validation_counts.items()):
        class_correct = count if label == prediction else 0
        print(f"{label}: {class_correct}/{count} = {class_correct / count:.1%}")


if __name__ == "__main__":
    main()

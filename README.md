# Leaf condition classification: baseline

This starter compares two classical image-classification baselines for a small
number of leaf-condition classes. It is an experiment, not a medical or
agricultural diagnostic tool.

The project is intentionally not a finished submission: you still need to choose
and document a dataset, run the experiments, analyze real results, and explain
the methods yourself. The course guidelines also say the submitted project must
represent your own work and understanding.

## What it does

- Builds a reproducible, stratified 70/15/15 train/validation/test split.
- Keeps byte-identical images in the same split, and rejects identical files
  assigned to different classes.
- Compares normalized RGB color histograms against the same color features plus
  a histogram of gradient orientations (a simple texture/edge feature).
- Fits a standardized nearest-centroid classifier for each feature set and
  selects one using validation accuracy.
- Evaluates the selected model on the original test images and on darkened and
  brightened versions of the same images. Saves confusion matrices, per-class
  precision, recall and F1, accuracy, and image-level predictions as JSON.

Exact-byte duplicate detection does **not** prevent near-duplicates or photos of
the same leaf from leaking across splits. If a dataset provides plant, leaf, or
collection-session identifiers, split by those groups instead before treating
results as reliable. The lighting experiment simulates brightness changes; it
does not establish performance on real photographs or new backgrounds.

## Setup

Requires Python 3.10 or later. From this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Download and extract a leaf image dataset yourself, following its license and
usage instructions. For example, look for a PlantVillage distribution with
class-labeled images. Check its documentation and cite the exact version and
source you use. This starter expects one folder per class:

```text
data/
  healthy/
    image1.jpg
    image2.jpg
  condition_a/
    image3.jpg
    image4.jpg
```

Folder names may be the original dataset class names. Start with 3-5 classes,
ideally from the **same plant species**, to reduce the chance of learning
species differences instead of condition differences. Each class must have at
least five distinct image files (more is strongly recommended). Select only the
specific class folders you want; other folders are ignored.

## Run

Replace the example class names with actual folder names:

```powershell
.\.venv\Scripts\python leaf_baseline.py --data data --classes healthy condition_a condition_b --output results\experiment.json
```

Use `--seed 42` to set the split seed and `--size 128` to change the square
resize dimension. The JSON file records the arguments and every image path,
hash, split, and prediction, along with the metrics. Do not select or tune a
method based on the test results: use the validation set for those decisions.

Run the included synthetic-data checks:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

## Suggested next steps

1. Run the baseline on your chosen real dataset; inspect the predictions and
   representative correct and incorrect images.
2. Add a pretrained CNN such as ResNet-18 with transfer learning, using the
   **same split** for a fair comparison. This starter does not yet include a CNN.
3. If possible, collect a separate, properly labeled set of photos with new
   backgrounds and lighting. Keep it separate from all training and tuning.
4. Report dataset provenance, class counts, results, limitations, and what you
   learned. Cite the dataset and all external methods or resources you use.

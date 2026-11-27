Apple Leaf Classification

This project classifies apple leaves as healthy, apple scab, black rot, or
cedar apple rust, then investigates how image brightness affects predictions.

Dataset: PlantVillage apple images
Mohanty, Hughes, and Salathé (2016), Using Deep Learning for
Image-Based Plant Disease Detection.
https://doi.org/10.3389/fpls.2016.01419
Dataset source: https://github.com/spMohanty/PlantVillage-Dataset

Name And Student Id (Group 20): 
Fatima Atweri (101259778)
Menna Migbar (101311712)
Liyana Eramasrayillath Najeer (101316968)

Files and folders

- scripts\ contains the data inspection, splitting, training, and evaluation scripts.
- data\PlantVillage-apple\raw\color\ contains the four image-class folders.
- data\PlantVillage-apple\leaf-map.json groups images of the same leaf.
- data\splits\ contains train.csv, model_train.csv, validation.csv, and test.csv.
- results\ contains training histories, checkpoints, and experiment outputs.
  It is ignored by git and may not be present in a fresh clone.
- requirements.txt lists the Python dependencies.

Setup (Windows)

Open a terminal in the project root. Python 3.13 was used for these runs.
Install dependencies:

    python -m pip install -r requirements.txt

The image folders and split manifests must be present at the paths above.
Use the existing manifests for the reported splits. To recreate them only if
needed, run the following in order. These commands overwrite the manifests:

    python scripts\split_data.py
    python scripts\prepare_validation.py

Running the experiments

Run the commands below from the project root, in order. The rerun folder must
be new: training WILL NOT OVERWRITE an existing checkpoint or history, and
evaluation requires a new output directory. Use a different rerun folder name
if needed. The first training run may download pretrained weights.

1. Inspect the dataset, check the loader, and run the majority-class baseline:

    python scripts\inspect_data.py
    python scripts\leaf_data.py
    python scripts\baseline.py

2. Train the three candidates. Each saves the epoch with the highest accuracy
   on unchanged validation images and writes a per-epoch history:

    python scripts\train_mobilenet.py --seed 42 --epochs 6 --batch-size 32 --learning-rate 0.0003 --checkpoint results\training\lr-0003\model.pt --history results\training\lr-0003\history.csv
    python scripts\train_mobilenet.py --seed 42 --epochs 6 --batch-size 32 --learning-rate 0.001 --checkpoint results\training\lr-001\model.pt --history results\training\lr-001\history.csv
    python scripts\train_mobilenet.py --seed 42 --epochs 6 --batch-size 32 --learning-rate 0.003 --checkpoint results\training\lr-003\model.pt --history results\training\lr-003\history.csv

3. Compare candidates on unchanged validation images:

    python scripts\experiment.py --checkpoint results\training\lr-0003\model.pt --split validation --brightness 1.0 --output results\selection\lr-0003
    python scripts\experiment.py --checkpoint results\training\lr-001\model.pt --split validation --brightness 1.0 --output results\selection\lr-001
    python scripts\experiment.py --checkpoint results\training\lr-003\model.pt --split validation --brightness 1.0 --output results\selection\lr-003

Compare original-image accuracy in each selection folder's results.json.
Choose the highest. If tied, use the highest average recall across the four
classes, then the lower learning rate.

The commands below use the retrained lr-003 checkpoint since this is what was 
most accurate in our runs, but any can be chosen if other learning rates have 
been tested.

4. Evaluate the selected model on validation images:

    python scripts\experiment.py --checkpoint results\training\lr-003\model.pt --split validation --brightness 0.85 0.7 0.5 --output results\validation-dark
    python scripts\experiment.py --checkpoint results\training\lr-003\model.pt --split validation --brightness 1.15 1.3 1.5 --output results\validation-bright
    python scripts\experiment.py --checkpoint results\training\lr-003\model.pt --split validation --brightness 0.5 0.7 0.85 1.15 1.3 1.5 --output results\validation-combined

5. After fixing the model and brightness factors (we stayed with lr-003), evaluate the test set:

    python scripts\experiment.py --checkpoint results\training\lr-003\model.pt --split test --brightness 0.85 0.7 0.5 --output results\test-dark
    python scripts\experiment.py --checkpoint results\training\lr-003\model.pt --split test --brightness 1.15 1.3 1.5 --output results\test-bright

The original images (factor 1.0) are automatically included in every run.
Evaluation uses the same images and checkpoint for each factor. It does not
retrain the model. The same seed helps repeatability, but retraining may differ
across hardware or library versions. 

Outputs

- results\training\<candidate>\model.pt: best saved checkpoint.
- results\training\<candidate>\history.csv: epoch loss and validation accuracy.
- results\selection\<candidate>\: original-image validation results.
- results\validation-... and results\test-...: brightness results.
- Each evaluation folder contains results.json (settings, hashes, metrics and
  predictions), accuracy.csv, and predictions.csv.
- summary.png shows accuracy and class recall; confusion.png shows confusion
  matrices; paired_changes.png counts prediction changes.
- conditions.png shows brightness examples; predictions.png shows original
  successes and errors; changes.png and changes-compact.png show examples whose
  predictions changed. 

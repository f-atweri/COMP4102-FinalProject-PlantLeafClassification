Apple Leaf Classification!

this project trains a small classifier to recognize four classes of apple
leaves and investigates how changes in image brightness affect its predictions.

implementation/project stages

1. inspect the data ([inspect_data.py](scripts/inspect_data.py)): count the images in each class
   and show their share of the dataset. (this is mostly just for some basic analaysis at the start because i think we need more healthy images)
2. split the data ([split_data.py](scripts/split_data.py)): create training and test manifests,
   keeping images of the same leaf together in one split.
3. prepare validation data ([prepare_validation.py](scripts/prepare_validation.py)): divide the training
   manifest into model-training and validation manifests, again by leaf. (fyi validation helps choose the model, not train it!)
4. load the data ([leaf_data.py](scripts/leaf_data.py)): read the manifests and provide images
   and labels to the model. Running the script also checks a sample batch.
5. establish a baseline ([baseline.py](scripts/baseline.py)): predict the most common training
   class for every validation image and report accuracy and per-class recall.
6. train the model ([train_mobilenet.py](scripts/train_mobilenet.py)): train the final classification
   layer of a pretrained MobileNetV2 and save the best validation checkpoint.
7. run the brightness experiment ([experiment.py](scripts/experiment.py)): evaluate the same
   validation or test images at different brightness levels and save metrics,
   plots, and prediction examples.

again, a note to self here to use the validation set to develop the model and choose settings. Keep the test
set for the final evaluation after those choices are fixed. The split scripts
overwrite their output manifests, so keep the existing splits for a consistent
experiment.

to run (windows): open a terminal in the project root (the folder
containing this README). all paths in the commands below are relative to it.

```text
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

the dataset images should be in `data\PlantVillage-apple\raw\color`, with a
folder for each class. the split manifests are in `data\splits`. to recreate
them if needed, make sure `data\PlantVillage-apple\leaf-map.json` is present,
then run these commands. they overwrite the existing manifests:

```text
.\.venv\Scripts\python.exe scripts\split_data.py
.\.venv\Scripts\python.exe scripts\prepare_validation.py
```

inspect the dataset, check a sample batch, and run the baseline:

```text
.\.venv\Scripts\python.exe scripts\inspect_data.py
.\.venv\Scripts\python.exe scripts\leaf_data.py
.\.venv\Scripts\python.exe scripts\baseline.py
```

optional: train a single model with the default settings and save its training
history (for comparing candidates, use the commands in experiments below instead):

```text
.\.venv\Scripts\python.exe scripts\train_mobilenet.py --history results\training.csv
```

the first training run may download pretrained weights. The best checkpoint is
saved to `models\mobilenet_best.pt`. Training will not overwrite an existing
checkpoint unless `--overwrite` is added; choose a new history file for each run.
if the default checkpoint already exists, use a new `--checkpoint` path or
explicitly add `--overwrite` only if you want to replace it.


experiments!
1. train three candidates from the project root (same data, seed, batch size and
   epochs, but different learning rates). each run saves its best epoch based on
   accuracy on the original validation images, plus its training history:

   ```text
   .\.venv\Scripts\python.exe scripts\train_mobilenet.py --seed 42 --epochs 6 --batch-size 32 --learning-rate 0.0003 --checkpoint results\training\lr-0003\model.pt --history results\training\lr-0003\history.csv
   .\.venv\Scripts\python.exe scripts\train_mobilenet.py --seed 42 --epochs 6 --batch-size 32 --learning-rate 0.001 --checkpoint results\training\lr-001\model.pt --history results\training\lr-001\history.csv
   .\.venv\Scripts\python.exe scripts\train_mobilenet.py --seed 42 --epochs 6 --batch-size 32 --learning-rate 0.003 --checkpoint results\training\lr-003\model.pt --history results\training\lr-003\history.csv
   ```

2. compare the saved candidates on the original validation images (1.0 means
   unchanged; no brightness changes yet):

   ```text
   .\.venv\Scripts\python.exe scripts\experiment.py --checkpoint results\training\lr-0003\model.pt --split validation --brightness 1.0 --output results\selection\lr-0003
   .\.venv\Scripts\python.exe scripts\experiment.py --checkpoint results\training\lr-001\model.pt --split validation --brightness 1.0 --output results\selection\lr-001
   .\.venv\Scripts\python.exe scripts\experiment.py --checkpoint results\training\lr-003\model.pt --split validation --brightness 1.0 --output results\selection\lr-003
   ```

   compare the original accuracy in `results.json` in each selection folder and choose
   the highest. if tied, choose the highest average recall across the four
   classes; if still tied, choose the lower learning rate. check the individual
   class recalls and confusion matrices too (accuracy alone can hide errors in
   smaller classes). keep the test set for after the model and settings are fixed!

   the commands below use `results\training\lr-001\model.pt` as an example.
   replace that checkpoint path in each command with the candidate that actually
   won, and use the same selected model for all the experiments.

3. validation experiments (use the same selected model for each):

   a) darker images (investigate reduced brightness):

   ```text
   .\.venv\Scripts\python.exe scripts\experiment.py --checkpoint results\training\lr-003\model.pt --split validation --brightness 0.85 0.7 0.5 --output results\validation-dark
   ```

   b) brighter images (investigate increased brightness):

   ```text
   .\.venv\Scripts\python.exe scripts\experiment.py --checkpoint results\training\lr-003\model.pt --split validation --brightness 1.15 1.3 1.5 --output results\validation-bright
   ```

   c) combined comparison (same comparisons for a single figure):

   ```text
   .\.venv\Scripts\python.exe scripts\experiment.py --checkpoint results\training\lr-003\model.pt --split validation --brightness 0.5 0.7 0.85 1.15 1.3 1.5 --output results\validation-combined
   ```

4. final testing (after choosing the model and brightness settings using validation):

   ```text
   .\.venv\Scripts\python.exe scripts\experiment.py --checkpoint results\training\lr-003\model.pt --split test --brightness 0.85 0.7 0.5 --output results\test-dark
   .\.venv\Scripts\python.exe scripts\experiment.py --checkpoint results\training\lr-003\model.pt --split test --brightness 1.15 1.3 1.5 --output results\test-bright
   .\.venv\Scripts\python.exe scripts\experiment.py --checkpoint results\training\lr-003\model.pt --split test --brightness 0.5 0.7 0.85 1.15 1.3 1.5 --output results\test-combined
   ```

prediction examples are chosen in manifest order, taking turns across actual
classes and preferring different leaves where possible. the brightness-change
examples take turns across correct-to-incorrect, incorrect-to-correct and
incorrect-to-incorrect changes when available, preferring different leaves,
classes and brightness levels where possible. they illustrate cases, not how
common the cases are; use the numerical results for that. the selection runs
only use original images, so they cannot show brightness-change examples.
each experiment needs a new output folder; choose a different name for a rerun.
`results` is git-ignored, so save the checkpoints, results and figures separately
when sharing them.

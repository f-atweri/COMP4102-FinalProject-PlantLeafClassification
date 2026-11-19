Apple Leaf Classification!

this project trains a small classifier to recognize four classes of apple
leaves and investigates how changes in image brightness affect its predictions.

implementation/project stages

1. inspect the data (scripts/inspect_data.py): count the images in each class
   and show their share of the dataset. (this is mostly just for some basic analaysis at the start because i think we need more healthy images)
2. split the data (scripts/split_data.py): create training and test manifests,
   keeping images of the same leaf together in one split.
3. prepare validation data (scripts/prepare_validation.py): divide the training
   manifest into model-training and validation manifests, again by leaf. (fyi validation helps choose the model, not train it!)
4. load the data (scripts/leaf_data.py): read the manifests and provide images
   and labels to the model. Running the script also checks a sample batch.
5. establish a baseline (scripts/baseline.py): predict the most common training
   class for every validation image and report accuracy and per-class recall.
6. train the model (scripts/train_mobilenet.py): train the final classification
   layer of a pretrained MobileNetV2 and save the best validation checkpoint.
7. run the brightness experiment (scripts/experiment.py): evaluate the same
   validation or test images at different brightness levels and save metrics,
   plots, and prediction examples.

again, a note to self here to use the validation set to develop the model and choose settings. Keep the test
set for the final evaluation after those choices are fixed. The split scripts
overwrite their output manifests, so keep the existing splits for a consistent
experiment.

to run (windows):

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

the dataset images should be in data\PlantVillage-apple\raw\color, with a
folder for each class. the split manifests are in data\splits. to recreate
them if needed, make sure data\PlantVillage-apple\leaf-map.json is present,
then run these commands. they overwrite the existing manifests:

.\.venv\Scripts\python.exe scripts\split_data.py
.\.venv\Scripts\python.exe scripts\prepare_validation.py

inspect the dataset, check a sample batch, and run the baseline:

.\.venv\Scripts\python.exe scripts\inspect_data.py
.\.venv\Scripts\python.exe scripts\leaf_data.py
.\.venv\Scripts\python.exe scripts\baseline.py

train the model and save its training history:

.\.venv\Scripts\python.exe scripts\train_mobilenet.py --history results\training.csv

the first training run may download pretrained weights. The best checkpoint is
saved to models\mobilenet_best.pt. Training will not overwrite an existing
checkpoint unless --overwrite is added; choose a new history file for each run.

run the brightness experiment on validation data:

.\.venv\Scripts\python.exe scripts\experiment.py --split validation --output results\validation

once the model and settings are final, run the experiment on the test data:

.\.venv\Scripts\python.exe scripts\experiment.py --split test --output results\test



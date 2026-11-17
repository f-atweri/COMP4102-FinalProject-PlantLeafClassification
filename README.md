Problem: A leaf can show visible symptoms such as spots, discoloration, or texture changes but identifying its condition from a photo is difficult, especially when lighting, background, or camera angle varies. Here we would classify a leaf image into a small set of labels, such as healthy, diseased, or a few specific conditions.

## first step: inspecting the data:

Total images: 3171

| Class | Images | Share |
|---|---:|---:|
| `Apple___Apple_scab` | 630 | 19.9% |
| `Apple___Black_rot` | 621 | 19.6% |
| `Apple___Cedar_apple_rust` | 275 | 8.7% |
| `Apple___healthy` | 1645 | 51.9% |

healthy leaves make up about half of the data. a model that always guesses “healthy” would already be about 52% accurate, without recognizing any diseases, so we'll need to evaluate each class, not just overall accuracy.

## second step: viewing example images and comparing classes
| Class |  |
|---|---|
| Apple scab | Dark, irregular spots or patches on the green leaf. |
| Black rot | Brown or dark damaged areas; some examples show damage near the leaf tip. |
| Cedar apple rust | Lighter brown or reddish spots and patches on the leaf surface. |
| Healthy | Mostly green leaves without obvious disease spots. |

these are patterns, not strict rules. symptoms can look different across images, and the disease classes can overlap visually and there's also different lighting conditions.

## third step: splitting the data

some leaves have multiple photos. to avoid data leakage, the script assigns
whole leaf groups to either training or testing, never both. it shuffles the
groups within each class using a fixed seed, then assigns about 80% of the
groups to training and the rest to testing. the image percentages may differ
because groups contain different numbers of photos.

the local PlantVillage `leaf-map.json` identifies the leaf groups. in this
file, the black rot images use the label `Apple_Frogeye Spot`; the script
maps that name to our `Apple___Black_rot` folder. missing or ambiguous group
information stops the script instead of silently assigning an image.

the training set is to develop the model and save the test set for
final evaluation. if we need validation data for tuning, it must also be split
by leaf group from the training set.

## foruth step: validation and a simple baseline

`python .\prepare_validation.py` to split `train.csv` into
`model_train.csv` and `validation.csv`. it assigns about 20% of each class's
leaf groups to validation using a fixed seed and it does not change `test.csv`.
running it again replaces the two generated CSV files.

use `model_train.csv` to train models, `validation.csv` to compare or tune
them, and `test.csv` only for final evaluation. `train.csv` remains the
original pool used to create the training and validation sets.

run `python .\baseline.py` for a simple reference: it chooses the most common
class in `model_train.csv` and predicts that class for every validation image.
it reports accuracy and recall for each class. recall is the fraction of a
class's images correctly identified. this baseline does not look at image
pixels and it shows why accuracy alone is not enough.
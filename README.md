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
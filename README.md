Problem: A leaf can show visible symptoms such as spots, discoloration, or texture changes but identifying its condition from a photo is difficult, especially when lighting, background, or camera angle varies. Here we would classify a leaf image into a small set of labels, such as healthy, diseased, or a few specific conditions.

first step: inspecting the data:


Total images: 3171

Class                            Images    Share
Apple___Apple_scab                  630   19.9%
Apple___Black_rot                   621   19.6%
Apple___Cedar_apple_rust            275    8.7%
Apple___healthy                    1645   51.9%

healthy leaves make up about half of the data. a model that always guesses “healthy” would already be about 52% accurate, without recognizing any diseases, so we'll need to evaluate each class, not just overall accuracy.

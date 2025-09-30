"""
Script to split folder into 70 | 15 | 15 split for train | validation | test 
"""

import splitfolders
import os

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "furniture_dataset" # This is the name we used in the training script

# This will split all images in INPUT_FOLDER into the OUTPUT_FOLDER
# It will create 'train', 'val', and 'test' subfolders.
# The 'seed' makes sure the split is reproducible.
# The ratio tuple is (train, validation, test).
print(f"Splitting files from '{INPUT_FOLDER}' into '{OUTPUT_FOLDER}'...")

splitfolders.ratio(
    INPUT_FOLDER,
    output=OUTPUT_FOLDER,
    seed=1337,
    ratio=(.7, .15, .15), # 70% train, 15% val, 15% test
    group_prefix=None,
    move=False # 'move=False' will COPY files, which is safer.
)

print("Done. Your dataset is now split and ready in the 'furniture_dataset' folder.")
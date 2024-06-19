#coding=utf-8
import os
from pathlib import Path
# import clip
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import torch
from PIL import Image

import math
import numpy as np
import pandas as pd



print("Available models:", available_models())

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./models')
# model, preprocess = load_from_name("ViT-H-14", device=device, download_root='./models')
features_path = Path("features")
model.eval()
# Function that computes the feature vectors for a batch of images
def compute_clip_features(photos_batch):
    # Load all the photos from the files
    photos = [Image.open(photo_file) for photo_file in photos_batch]

    # Preprocess all photos
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)

    with torch.no_grad():
        # Encode the photos batch to compute the feature vectors and normalize them
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)

    # Transfer the feature vectors back to the CPU and convert to numpy
    return photos_features.cpu().numpy()



def process_pic(photos_files):
    # Define the batch size so that it fits on your GPU. You can also do the processing on the CPU, but it will be slower.
    batch_size = 16

    # Path where the feature vectors will be stored

    # Compute how many batches are needed
    batches = math.ceil(len(photos_files) / batch_size)
    print(batches)
    # Process each batch
    for i in range(batches):
        print(f"Processing batch {i + 1}/{batches}")

        batch_ids_path = features_path / f"{i:010d}.csv"
        batch_features_path = features_path / f"{i:010d}.npy"

        # Only do the processing if the batch wasn't processed yet
        if not batch_features_path.exists():
            try:
                # Select the photos for the current batch
                batch_files = photos_files[i * batch_size: (i + 1) * batch_size]

                # Compute the features and save to a numpy file
                batch_features = compute_clip_features(batch_files)
                np.save(batch_features_path, batch_features)

                # Save the photo IDs to a CSV file
                photo_ids = [photo_file.name.split(".")[0] for photo_file in batch_files]
                photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
                photo_ids_data.to_csv(batch_ids_path, index=False)
            except:
                # Catch problems with the processing to make the process more robust
                print(f'Problem with batch {i}')
    load_file()

def load_file():
    # Load all numpy files
    features_list = [np.load(features_file) for features_file in sorted(features_path.glob("00*.npy"))]

    # Concatenate the features and store in a merged file
    features = np.concatenate(features_list)
    np.save(features_path / "features.npy", features)

    # Load all the photo IDs
    photo_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(features_path.glob("00*.csv"))])
    photo_ids.to_csv(features_path / "photo_ids.csv", index=False)



if __name__ == '__main__':
    # path = 'images/gallery'
    path = r'/Users/songyanan/MUGE/'

    photos_path = Path(path)
    # print(os.listdir(path))
    photos_files = list(photos_path.glob("**/*.jpg"))
    print(len(photos_files))
    #
    process_pic(photos_files)
    # load_file()


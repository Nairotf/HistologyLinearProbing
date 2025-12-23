import os
import tqdm
import h5py
import numpy as np
import sys
import pandas as pd

def get_features(slide_ids, features_dir):
    """
    Get the features for the given slide ids from the features directory.
    """
    data = []
    for slide_id in tqdm.tqdm(slide_ids):
        if ".svs" in slide_id:
            slide_id = slide_id.replace(".svs", "")
        if ".tif" in slide_id:
            slide_id = slide_id.replace(".tif", "")
        features_file = os.path.join(features_dir, f"{slide_id}.h5")
        with h5py.File(features_file, 'r') as f:
            features = f['features'][:]
        data.append(features)
    return np.array(data)
    

if __name__ == "__main__":
    dataset_path = sys.argv[1] 
    target_column = sys.argv[2]
    features_dir = sys.argv[3]
    feature_extractor = sys.argv[4]

    data = pd.read_csv(dataset_path)

    X = data.drop(columns=[target_column])
    X_features = get_features(X['slide_id'], features_dir)
    y = data[target_column]

    with h5py.File(f'{feature_extractor}.h5', 'w') as f:
        f['features'] = X_features
        f['target'] = y.values
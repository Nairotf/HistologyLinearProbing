import sys

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import BaseCrossValidator


def eval_test_metrics(y_true, y_pred):
    """Return test metrics as a single-row DataFrame."""
    return pd.DataFrame(
        {
            "r2": [r2_score(y_true, y_pred)],
            "mse": [mean_squared_error(y_true, y_pred)],
            "mae": [mean_absolute_error(y_true, y_pred)],
            "rmse": [root_mean_squared_error(y_true, y_pred)],
        }
    )

def iqr(X, y):
    """
    Remove outliers in y using the IQR rule and keep X aligned.
    X: numpy array of shape (n_samples, n_features)
    y: 1D numpy array of shape (n_samples,)
    Returns: filtered X, filtered y, and mask of remaining indices
    """
    y = np.asarray(y)
    X = np.asarray(X)

    y_q1 = np.quantile(y, 0.25)
    y_q3 = np.quantile(y, 0.75)
    iqr_factor = 1.5 * (y_q3 - y_q1)
    y_min = y_q1 - iqr_factor
    y_max = y_q3 + iqr_factor

    mask = (y >= y_min) & (y <= y_max)
    return X[mask], y[mask], mask

def create_pipeline(model):
    """Creates a pipeline with some params based in the provided option"""
    if model == "ridge":
        param_grid = {
            "pca__n_components": [0.8, 0.9],
            "ridge__alpha": [1e-1, 1],
        }
        pipeline = Pipeline(
            [
                ("pca", PCA()),
                ("ridge", Ridge()),
            ]
        )
    elif model == "lasso":
        param_grid = {
            "pca__n_components": [0.8, 0.9],
            "lasso__alpha": [1e-1, 1],
        }
        pipeline = Pipeline(
            [
                ("pca", PCA()),
                ("lasso", Lasso()),
            ]
        )
    elif model == "linear":
        param_grid = {
            "pca__n_components": [0.8, 0.9],
        }
        pipeline = Pipeline(
            [
                ("pca", PCA()),
                ("linear", LinearRegression()),
            ]
        )
    elif model == "mlp":
        param_grid = {
            "mlp__hidden_layer_sizes": [(100,), (200,), (300,), (100, 100), (200, 200), (300, 300), (300, 200, 100)],
            "mlp__activation": ["relu"],
            "mlp__solver": ["adam"],
            "mlp__alpha": [0.001, 0.01],
            "mlp__max_iter": [10, 20, 30, 40, 50, 80, 100],
        }
        pipeline = Pipeline(
            [
                ("mlp", MLPRegressor(random_state=42)),
            ]
        )
    else:
        raise ValueError(f"Invalid model: {model}")
    return pipeline, param_grid


# Create custom cross-validator that uses predefined splits
class PredefinedSplitsCV(BaseCrossValidator):
    """Custom cross-validator using predefined splits from files"""
    def __init__(self, load_splits_func, num_folds, global_to_local):
        self.load_splits_func = load_splits_func
        self.num_folds = num_folds
        self.global_to_local = global_to_local
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.num_folds
    
    def split(self, X, y=None, groups=None):
        """Generate train/validation splits with local indices"""
        for fold_idx in range(self.num_folds):
            train_indices_global, val_indices_global, _ = self.load_splits_func(fold_idx)
            # Map to local indices
            train_indices_local = np.array([self.global_to_local[idx] for idx in train_indices_global if idx in self.global_to_local])
            val_indices_local = np.array([self.global_to_local[idx] for idx in val_indices_global if idx in self.global_to_local])
            yield train_indices_local, val_indices_local




# Arguments: dataset.h5 model feature_extractor [splits_dir] [num_folds]
dataset_path = sys.argv[1]
model = sys.argv[2]
feature_extractor = sys.argv[3]
splits_dir = sys.argv[4] if len(sys.argv) > 4 else "./bin/class_MKI67"
num_folds = int(sys.argv[5]) if len(sys.argv) > 5 else 10

# Load dataset CSV to get slide_id order
dataset_csv_path = f"{splits_dir}/dataset.csv"
dataset_df = pd.read_csv(dataset_csv_path)

# Load features and targets from h5
with h5py.File(dataset_path, "r") as f:
    X_features = f["features"][:]
    y = f["target"][:]
    # Get slide_ids if available, otherwise use dataset.csv order
    if 'slide_ids' in f:
        slide_ids = [s.decode('utf-8') for s in f['slide_ids'][:]]
    else:
        # Assume h5 order matches dataset.csv order (as created by import_features)
        slide_ids = dataset_df['slide_id'].values.tolist()

# Apply IQR outlier removal and track which indices remain
X_features, y, mask = iqr(X_features, y)

# Filter slide_ids using the mask to keep only remaining samples
remaining_slide_ids = [slide_ids[i] for i in range(len(slide_ids)) if mask[i]]

# Create mapping from slide_id to index in filtered dataset
slide_to_idx = {slide_id: idx for idx, slide_id in enumerate(remaining_slide_ids)}

# Load all folds and prepare data splits
def load_splits(fold_idx):
    """Load splits for a given fold and return train/val/test indices"""
    splits_path = f"{splits_dir}/splits_{fold_idx}_bool.csv"
    splits_df = pd.read_csv(splits_path, index_col=0)
    
    train_slides = splits_df[splits_df['train']].index.tolist()
    val_slides = splits_df[splits_df['val']].index.tolist()
    test_slides = splits_df[splits_df['test']].index.tolist()
    
    train_indices = np.array([slide_to_idx[s] for s in train_slides if s in slide_to_idx])
    val_indices = np.array([slide_to_idx[s] for s in val_slides if s in slide_to_idx])
    test_indices = np.array([slide_to_idx[s] for s in test_slides if s in slide_to_idx])
    
    return train_indices, val_indices, test_indices

# Load fold 0 for final test evaluation (test set is the same across folds)
_, _, test_indices = load_splits(0)
X_test = X_features[test_indices]
y_test = y[test_indices]

# Collect all unique train+val indices from all folds
all_train_val_indices_set = set()
for fold_idx in range(num_folds):
    train_indices, val_indices, _ = load_splits(fold_idx)
    all_train_val_indices_set.update(train_indices)
    all_train_val_indices_set.update(val_indices)

# Create sorted array of all train+val indices
all_train_val_indices_array = np.array(sorted(all_train_val_indices_set))
X_train_val = X_features[all_train_val_indices_array]
y_train_val = y[all_train_val_indices_array]

# Create mapping from global indices (in full dataset) to local indices (in X_train_val)
global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(all_train_val_indices_array)}

pipeline, param_grid = create_pipeline(model)

# Create cross-validator
cv_splitter = PredefinedSplitsCV(load_splits, num_folds, global_to_local)

# Use GridSearchCV with custom cross-validator
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_splitter,
    scoring=["r2", "neg_mean_squared_error"],
    verbose=3,
    # Use single core to avoid loky resource_tracker issues inside containers
    n_jobs=16,
    return_train_score=True,
    refit="r2",
)

grid_search.fit(X_train_val, y_train_val)

# Save CV results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv(f"{feature_extractor}.{model}.cv_result.csv", index=False)

best_pipeline = grid_search.best_estimator_
results = []
for fold in range(num_folds):
    train_indices, val_indices, test_indices = load_splits(fold)
    X_train = X_features[train_indices]
    y_train = y[train_indices]
    X_test = X_features[test_indices]
    y_test = y[test_indices]
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    test_metrics = eval_test_metrics(y_test, y_pred)
    test_metrics["fold"] = fold
    results.append(test_metrics)
    y_predictions = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    y_predictions.to_csv(f"{feature_extractor}.{model}.{fold}.test_predictions.csv", index=False)

test_metrics = pd.concat(results)
test_metrics["feature_extractor"] = feature_extractor
test_metrics["model"] = model
print(test_metrics)
test_metrics.to_csv(f"{feature_extractor}.{model}.test_metrics.csv", index=False)

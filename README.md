## HistologyLinearProbing

<p align="center">
  <img src="imgs/logo.png" alt="Histology Linear Probing Pipeline" width="40%"/>
</p>

**Linear probing pipeline** for histopathology to evaluate different **feature extractors** (foundation models) using **Elastic Net** regression/classification on genes of interest (for example, *MKI67* and *ESR1*).

The workflow is implemented in **Nextflow DSL2** and uses containers (Wave/Singularity) to run both the Python part (feature import and grid search) and the R part (visualizations).

---

### Pipeline overview

- **`main.nf`**  
  Orchestrates the pipeline:
  - Reads the clinical/gene-expression dataset (`params.dataset`).
  - Reads the list of feature extractors from `params/feature_extractors.csv` (automatically loaded).
  - Uses `params.features_dir` to construct feature directory paths.
  - Launches:
    - `split_dataset`: splits the dataset into train/val/test folds for cross-validation.
    - `import_features`: builds `.h5` files with features + target for each feature extractor.
    - `grid_search_workflow`: runs grid-search for regression or binary classification, depending on `params.task`.
    - `concat_results`: concatenates all test metrics into a single summary file.
    - `summary_plot`: generates a global performance boxplot (R² or ROC AUC).

- **`modules/grid_search.nf`**
  - `process split_dataset`: runs `bin/make_splits.py` to create train/val/test splits for cross-validation.
  - `process import_features`: runs `bin/import_features.py` to combine features and targets into `.h5` files.
  - `process grid_search`: runs either the regression or classification script for each `feature_extractor × model` combination and publishes:
    - `*.cv_result.csv` (full cross-validation results)
    - `*.test_metrics.csv` (test set metrics)
    - `*.test_predictions.csv` (test set predictions)
    - `*.pipeline.joblib` (trained model pipeline)
    - `*.best_params.json` (best hyperparameters)
  - `process concat_results`: concatenates all test metrics into a single `summary.csv` file.

- **`workflows/grid_search.nf`**
  - Defines the `grid_search_workflow` workflow, which:
    - Runs `grid_search` with:
      - `grid_search_classification.py` when `params.task == "classification"`.
      - `grid_search_regression.py` when `params.task == "regression"`.
    - For regression, generates scatterplots for each prediction file with `scatter.R`.
    - For classification, generates ROC curves for each prediction file with `roc_curve.R`.

- **`workflows/visualization.nf`**
  - Defines `summary_plot`, which:
    - For regression, calls `boxplot` with `boxplot_r2.R` (R² boxplot).
    - For classification, calls `boxplot` with `boxplot_auc.R` (ROC AUC boxplot).

- **`modules/visualization.nf`**
  - `process scatterplot`: generates `y_true` vs `y_pred` scatterplots from regression outputs.
  - `process roc_auc_curve`: generates ROC curves from classification outputs.
  - `process boxplot`: wraps the R boxplot scripts (`boxplot_r2.R` or `boxplot_auc.R`).

- **`bin/`**
  - `make_splits.py`: creates train/val/test splits for cross-validation (5-fold by default).
  - `import_features.py`: loads the clinical/expression CSV, collects features by `slide_id`, and writes one `.h5` per extractor.
  - `grid_search_regression.py`: applies optional IQR filtering, runs `GridSearchCV` with Elastic Net regression, and saves results and predictions for regression tasks.
  - `grid_search_classification.py`: runs `GridSearchCV` with Elastic Net logistic regression, and saves results and predictions for binary classification tasks.
  - `scatter.R`: reads each regression `*test_predictions.csv` and generates `*.scatterplot.png`.
  - `roc_curve.R`: reads each classification `*test_predictions.csv` and generates `*.roc_auc_curve.png`.
  - `boxplot_r2.R`: reads all regression `*cv_result.csv` files and generates an R² `boxplot.png`.
  - `boxplot_auc.R`: reads all classification `*cv_result.csv` files and generates a ROC AUC `boxplot.png`.

---

### Inputs

- **Expression/metadata file** (`params.dataset`)
  - CSV with at least:
    - A `slide_id` column to link samples with feature files.
    - Columns with genes of interest (for example `MKI67`, `ESR1`).
  - Example structure:
    ```csv
    slide_id,ESR1,MKI67
    slide_1,0.534,0.123
    slide_2,0.868,0.456
    ...
    ```

- **Feature extractors configuration** (`params/feature_extractors.csv`)
  - CSV file automatically loaded by the pipeline (located in `params/` directory).
  - Required columns:
    - `patch_encoder`: patch-level encoder name (e.g. `uni_v1`, `virchow`, `ctranspath`).
    - `slide_encoder`: slide-level aggregation method (e.g. `mean-uni_v1`, `titan`, `chief`, `prism`).
    - `patch_size`: patch size in pixels (e.g. `256`, `224`, `512`).
    - `mag`: magnification level (e.g. `20`).
    - `batch_size`: batch size used during feature extraction (e.g. `200`).
    - `overlap`: overlap in pixels (e.g. `0`).
  - Example:
    ```csv
    patch_encoder,slide_encoder,patch_size,mag,batch_size,overlap
    uni_v1,mean-uni_v1,256,20,200,0
    virchow,mean-virchow,224,20,200,0
    ctranspath,chief,256,20,200,0
    ```

- **Features directory** (`params.features_dir`)
  - Base directory path where feature directories are located.
  - Feature directories follow the pattern: `{features_dir}{mag}x_{patch_size}px_{overlap}px_overlap/slide_features_{slide_encoder}/`
  - Each feature directory should contain one `.h5` file per slide (named `{slide_id}.h5`).

- **Pipeline parameters** (YAML files in `params/`)
  - The key parameters are:
    - `dataset`: path to the CSV with expression/metadata.
    - `features_dir`: base directory path where feature directories are located.
    - `outdir`: output directory for this run (default: `./results/`).
    - `target`: column name of the gene/target variable.
    - `task`: `"regression"` or `"classification"`.

  - Examples:
    - **ESR1 regression** (`params/params_esr1_regr.yml`):
      ```yaml
      dataset: './params/regr_MKI67_ESR1.csv'
      features_dir: "/path/to/features/base/directory/"
      outdir: "./results_esr1_regr/"
      target: "ESR1"
      task: "regression"
      ```
    - **ESR1 binary classification** (`params/params_esr1_class.yml`):
      ```yaml
      dataset: './params/class_MKI67_ESR1.csv'
      features_dir: "/path/to/features/base/directory/"
      outdir: "./results_esr1_class/"
      target: "ESR1"
      task: "classification"
      ```
  - Additional param files (e.g. `params_mki67_*`) allow running the same pipeline for MKI67 or other targets.

---

### Outputs

All outputs are written under `params.outdir` (configured in the selected params file):

- **Grid search results**
  - `cv_result/`
    - `feature_extractor.model.cv_result.csv` (full `GridSearchCV` table with cross-validation results).
  - `test_metrics/`
    - `feature_extractor.model.test_metrics.csv` with metrics per fold.
    - `summary.csv` (concatenated test metrics from all feature extractors and models).
    - Regression metrics: `r2`, `mse`, `mae`, `rmse`.
    - Classification metrics: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`.
  - `test_predictions/`
    - Regression: `feature_extractor.model.fold_X.test_predictions.csv` with `y_true`, `y_pred` (continuous).
    - Classification: `feature_extractor.model.fold_X.test_predictions.csv` with `y_true`, `y_score` (score/probability for the positive class).
  - `models/`
    - `feature_extractor.model.pipeline.joblib` (trained model pipeline for each fold).
  - `best_params/`
    - `feature_extractor.model.best_params.json` (best hyperparameters found during grid search).
  - `splits/`
    - Train/val/test split files for cross-validation.
  - `features/`
    - `feature_extractor.h5` (combined features and targets for each extractor).

- **Plots**
  - Regression:
    - `plots/boxplot.png` (when `task: regression`):  
      Distribution of R² by `feature_extractor` and `algorithm`.
    - `plots/*.scatterplot.png`:  
      One scatterplot per regression `*test_predictions.csv` (y_true vs y_pred, with R² in the title).
  - Classification:
    - `plots/boxplot.png` (when `task: classification`):  
      Distribution of ROC AUC by `feature_extractor` and `algorithm`.
    - `plots/*.roc_auc_curve.png`:  
      One ROC curve per classification `*test_predictions.csv` (FPR vs TPR, AUC in the title).

- **Pipeline information**
  - `pipeline_info/` (timeline, report, trace, DAG HTML) generated automatically by Nextflow.

---

### Requirements

- **Nextflow** ≥ 22.x
- Access to Singularity/Wave containers (configured in `nextflow.config`).
- Cluster with **SLURM** if using the `kutral` profile (default in this repo).
- Python dependencies (provided via containers):
  - `h5py`, `numpy`, `pandas`, `scikit-learn`, `tqdm`
- R dependencies (provided via containers):
  - `ggplot2`, `tidyverse`

> **Note**: You do **not** need to manually install the Python/R dependencies: they are provided through the containers declared in `nextflow.config`. The pipeline uses Wave containers from the Seqera community registry.

---

### Basic usage

1. **Load the environment** where Nextflow and Singularity are available.
2. **Configure feature extractors**: Ensure `params/feature_extractors.csv` exists and contains the feature extractor configurations you want to evaluate.
3. **Choose or edit a params file** in `params/` directory:
   - Set `dataset`: path to your CSV with expression/metadata.
   - Set `features_dir`: base directory where feature directories are located.
   - Set `target`: column name of the gene/target variable (e.g., `ESR1`, `MKI67`).
   - Set `outdir`: output directory for this run.
   - Set `task`: `"regression"` or `"classification"`.
4. **Run the pipeline**:

```bash
# ESR1 regression
nextflow run main.nf -profile kutral -params-file params/params_esr1_regr.yml

# ESR1 binary classification
nextflow run main.nf -profile kutral -params-file params/params_esr1_class.yml

# MKI67 regression
nextflow run main.nf -profile kutral -params-file params/params_mki67_regr.yml
```

For **local execution** (without SLURM), you can use the `local` profile defined in `nextflow.config`:

```bash
nextflow run main.nf -profile local -params-file params/params_esr1_regr.yml
```

### Supported models

The pipeline currently uses **Elastic Net** for both regression and classification tasks:

- **Regression**: Elastic Net regression (combines L1 and L2 regularization)
  - Hyperparameters: `alpha` (regularization strength) and `l1_ratio` (mixing parameter)
  - No PCA is applied for Elastic Net (unlike other models that could be added)

- **Classification**: Elastic Net logistic regression (combines L1 and L2 regularization)
  - Hyperparameters: `C` (inverse regularization strength) and `l1_ratio` (mixing parameter)
  - No PCA is applied for Elastic Net

> **Note**: The underlying scripts (`grid_search_regression.py` and `grid_search_classification.py`) support other models (ridge, lasso, linear, MLP), but the workflow is currently configured to only run Elastic Net. To use other models, modify `workflows/grid_search.nf` to include additional algorithms in the `algorithms` list.

Hyperparameters are optimized via `GridSearchCV` with 5-fold cross-validation.

---

### Output directory structure

After running the pipeline, the output directory (`params.outdir`) will have the following structure:

```
results/
├── best_params/              # Best hyperparameters for each model
│   ├── feature_extractor.model.best_params.json
│   └── ...
├── cv_result/                # Full cross-validation results
│   ├── feature_extractor.model.cv_result.csv
│   └── ...
├── features/                 # Combined features and targets
│   ├── feature_extractor.h5
│   └── ...
├── models/                   # Trained model pipelines
│   ├── feature_extractor.model.pipeline.joblib
│   └── ...
├── plots/                    # All generated plots
│   ├── boxplot.png           # Summary boxplot (R² or ROC AUC)
│   ├── *.scatterplot.png     # Regression scatterplots
│   └── *.roc_auc_curve.png   # Classification ROC curves
├── splits/                   # Train/val/test splits
│   ├── target/
│   └── ...
├── test_metrics/             # Test set metrics
│   ├── feature_extractor.model.test_metrics.csv
│   ├── summary.csv           # Concatenated summary
│   └── ...
├── test_predictions/         # Test set predictions
│   ├── feature_extractor.model.fold_X.test_predictions.csv
│   └── ...
└── pipeline_info/            # Nextflow execution reports
    ├── execution_report_*.html
    ├── execution_timeline_*.html
    ├── execution_trace_*.txt
    └── pipeline_dag_*.html
```

---

### Tips and best practices

1. **Feature extractor configuration**: Make sure the `patch_encoder` and `slide_encoder` names in `params/feature_extractors.csv` match the directory structure in your `features_dir`.

2. **Cross-validation**: The pipeline uses 5-fold cross-validation by default. Each fold generates separate test metrics and predictions.

3. **Model selection**: The pipeline currently uses Elastic Net for all feature extractors. The scripts support other models (ridge, lasso, linear, MLP), but they need to be enabled in the workflow configuration.

4. **Memory requirements**: Grid search processes can be memory-intensive. The default configuration allocates 100G for grid search processes. Adjust in `nextflow.config` if needed.

5. **Resume execution**: Nextflow supports resuming failed runs. Use `-resume` flag:
   ```bash
   nextflow run main.nf -profile kutral -params-file params/params_esr1_regr.yml -resume
   ```

---

### Contact

Author: **Gabriel Cabas**  
For questions or suggestions, please open an *issue* or *pull request* in this repository. 
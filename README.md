## HistologyLinearProbing

<p align="center">
  <img src="imgs/logo.png" alt="Histology Linear Probing Pipeline" width="40%"/>
</p>

**Linear probing pipeline** for histopathology to evaluate different **feature extractors** (foundation models) using simple linear models (ridge, lasso and linear/logistic regression) on genes of interest (for example, *MKI67* and *ESR1*).

The workflow is implemented in **Nextflow DSL2** and uses containers (Wave/Singularity) to run both the Python part (feature import and grid search) and the R part (visualizations).

---

### Pipeline overview

- **`main.nf`**  
  Orchestrates the pipeline:
  - Reads the clinical/gene-expression dataset (`params.dataset`).
  - Reads the list of feature extractors from `params/feature_extractors.csv` (automatically loaded).
  - Uses `params.features_dir` to construct feature directory paths.
  - Launches:
    - `import_features`: builds `.h5` files with features + target.
    - `grid_search_workflow`: runs grid-search for regression or binary classification, depending on `params.task`.
    - `summary_plot`: generates a global performance boxplot (R² or ROC AUC).

- **`modules/grid_search.nf`**
  - `process import_features`: runs `bin/import_features.py`.
  - `process grid_search`: runs either the regression or classification script for each `feature_extractor × model (ridge, lasso, linear)` combination and publishes:
    - `*.cv_result.csv`
    - `*.test_metrics.csv`
    - `*.test_predictions.csv`

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
  - `import_features.py`: loads the clinical/expression CSV, collects features by `slide_id`, and writes one `.h5` per extractor.
  - `grid_search_regression.py`: applies optional IQR filtering, runs `GridSearchCV` with PCA + linear model (ridge/lasso/elastic-net/linear regression), and saves results and predictions for regression tasks.
  - `grid_search_classification.py`: runs `GridSearchCV` with PCA + logistic regression variants (ridge/lasso/"linear" no-penalty), and saves results and predictions for binary classification tasks.
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
    - `feature_extractor.model.cv_result.csv` (full `GridSearchCV` table).
  - `test_metrics/`
    - Regression: `r2`, `mse`, `mae`, `rmse`.
    - Classification: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`.
  - `test_predictions/`
    - Regression: `feature_extractor.model.test_predictions.csv` with `y_true`, `y_pred` (continuous).
    - Classification: `feature_extractor.model.test_predictions.csv` with `y_true`, `y_score` (score/probability for the positive class).

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

> Note: you do **not** need to manually install the Python/R dependencies: they are provided through the containers declared in `nextflow.config`.

---

### Basic usage

1. Load the environment where Nextflow and Singularity are available.
2. Ensure `params/feature_extractors.csv` exists and contains the feature extractor configurations you want to evaluate.
3. Choose or edit a params file in `params/` (dataset, features_dir, target, outdir, task).
4. Run the pipeline, for example:

```bash
# ESR1 regression
nextflow run main.nf -profile kutral -params-file params/params_esr1_regr.yml

# ESR1 binary classification
nextflow run main.nf -profile kutral -params-file params/params_esr1_class.yml
```

For local execution (without SLURM), you can use the `local` profile defined in `nextflow.config`:

```bash
nextflow run main.nf -profile local -params-file params/params_esr1_regr.yml
```

---

### Contact

Author: **Gabriel Cabas**  
For questions or suggestions, please open an *issue* or *pull request* in this repository. 
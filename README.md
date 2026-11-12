# Enterprise AutoML Benchmarking Dashboard

A Streamlit application for exploring and comparing a small, fixed set of
regression and classification models on tabular datasets. It can load a few
built-in scikit-learn datasets or accept CSV and Excel uploads, preprocess the
data, train benchmarks, show evaluation metrics, and let you try a single
interactive prediction.

This is an educational and exploratory benchmarking tool. It is not a
production AutoML platform, and its scores should not be treated as a guarantee
of real-world model performance.

## Why this project exists

The project brings common tabular machine-learning steps into one interactive
application:

- Load a built-in example dataset or upload a dataset.
- Select a target column and configure a holdout split, random seed, and row
  limit.
- Apply basic missing-value handling, categorical encoding, and feature
  scaling.
- Compare a fixed collection of regression or classification estimators.
- Inspect metric tables and charts, then make an example prediction with the
  best-scoring model.

Its architectural goal is to keep the Streamlit presentation and routing
separate from data loading, preprocessing, and model evaluation. The components
under `core/` do not import Streamlit, while the `ui/` modules render pages and
call those components.

## Features

### Regression

The Regression Workspace defaults to the California Housing dataset. If that
dataset cannot be fetched, the data loader falls back to scikit-learn's
Diabetes dataset. For uploaded data, the target can be any column; a
non-numeric target is label-encoded, although regression is generally most
meaningful for a continuous numeric target.

The benchmark compares:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Support Vector Regressor (SVR)
6. K-Neighbors Regressor

Metrics shown are R², mean absolute error (MAE), mean squared error (MSE), and
root mean squared error (RMSE). The workspace highlights the model with the
highest R² on the holdout set.

### Classification

The Classification Workspace defaults to the Iris dataset. The data loader
also supports the Wine dataset, although it is not currently offered as a
selection in the UI.

The benchmark compares:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Support Vector Classifier (SVC)
5. K-Nearest Neighbors (KNN)
6. Decision Tree Classifier

Metrics shown are accuracy, macro-averaged precision, macro-averaged recall,
and macro-averaged F1. The workspace highlights the model with the highest
macro F1 on the holdout set. The prediction view also displays class
probabilities when the selected trained model supports them.

### Interactive prediction

After benchmarking, each workspace uses the best-scoring successfully trained
model to make a prediction from controls generated for the feature columns.
Numeric controls are initialized to their training-data means; categorical
controls offer observed category values. When the dataset has more than 12
features, the UI exposes only the first 10 and fills the remaining features
with their training mean or mode.

## How it works

1. **Load data.** The app reads an uploaded CSV or Excel file, or loads a
   built-in dataset through scikit-learn.
2. **Choose the target.** The selected target is removed from the feature
   matrix. The UI defaults to the built-in dataset's target when available,
   otherwise to the last column.
3. **Limit rows.** If the dataset exceeds the selected maximum, the app takes a
   reproducible random sample using the configured seed.
4. **Preprocess.** Entirely empty columns are dropped. Missing targets are
   imputed (median for numeric targets, most frequent value for categorical
   targets). Categorical targets are label-encoded. The data is split into
   training and test partitions. Feature missing values are imputed using
   training-set means or most-frequent values; categorical features are
   label-encoded; and a `StandardScaler` is fitted on training features and
   applied to both partitions.
5. **Benchmark.** Each estimator is fitted on the training partition and
   evaluated on the test partition. The results and fitted estimators are
   retained in Streamlit session state so widget interactions do not
   repeatedly fit models for the same configuration.
6. **Explore results.** The UI shows the metrics, comparison charts, and
   prediction controls.

The configured train split ranges from 50% to 90%, with a default of 80%.
The random seed defaults to 42. The row-limit control ranges from 500 to
10,000 rows, with a default of 2,000; datasets smaller than the selected limit
are used as-is.

## Project layout

```text
.
├── main.py                    # Streamlit app configuration and page routing
├── requirements.txt           # Python package requirements
├── core/
│   ├── __init__.py
│   ├── data_handler.py        # Uploaded-file and built-in dataset loading
│   ├── preprocessor.py        # Imputation, encoding, split, and scaling
│   └── ml_engine.py           # Regression/classification training and metrics
├── ui/
│   ├── __init__.py
│   ├── home.py                # Overview page
│   ├── regression_ui.py       # Regression controls, results, and predictions
│   └── classification_ui.py   # Classification controls, results, and predictions
└── .devcontainer/
    └── devcontainer.json      # Development-container configuration
```

### Main modules

- **`main.py`** sets the Streamlit page title, icon, wide layout, and sidebar
  state; adds global card styling; and registers the three pages using
  Streamlit's `st.Page` and `st.navigation` APIs.
- **`ui/home.py`** introduces the workspaces and describes the project
  architecture.
- **`ui/regression_ui.py`** and **`ui/classification_ui.py`** implement the
  respective data-source controls, target selection, benchmark displays,
  charts, and prediction playgrounds. They also cache benchmark state in the
  current Streamlit session and clear that state when relevant configuration
  changes.
- **`core/data_handler.py`** parses uploaded files and loads supported
  scikit-learn datasets. Built-in data is cached in-process.
- **`core/preprocessor.py`** cleans and transforms the selected dataset and
  returns the transformed partitions plus fitted encoders and scaler used by
  the prediction views.
- **`core/ml_engine.py`** defines the fixed estimators and computes the
  benchmark metrics. It does not depend on Streamlit.

## Requirements and dependencies

The versions below are the minimums declared in `requirements.txt`; there is
no lockfile or declared upper version bound.

| Dependency | Declared minimum | Use |
| --- | --- | --- |
| Python | Not specified in package metadata; the devcontainer uses Python 3.11 | Runtime |
| Streamlit | `>=1.35.0` | Web UI, routing, widgets, and session state |
| pandas | `>=1.5.0` | Tabular data loading and processing |
| NumPy | `>=1.20.0` | Numeric operations and array handling |
| scikit-learn | `>=1.0.0` | Built-in datasets, preprocessing, estimators, and metrics |
| Plotly | `>=5.0.0` | Interactive benchmark charts |
| openpyxl | `>=3.0.0` | Excel `.xlsx` file reading |

Pandas' Excel readers use format-specific engines. The declared
`openpyxl` dependency covers `.xlsx`; `.xls` uploads may also require the
optional `xlrd` package, which is not currently listed in `requirements.txt`.

## Run locally

Use Python 3.11 to match the supplied development-container configuration.
Other versions may work, but are not specified as supported or tested by this
repository.

```bash
git clone <repository-url>
cd <repository-directory>

python -m venv .venv
source .venv/bin/activate
# On Windows PowerShell, activate with:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run main.py
```

Streamlit prints a local URL (typically `http://localhost:8501`) when the
server starts. Open it in a browser and choose a page from the sidebar.

The devcontainer also starts the app with `streamlit run main.py` and forwards
port 8501. Its startup command disables CORS and XSRF protection for that
development environment; those settings should not be copied into a public or
production deployment without a deliberate security review.

## Using uploaded data

1. Open either workspace.
2. Select **Upload CSV/Excel** in the sidebar and choose a `.csv`, `.xlsx`, or
   `.xls` file.
3. Select the target column.
4. Set the train split ratio, random seed, and maximum row count as needed.
5. Review the benchmark table and charts, then adjust the feature controls to
   try a prediction.

Use a rectangular tabular dataset with a header row and a distinct target
column. The selected target must exist and should contain usable values. A
dataset with only a target and no features, too few rows, problematic values,
or data unsuitable for the selected task can fail during preprocessing or
model fitting. Some models may not support every possible input shape or
target distribution.

The file reader chooses the parser from the filename suffix. The current
extension checks are case-sensitive, so use lowercase `.csv`, `.xlsx`, or
`.xls` suffixes.

## Important limitations

- **Fixed model set, not model search.** The app trains a preset group of
  estimators with mostly default parameters. It does not perform automated
  feature selection, hyperparameter tuning, cross-validation, or model
  deployment.
- **Single holdout evaluation.** Metrics come from one random train/test split
  and can vary with the split, dataset size, class imbalance, and random seed.
  Classification splitting is not stratified.
- **Preprocessing is intentionally basic.** Numeric features use mean
  imputation; categorical features use most-frequent imputation and
  `LabelEncoder`; all features are standardized. Label encoding gives
  categories integer order, which can create misleading ordinal relationships
  for some algorithms. There is no dedicated handling for dates, text,
  identifiers, or domain-specific data.
- **Target imputation occurs before the train/test split.** In
  `core/preprocessor.py`, missing target values are imputed using the full
  target column before splitting. This means target-imputation statistics
  are not isolated to the training set and can influence evaluation.
- **Unseen feature categories are simplified.** A category in the test set
  that was not observed in training is mapped to the first training category.
  That is a fallback, not a learned representation of the unknown category.
- **Prediction controls are illustrative.** Numeric controls are bounded by
  observed feature minima and maxima, and hidden features in wide datasets are
  fixed to their mean or mode. These controls do not represent a full data
  collection or validation workflow.
- **Large workloads can be slow or memory-intensive.** Training cost depends
  on the dataset, number of features, and model. The row limit helps, but does
  not guarantee responsive training for all datasets.
- **Model failures are not a full diagnostic system.** The engine catches
  individual estimator exceptions, prints an error, and records missing
  metrics for that model. If the chosen result has no fitted model, the
  prediction view may not be usable.
- **Session-scoped state only.** Fitted models and benchmark results are kept
  in Streamlit session state; they are not saved as reusable model artifacts
  and are not shared across sessions.
- **No authentication or persistence layer is included.** The repository
  does not implement user accounts, database storage, audit logging, or a
  deployment access-control policy.
- **No automated test suite is present in the repository layout.** Validate
  changes and representative datasets manually or add tests before relying on
  the application in a critical workflow.

## Data handling and responsible use

Uploaded data is read into the running application process and processed in
memory by pandas and scikit-learn. The code does not define a database or
explicitly save uploads or fitted models to disk. Deployment platforms may
have their own logging, storage, and retention behavior, so verify the
deployment environment before uploading confidential or personal data.

Review the dataset for sensitive fields before use. The application is intended
for experimentation and education; validate data quality, leakage risks,
fairness, and performance on representative data before using model outputs
to inform consequential decisions.

## Troubleshooting

- **`ModuleNotFoundError`**: activate the intended virtual environment and
  install `requirements.txt`.
- **Excel `.xlsx` read error**: verify that `openpyxl` was installed.
- **Excel `.xls` read error**: install `xlrd` in the environment if needed;
  it is not currently declared as a dependency.
- **California Housing does not load**: the first fetch may require an
  internet connection. If fetching fails, the code attempts to use the
  Diabetes dataset instead.
- **Upload does not appear to work**: check the lowercase extension, the
  selected workspace's required upload control, and that the file contains a
  valid tabular dataset.
- **A benchmark metric is missing**: an individual model may have failed to
  train. Check the Streamlit process output for its `[ERROR]` message and
  inspect the target and feature data.

## Development notes

- Keep UI-specific code in `ui/` and reusable data/model operations in `core/`.
- The current app has no configured formatter, linter, or automated test
  command in the repository files.
- The `.devcontainer/devcontainer.json` configures a Python 3.11 development
  container and installs the listed requirements.

## License

No license file is present in the repository. Unless a license is added, do
not assume that the code is available for redistribution or reuse under an
open-source license.

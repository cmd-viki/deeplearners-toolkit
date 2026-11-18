
# NumCompute-Stream — Assignment 2.2

## Overview

NumCompute-Stream is an individual extension of the original NumCompute group toolkit, developed for Assignment 2.2: Programming Task 2 at the University of Adelaide.

The project implements a decision tree-based machine learning framework designed to process data incrementally in chunks. It supports buffered decision-tree retraining, incremental preprocessing, streaming statistics, classification metrics, pipeline orchestration, and real-time visualisation.

The decision tree is implemented from scratch using Python and NumPy, without external machine learning libraries.

The original `numcompute/` and `deeplearners_toolkit/` implementations are retained separately. All individual assignment development is maintained on the `veeksha-assignment-2-2` branch.

## 1. Requirements

- Python 3.10 or later
- NumPy
- Matplotlib
- Jupyter Notebook (for the demonstration)

The project uses Python's built-in `unittest` framework for automated testing.

No external machine learning or data processing libraries such as scikit-learn, pandas, or PyTorch are required.

### Installation

Install the dependencies:

```bash
python -m pip install numpy matplotlib notebook
```

## 2. Project Structure

The individual assignment is located inside the `numcompute_stream_work/` directory.

```text
numcompute_stream_work/
|
|-- numcompute_stream/
|   |-- tree.py
|   |-- preprocessing.py
|   |-- stats.py
|   |-- metrics.py
|   |-- pipeline.py
|   |-- stream.py
|   |-- io.py
|   |-- visualise.py
|
|-- tests/
|   |-- test_streaming.py
|
|-- benchmark/
|   |-- stream_benchmark.py
|
|-- demo/
|   |-- stream_demo.ipynb
|
|-- README.md
|-- README_STREAM.md
|-- README_ORIGINAL_GROUP.md
|-- NumCompute_Report.pdf
|-- pyproject.toml
```

Additional files, including the original group toolkit and demonstration dataset, are retained within the assignment directory.

## 3. Quick Start

Open a terminal in the `numcompute_stream_work/` directory before executing the commands below.

### Step 1: Run Unit Tests

```bash
python -m unittest discover -s tests -p "test_streaming.py" -v
```

The individual extension contains 48 unit tests covering normal functionality, streaming behaviour, numerical stability, and edge cases.

### Step 2: Run Benchmarks

```bash
python -m benchmark.stream_benchmark
```

This benchmark evaluates:

- Python-loop versus NumPy-vectorised split calculations.
- Decision-tree retraining with different buffer sizes.
- Training time and retraining frequency.
- Prequential accuracy and holdout accuracy.

### Step 3: Launch the Demonstration

```bash
jupyter notebook demo/stream_demo.ipynb
```

Start Jupyter from the `numcompute_stream_work/` directory so the demonstration can resolve relative dataset paths.

The notebook demonstrates CSV loading, chunk-wise processing, incremental training, prediction, metric logging, and visualisation.

## 4. Decision Tree Implementation

The `DecisionTreeClassifier` implements a classification tree from scratch using NumPy.

### Supported Features

- Gini impurity and entropy splitting criteria.
- Configurable maximum tree depth.
- Minimum samples required for splitting.
- Configurable feature selection through `max_features`.
- Numerical feature threshold evaluation.
- Deterministic class tie resolution.
- Handling of missing feature values.
- Constant-feature and unsplittable-node handling.
- Buffered retraining for streaming data.

### Buffered Retraining

The `partial_fit()` method adds incoming samples to an internal buffer.

Once the buffer contains at least `buffer_size` samples, the tree is retrained on the buffered data and the buffer is cleared.

Between retraining events, predictions use the most recently fitted tree.

### Example

```python
from numcompute_stream import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=5,
    buffer_size=100
)

for X_chunk, y_chunk in stream:
    tree.partial_fit(X_chunk, y_chunk)

    if tree.root_ is not None:
        predictions = tree.predict(X_chunk)
        print(predictions)
```

Here, `stream` represents an iterable yielding feature and label chunks.

The buffered implementation uses periodic retraining rather than continuously updating an existing tree.

## 5. Incremental Preprocessing

The `preprocessing.py` module provides streaming-compatible preprocessing components.

### StandardScaler

Maintains running sample counts, means, and variances using numerically stable incremental updates.

Zero-variance features are handled using a safe scaling value.

### Imputer

Updates missing-value estimates incrementally using observed values from incoming chunks.

### OneHotEncoder

Learns new categories as data arrives and expands its encoded feature representation.

These components support incremental updates through `partial_fit()`.

## 6. Streaming Statistics

The `stats.py` module provides the `StreamingStats` component.

Supported functionality includes:

- Incremental mean calculation.
- Incremental variance calculation.
- Quantiles.
- Histograms.
- Chunk-wise statistical updates.
- Optional retained-value windows.

Running moments are maintained incrementally.

Exact quantile and histogram calculations use retained observations. Consequently, these operations are not constant-memory.

## 7. Streaming Classification Metrics

The `metrics.py` module provides `StreamingMetrics` for evaluating predictions across incoming data chunks.

Supported metrics include:

- Accuracy.
- Error rate.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Binary AUC.
- Rolling-window metrics.

The module supports:

```python
update(y_true_chunk, y_pred_chunk)
reset()
result()
```

Metrics can accumulate across chunks, allowing model performance to be monitored throughout the streaming process.

Binary AUC requires prediction scores and retains the scores needed for exact calculation.

## 8. Incremental Pipeline

The `pipeline.py` module supports chained preprocessing and model components.

It enables incremental training using `partial_fit()` and prediction using the configured transformation pipeline.

Example:

```python
from numcompute_stream import (
    Pipeline,
    StandardScaler,
    DecisionTreeClassifier
)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", DecisionTreeClassifier(
        max_depth=5,
        buffer_size=100
    ))
])

for X_chunk, y_chunk in stream:
    pipe.partial_fit(X_chunk, y_chunk)
```

The pipeline allows preprocessing and model training to be combined into a reusable workflow.

## 9. Streaming Trainer

The `stream.py` module implements `StreamTrainer`.

It coordinates chunk-wise evaluation, model training, and performance logging.

The trainer supports:

- `fit_chunk(X, y)`
- `score_chunk(X, y)`
- Per-chunk metric logging.
- Cumulative accuracy tracking.
- Retraining-count tracking.
- Buffer-memory monitoring.

The streaming workflow uses prequential evaluation, where incoming data is evaluated before the model is updated using that chunk.

This helps simulate real-world scenarios in which predictions are made before new labels are incorporated into training.

## 10. CSV Input and Output

The `io.py` module provides custom CSV data-loading functionality without pandas.

It includes chunk-based CSV iteration for processing datasets incrementally.

This allows the framework to simulate streaming data from a CSV source while avoiding the need to load the entire dataset into memory at once.

## 11. Visualisation

The `visualise.py` module provides reusable Matplotlib plotting functions.

### Available Functions

**1. `plot_metric_over_time()`**

Plots a performance metric across incoming data chunks.

```python
plot_metric_over_time(
    metric_values,
    title,
    ylabel
)
```

**2. `compare_models()`**

Compares streaming performance between two models or configurations.

```python
compare_models(
    metric1,
    metric2,
    labels
)
```

**3. `plot_predictions_vs_ground_truth()`**

Visualises model predictions against actual labels.

```python
plot_predictions_vs_ground_truth(
    y_true,
    y_pred
)
```

These functions support visual analysis of streaming performance and are used in the demonstration workflow.

## 12. Testing and Debugging

The individual extension includes 48 unit tests implemented using Python's standard-library `unittest` framework.

Tests cover:

- Buffered retraining thresholds.
- Predictions between retraining events.
- Oversized input chunks.
- Retraining counts.
- Gini and entropy calculations.
- Deterministic class ties.
- NaN handling and routing.
- Constant feature columns.
- Multiclass classification.
- Invalid hyperparameters.
- Input shape mismatches.
- Incremental mean and variance.
- Missing-value imputation.
- One-hot category expansion.
- Rolling classification metrics.
- AUC tie handling.
- CSV chunk iteration.
- Pipeline integration.

### Running Tests

```bash
python -m unittest discover -s tests -p "test_streaming.py" -v
```

The individual test suite contains 48 passing tests in the verified execution.

The original group toolkit's complete legacy test suite is not claimed to have been independently verified.

## 13. Benchmarking and Performance

Benchmarking compares conventional Python-loop calculations with NumPy-vectorised implementations.

It also examines how different buffer sizes affect retraining frequency, execution time, and predictive performance.

### Buffer Size Comparison

The following results were recorded during the benchmark execution documented in the technical report.

| Buffer Size | Retrains | Train Time (s) | Prequential Accuracy | Holdout Accuracy |
|---|---:|---:|---:|---:|
| 50 | 20 | 0.0780 | 88.95% | 93.0% |
| 150 | 6 | 0.0399 | 93.76% | 96.0% |
| 300 | 3 | 0.0313 | 96.86% | 99.0% |
| 600 | 1 | 0.0209 | 97.50% | 96.5% |

### Vectorisation Benchmark

Recorded results:

- Python-loop split scan: 0.79975 seconds.
- NumPy-vectorised split scan: 0.000905 seconds.
- Measured speedup: approximately 884.09 times.

Both approaches produced the same reported impurity gain and split threshold.

These results demonstrate the performance advantage of vectorised split calculations for the tested workload.

The measured speedup applies to the split-search calculation, not the complete decision-tree training process.

Execution times are hardware-dependent and may differ between runs.

## 14. Design Decisions and Limitations

The framework prioritises transparent algorithmic implementation, numerical stability, modularity, and streaming compatibility.

Several limitations are acknowledged.

### Buffered Learning

The decision tree is retrained using buffered observations only.

Previously consumed training batches are not retained after the buffer is cleared.

Consequently, the model may not respond optimally to every form of concept drift.

### Feature Selection

The current `max_features` implementation selects a deterministic prefix of features rather than a random feature subset.

### Dynamic One-Hot Encoding

The one-hot encoder can discover new categories during streaming.

This can increase the feature dimension after a decision tree has already been fitted.

Production use should freeze the category schema or coordinate tree rebuilding when the schema changes.

### Memory Usage

Exact AUC, quantile, and histogram calculations retain observations or prediction scores.

These operations therefore do not guarantee constant-memory usage.

### Rolling Statistics

The optional retained-value window applies to quantile and histogram calculations, while running moments continue to represent cumulative observations.

### Preprocessing and Evaluation

Pipeline preprocessors are updated before transforming the current training chunk.

For strict prequential evaluation, `StreamTrainer` performs scoring before updating the model.

### Benchmark Interpretation

Benchmark results depend on the input dataset, buffer configuration, and execution environment.

Prequential accuracy comparisons may also be influenced by when trained models become available.

## 15. Future Improvements

Potential improvements include:

- Feature-schema versioning.
- Coordinated model rebuilding after category expansion.
- Approximate bounded-memory quantiles and histograms.
- Concept-drift monitoring.
- Randomised feature subset selection.
- Additional multi-seed benchmarking.
- Improved support for changing data distributions.

## 16. Reproducibility

To reproduce the main results:

1. Open the `numcompute_stream_work/` directory.
2. Install NumPy, Matplotlib, and Jupyter Notebook.
3. Run the streaming unit tests.
4. Execute the benchmark module.
5. Launch `demo/stream_demo.ipynb`.
6. Run the notebook cells in sequence.
7. Inspect the generated metrics, predictions, and plots.

The benchmark uses seeded synthetic data to support reproducible comparisons.

Wall-clock execution times may vary across systems.

## 17. Assignment Submission

The individual assignment submission consists of:

1. GitHub repository link pointing to the individual assignment branch.
2. PDF technical report (maximum four pages).
3. Demonstration video (maximum five minutes).

### GitHub Branch

`veeksha-assignment-2-2`

### Assignment Directory

`numcompute_stream_work/`

### Repository

https://github.com/cmd-viki/deeplearners-toolkit/tree/veeksha-assignment-2-2/numcompute_stream_work

### Technical Report

`NumCompute_Report.pdf`

### Demonstration Notebook

`demo/stream_demo.ipynb`

---

**NumCompute-Stream — Assignment 2.2**

Individual implementation extending the original NumCompute toolkit with buffered decision-tree learning, incremental data processing, streaming evaluation, and Matplotlib visualisation.

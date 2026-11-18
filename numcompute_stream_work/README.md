# NumCompute-Stream — Assignment 2.2

Individual extension of the original group toolkit. The original `numcompute/` and `deeplearners_toolkit/` directories are retained unchanged. **Do not commit to the group main branch.**

## Requirements
Python 3.10+, NumPy, matplotlib. Tests use standard-library `unittest`. No sklearn/pandas.

## Quick start (from repository root)
```bash
python -m unittest discover -s tests -p 'test_streaming.py' -v
python -m benchmark.stream_benchmark
jupyter notebook demo/stream_demo.ipynb
```
For the notebook, start Jupyter in the repository root so relative CSV paths work. If Jupyter is unavailable, run the equivalent imports and pipeline from a Python script.

## Example
```python
from numcompute_stream import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=5, buffer_size=100)
for X_chunk, y_chunk in stream:
    tree.partial_fit(X_chunk, y_chunk)
    if tree.root_ is not None:
        print(tree.predict(X_chunk))
```

## Architecture
- `tree.py`: deterministic CART classification tree, entropy/Gini, learned NaN routing, buffer retraining.
- `preprocessing.py`: incremental mean imputer, running-moment standard scaler, expanding one-hot categories.
- `stats.py`: running moments, exact quantiles from retained values, histogram.
- `metrics.py`: cumulative and rolling binary metrics, confusion matrix, exact AUC from stored scores.
- `pipeline.py`: chained incremental transformers and model.
- `stream.py`: prequential training and metric/buffer-memory logs.
- `io.py`: NumPy CSV loader and bounded-memory CSV chunk iterator.
- `visualise.py`: three reusable matplotlib plotting functions.

## Important limitations and design choices
- Trees are retrained from buffered samples **only**; they do not retain earlier training samples.
- The tree's `max_features` uses the first N features deterministically, rather than a random feature subset.
- One-hot category expansion changes feature width. With a fitted tree, a new category before the next buffer flush can cause a feature-count mismatch; use fixed categories or rebuild the tree when the schema changes.
- StreamingMetrics is binary; AUC needs supplied scores. Exact AUC retains individual scores and is not constant-memory.
- Exact quantiles/histograms retain values; `window` limits retained values for those operations, not lifetime moments.
- Pipeline preprocessors update before the current chunk is transformed; for strict prequential evaluation, `StreamTrainer` scores before updating.
- The benchmark measures elapsed time on the current machine and must be run to obtain genuine results.

## Submission checklist
Create your **own fork and individual branch**, commit changes, run tests/benchmark, execute notebook and capture figures, record an <=5-minute MP4 demo, and submit the repository link, report PDF and video.

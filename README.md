# GOSEIDATASET

A Python library for synthetic dataset generation, missing value simulation, time series generation, dataset merging, and supervised learning utilities.

## Features

### Dataset Generation

* Generate completely random datasets
* Generate datasets with user-defined correlations
* Generate datasets using mathematical formulas
* Control data types (`int` or `float`)
* Apply bias towards high, low, or center values

### Missing Value Simulation

* MCAR (Missing Completely At Random)
* MAR (Missing At Random)
* MNAR (Missing Not At Random)
* Random Missing Values
* Column-wise Missing Values
* Row-wise Missing Values
* Block Missing Values
* Consecutive Missing Values
* Correlation-Based Missing Values

### Time Series Generation

* Timestamp Series
* Year Series
* Month Series
* Week Series
* Day Series
* Hour Series
* Minute Series
* Second Series
* Manual Time Insertion

### Dataset Merging

* Merge datasets with the same target
* Vertical dataset stacking
* Missing value imputation after merging

### Supervised Learning Utilities

* Compare models across datasets
* Weighted ensemble generation
* Mean imputation
* Median imputation
* Mode imputation
* KNN imputation
* Iterative imputation
* Random Forest imputation
* XGBoost imputation

---

# Installation

```bash
pip install goseidataset
```

---

# Quick Start

```python
from goseidataset import DatasetGenerator

dg = DatasetGenerator()

df = dg.generate_random(
    n_rows=100,
    constraints={
        "age": [18, 60],
        "salary": [30000, 100000]
    }
)

print(df.head())
```

---

# DatasetGenerator

## Random Dataset Generation

```python
from goseidataset import DatasetGenerator

dg = DatasetGenerator()

df = dg.generate_random(
    n_rows=100,
    constraints={
        "sleep": [4, 10],
        "revision": [0, 8],
        "session": ["Morning", "Evening"]
    }
)

print(df.head())
```

---

## Random Dataset with Bias

```python
df = dg.generate_random(
    n_rows=100,
    constraints={
        "sleep": [4, 10],
        "revision": [0, 8]
    },
    bias={
        "sleep": "high",
        "revision": "center"
    }
)
```

Supported Bias Types:

* `"high"`
* `"low"`
* `"center"`
* Numeric value representing desired mean

---

## Data Type Control

```python
df = dg.generate_random(
    n_rows=100,
    constraints={
        "age": [18, 60],
        "cgpa": [6, 10]
    },
    dtypes={
        "age": "int",
        "cgpa": "float"
    }
)
```

---

## Correlated Dataset Generation

```python
df = dg.generate_correlated(
    n_rows=1000,
    target="marks",
    correlations={
        "hours": 0.8,
        "stress": -0.5
    },
    constraints={
        "marks": [0, 100],
        "hours": [0, 12],
        "stress": [0, 100]
    }
)

print(df.head())
```

---

## Correlation Report

```python
result = dg.generate_correlated(
    n_rows=1000,
    target="marks",
    correlations={
        "hours": 0.8
    },
    constraints={
        "marks": [0, 100],
        "hours": [0, 12]
    },
    return_report=True
)

dataset = result["dataset"]
report = result["correlation_report"]

print(report)
```

---

## Formula-Based Dataset Generation

```python
df = dg.generate_formula(
    n_rows=500,
    formula="hours*10 + revision*5",
    constraints={
        "hours": [1, 10],
        "revision": [0, 5],
        "marks": [0, 125]
    },
    target="marks"
)

print(df.head())
```

---

# MissingValueGenerator

```python
from goseidataset import MissingValueGenerator

mv = MissingValueGenerator(df)
```

---

## Random Missing Values

```python
result = mv.random_missing(
    percentage=20
)
```

---

## Column Missing Values

```python
result = mv.column_missing(
    columns=["marks"],
    percentage=30
)
```

---

## Row Missing Values

```python
result = mv.row_missing(
    percentage=20
)
```

---

## MCAR

```python
result = mv.mcar(
    column="marks",
    percentage=25
)
```

---

## MAR

```python
result = mv.mar(
    source_column="hours",
    target_column="marks",
    percentage=25
)
```

---

## MNAR

```python
result = mv.mnar(
    column="marks",
    percentage=25
)
```

---

## Consecutive Missing

```python
result = mv.consecutive_missing(
    column="marks",
    percentage=20
)
```

---

## Correlation-Based Missing

```python
result = mv.correlation_based_missing(
    source_column="hours",
    target_column="marks",
    percentage=20,
    sort="descending"
)
```

Supported sort values:

* `"descending"`
* `"ascending"`

---

# TimeSeriesGenerator

```python
from goseidataset import TimeSeriesGenerator

ts = TimeSeriesGenerator(df)
```

---

## Timestamp Series

```python
result = ts.timestamp_series()
```

---

## Year Series

```python
result = ts.year_series()
```

---

## Month Series

```python
result = ts.month_series()
```

---

## Week Series

```python
result = ts.week_series()
```

---

## Day Series

```python
result = ts.day_series()
```

---

## Hour Series

```python
result = ts.hour_series()
```

---

## Minute Series

```python
result = ts.minute_series()
```

---

## Second Series

```python
result = ts.second_series()
```

---

# Supervised Learning

```python
from goseidataset import Supervised_learning
```

---

## Initialize

```python
sl = Supervised_learning(
    dataset_a,
    dataset_b,
    target="Retention"
)
```

---

## Merge Datasets

```python
merged = sl.merge_same_target()
```

---

## Compare Models

```python
from sklearn.ensemble import RandomForestRegressor

result = sl.compare_models(
    model=RandomForestRegressor()
)

print(result)
```

---

## Weighted Ensemble

```python
result = sl.weighted_ensemble(
    model=RandomForestRegressor()
)

ensemble_dataset = result["ensemble_dataset"]
```

---

## Vertical Stack

### Mean Imputation

```python
result = sl.vertical_stack(
    method="mean"
)
```

### Median Imputation

```python
result = sl.vertical_stack(
    method="median"
)
```

### Mode Imputation

```python
result = sl.vertical_stack(
    method="mode"
)
```

### KNN Imputation

```python
result = sl.vertical_stack(
    method="knn",
    n_neighbour=3
)
```

### Iterative Imputation

```python
result = sl.vertical_stack(
    method="iterative"
)
```

### Random Forest Imputation

```python
result = sl.vertical_stack(
    method="random_forest",
    predictors={
        "salary": [
            "age",
            "experience"
        ]
    }
)
```

### XGBoost Imputation

```python
result = sl.vertical_stack(
    method="xgboost",
    predictors={
        "salary": [
            "age",
            "experience"
        ]
    }
)
```

---

# Requirements

* Python 3.9+
* pandas
* numpy
* scikit-learn
* xgboost (optional)

---

# License

MIT License

---

# Author

Harihara Suthan

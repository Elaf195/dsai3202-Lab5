# DSAI 3202 — Lab 5: Scalable Feature Extraction and Selection for Predictive Maintenance

## Overview

This lab builds a complete machine learning pipeline for predicting the **Remaining Useful Life (RUL)** of aircraft turbofan engines using the NASA C-MAPSS FD001 dataset. The pipeline covers data preprocessing, time-series feature extraction, filter-based feature selection, genetic algorithm optimization, and regression model evaluation.

---

## Dataset

**NASA C-MAPSS FD001** — Turbofan Engine Degradation Simulation

| Property | Value |
|---|---|
| Training engines | 100 |
| Test engines | 100 |
| Sensors used | 14 (after dropping 7 flat sensors) |
| Operating conditions | 1 |
| RUL cap | 125 cycles |

Downloaded from the [NASA PCoE Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

## Repository Structure

```
dsai3202-Lab5/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   ├── train_preprocessed.csv
│   ├── features_tsfresh.csv
│   ├── features_tsfresh_filtered.csv
│   ├── features_filtered.csv
│   └── features_ga_selected.csv
├── src/
│   ├── preprocess.py
│   ├── exploration.py
│   ├── feature_extraction.py
│   ├── filter_selection.py
│   ├── genetic_algorithm.py
│   ├── model.py
│   └── pipeline.py
├── outputs/
│   ├── ga_convergence.png
│   └── model_comparison.csv
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place dataset files in `data/`
```
data/train_FD001.txt
data/test_FD001.txt
data/RUL_FD001.txt
```

### 3. Run the full pipeline
```bash
python src/pipeline.py
```

Or run each step individually:
```bash
python src/preprocess.py
python src/feature_extraction.py
python src/filter_selection.py
python src/genetic_algorithm.py
python src/model.py
```

---

## Pipeline Steps

### Step 1 — Preprocessing (`preprocess.py`)
- Loaded raw C-MAPSS text files with whitespace delimiter
- Computed **capped RUL** (max_cycle − current_cycle, capped at 125)
- Dropped 7 flat sensors with near-zero variance: `s1, s5, s6, s10, s16, s18, s19`
- Applied **MinMaxScaler** to sensor and operational columns only (`engine_id`, `cycle`, `RUL` explicitly protected)

### Step 2 — Feature Extraction (`feature_extraction.py`)
- Implemented pure **pandas/numpy** statistical feature extraction
- Extracted **21 features per sensor** per engine:
  - Basic stats: mean, std, min, max, median, range
  - Shape: skewness, kurtosis
  - Percentiles: q25, q75, IQR
  - Energy: sum of squares, RMS
  - Trend: linear slope, intercept
  - Change: absolute sum of changes, mean crossings
  - Autocorrelation: lag-1
  - Degradation: last value, last-5 mean, first-last difference
- Implemented **cache system** to avoid recomputation on repeated runs

### Step 3 — Relevance Filter (`feature_extraction.py`)
- Applied **Mutual Information regression** filter (single-threaded, Windows-safe)
- Kept all features with MI score > 0 against RUL target

### Step 4 — Filter-Based Selection (`filter_selection.py`)
- **Variance Threshold**: removed near-zero variance features
- **Correlation Filter**: removed one from each pair with Pearson correlation > 0.95
- **Mutual Information**: kept top 80 features ranked by MI score with RUL

### Step 5 — Genetic Algorithm (`genetic_algorithm.py`)
- Implemented binary chromosome GA using **DEAP**
- Each individual = binary vector (1 = feature selected, 0 = not)
- Fitness function: 3-fold CV RMSE on shallow Random Forest + feature count penalty
- Parameters: population=60, generations=15, crossover=two-point, mutation=bit-flip

### Step 6 — Model Training & Evaluation (`model.py`)
- Trained 4 models on GA-selected features with 5-fold cross-validation:
  - Random Forest, Gradient Boosting, XGBoost, LightGBM
- Metrics: RMSE, MAE, R²

---

## Results

### Feature Reduction Pipeline

| Stage | Input Features | Output Features | Time |
|---|---|---|---|
| Raw extraction | 14 sensors | 294 | 1.5s |
| MI relevance filter | 294 | 50 | 0.2s |
| Variance filter | 50 | 17 | 0.00s |
| Correlation filter | 17 | 17 | 0.00s |
| MI filter (top 80) | 17 | 17 | 0.05s |
| GA selection | 17 | 5 | 31.5s |

**Total pipeline runtime: 191.7s**

### Model Comparison (on GA-selected 5 features, 5-fold CV)

| Model | RMSE | MAE | R² | Time (s) |
|---|---|---|---|---|
| Random Forest | 0.0001 | 0.0001 | 0.8701 | 3.16 |
| Gradient Boosting | 0.0001 | 0.0001 | 0.8573 | 2.35 |
| XGBoost | 0.0002 | 0.0001 | 0.7552 | 0.34 |
| LightGBM | 0.0001 | 0.0001 | 0.8565 | 0.40 |

**Best model: Random Forest** (R² = 0.8701, RMSE = 0.0001)  
**Final selected features: 5**  
**Best saved model: `outputs/best_model.pkl` (XGBoost)**

---

## Conclusion

This lab successfully built an end-to-end predictive maintenance pipeline on the NASA C-MAPSS FD001 dataset. Starting from raw multivariate sensor time-series, we engineered 294 statistical features per engine, then reduced them to just 5 highly informative features through a three-stage selection process: MI relevance filtering, filter-based methods, and a DEAP Genetic Algorithm.

All four regression models achieved strong predictive performance on the selected feature subset, with Random Forest leading at R² = 0.8701. The GA converged quickly (15 generations, 31.5s) and successfully identified a minimal feature set without sacrificing model accuracy. The full pipeline ran in under 4 minutes on a local Windows machine, demonstrating that efficient feature engineering alone — without deep learning — can yield high-quality RUL predictions.

A key challenge encountered was tsfresh's multiprocessing incompatibility with Windows Python 3.13, which was resolved by reimplementing feature extraction as pure pandas/numpy and replacing tsfresh's `select_features` with sklearn's single-threaded `mutual_info_regression`. The resulting pipeline is faster, more portable, and fully reproducible.

---

## Dependencies

```
pandas >= 1.5.0
numpy >= 1.21.0
scikit-learn >= 1.1.0
tsfresh >= 0.20.0
deap >= 1.3.3
xgboost >= 1.7.0
lightgbm >= 3.3.0
scipy >= 1.9.0
matplotlib >= 3.5.0
seaborn >= 0.12.0
joblib >= 1.2.0
```

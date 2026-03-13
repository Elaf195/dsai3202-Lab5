# src/filter_selection.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import (VarianceThreshold,
                                        mutual_info_regression)
import time

def variance_threshold_filter(df: pd.DataFrame,
                                threshold: float = 0.01) -> pd.DataFrame:
    """Remove features with variance below threshold."""
    t0 = time.time()
    selector = VarianceThreshold(threshold=threshold)
    arr = selector.fit_transform(df)
    kept = df.columns[selector.get_support()]
    result = pd.DataFrame(arr, index=df.index, columns=kept)
    print(f'Variance filter: {df.shape[1]} -> {result.shape[1]} features'
          f' in {time.time()-t0:.2f}s')
    return result

def correlation_filter(df: pd.DataFrame,
                         threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove one feature from each pair with Pearson correlation > threshold.
    Keeps the first in each correlated pair.
    """
    t0 = time.time()
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns
               if any(upper[col] > threshold)]
    result = df.drop(columns=to_drop)
    print(f'Correlation filter: {df.shape[1]} -> {result.shape[1]} features'
          f'  ({len(to_drop)} dropped) in {time.time()-t0:.2f}s')
    return result

def mutual_info_filter(df: pd.DataFrame,
                         target: pd.Series,
                         top_k: int = 80) -> pd.DataFrame:
    """Keep top_k features ranked by mutual information with target."""
    t0 = time.time()
    mi_scores = mutual_info_regression(df, target, random_state=42)
    mi_series = pd.Series(mi_scores, index=df.columns).sort_values(ascending=False)
    top_features = mi_series.head(top_k).index.tolist()
    result = df[top_features]
    print(f'MI filter: {df.shape[1]} -> {result.shape[1]} features'
          f' in {time.time()-t0:.2f}s')
    return result, mi_series

def run_filter_pipeline(features: pd.DataFrame,
                          target: pd.Series,
                          var_thresh: float = 0.01,
                          corr_thresh: float = 0.95,
                          top_k: int = 80) -> pd.DataFrame:
    """Run all three filter steps in sequence."""
    print('=== Filter-Based Feature Selection ===')
    f1 = variance_threshold_filter(features, var_thresh)
    f2 = correlation_filter(f1, corr_thresh)
    f3, mi_scores = mutual_info_filter(f2, target, top_k)
    print(f'Final after filters: {f3.shape[1]} features')
    return f3, mi_scores

if __name__ == '__main__':
    features = pd.read_csv('data/features_tsfresh.csv', index_col='engine_id')
    import pandas as pd
    train_df = pd.read_csv('data/train_preprocessed.csv')
    rul_target = (train_df.sort_values('cycle')
                  .groupby('engine_id')['RUL'].first())
    features_aligned = features.loc[rul_target.index]
    filtered, mi = run_filter_pipeline(features_aligned, rul_target)
    filtered.to_csv('data/features_filtered.csv')
    print('Saved: data/features_filtered.csv')

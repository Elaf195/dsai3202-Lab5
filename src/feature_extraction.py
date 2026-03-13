import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import time
import warnings
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore', category=RuntimeWarning)


def compute_sensor_features(values: np.ndarray, sensor_name: str) -> dict:
    n = len(values)
    result = {}
    p = f'{sensor_name}__'
    v_std = float(np.std(values, ddof=1)) if n > 1 else 0.0

    result[p + 'mean']   = float(np.mean(values))
    result[p + 'std']    = v_std
    result[p + 'min']    = float(np.min(values))
    result[p + 'max']    = float(np.max(values))
    result[p + 'median'] = float(np.median(values))
    result[p + 'range']  = float(np.max(values) - np.min(values))
    result[p + 'skewness'] = float(scipy_stats.skew(values))     if n > 2 else 0.0
    result[p + 'kurtosis'] = float(scipy_stats.kurtosis(values)) if n > 3 else 0.0

    q25, q75 = np.percentile(values, [25, 75])
    result[p + 'q25'] = float(q25)
    result[p + 'q75'] = float(q75)
    result[p + 'iqr'] = float(q75 - q25)

    result[p + 'energy'] = float(np.sum(values ** 2))
    result[p + 'rms']    = float(np.sqrt(np.mean(values ** 2)))

    if n > 1:
        x = np.arange(n, dtype=float)
        slope, intercept, _, _, _ = scipy_stats.linregress(x, values)
        result[p + 'trend_slope']     = float(slope)     if np.isfinite(slope)     else 0.0
        result[p + 'trend_intercept'] = float(intercept) if np.isfinite(intercept) else 0.0
    else:
        result[p + 'trend_slope']     = 0.0
        result[p + 'trend_intercept'] = float(values[0]) if n == 1 else 0.0

    result[p + 'abs_change_sum'] = float(np.sum(np.abs(np.diff(values)))) if n > 1 else 0.0

    if n > 1:
        mean_val = np.mean(values)
        signs = np.sign(values - mean_val)
        signs[signs == 0] = 1
        result[p + 'mean_crossings'] = int(np.sum(np.diff(signs) != 0))
    else:
        result[p + 'mean_crossings'] = 0

    if n > 2 and v_std > 0:
        a, b = values[:-1], values[1:]
        if np.std(a) > 0 and np.std(b) > 0:
            corr = np.corrcoef(a, b)[0, 1]
            result[p + 'autocorr_lag1'] = float(corr) if np.isfinite(corr) else 0.0
        else:
            result[p + 'autocorr_lag1'] = 0.0
    else:
        result[p + 'autocorr_lag1'] = 0.0

    result[p + 'last_value']      = float(values[-1])
    result[p + 'last5_mean']      = float(np.mean(values[-5:]))
    result[p + 'first_last_diff'] = float(values[-1] - values[0])

    return result


def extract_tsfresh_features(df: pd.DataFrame,
                              sensor_cols: list,
                              n_jobs: int = None) -> pd.DataFrame:
    df = df.copy()
    df['engine_id'] = df['engine_id'].astype(float).astype(int)
    engine_ids = sorted(df['engine_id'].unique().tolist())
    n_engines  = len(engine_ids)
    t0 = time.time()

    print(f'Extracting features: {n_engines} engines | '
          f'{len(sensor_cols)} sensors | '
          f'~{len(sensor_cols) * 21} features/engine')

    df_sorted = (df[['engine_id', 'cycle'] + sensor_cols]
                 .sort_values(['engine_id', 'cycle'])
                 .reset_index(drop=True))

    rows = []
    for i, eid in enumerate(engine_ids):
        if (i + 1) % 30 == 0 or (i + 1) == n_engines:
            print(f'  Processed {i+1}/{n_engines} engines...')
        mask      = df_sorted['engine_id'] == eid
        engine_df = df_sorted.loc[mask, ['cycle'] + sensor_cols].sort_values('cycle')
        row = {'engine_id': eid}
        for sensor in sensor_cols:
            values = engine_df[sensor].astype(float).values
            row.update(compute_sensor_features(values, sensor))
        rows.append(row)

    result_df = pd.DataFrame(rows)
    assert len(result_df) == n_engines, \
        f'Row count mismatch: {len(result_df)} vs {n_engines}'
    assert result_df['engine_id'].nunique() == n_engines, \
        'Duplicate engine_ids in output!'

    result_df = result_df.set_index('engine_id').sort_index()
    result_df.index = result_df.index.astype(int)
    result_df = result_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    elapsed = time.time() - t0
    print(f'Extraction done in {elapsed:.1f}s | Shape: {result_df.shape}')
    return result_df


def filter_features_with_tsfresh(features: pd.DataFrame,
                                  target: pd.Series) -> pd.DataFrame:
    """
    MI-based relevance filter — no multiprocessing, safe on Windows.
    Uses n_neighbors=1 to handle small sample sizes safely.
    """
    from sklearn.feature_selection import mutual_info_regression

    common = features.index.intersection(target.index)
    if len(common) == 0:
        raise ValueError('No common engine IDs between features and target!')

    X = features.loc[common]
    y = target.loc[common]
    n_samples = len(X)

    print(f'MI relevance filter: {X.shape[1]} features, {n_samples} engines...')
    t0 = time.time()

    # n_neighbors must be < n_samples — use min(3, n_samples-1)
    n_neighbors = max(1, min(3, n_samples - 1))

    mi_scores = mutual_info_regression(
        X, y,
        random_state=42,
        n_neighbors=n_neighbors,
        n_jobs=1              # single-threaded — no Windows spawn issues
    )
    mi_series = pd.Series(mi_scores, index=X.columns)

    # Keep features with MI > 0
    selected_cols = mi_series[mi_series > 0].index.tolist()

    # Safety floor: always keep at least top 50
    if len(selected_cols) < 50:
        n_keep = min(50, len(mi_series))
        selected_cols = mi_series.nlargest(n_keep).index.tolist()

    selected = X[selected_cols]
    print(f'MI filter: {X.shape[1]} -> {selected.shape[1]} features '
          f'in {time.time()-t0:.1f}s')
    return selected


CACHE_PATH = 'data/features_tsfresh.csv'


def is_cache_valid(cache_path: str, expected_engines: int) -> bool:
    if not os.path.exists(cache_path):
        return False
    try:
        row_count = sum(1 for _ in open(cache_path)) - 1
        if row_count != expected_engines:
            print(f'Cache invalid: {row_count} rows, '
                  f'expected {expected_engines}. Deleting...')
            os.remove(cache_path)
            return False
        return True
    except Exception as e:
        print(f'Cache unreadable ({e}). Deleting...')
        os.remove(cache_path)
        return False


def extract_or_load(train_df: pd.DataFrame,
                    sensor_cols: list,
                    n_jobs: int = None) -> pd.DataFrame:
    expected = train_df['engine_id'].astype(float).astype(int).nunique()
    if is_cache_valid(CACHE_PATH, expected):
        print(f'Loading cached features from {CACHE_PATH}...')
        cached = pd.read_csv(CACHE_PATH, index_col='engine_id')
        cached.index = cached.index.astype(int)
        print(f'Loaded shape: {cached.shape}')
        return cached
    print('No valid cache — running extraction...')
    features = extract_tsfresh_features(train_df, sensor_cols, n_jobs)
    os.makedirs('data', exist_ok=True)
    features.to_csv(CACHE_PATH)
    print(f'Cached to {CACHE_PATH}')
    return features


if __name__ == '__main__':
    train_df = pd.read_csv('data/train_preprocessed.csv')
    train_df['engine_id'] = train_df['engine_id'].astype(float).astype(int)

    print(f'Train shape: {train_df.shape}')
    print(f'Engines:     {train_df["engine_id"].nunique()}')

    sensor_cols = [c for c in train_df.columns
                   if c.startswith('s') and c != 'engine_id']

    features = extract_or_load(train_df, sensor_cols)
    print(f'Features shape: {features.shape}')

    rul_per_engine = (train_df.sort_values('cycle')
                              .groupby('engine_id')['RUL'].first())
    rul_per_engine.index = rul_per_engine.index.astype(int)

    common_ids     = features.index.intersection(rul_per_engine.index)
    features       = features.loc[common_ids]
    rul_per_engine = rul_per_engine.loc[common_ids]

    features_filtered = filter_features_with_tsfresh(features, rul_per_engine)
    features_filtered.to_csv('data/features_tsfresh_filtered.csv')
    print(f'Final shape: {features_filtered.shape}')
    print('Saved: data/features_tsfresh_filtered.csv')
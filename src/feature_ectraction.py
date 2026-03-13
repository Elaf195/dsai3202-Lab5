# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import time

COLUMNS = [
    'engine_id', 'cycle',
    'op1', 'op2', 'op3',
    's1',  's2',  's3',  's4',  's5',  's6',  's7',
    's8',  's9',  's10', 's11', 's12', 's13', 's14',
    's15', 's16', 's17', 's18', 's19', 's20', 's21'
]

# Sensors with near-zero variance in FD001
DROP_SENSORS = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']

RUL_CAP = 125


def load_cmapss(data_dir: str, subset: str = 'FD001'):
    train = pd.read_csv(
        os.path.join(data_dir, f'train_{subset}.txt'),
        sep=r'\s+', header=None, names=COLUMNS
    )
    test = pd.read_csv(
        os.path.join(data_dir, f'test_{subset}.txt'),
        sep=r'\s+', header=None, names=COLUMNS
    )
    rul = pd.read_csv(
        os.path.join(data_dir, f'RUL_{subset}.txt'),
        sep=r'\s+', header=None, names=['RUL']
    )
    return train, test, rul


def compute_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    max_cycle = (df.groupby('engine_id')['cycle']
                   .max()
                   .reset_index()
                   .rename(columns={'cycle': 'max_cycle'}))
    df = df.merge(max_cycle, on='engine_id')
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=cap)
    df.drop(columns=['max_cycle'], inplace=True)
    return df


def drop_flat_sensors(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in DROP_SENSORS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def normalize_sensors(train: pd.DataFrame,
                      test: pd.DataFrame):
    """
    Normalize ONLY sensor and operational columns.
    engine_id, cycle, and RUL are explicitly excluded.
    """
    # Identify columns to normalize — never touch engine_id, cycle, RUL
    exclude = {'engine_id', 'cycle', 'RUL'}
    sensor_cols = [
        c for c in train.columns
        if c not in exclude
    ]

    print(f'Normalizing {len(sensor_cols)} columns: {sensor_cols}')
    print(f'Protecting: {exclude}')

    scaler = MinMaxScaler()

    # Work on copies to be safe
    train = train.copy()
    test  = test.copy()

    train[sensor_cols] = scaler.fit_transform(train[sensor_cols])
    test[sensor_cols]  = scaler.transform(test[sensor_cols])

    return train, test, scaler


def preprocess_pipeline(data_dir: str, subset: str = 'FD001'):
    t0 = time.time()
    print(f'Loading {subset}...')
    train, test, rul = load_cmapss(data_dir, subset)

    # Verify engine_id before anything
    print(f'Raw train engines: {train["engine_id"].nunique()} '
          f'| IDs: {sorted(train["engine_id"].unique())[:5]}...')

    train = compute_rul(train)
    train = drop_flat_sensors(train)
    test  = drop_flat_sensors(test)

    # Verify engine_id survived preprocessing
    print(f'After drop_sensors engines: {train["engine_id"].nunique()}')

    train, test, scaler = normalize_sensors(train, test)

    # Final verification
    print(f'After normalization engines: {train["engine_id"].nunique()}')
    print(f'engine_id dtype: {train["engine_id"].dtype}')
    print(f'engine_id sample: {train["engine_id"].unique()[:5].tolist()}')
    print(f'Preprocessing done in {time.time()-t0:.2f}s')
    print(f'Train shape: {train.shape} | Test shape: {test.shape}')

    return train, test, rul


if __name__ == '__main__':
    train_df, test_df, rul_df = preprocess_pipeline('data/')

    # Sanity check before saving
    assert train_df['engine_id'].nunique() == 100, (
        f"Expected 100 engines, got {train_df['engine_id'].nunique()}. "
        f"Check normalization — engine_id may have been overwritten."
    )
    assert train_df['engine_id'].min() >= 1, (
        f"engine_id starts at {train_df['engine_id'].min()}, expected 1."
    )

    train_df.to_csv('data/train_preprocessed.csv', index=False)
    test_df.to_csv('data/test_preprocessed.csv',   index=False)
    print('Saved: data/train_preprocessed.csv')
    print('Saved: data/test_preprocessed.csv')
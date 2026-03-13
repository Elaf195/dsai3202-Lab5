import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time

COLUMNS = [
    'engine_id', 'cycle',
    'op1', 'op2', 'op3',
    's1',  's2',  's3',  's4',  's5',  's6',  's7',
    's8',  's9',  's10', 's11', 's12', 's13', 's14',
    's15', 's16', 's17', 's18', 's19', 's20', 's21'
]

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
    # NEVER normalize engine_id, cycle, or RUL
    exclude = {'engine_id', 'cycle', 'RUL'}
    sensor_cols = [c for c in train.columns if c not in exclude]

    train = train.copy()
    test  = test.copy()

    scaler = MinMaxScaler()
    train[sensor_cols] = scaler.fit_transform(train[sensor_cols])
    test[sensor_cols]  = scaler.transform(test[sensor_cols])

    return train, test, scaler


def preprocess_pipeline(data_dir: str, subset: str = 'FD001'):
    t0 = time.time()
    train, test, rul = load_cmapss(data_dir, subset)

    # Force clean integer engine_id immediately
    train['engine_id'] = train['engine_id'].astype(float).astype(int)
    test['engine_id']  = test['engine_id'].astype(float).astype(int)

    train = compute_rul(train)
    train = drop_flat_sensors(train)
    test  = drop_flat_sensors(test)
    train, test, scaler = normalize_sensors(train, test)

    # Verify engine_id survived
    assert train['engine_id'].nunique() > 1, (
        f"ERROR: engine_id has only {train['engine_id'].nunique()} unique value after preprocessing!"
    )

    print(f'Preprocessing done in {time.time()-t0:.2f}s')
    print(f'Train shape: {train.shape}, Test shape: {test.shape}')
    return train, test, rul


if __name__ == '__main__':
    train_df, test_df, rul_df = preprocess_pipeline('data/')

    assert train_df['engine_id'].nunique() == 100, (
        f"Expected 100 engines, got {train_df['engine_id'].nunique()}"
    )

    train_df.to_csv('data/train_preprocessed.csv', index=False)
    test_df.to_csv('data/test_preprocessed.csv',   index=False)
    print(f'Engines: {train_df["engine_id"].nunique()}')
    print(f'engine_id sample: {sorted(train_df["engine_id"].unique())[:5]}')
    print('Saved: data/train_preprocessed.csv')
    print('Saved: data/test_preprocessed.csv')
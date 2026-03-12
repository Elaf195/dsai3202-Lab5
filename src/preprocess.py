import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os, time

COLUMNS = ['engine_id','cycle',
           'op1','op2','op3',
           's1','s2','s3','s4','s5','s6','s7','s8','s9','s10',
           's11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

# Sensors with near-zero variance in FD001 - safe to drop
DROP_SENSORS = ['s1','s5','s6','s10','s16','s18','s19']
RUL_CAP = 125  # piecewise linear RUL cap

def load_cmapss(data_dir: str, subset: str = 'FD001'):
    """Load train, test, and RUL files for a given C-MAPSS subset."""
    train = pd.read_csv(
        os.path.join(data_dir, f'train_{subset}.txt'),
        sep=' ', header=None, names=COLUMNS
    ).dropna(axis=1)
    test = pd.read_csv(
        os.path.join(data_dir, f'test_{subset}.txt'),
        sep=' ', header=None, names=COLUMNS
    ).dropna(axis=1)
    rul = pd.read_csv(
        os.path.join(data_dir, f'RUL_{subset}.txt'),
        sep=' ', header=None, names=['RUL']
    ).dropna(axis=1)
    return train, test, rul

def compute_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Add capped RUL column to training dataframe."""
    max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycle, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df['RUL'] = df['RUL'].clip(upper=cap)
    df.drop(columns=['max_cycle'], inplace=True)
    return df

def drop_flat_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Remove sensors with zero or near-zero variance."""
    cols_to_drop = [c for c in DROP_SENSORS if c in df.columns]
    return df.drop(columns=cols_to_drop)

def normalize_sensors(train: pd.DataFrame,
                       test: pd.DataFrame) -> tuple:
    """Fit MinMaxScaler on train, apply to both train and test."""
    sensor_cols = [c for c in train.columns
                   if c.startswith('s') or c.startswith('op')]
    scaler = MinMaxScaler()
    train[sensor_cols] = scaler.fit_transform(train[sensor_cols])
    test[sensor_cols] = scaler.transform(test[sensor_cols])
    return train, test, scaler

def preprocess_pipeline(data_dir: str, subset: str = 'FD001'):
    """Full preprocessing pipeline. Returns train_df, test_df, rul_df."""
    t0 = time.time()
    train, test, rul = load_cmapss(data_dir, subset)
    train = compute_rul(train)
    train = drop_flat_sensors(train)
    test  = drop_flat_sensors(test)
    train, test, _ = normalize_sensors(train, test)
    print(f'Preprocessing done in {time.time()-t0:.2f}s')
    print(f'Train shape: {train.shape}, Test shape: {test.shape}')
    return train, test, rul

if __name__ == '__main__':
    train_df, test_df, rul_df = preprocess_pipeline('data/')
    train_df.to_csv('data/train_preprocessed.csv', index=False)
    test_df.to_csv('data/test_preprocessed.csv', index=False)

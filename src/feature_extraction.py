import pandas as pd
import numpy as np
import time
import os
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters

def extract_tsfresh_features(df: pd.DataFrame, 
                             sensor_cols: list, 
                             n_jobs: int = 1) -> pd.DataFrame:
    """
    Extract time-series features using tsfresh EfficientFCParameters.
    Correctly handles multiple sensors by using engine_id as the ID.
    Returns a DataFrame indexed by engine_id.
    """
    t0 = time.time()
    print(f'Preparing tsfresh input for {df["engine_id"].nunique()} engines...')
    
    # tsfresh expects a dataframe where one column is the ID.
    # We use 'engine_id' as the ID and 'cycle' as the sort column.
    # We only pass the sensor columns for feature extraction.
    ts_df = df[['engine_id', 'cycle'] + sensor_cols]
    
    print('Running tsfresh feature extraction...')
    # tsfresh will automatically handle multiple value columns and 
    # prefix feature names with the column (sensor) name.
    extracted = extract_features(
        ts_df,
        column_id='engine_id',
        column_sort='cycle',
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=n_jobs,
        disable_progressbar=False,
        impute_function=impute # handles NaN/Inf automatically
    )
    
    # The index is already engine_id
    extracted.index.name = 'engine_id'
    
    elapsed = time.time() - t0
    print(f'Extraction done in {elapsed:.1f}s | Shape: {extracted.shape}')
    return extracted

def filter_features_with_tsfresh(features: pd.DataFrame, 
                                 target: pd.Series) -> pd.DataFrame:
    """Use tsfresh built-in statistical filter to keep relevant features."""
    t0 = time.time()
    # Ensure features and target are aligned by index (engine_id)
    common_ids = features.index.intersection(target.index)
    features_aligned = features.loc[common_ids]
    target_aligned = target.loc[common_ids]
    
    selected = select_features(features_aligned, target_aligned, fdr_level=0.05)
    print(f'tsfresh filter: {features.shape[1]} -> {selected.shape[1]} features'
          f' in {time.time()-t0:.1f}s')
    return selected

if __name__ == '__main__':
    # This block is for standalone testing
    if os.path.exists('data/train_preprocessed.csv'):
        train_df = pd.read_csv('data/train_preprocessed.csv')
        sensor_cols = [c for c in train_df.columns 
                      if c.startswith('s') and c != 'engine_id']
        
        features = extract_tsfresh_features(train_df, sensor_cols)
        
        # Build RUL target indexed by engine_id
        # Taking the first RUL (initial RUL) as the target for the whole series
        rul_per_engine = (train_df.sort_values('cycle')
                          .groupby('engine_id')['RUL'].first())
        
        # Initial tsfresh filter
        features_filtered = filter_features_with_tsfresh(features, rul_per_engine)
        
        os.makedirs('data', exist_ok=True)
        features_filtered.to_csv('data/features_tsfresh.csv')
        print('Saved: data/features_tsfresh.csv')
    else:
        print("Data file not found. Please run preprocess.py first.")

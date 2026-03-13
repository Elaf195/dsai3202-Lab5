"""
End-to-end pipeline for Lab 5.
Run: python src/pipeline.py
"""
import time, os, pandas as pd
from preprocess import preprocess_pipeline
from feature_extraction import extract_tsfresh_features, filter_features_with_tsfresh
from filter_selection import run_filter_pipeline
from genetic_algorithm import run_ga, plot_ga_convergence
from model import run_all_models, train_best_model, MODELS

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

PIPELINE_START = time.time()

# Step 1: Preprocess 
print('\n[1/6] Preprocessing...')
train_df, test_df, rul_df = preprocess_pipeline('data/', subset='FD001')

# Identify sensor columns
sensor_cols = [c for c in train_df.columns 
              if c.startswith('s') and c != 'engine_id']

# Build RUL target indexed by engine_id (using the initial RUL for the series)
rul_target = (train_df.sort_values('cycle')
              .groupby('engine_id')['RUL'].first())

# Step 2: tsfresh Extraction
print('\n[2/6] tsfresh feature extraction...')
# Note: n_jobs=1 is safer for some environments, but -1 uses all cores
features_raw = extract_tsfresh_features(train_df, sensor_cols, n_jobs=1)

# Align features with target
features_raw = features_raw.loc[rul_target.index]
features_raw.to_csv('data/features_tsfresh.csv')

# Step 3: tsfresh built-in filter 
print('\n[3/6] tsfresh relevance filter...')
features_ts = filter_features_with_tsfresh(features_raw, rul_target)

# Step 4: Filter methods 
print('\n[4/6] Filter-based selection (Var + Corr + MI)...')
features_filtered, mi_scores = run_filter_pipeline(
    features_ts, rul_target, top_k=80
)
features_filtered.to_csv('data/features_filtered.csv')

# Step 5: Genetic Algorithm 
print('\n[5/6] Genetic Algorithm feature selection...')
sel_df, best, log, sel_cols = run_ga(features_filtered, rul_target)
sel_df.to_csv('data/features_ga_selected.csv')
plot_ga_convergence(log, save_path='outputs/ga_convergence.png')

# Step 6: Model Training & Evaluation
print('\n[6/6] Model Training & Evaluation...')
# Ensure final alignment
sel_df = sel_df.loc[rul_target.index]
results_df = run_all_models(sel_df, rul_target)

results_df.to_csv('outputs/model_comparison.csv', index=False)

# Train best model (e.g. XGBoost) on full data
train_best_model(MODELS['XGBoost'], sel_df.values, rul_target.values, 
                save_path='outputs/best_model.pkl')

total = time.time() - PIPELINE_START
print(f'\n=== PIPELINE COMPLETE in {total:.1f}s ===')
print(f'Features after GA: {sel_df.shape[1]}')
print(results_df.to_string(index=False))

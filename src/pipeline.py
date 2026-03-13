import sys
import os

# fixes imports and Windows multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multiprocessing
multiprocessing.freeze_support()

import time
import pandas as pd


def main():
    from preprocess         import preprocess_pipeline
    from feature_extraction import extract_or_load, filter_features_with_tsfresh
    from filter_selection   import run_filter_pipeline
    from genetic_algorithm  import run_ga, plot_ga_convergence
    from model              import run_all_models, train_best_model, MODELS

    os.makedirs('data',    exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    PIPELINE_START = time.time()

    # Step 1: Preprocess 
    print('\n[1/5] Preprocessing...')
    train_df, test_df, rul_df = preprocess_pipeline('data/', subset='FD001')
    train_df['engine_id'] = train_df['engine_id'].astype(float).astype(int)

    sensor_cols = [c for c in train_df.columns
                   if c.startswith('s') and c != 'engine_id']

    rul_target = (train_df.sort_values('cycle')
                          .groupby('engine_id')['RUL'].first())
    rul_target.index = rul_target.index.astype(int)
    print(f'Engines: {len(rul_target)} | Sensors: {len(sensor_cols)}')

    # Step 2: Feature Extraction
    print('\n[2/5] Feature extraction...')
    features_raw = extract_or_load(train_df, sensor_cols)
    common       = features_raw.index.intersection(rul_target.index)
    features_raw = features_raw.loc[common]
    rul_target   = rul_target.loc[common]
    print(f'Features shape: {features_raw.shape}')

    # Step 3: Relevance filter (MI-based, no multiprocessing) 
    print('\n[3/5] Relevance filter...')
    features_ts = filter_features_with_tsfresh(features_raw, rul_target)
    features_ts.to_csv('data/features_tsfresh_filtered.csv')

    # Step 4: Filter methods (Var + Corr + MI) 
    print('\n[4/5] Filter-based selection...')
    features_filtered, mi_scores = run_filter_pipeline(
        features_ts, rul_target, top_k=80
    )
    features_filtered.to_csv('data/features_filtered.csv')
    print(f'After filters: {features_filtered.shape[1]} features')

    # Step 5: Genetic Algorithm 
    print('\n[5/5] Genetic Algorithm...')
    sel_df, best, log, sel_cols = run_ga(features_filtered, rul_target)
    sel_df.to_csv('data/features_ga_selected.csv')
    plot_ga_convergence(log, save_path='outputs/ga_convergence.png')
    print(f'GA selected: {sel_df.shape[1]} features')

    # Step 6: Model Training & Evaluation 
    print('\n[6/6] Model Training & Evaluation...')
    results_df = run_all_models(sel_df, rul_target)
    results_df.to_csv('outputs/model_comparison.csv', index=False)
    train_best_model(MODELS['XGBoost'], sel_df.values, rul_target.values)

    total = time.time() - PIPELINE_START
    print(f'\n{"="*50}')
    print(f'PIPELINE COMPLETE in {total:.1f}s')
    print(f'Final feature count: {sel_df.shape[1]}')
    print(f'{"="*50}')
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
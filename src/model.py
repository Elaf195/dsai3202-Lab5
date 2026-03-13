import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb

MODELS = {
    'RandomForest': RandomForestRegressor(
        n_estimators=200, max_depth=10, 
        random_state=42, n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=200, max_depth=5, 
        learning_rate=0.05, random_state=42
    ),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=200, max_depth=6, 
        learning_rate=0.05, random_state=42, 
        n_jobs=-1, verbosity=0
    ),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, 
        learning_rate=0.05, random_state=42, 
        n_jobs=-1, verbose=-1,
        importance_type='split' # Explicitly set to avoid warnings
    ),
}

def evaluate_model(name: str, model, X: np.ndarray, 
                   y: np.ndarray, cv: int = 5) -> dict:
    """Train model with cross-validation and return metrics."""
    t0 = time.time()
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Use negative RMSE as scoring, then negate it back
    rmse_scores = -cross_val_score(model, X, y, cv=kf, 
                                   scoring='neg_root_mean_squared_error')
    mae_scores = -cross_val_score(model, X, y, cv=kf, 
                                  scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, X, y, cv=kf, 
                                scoring='r2')
    
    elapsed = time.time() - t0
    result = {
        'Model': name,
        'RMSE (mean)': round(rmse_scores.mean(), 4),
        'RMSE (std)': round(rmse_scores.std(), 4),
        'MAE (mean)': round(mae_scores.mean(), 4),
        'R2 (mean)': round(r2_scores.mean(), 4),
        'Time (s)': round(elapsed, 2),
    }
    print(result)
    return result

def train_best_model(model, X: np.ndarray, y: np.ndarray, 
                     save_path: str = 'outputs/best_model.pkl'):
    """Fit model on full training set and save to disk."""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.fit(X, y)
    joblib.dump(model, save_path)
    print(f'Model saved to {save_path}')
    return model

def run_all_models(features: pd.DataFrame, 
                   target: pd.Series) -> pd.DataFrame:
    """Run all models and return comparison table."""
    X = features.values
    y = target.values
    results = []
    for name, model in MODELS.items():
        print(f'\n--- {name} ---')
        results.append(evaluate_model(name, model, X, y))
    return pd.DataFrame(results)

if __name__ == '__main__':
    # This block is for standalone testing
    if os.path.exists('data/features_ga_selected.csv'):
        features = pd.read_csv('data/features_ga_selected.csv', 
                               index_col='engine_id')
        train_df = pd.read_csv('data/train_preprocessed.csv')
        
        # Build RUL target indexed by engine_id
        rul_target = (train_df.sort_values('cycle')
                      .groupby('engine_id')['RUL'].first())
        
        # Align features with target
        features = features.loc[rul_target.index]
        
        results_df = run_all_models(features, rul_target)
        
        print('\n=== Model Comparison ===')
        print(results_df.to_string(index=False))
        
        os.makedirs('outputs', exist_ok=True)
        results_df.to_csv('outputs/model_comparison.csv', index=False)
        
        train_best_model(MODELS['XGBoost'], features.values, 
                         rul_target.values)
    else:
        print("Data file not found. Please run genetic_algorithm.py first.")

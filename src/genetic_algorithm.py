import pandas as pd
import numpy as np
import time
import random
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# GA Hyperparameters
POP_SIZE = 60
N_GEN = 15
CX_PROB = 0.7
MUT_PROB = 0.2
INDPB = 0.05
TOURNAMENT = 3
ALPHA = 0.1  # Feature count penalty
MIN_FEATURES = 5

# Setup DEAP creator (must be outside functions to avoid re-creation errors)
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

def setup_deap(n_features: int):
    """Initialize DEAP toolbox with selection, crossover, and mutation."""
    toolbox = base.Toolbox()
    
    # Attribute generator: random bit (0 or 1)
    toolbox.register('attr_bool', random.randint, 0, 1)
    
    # Structure initializers
    toolbox.register('individual', tools.initRepeat, 
                     creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register('population', tools.initRepeat, list, 
                     toolbox.individual)
    
    # Genetic operators
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=INDPB)
    toolbox.register('select', tools.selTournament, 
                     tournsize=TOURNAMENT)
    
    return toolbox

def evaluate(individual, X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Fitness function: cross-validated RMSE + feature count penalty.
    Uses a shallow Random Forest for speed inside the GA loop.
    """
    selected_idx = [i for i, bit in enumerate(individual) if bit == 1]
    
    if len(selected_idx) < MIN_FEATURES:
        return (1e6,)  # penalise degenerate chromosomes
    
    X_sub = X[:, selected_idx]
    
    # Shallow RF for speed
    model = RandomForestRegressor(
        n_estimators=30, max_depth=5, 
        random_state=42, n_jobs=1
    )
    
    # Use negative RMSE as scoring, then negate it back
    scores = cross_val_score(model, X_sub, y, 
                             cv=3, 
                             scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    
    # Penalty for using too many features
    penalty = ALPHA * len(selected_idx)
    
    return (rmse + penalty,)

def run_ga(features: pd.DataFrame, 
           target: pd.Series, 
           verbose: bool = True) -> tuple:
    """
    Run the Genetic Algorithm.
    Returns: (selected_features_df, best_individual, logbook, selected_cols)
    """
    X = features.values
    y = target.values
    n_features = X.shape[1]
    t0 = time.time()
    
    toolbox = setup_deap(n_features)
    toolbox.register('evaluate', evaluate, X=X, y=y)
    
    # Statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register('min', np.min)
    stats.register('avg', np.mean)
    stats.register('max', np.max)
    
    hof = tools.HallOfFame(1)  # keep best individual ever seen
    pop = toolbox.population(n=POP_SIZE)
    
    print(f'Running GA: {POP_SIZE} individuals, {N_GEN} generations...')
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, 
        cxpb=CX_PROB, mutpb=MUT_PROB, ngen=N_GEN, 
        stats=stats, halloffame=hof, verbose=verbose
    )
    
    best = hof[0]
    selected_idx = [i for i, b in enumerate(best) if b == 1]
    selected_cols = features.columns[selected_idx].tolist()
    selected_df = features[selected_cols]
    
    elapsed = time.time() - t0
    print(f'GA done in {elapsed:.1f}s')
    print(f'Best RMSE (approx): {best.fitness.values[0]:.4f}')
    print(f'Selected features: {len(selected_cols)}')
    
    return selected_df, best, logbook, selected_cols

def plot_ga_convergence(logbook, save_path: str = None):
    """Plot GA fitness over generations."""
    import matplotlib.pyplot as plt
    gen = logbook.select('gen')
    mins = logbook.select('min')
    avgs = logbook.select('avg')
    
    plt.figure(figsize=(9, 4))
    plt.plot(gen, mins, label='Best Fitness', color='navy')
    plt.plot(gen, avgs, label='Avg Fitness', color='steelblue', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (RMSE + penalty)')
    plt.title('GA Convergence')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

if __name__ == '__main__':
    import os
    if os.path.exists('data/features_filtered.csv'):
        features = pd.read_csv('data/features_filtered.csv', index_col='engine_id')
        train_df = pd.read_csv('data/train_preprocessed.csv')
        
        # Build RUL target indexed by engine_id
        rul_target = (train_df.sort_values('cycle')
                      .groupby('engine_id')['RUL'].first())
        
        # Align features with target
        features = features.loc[rul_target.index]
        
        sel_df, best_ind, log, sel_cols = run_ga(features, rul_target)
        
        os.makedirs('data', exist_ok=True)
        sel_df.to_csv('data/features_ga_selected.csv')
        
        os.makedirs('outputs', exist_ok=True)
        plot_ga_convergence(log, save_path='outputs/ga_convergence.png')
        print('Saved: data/features_ga_selected.csv')
    else:
        print("Data file not found. Please run filter_selection.py first.")

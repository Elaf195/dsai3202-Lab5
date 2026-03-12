import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sensor_trends(df: pd.DataFrame, engine_ids: list,
                        sensors: list, save_path: str = None):
    """Plot sensor readings over cycle for selected engines."""
    fig, axes = plt.subplots(len(sensors), 1,
                              figsize=(12, 3 * len(sensors)))
    if len(sensors) == 1:
        axes = [axes]
    for ax, sensor in zip(axes, sensors):
        for eid in engine_ids:
            subset = df[df['engine_id'] == eid]
            ax.plot(subset['cycle'], subset[sensor],
                    label=f'Engine {eid}', alpha=0.7)
        ax.set_title(f'Sensor: {sensor}')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Value')
        ax.legend(fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_rul_distribution(df: pd.DataFrame, save_path: str = None):
    """Show RUL distribution across all engines."""
    plt.figure(figsize=(8, 4))
    sns.histplot(df['RUL'], bins=50, kde=True, color='steelblue')
    plt.title('RUL Distribution (capped at 125)')
    plt.xlabel('Remaining Useful Life')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def sensor_variance_report(df: pd.DataFrame):
    """Print variance of each sensor column."""
    sensor_cols = [c for c in df.columns if c.startswith('s')]
    var_df = df[sensor_cols].var().sort_values()
    print(var_df.to_string())
    return var_df

if __name__ == '__main__':
    import pandas as pd
    train_df = pd.read_csv('data/train_preprocessed.csv')
    sensors = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14']
    plot_sensor_trends(train_df, [1,2,3], sensors,
                        save_path='outputs/sensor_trends.png')
    plot_rul_distribution(train_df,
                           save_path='outputs/rul_distribution.png')
    sensor_variance_report(train_df)

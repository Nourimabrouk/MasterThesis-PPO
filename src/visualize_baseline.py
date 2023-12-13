import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import ENVIRONMENTS
import os

sns.set_theme(style="darkgrid")
sns.set_palette('colorblind') 

def plot_baseline_analysis():

    save_dir_baseline = "./visualizations/baseline_analysis/"
    os.makedirs(save_dir_baseline, exist_ok=True)

    # Iterate over each environment for plotting
    for env_name in ENVIRONMENTS:
        file_path = f"./output/baseline_analysis/combined_rewards_{env_name}.csv"
        df = pd.read_csv(file_path)

        # Apply a rolling mean for smoother plots
        window_size = 10 
        df_smoothed = df.rolling(window=window_size, min_periods=1).mean()

        steps = range(1, len(df) + 1)

        # Plotting the smoothed rewards
        plt.figure(figsize=(10, 6))
        for agent in df.columns:
            plt.plot(steps, df_smoothed[agent], label=agent)
            plt.fill_between(steps, df_smoothed[agent] - df_smoothed[agent].std(), df_smoothed[agent] + df_smoothed[agent].std(), alpha=0.1)

        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title(f'Baseline Analysis Results for {env_name}')
        plt.legend(loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Adjust y-axis scale for clarity in FrozenLake environment
        if env_name == 'FrozenLake-v1':
            plt.ylim(0, 1)  

        # Saving the plot
        plt.savefig(f"{save_dir_baseline}baseline_results_{env_name}.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    plot_baseline_analysis()

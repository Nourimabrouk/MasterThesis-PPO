import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from config import ENVIRONMENTS
import warnings

sns.set_theme(style="darkgrid")
sns.set_palette('colorblind')

# Load data from CSV files
evaluation_results = pd.read_csv('output/comparative_analysis/evaluation_results.csv')
summary_statistics = pd.read_csv('output/comparative_analysis/summary_statistics.csv')
t_test_results = pd.read_csv('output/comparative_analysis/t_test_results.csv')
anova_results = pd.read_csv('output/comparative_analysis/anova_results.csv')

# Standardize column names for merging
summary_statistics.rename(columns={'Agent': 'agent', 'Environment': 'environment'}, inplace=True)
merged_data = pd.merge(evaluation_results, summary_statistics, on=['agent', 'environment'])

def preprocess_evaluation_results(df, summary_df):
    # Preprocess evaluation results to extract reward mean and standard deviation
    df['reward_mean'] = df['reward'].apply(lambda x: np.mean(eval(x)))
    # Join to get standard deviation
    df = df.join(summary_df.set_index(['agent', 'environment']), on=['agent', 'environment'])
    return df[['agent', 'environment', 'reward_mean', 'reward_std']]

def transform_t_test_for_heatmap(df):
    # Transform t-test results for heatmap visualization
    df_agg = df.groupby(['Agent Pair', 'Environment'])['p-value'].mean().reset_index()
    pivot_df = df_agg.pivot(index='Agent Pair', columns='Environment', values='p-value')
    return pivot_df

def plot_mean_rewards_per_environment(df, environments, save_dir):
    for env in environments:
        env_data = df[df['environment'] == env]

        if env_data.empty:
            print(f"Warning: No data available for environment {env}")
            continue

        if env_data['reward_mean'].isna().any():
            print(f"Warning: NaN values found in reward_mean for environment {env}")
            continue

        plt.figure(figsize=(12, 8))
        sns.barplot(x="agent", y="reward_mean", yerr=env_data['reward_std'], data=env_data, capsize=.2)
        plt.title(f"Mean Rewards of Agents in {env}")
        plt.ylabel("Mean Reward")
        plt.xlabel("Agent")
        # Adding gridlines for better readability
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.savefig(os.path.join(save_dir, f'mean_rewards_{env}.png'))
        plt.close()

def heatmap_t_test(transformed_data, save_dir):
    # Create a heatmap for the t-test results
    plt.figure(figsize=(12, 8))
    sns.heatmap(transformed_data, annot=True, fmt=".2g", cmap="viridis", cbar_kws={'label': 'p-value'})
    plt.title("T-Test P-Values Heatmap")
    plt.ylabel("Agent Pair")
    plt.xlabel("Environment")
    plt.savefig(os.path.join(save_dir, 't_test_heatmap.png'))
    plt.close()

def plot_anova_f_values(anova_results, save_dir):
    # Plot the ANOVA F-Values for different environments
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Environment", y="F", hue="Statistic", data=anova_results)
    plt.title("ANOVA F-Values for Different Environments")
    # Adding error bars if standard deviation is available
    if 'reward_std' in anova_results:
        sns.barplot(x="Environment", y="F", yerr=anova_results['reward_std'], hue="Statistic", data=anova_results, capsize=.2)
    plt.ylabel("ANOVA F-Value")
    plt.xlabel("Environment")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(title='Statistic')
    plt.savefig(os.path.join(save_dir, 'anova_f_values.png'))
    plt.close()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    save_dir = 'visualizations/comparative_analysis'
    os.makedirs(save_dir, exist_ok=True)

    preprocessed_evaluation_results = preprocess_evaluation_results(evaluation_results, summary_statistics)
    plot_mean_rewards_per_environment(preprocessed_evaluation_results, ENVIRONMENTS, save_dir)

   

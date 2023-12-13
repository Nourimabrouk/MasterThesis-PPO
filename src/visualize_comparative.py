import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from config import ENVIRONMENTS

sns.set_theme(style="darkgrid")
sns.set_palette('colorblind') 

# Load data from CSV files
evaluation_results = pd.read_csv('./output/comparative_analysis/evaluation_results.csv')
summary_statistics = pd.read_csv('./output/comparative_analysis/summary_statistics.csv')
t_test_results = pd.read_csv('./output/comparative_analysis/t_test_results.csv')
anova_results = pd.read_csv('./output/comparative_analysis/anova_results.csv')

# Standardize column names for merging
summary_statistics.rename(columns={'Agent': 'agent', 'Environment': 'environment'}, inplace=True)
merged_data = pd.merge(evaluation_results, summary_statistics, on=['agent', 'environment'])

def preprocess_evaluation_results(df):
    # Preprocess evaluation results to extract reward mean
    df['reward_mean'] = df['reward'].apply(lambda x: np.mean(eval(x)))
    return df[['agent', 'environment', 'reward_mean']]

def transform_t_test_for_heatmap(df):
    # Transform t-test results for heatmap visualization
    df_agg = df.groupby(['Agent Pair', 'Environment'])['p-value'].mean().reset_index()
    pivot_df = df_agg.pivot(index='Agent Pair', columns='Environment', values='p-value')
    return pivot_df

def plot_mean_rewards_per_environment(df, environments):
    # Plot the mean rewards for each agent in different environments
    for env in environments:
        plt.figure(figsize=(12, 8))
        env_data = df[df['environment'] == env]
        sns.barplot(x="agent", y="reward_mean", data=env_data)
        plt.title(f"Mean Rewards of Agents in {env}")
        plt.ylabel("Mean Reward")
        plt.xlabel("Agent")
        plt.show()
        plt.savefig(os.path.join(save_dir, f'mean_rewards_{env}.png'))

def heatmap_t_test():
    # Create a heatmap for the t-test results
    transformed_data = transform_t_test_for_heatmap(t_test_results)
    plt.figure(figsize=(12, 8))
    sns.heatmap(transformed_data, annot=True)
    plt.title("T-Test P-Values Heatmap")
    plt.show()

def plot_anova_f_values():
    # Plot the ANOVA F-Values for different environments
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Environment", y="F", hue="Statistic", data=anova_results)
    plt.title("ANOVA F-Values for Different Environments")
    plt.ylabel("F-Value")
    plt.xlabel("Environment")
    plt.show()

    
if __name__ == "__main__":

    save_dir = './visualizations/comparative_analysis'
    os.makedirs(save_dir, exist_ok=True)

    preprocessed_evaluation_results = preprocess_evaluation_results(evaluation_results)
    plot_mean_rewards_per_environment(preprocessed_evaluation_results, ENVIRONMENTS)

    plt.savefig(os.path.join(save_dir, 'mean_rewards.png'))

    heatmap_t_test()
    plt.savefig(os.path.join(save_dir, 't_test_heatmap.png'))

    plot_anova_f_values()
    plt.savefig(os.path.join(save_dir, 'anova_f_values.png'))

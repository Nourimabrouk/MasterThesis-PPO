import warnings
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import os
import ast
from utils import save_results_to_csv

# Process reward strings into numerical mean and standard deviation
def process_rewards(reward_str):
    try:
        rewards_list = ast.literal_eval(reward_str)
        if isinstance(rewards_list, list) and len(rewards_list) > 0:
            return np.mean(rewards_list), np.std(rewards_list)
        return None, None
    except:
        return None, None

# Load and preprocess evaluation results from a CSV file
def load_evaluation_results(file_path):
    df = pd.read_csv(file_path)
    df[['reward_mean', 'reward_std']] = df['reward'].apply(lambda x: pd.Series(process_rewards(x)))
    df.dropna(subset=['reward_mean', 'reward_std'], inplace=True)
    return df

# Perform t-tests between agents for a given environment and statistic
def perform_t_tests(df, agents, env, stat):
    t_test_results = {}
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            agent1, agent2 = agents[i], agents[j]
            rewards_agent1 = df[(df['agent'] == agent1) & (df['environment'] == env)][stat]
            rewards_agent2 = df[(df['agent'] == agent2) & (df['environment'] == env)][stat]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    t_stat, p_val = stats.ttest_ind(rewards_agent1, rewards_agent2)
                t_test_results[(agent1, agent2)] = p_val
            except Exception as e:
                print(f"Error in t-test between {agent1} and {agent2}: {e}")
                t_test_results[(agent1, agent2)] = None

    return t_test_results

# Perform ANOVA for a given environment and statistic
def perform_anova(df, env, stat):
    formula = f'{stat} ~ C(agent)'
    df_env = df[df['environment'] == env]
    model = ols(formula, data=df_env).fit()
    return sm.stats.anova_lm(model, typ=2)

# Main function to run comparative statistics on evaluation results
def run_comparative_statistics(file_path):
    df_results = load_evaluation_results(file_path)
    agents = df_results['agent'].unique()
    environments = df_results['environment']. unique()

    # Adjust alpha value for multiple testing
    total_tests = 2 * sum(len(agents) * (len(agents) - 1) / 2 for _ in environments)
    adjusted_alpha = 0.05 / total_tests

    summary_statistics = pd.DataFrame()
    t_test_results_data = []
    anova_results_data = []

    print("Running Comparative Statistics with Bonferroni Correction...")
    for env in environments:
        print(f"\nEnvironment: {env}")
        env_df = df_results[df_results['environment'] == env]

        for stat in ['reward_mean', 'reward_std']:
            t_test_results = perform_t_tests(env_df, agents, env, stat)
            for pair, p_val in t_test_results.items():
                significant = 'Yes' if p_val < adjusted_alpha else 'No'
                t_test_results_data.append({
                    'Environment': env,
                    'Statistic': stat,
                    'Agent Pair': pair,
                    'p-value': p_val,
                    'Significant': significant
                })

            anova_result = perform_anova(env_df, env, stat)
            anova_result.reset_index(inplace=True)
            anova_result['Environment'] = env
            anova_result['Statistic'] = stat
            anova_results_data.append(anova_result)

        # Append summary statistics for each agent
        for agent in agents:
            agent_df = env_df[env_df['agent'] == agent]
            summary = agent_df[['reward_mean', 'reward_std']].agg(['mean', 'std']).reset_index()
            summary['Agent'] = agent
            summary['Environment'] = env
            summary_statistics = pd.concat([summary_statistics, summary], ignore_index=True)

    # Save all results to CSV files
    save_results_to_csv(summary_statistics, 'summary_statistics.csv')
    save_results_to_csv(pd.DataFrame(t_test_results_data), 't_test_results.csv')
    save_results_to_csv(pd.concat(anova_results_data, ignore_index=True), 'anova_results.csv')

if __name__ == "__main__":
    results_file_path = 'output/comparative_analysis/evaluation_results.csv'
    run_comparative_statistics(results_file_path)

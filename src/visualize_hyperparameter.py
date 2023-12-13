import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

sns.set_theme(style="darkgrid")
sns.set_palette('colorblind')

# Function to load results from multiple CSV files into a single Dataframe
def load_all_results(directory):
    all_files = glob.glob(os.path.join(directory, "optuna_results_*.csv"))
    all_df = []

    # Loop through each file and append to a list after adding environment and agent columns
    for file in all_files:
        df = pd.read_csv(file)
        env_agent = os.path.basename(file).replace('optuna_results_', '').replace('.csv', '').split('_')
        env_name, agent_name = env_agent[0], env_agent[1]
        df['Environment'] = env_name
        df['Agent'] = agent_name
        all_df.append(df)
        
    all_results = pd.concat(all_df, ignore_index=True)
    all_results.to_csv(os.path.join(directory, 'all_results.csv'), index=False)
    return all_results
    
# Function to plot the best performers by environment
def plot_best_performers_by_environment(df, visualization_directory):
    environments = df['Environment'].unique()

    # Loop through each environment to plot best performers
    for env in environments:
        env_df = df[df['Environment'] == env]
        best_performers = env_df.groupby(['Agent'])['value'].apply(lambda x: -1 * x.mean()).sort_values().reset_index()

        plt.figure(figsize=(10, 6))
        barplot = sns.barplot(x='value', y='Agent', data=best_performers, 
                              palette=[agent_colors[agent] for agent in best_performers['Agent']])
        for index, row in best_performers.iterrows():
            barplot.text(row.value, index, f' {row.value:.2f}', color='black', ha="left", va="center")

        plt.title(f"Average Reward in {env}")
        plt.xlabel("Average Reward")
        plt.ylabel("Agent")
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_directory, f'average_reward_{env}.png'))
        plt.close()
        
# Function to plot hyperparameter importance as a heatmap for each agent
def plot_hyperparameter_importance(df, visualization_directory):
    relevant_params = {
        'A2C': ['n_steps', 'vf_coef', 'learning_rate', 'gamma'],
        'DQN': ['learning_rate', 'gamma', 'exploration_fraction', 'target_update_interval'],
        'PPO': ['n_steps', 'ent_coef', 'learning_rate', 'gamma']
    }

    os.makedirs(visualization_directory, exist_ok=True)

    # Loop through each agent and plot the correlation matrix
    for agent, group in df.groupby('Agent'):
        agent_hyperparams = group.filter(regex='^params_').copy()
        agent_hyperparams.columns = [col.replace('params_', '') for col in agent_hyperparams.columns]
        existing_params = agent_hyperparams.columns.intersection(relevant_params[agent])
        relevant_hyperparams = agent_hyperparams[existing_params]
        corr = relevant_hyperparams.corr()

        if not corr.empty:
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f"Hyperparameter Correlation Heatmap for {agent}")
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_directory, f'hyperparameter_correlation_{agent}.png'))
            plt.close()
        else:
            print(f"Not enough hyperparameters to plot for {agent}")

# Function to plot performance distribution of top agents
def plot_performance_distribution(df, visualization_directory, top_n=3):
    if (df['value'] < 0).all():
        df['value'] = df['value'] * -1
    
    best_performers = df.groupby(['Agent'])['value'].mean().sort_values(ascending=False).head(top_n).index
    top_df = df[df['Agent'].isin(best_performers)].copy()
    top_df['mean_value'] = top_df.groupby('Agent')['value'].transform('mean')
    top_df.sort_values('mean_value', ascending=False, inplace=True)

    plt.figure(figsize=(10, 6))
    boxplot = sns.boxplot(x='Agent', y='value', data=top_df, palette=color_list)
    for i, agent in enumerate(top_df['Agent'].unique()):
        mean_val = top_df[top_df['Agent'] == agent]['value'].mean()
        plt.text(i, mean_val, f'{mean_val:.2f}', ha='center', va='bottom', color='black')

    plt.title("Performance Distribution of Top Agents")
    plt.xlabel("Agent")
    plt.ylabel("Average Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_directory, 'performance_distribution.png'))
    plt.close()

if __name__ == "__main__":
    results_directory = 'output/hyperparameter_analysis'
    visualization_directory = 'visualizations/hyperparameter_analysis'
    os.makedirs(visualization_directory, exist_ok=True)

    df = load_all_results(results_directory)
    plot_best_performers_by_environment(df, visualization_directory)
    plot_hyperparameter_importance(df, visualization_directory)
    plot_performance_distribution(df, visualization_directory)

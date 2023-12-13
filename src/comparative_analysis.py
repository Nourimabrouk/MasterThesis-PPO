import os
import gymnasium as gym
import stable_baselines3 as sb3
import numpy as np
import random
import torch
import optuna
import pandas as pd
from utils import save_model, load_model, save_rewards_to_csv
from config import AGENTS, ENVIRONMENTS, TOTAL_TIMESTEPS, NUM_RUNS
from agents import evaluate_agent, train_agent

# Set up the directory for TensorBoard logs
tensorboard_base_dir = "./output/tensorboard_logs/"

# Seed setup for reproducibility
SEED = 1337
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Main function to run the comparative analysis
def run_comparative_analysis():
    print("Running comparative analysis...")
    train_all_agents()
    evaluate_all_agents()
    print("Analysis complete.")

# Function to load best hyperparameters from a CSV file
def get_best_hyperparameters(env_name, agent_name):
    print(f"Loading best hyperparameters for {agent_name} on {env_name}...")
    file_path = f'output/hyperparameter_analysis/best_hyperparams_{env_name}_{agent_name}.csv'
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No hyperparameter file found for {env_name} on {agent_name}")

    df_best = pd.read_csv(file_path)
    best_hyperparams = df_best.iloc[0].to_dict()
    best_hyperparams.pop('best_reward', None)

    # Filter and adjust hyperparameters based on the agent
    valid_hyperparams = {}
    for key, value in best_hyperparams.items():
        if key not in ['Unnamed: 0', 'best_reward']:
            new_key = key.split('_', 1)[1] if agent_name in key else key
            valid_hyperparams[new_key] = int(value) if new_key == 'n_steps' else value

    return valid_hyperparams

# Function to train all agents with the best hyperparameters
def train_all_agents():
    for agent_name in AGENTS:              
        for env_name in ENVIRONMENTS:   
            for run in range(NUM_RUNS):
                try:
                    print(f"Training {agent_name} on {env_name}, Run {run + 1}/{NUM_RUNS}...")
                    hyperparams = get_best_hyperparameters(env_name, agent_name)
                    env = gym.make(env_name)
                    if hasattr(env, 'seed'):
                        env.seed(SEED + run)
                    tensorboard_log_dir = os.path.join(tensorboard_base_dir, f"{agent_name}_{env_name}_run{run + 1}")

                    trained_agent = train_agent(agent_name, env, hyperparameters=hyperparams, tensorboard_log=tensorboard_log_dir, callback=None, total_timesteps=TOTAL_TIMESTEPS)

                    save_path = os.path.join('/output/comparative_analysis/models', f'{agent_name}_{env_name}_run{run + 1}.model')
                    save_model(trained_agent, save_path)

                    env.close()
                except Exception as e:
                    print(f"Error in run {run + 1} training {agent_name} on {env_name}. Details: {e}")

# Function to evaluate all trained agents
def evaluate_all_agents():
    results = []
    for agent_name in AGENTS:
        for env_name in ENVIRONMENTS:
            for run in range(NUM_RUNS):
                try:
                    print(f"Evaluating {agent_name} on {env_name}, Run {run + 1}/{NUM_RUNS}...")

                    env = gym.make(env_name)
                    if hasattr(env, 'seed'):
                        env.seed(SEED + run)

                    model_path = os.path.join('/output/comparative_analysis/models', f'{agent_name}_{env_name}_run{run + 1}.model')
                    trained_agent = load_model(agent_name=agent_name, load_path=model_path) 

                    total_reward = evaluate_agent(trained_agent, env)
                    print(f"Evaluation reward for {agent_name} on {env_name}, Run {run + 1}: {total_reward}")
                    
                    results.append({
                        "agent": agent_name,
                        "environment": env_name,
                        "run": run + 1,
                        "reward": total_reward
                    })

                    env.close()
                except Exception as e:
                    print(f"Error in run {run + 1} evaluating {agent_name} on {env_name}. Details: {e}")

    df_results = pd.DataFrame(results)
    df_results.to_csv('/output/comparative_analysis/evaluation_results.csv', index=False)

def main():
    run_comparative_analysis()

if __name__ == "__main__":
    main()

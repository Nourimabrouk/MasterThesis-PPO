import os
import gymnasium as gym
import stable_baselines3 as sb3
import numpy as np
import random
import torch
import optuna
import pandas as pd
from src.utils import save_model, load_model, save_rewards_to_csv
from src.config import AGENTS, ENVIRONMENTS, TOTAL_TIMESTEPS
from src.agents import evaluate_agent, train_agent

tensorboard_base_dir = "./tensorboard_logs/"

SEED = 1337
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

def run_comparative_analysis():
    print("Running comparative analysis...")
    train_all_agents()
    evaluate_all_agents()
    print("Analysis complete.")
    
def get_best_hyperparameters(ENV_NAME, agent_name):
    """Load Optuna results and return the best hyperparameters for a given agent-environment pair."""
    
    print(f"Loading hyperparameters for {ENV_NAME} on {agent_name}...")
    df = pd.read_csv(f'output/optuna_results_{ENV_NAME}_{agent_name}.csv')
    
    best_trial = df.sort_values(by='value').iloc[0]

    hyperparams = {}
    for col in df.columns:
        if col.startswith('params_'):
            param_name = col.replace('params_', '')
            # Remove agent name prefix and leading underscores
            clean_param_name = param_name.replace(f"{agent_name}_", "").lstrip('_')
            hyperparams[clean_param_name] = best_trial[col]

    return hyperparams

def train_all_agents():
    """Train all agents with best hyperparameters."""
    for agent_name in AGENTS:              
        for env_name in ENVIRONMENTS:   
            try:
                print(f"Training {agent_name} on {env_name}...")
                hyperparams = get_best_hyperparameters(env_name, agent_name)
                env = gym.make(env_name)
                if hasattr(env, 'seed'):
                    env.seed(SEED)
                tensorboard_log_dir = os.path.join(tensorboard_base_dir, f"{agent_name}_{env_name}")
                trained_agent = train_agent(agent_name, env, hyperparameters=hyperparams, tensorboard_log=tensorboard_log_dir)

                save_path = os.path.join('models', f'{agent_name}_{env_name}.model')
                save_model(trained_agent, save_path)
                
                env.close()
            except Exception as e:
                print(f"Error training {agent_name} on {env_name}. Details: {e}")

def evaluate_all_agents():
    """Evaluate all trained agents."""
    for agent_name in AGENTS:
        for env_name in ENVIRONMENTS:
            try:
                print(f"Evaluating {agent_name} on {env_name}...")

                env = gym.make(env_name)
                if hasattr(env, 'seed'):
                    env.seed(SEED)

                model_path = os.path.join('models', f'{agent_name}_{env_name}.model')
                trained_agent = load_model(model_path)

                total_reward = evaluate_agent(trained_agent, env)
                print(f"Evaluation reward for {agent_name} on {env_name}: {total_reward}")
                save_rewards_to_csv(agent_name, env_name, total_reward)
                
                env.close()
            except Exception as e:
                print(f"Error evaluating {agent_name} on {env_name}. Details: {e}")

def main():
    run_comparative_analysis()
    # TODO: Add a function to visualize and analyze results.
    # View tensorboard logs: tensorboard --logdir ./tensorboard_logs/

if __name__ == "__main__":
    main()
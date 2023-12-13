import pandas as pd
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.save_util import save_to_zip_file
from config import AGENTS
import csv

# Function to save rewards for a specific agent and environment
def save_rewards_to_csv(agent_name, env_name, rewards, folder_name):
    filepath = f'output/{folder_name}/rewards_{agent_name}_{env_name}.csv'
    df = pd.DataFrame({"reward": rewards})
    df.to_csv(filepath, index=False)
    print(f"Saved rewards for {agent_name} on {env_name} to {filepath}")

# Function to save combined rewards from different agents in a given environment
def save_combined_rewards_to_csv(env_name, reward_data, folder_name):
    filepath = f'output/{folder_name}/combined_rewards_{env_name}.csv'
    df = pd.DataFrame(reward_data)
    df.to_csv(filepath, index=False)
    print(f"Saved combined rewards for {env_name} to {filepath}")
            
# Collate results by computing means and standard deviations of rewards
def collate_results(all_rewards):
    results = {}

    for agent, rewards in all_rewards.items():
        rewards_array = np.array(rewards)
        means = np.mean(rewards_array, axis=0)
        std_devs = np.std(rewards_array, axis=0)
        results[agent] = {'mean': means, 'std_dev': std_devs}

    return results

# Save an agent model to a specified path
def save_model(agent, save_path):
    agent.save(save_path)

# Load a model for a given agent from a specified path
def load_model(agent_name, load_path):
    algorithms = {
        "DQN": sb3.DQN,
        "A2C": sb3.A2C,
        "PPO": sb3.PPO,
    }
    
    if agent_name not in algorithms:
        raise ValueError(f"Unsupported agent: {agent_name}")

    return algorithms[agent_name].load(load_path)

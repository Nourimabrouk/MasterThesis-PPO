import pandas as pd
import numpy as np

import stable_baselines3 as sb3
from stable_baselines3.common.save_util import save_to_zip_file
from config import AGENTS
import numpy as np
import csv


def save_rewards_to_csv(agent_name, env_name, rewards):
    """Save rewards to a CSV file."""
    filepath = f'output/rewards_{agent_name}_{env_name}.csv'
    df = pd.DataFrame({"reward": rewards})
    df.to_csv(filepath, index=False)
    print(f"Saved rewards for {agent_name} on {env_name} to {filepath}")
            
def collate_results(all_rewards):
    results = {}

    for agent, rewards in all_rewards.items():
        # Convert rewards to numpy array
        rewards_array = np.array(rewards)

        # Calculate mean and standard deviation
        means = np.mean(rewards_array, axis=0)
        std_devs = np.std(rewards_array, axis=0)

        results[agent] = {
            'mean': means,
            'std_dev': std_devs
        }

    return results

def save_model(agent, save_path):
    """Save the agent model to the specified path."""
    agent.save(save_path)

def load_model(agent_name, load_path):
    """
    Load the agent model from the specified path.
    
    Parameters:
    - agent_name (str): The name of the agent ('PPO', 'DQN', 'A2C', 'SAC', 'DDPG', etc.)
    - load_path (str): The path to the saved model.

    Returns:
    - agent (BaseAlgorithm): Loaded agent model.
    """
    
    algorithms = {
        "DQN": sb3.DQN,
        "A2C": sb3.A2C,
        "PPO": sb3.PPO,
    }
    
    if agent_name not in algorithms:
        raise ValueError(f"Unsupported agent: {agent_name}")

    return algorithms[agent_name].load(load_path)
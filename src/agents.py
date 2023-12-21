#### Functions for creating models and training and evaluating agents

import stable_baselines3 as sb3
from config import SAVE_PATH, LOG_PATH, TOTAL_TIMESTEPS, EVALUATION_EPISODES
import json
import os
import numpy as np

class RandomAgent:
    def __init__(self, action_space):
        # Initialize the agent with the given action space
        self.action_space = action_space

    def predict(self, _observation, deterministic=True):
        # Randomly sample an action from the action space
        return self.action_space.sample(), None

def create_model(agent_name, env, log_path, hyperparameters=None, tensorboard_log=None):
    # Mapping agent names to their respective SB3 algorithms
    algorithms = {
        "PPO": sb3.PPO,
        "DQN": sb3.DQN,
        "A2C": sb3.A2C,
    }

    # Check if the specified agent name is valid
    if agent_name not in algorithms:
        raise ValueError(f"Unsupported agent: {agent_name}")

    # Set up the path for TensorBoard logging
    tensorboard_log_path = None
    if tensorboard_log:
        tensorboard_log_path = os.path.join(tensorboard_log, agent_name.lower())

    # Create the model with the specified algorithm, policy, and hyperparameters
    model = algorithms[agent_name]("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log_path, **(hyperparameters or {}))
    
    return model

def evaluate_agent(model, env, num_episodes=EVALUATION_EPISODES):
    """Evaluate the agent's performance over a fixed number of episodes."""
    print(f"Evaluating agent over {num_episodes} episodes...")
    
    rewards = []
    for _ in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        
        while True:
            # Predict the next action based on the current observation
            action, _states = model.predict(observation, deterministic=False)
            
            # Ensure the action format is compatible with the environment
            if isinstance(action, np.ndarray):
                if action.ndim == 0:
                    action = int(action.item())
                elif action.size == 1:
                    action = int(action[0])
            
            # Perform the action in the environment and observe the results
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Break the loop if the episode is terminated or truncated
            if terminated or truncated:
                break

        # Accumulate the rewards gained in each episode
        rewards.append(episode_reward)
    
    env.close()
    return rewards

def train_agent(agent_name, env, hyperparameters=None, tensorboard_log=None, callback=None, total_timesteps=None):
    # Create a model for the agent using the specified configuration
    model = create_model(agent_name, env, LOG_PATH, hyperparameters, tensorboard_log)

    # Train the model for a given number of timesteps
    model.learn(total_timesteps, callback=callback)
    env_name = env.spec.id
    model_save_path = f"{SAVE_PATH}{env_name}_{agent_name}.zip"

    # Save the trained model
    model.save(model_save_path)
    print(f"{agent_name} model saved to {model_save_path}")
    return model

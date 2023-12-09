import stable_baselines3 as sb3
from config import SAVE_PATH, LOG_PATH, TOTAL_TIMESTEPS
import json
import os
import numpy as np

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, _observation, deterministic=True):
        return self.action_space.sample(), None

def create_model(agent_name, env, log_path, hyperparameters=None, tensorboard_log=None):
    algorithms = {
        "PPO": sb3.PPO,
        "DQN": sb3.DQN,
        "A2C": sb3.A2C,
    }

    if agent_name not in algorithms:
        raise ValueError(f"Unsupported agent: {agent_name}")

    tensorboard_log_path = None
    if tensorboard_log:
        tensorboard_log_path = os.path.join(tensorboard_log, agent_name.lower())

    model = algorithms[agent_name]("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log_path, **(hyperparameters or {}))
    
    return model

def evaluate_agent(model, env, num_episodes=10):
    """Evaluate the agent's performance over a fixed number of episodes."""
    print(f"Evaluating agent over {num_episodes} episodes...")
    
    rewards = []
    for _ in range(num_episodes):
        # print(f"Episode {_ + 1}/{num_episodes}")
        observation, info = env.reset()
        episode_reward = 0
        
        while True:
            action, _states = model.predict(observation, deterministic=False)
            
            # Convert action if it's a numpy array
            if isinstance(action, np.ndarray):
                if action.ndim == 0:
                    action = int(action.item())
                elif action.size == 1:
                    action = int(action[0])
            
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break

        rewards.append(episode_reward)
    
    env.close()
    return rewards

def train_agent(agent_name, env, hyperparameters=None, tensorboard_log=None, callback=None):
    model = create_model(agent_name, env, LOG_PATH, hyperparameters, tensorboard_log)

    model.learn(TOTAL_TIMESTEPS, callback=callback)
    env_name = env.spec.id
    model_save_path = f"{SAVE_PATH}{env_name}_{agent_name}.zip"
    model.save(model_save_path)
    print(f"{agent_name} model saved to {model_save_path}")
    return model

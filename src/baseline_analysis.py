import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
from config import AGENTS, ENVIRONMENTS, SAVE_PATH
from agents import create_model, evaluate_agent, train_agent
from utils import save_rewards_to_csv, save_combined_rewards_to_csv
from stable_baselines3 import PPO, DQN, A2C  # Import specific classes


EPISODES = 100

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, _observation, deterministic=True):
        return self.action_space.sample(), None

def run_baseline_analysis():
    print("Running baseline analysis...")

    all_agents = AGENTS + ["Random"]
    combined_rewards = {}  # Dictionary to store combined rewards for each environment

    for env_name in ENVIRONMENTS:
        env_rewards = {}  # Store rewards for the current environment

        for agent_name in all_agents:
            rewards = baseline_analysis(agent_name, env_name)
            env_rewards[agent_name] = rewards  # Store rewards for each agent
        
        combined_rewards[env_name] = env_rewards
        save_combined_rewards_to_csv(env_name, env_rewards)  # Save combined rewards

    print("\nBaseline analysis completed for all agent-environment combinations.")

def baseline_analysis(agent_name, env_name, hyperparameters=None, tensorboard_log=None):
    print(f"\nTraining and evaluating {agent_name} on {env_name}...")

    env = gym.make(env_name)

    if agent_name != "Random":
        model = train_agent(agent_name, env, hyperparameters, tensorboard_log)
        model_path = f"{SAVE_PATH}{env_name}_{agent_name}.zip"

        # Load the model based on the agent's name
        if agent_name == 'PPO':
            model = PPO.load(model_path, env=env)
        elif agent_name == 'DQN':
            model = DQN.load(model_path, env=env)
        elif agent_name == 'A2C':
            model = A2C.load(model_path, env=env)
    else:
        model = RandomAgent(env.action_space)

    rewards = evaluate_agent(model, env, num_episodes=EPISODES)
    mean_reward, std_reward = np.mean(rewards), np.std(rewards)

    print(f"Agent: {agent_name}, Environment: {env_name}, Mean Reward: {mean_reward:.2f}, Std Deviation: {std_reward:.2f}")
    save_rewards_to_csv(agent_name, env_name, rewards)

    return rewards

if __name__ == "__main__":
    run_baseline_analysis()
    

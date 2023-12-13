import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
from config import AGENTS, ENVIRONMENTS, SAVE_PATH
from agents import evaluate_agent, train_agent
from utils import save_rewards_to_csv, save_combined_rewards_to_csv
from stable_baselines3 import PPO, DQN, A2C 

EPISODES = 100  # Number of episodes for evaluation
TIMESTEPS = 10000  # Number of timesteps for training

class RandomAgent:
    def __init__(self, action_space):
        # Initialize with the action space of the environment
        self.action_space = action_space

    def predict(self, _observation, deterministic=True):
        # Random action selection
        return self.action_space.sample(), None

def run_baseline_analysis():
    # Main function to run baseline analysis across environments and agents
    print("Running baseline analysis...")

    all_agents = AGENTS + ["Random"]  # Include the random agent in the analysis
    combined_rewards = {} 

    for env_name in ENVIRONMENTS:
        env_rewards = {}  

        for agent_name in all_agents:
            # Run baseline analysis for each agent in the environment
            rewards = baseline_analysis(agent_name, env_name)
            env_rewards[agent_name] = rewards 
        
        combined_rewards[env_name] = env_rewards
        # Save combined rewards of all agents for the current environment
        save_combined_rewards_to_csv(env_name, env_rewards, "baseline_analysis")

    print("\nBaseline analysis completed for all agent-environment combinations.")

def baseline_analysis(agent_name, env_name, hyperparameters=None, tensorboard_log=None):
    # Function to train and evaluate an agent in a given environment
    print(f"\nTraining and evaluating {agent_name} on {env_name}...")

    # Create the environment
    env = gym.make(env_name)

    if agent_name != "Random":
        # Train the agent
        model = train_agent(agent_name, env, hyperparameters, tensorboard_log, TIMESTEPS)
        model_path = f"{SAVE_PATH}{env_name}_{agent_name}.zip"

        # Load the trained model based on the agent's name
        if agent_name == 'PPO':
            model = PPO.load(model_path, env=env)
        elif agent_name == 'DQN':
            model = DQN.load(model_path, env=env)
        elif agent_name == 'A2C':
            model = A2C.load(model_path, env=env)
    else:
        model = RandomAgent(env.action_space)

    # Evaluate the agent
    rewards = evaluate_agent(model, env, num_episodes=EPISODES)
    mean_reward, std_reward = np.mean(rewards), np.std(rewards)

    # Print and save the evaluation results
    print(f"Agent: {agent_name}, Environment: {env_name}, Mean Reward: {mean_reward:.2f}, Std Deviation: {std_reward:.2f}")
    save_rewards_to_csv(agent_name, env_name, rewards, "baseline_analysis")

    return rewards

if __name__ == "__main__":
    run_baseline_analysis()

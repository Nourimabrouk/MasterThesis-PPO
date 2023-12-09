import gymnasium as gym
import numpy as np
from config import AGENTS, ENVIRONMENTS
from agents import create_model, evaluate_agent
from utils import save_rewards_to_csv

EPISODES = 100

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, _observation, deterministic=True):
        return self.action_space.sample(), None

def run_baseline_analysis():
    print("Running baseline analysis...")
    all_agents = AGENTS + ["Random"]

    for env_name in ENVIRONMENTS:
        for agent_name in all_agents:
            baseline_analysis(agent_name, env_name)
    print("\nBaseline analysis completed for all agent-environment combinations.")                    

def baseline_analysis(agent_name, env_name):
    print(f"\nEvaluating {agent_name} on {env_name} for Baseline Analysis...")

    env = gym.make(env_name)

    if agent_name == "Random":
        agent = RandomAgent(env.action_space)
    else:
        agent = create_model(agent_name, env, None)

    rewards = evaluate_agent(agent, env, num_episodes=EPISODES)
    mean_reward, std_reward = np.mean(rewards), np.std(rewards)

    print(f"Agent: {agent_name}, Environment: {env_name}, Mean Reward: {mean_reward:.2f}, Std Deviation: {std_reward:.2f}")
    save_rewards_to_csv(agent_name, env_name, rewards)

if __name__ == "__main__":
    run_baseline_analysis()
    

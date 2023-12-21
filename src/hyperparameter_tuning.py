#### Functions for hyperparameter tuning

from optuna.integration import TensorBoardCallback
import optuna
import logging
import stable_baselines3 as sb3
import numpy as np
import random
import torch
import pandas as pd
import gymnasium as gym
from agents import evaluate_agent
from config import ENVIRONMENTS, AGENTS

# Setting logging level for Optuna
optuna.logging.set_verbosity(optuna.logging.DEBUG)

# Setting seed for reproducibility
SEED = 1337
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Define number of trials and total timesteps for training
TRIALS = 100
TOTAL_TIMESTEPS = 100000

def run_hyperparameter_tuning():
    # Run hyperparameter tuning for all agent-environment combinations
    print("Running hyperparameter tuning...")
    for env_name in ENVIRONMENTS:
        for agent_name in AGENTS:
            run_optimization_for_agent(agent_name, env_name)

    print("\nHyperparameter tuning completed for all agent-environment combinations.")    

def create_model(agent_name, env, trial):
    """Instantiate the agent model based on the specified agent type."""
    # Suggest hyperparameters for each agent
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)

    # Create a model based on the agent type and suggested hyperparameters
    if agent_name == "PPO":
        n_steps_values = [i for i in range(64, 1025, 64)]
        n_steps = trial.suggest_categorical("PPO_n_steps", n_steps_values)
        ent_coef = trial.suggest_float("ent_coef", 1e-5, 0.1, log=True)
        return sb3.PPO("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps, ent_coef=ent_coef, verbose=0)

    elif agent_name == "A2C":
        n_steps = trial.suggest_int("A2C_n_steps", 5, 500)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 0.5)
        return sb3.A2C("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps, vf_coef=vf_coef, verbose=0)

    elif agent_name == "DQN":
        target_update_interval = trial.suggest_int("target_update_interval", 500, 2000)
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
        return sb3.DQN("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, target_update_interval=target_update_interval, exploration_fraction=exploration_fraction, verbose=0)

def objective(trial, env_name, agent_name):
    # Objective function for optimization
    print(f"Running trial {trial.number} for {agent_name} on {env_name}.")
    env = gym.make(env_name)
    if hasattr(env, 'seed'):
        env.seed(SEED)

    # Train the model and evaluate its performance
    model = create_model(agent_name, env, trial)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    rewards = evaluate_agent(model, env)
    env.close()
    
    # Negative mean reward as optimization target
    return -1 * np.mean(rewards)  

def run_optimization_for_agent(agent_name, env_name):
    # Run hyperparameter optimization for a specific agent-environment pair
    print(f"\nOptimizing hyperparameters for agent: {agent_name} on environment: {env_name}")

    study_name = f"{env_name}_{agent_name}"
    study = optuna.create_study(study_name=study_name, direction="minimize")

    tensorboard_dir = f"~/tensorboard_logs/{study_name}"
    tensorboard_callback = TensorBoardCallback(dirname=tensorboard_dir, metric_name="reward")

    try:
        study.optimize(lambda trial: objective(trial, env_name, agent_name), n_trials=TRIALS, callbacks=[tensorboard_callback])

        # Save study results to CSV files
        df = study.trials_dataframe()
        df.to_csv(f'~/output/hyperparameter_analysis/optuna_results_{env_name}_{agent_name}.csv', index=False)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_params['best_reward'] = -best_trial.value

        df_best = pd.DataFrame([best_params])
        df_best.to_csv(f'~/output/hyperparameter_analysis/best_hyperparams_{env_name}_{agent_name}.csv', index=False)

        print(f"Finished optimizing {agent_name} on {env_name}. Best reward: {-study.best_value}")

    except Exception as e:
        print(f"Error encountered during optimization for agent {agent_name} on {env_name}: {e}")

    return study

if __name__ == "__main__":
    run_hyperparameter_tuning()

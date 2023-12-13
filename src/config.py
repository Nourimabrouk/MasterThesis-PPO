#### Config file with parameters for the comparative analysis ####

NUM_RUNS = 10 # 100 used for the paper
TOTAL_TIMESTEPS = 1000 # 1000000 used for the paper

LOG_PATH = './output/tensorboard_logs/'
SAVE_PATH = './output/models/'	
ENVIRONMENTS = ["CartPole-v1", "FrozenLake-v1", "MountainCar-v0"]
AGENTS = ["PPO", "A2C", "DQN"] 


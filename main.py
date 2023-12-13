#### Main file for my master thesis project on the PPO algorithm in Reinforcement Learning
#### By Nouri Mabrouk
#### 2023

from src.baseline_analysis import run_baseline_analysis
from src.hyperparameter_tuning import run_hyperparameter_tuning
from src.comparative_analysis import run_comparative_analysis

def main():
    print("Starting analysis...")

    # Run baseline analysis
    run_baseline_analysis()

    # Perform hyperparameter tuning
    run_hyperparameter_tuning()

    # Conduct comparative analysis
    run_comparative_analysis()

    print("Analysis complete.")

if __name__ == "__main__":
    main()
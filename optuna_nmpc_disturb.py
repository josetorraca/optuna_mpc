import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import optuna
import matplotlib

# Setting the backend for matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI issues

# Add the 'pc-gym/src' directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'pc-gym', 'src'))

import pcgym
from pcgym.pcgym import make_env

# Directory configurations
server_dir = '/rds/general/user/jrn23/home/'
save_dir = os.path.join(server_dir, "saferl-pcgym/optuna-mpc/")
os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

# print(os.listdir('pc-gym'))  # Debugging: list contents of pc-gym
# print(os.listdir(server_dir))  # Debugging: list contents of the server directory

##################################################################################
# Environment and RL Definition
##################################################################################

# Global params
T = 26
nsteps = 100

#Enter required setpoints for each state. Enter None for states without setpoints.
SP = {
    # 'Ca': [0.85 for i in range(int(nsteps/2))] + [0.9 for i in range(int(nsteps/2))],
    # 'Ca': [0.75 for i in range(int(nsteps/2))] + [0.7 for i in range(int(nsteps/2))],
    'T': [324.475443431599 for _ in range(5)] + [340.0 for _ in range(nsteps - 5)],
}

#Continuous box action space
action_space = {
    # 'low': np.array([295]),
    # 'high':np.array([302])
    'low': np.array([250]),
    'high':np.array([350])
}
#Continuous box observation space ([CA, T, CA_Setpoint, T_Setpoint])
observation_space = {
    # 'low' : np.array([0.7,300,0.8]),
    # 'high' : np.array([1,350,0.9])
    'low' : np.array([0.0,200,300]),
    'high' : np.array([1,600,400])
}

r_scale ={
    # 'Ca': 5, #Reward scale for each state,
    'T': 1e-6 #Reward scale for each state,
}

# Environment parameters
env_params = {
    'Nx': 2,
    'N': 100,
    'tsim': 26,
    'Nu': 1,
    'SP': {'T': [324.475443431599] * 5 + [340.0] * 95},
    'o_space': {'low': np.array([0.0, 200, 300]), 'high': np.array([1, 600, 400])},
    'a_space': {'low': np.array([250]), 'high': np.array([350])},
    'x0': np.array([0.87725294608097, 324.475443431599, 324.475443431599]),
    'model': 'cstr_ode',
    'r_scale': {'T': 1e-6},
    'normalise_a': True,
    'normalise_o': True,
    'noise': True,
    'integration_method': 'casadi',
    'noise_percentage': 0.001
}

# Base disturbances
disturbances = {'Ti': np.repeat([400, 420, 380], [nsteps//4, nsteps//2, nsteps//4])}

# Disturbance bounds for 'Ti'
disturbance_space = {
    'low': np.array([100]),  # Lower bound for 'Ti'
    'high': np.array([600])  # Upper bound for 'Ti'
}

# Update env_params with disturbances and their bounds
env_params.update({
    'disturbances': disturbances,
    'disturbance_bounds': disturbance_space, # Disturbance space
})

# Configuration for reinforcement learning model
config = {
    "policy": 'MlpPolicy',
    "learning_rate": lambda p: 0.0004 - (0.0001 * p ** 5),
    "gamma": 0.99,
    "total_timesteps": 1000,
    "gae_lambda": 0.99,
    "ent_coef": 0.005,
    "batch_size": 64,
    "n_steps": 128,
    "n_epochs": 10,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "seed": 1990,
    "check_freq": 100,
    "n_eval_episodes": 10,
    "positive_definiteness_penalty_weight": 6,
    "derivative_penalty_weight": 6,
    "noise_percentage": 0.001
}

# Function to create the environment
def create_env(env_params):
    return make_env(env_params)

# Instantiate the environment
env = create_env(env_params)
eval_env = create_env(env_params)  # Evaluation environment

##################################################################################
# Training a common RL agent
##################################################################################

# Setup the model
model = PPO(
    config['policy'],
    env,
    learning_rate=config['learning_rate'],
    clip_range=config['clip_range'],
    clip_range_vf=config['clip_range_vf'],
    batch_size=config['batch_size'],
    n_steps=config['n_steps'],
    seed=config['seed'],
    gamma=config['gamma'],
    gae_lambda=config['gae_lambda'],
    ent_coef=config['ent_coef'],
    n_epochs=config['n_epochs'],
    policy_kwargs={'activation_fn': th.nn.Tanh, 'net_arch': {'pi': [32, 32], 'vf': [32, 32]}},
    tensorboard_log=f"{save_dir}/runs/ppo",
    verbose=1
)

# Callback for saving the best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=save_dir,
    log_path=save_dir,
    eval_freq=config['check_freq'],
    n_eval_episodes=config['n_eval_episodes'],
    deterministic=True,
    render=False
)

# Train the model
model.learn(total_timesteps=config['total_timesteps'], callback=eval_callback)

# Save the trained model
model.save(os.path.join(save_dir, "PPO_cstr"))

##################################################################################
# Optuna - NMPC hyperparameter tuning
##################################################################################

# Function to run the Optuna study
def run_base_disturb_study(env, save_dir, n_trials, rollout_number):
    def objective(trial):
        N = trial.suggest_int('N', 5, 20)
        # R = trial.suggest_loguniform('R', 1e-10, 1e-6)
        R = trial.suggest_uniform('R', 1e-6, 1e-5)  # Suggesting a range for regularization term using linear scale
        try:
            evaluator, data = env.plot_rollout({'PPO': PPO.load(os.path.join(save_dir, "best_model"))}, rollout_number, oracle=True, dist_reward=False, MPC_params={'N': N, 'R': R})
            return np.mean(data['oracle']['r'][0])
        except Exception as e:
            print(f"Optuna trial failed with N={N}, R={R}: {str(e)}")
            return float('-inf')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Save Optuna results
    results_path = os.path.join(save_dir, "nmpc_optimization_disturb_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Best trial number: {study.best_trial.number}\n")
        f.write(f"Best parameters: {study.best_params}\n")
        f.write(f"Best observed reward: {study.best_value}\n")

    return study

##################################################################################
# Optuna - Testing alternative disturbance (higher)
##################################################################################

def run_alternative_disturb_study(env_params, save_dir, n_trials, rollout_number):
    # Update the environment parameters for the new disturbances
    disturbances3 = {'Ti': np.repeat([400, 440, 380], [nsteps//4, nsteps//2, nsteps//4])}

    # Update env_params with alternative disturbances
    env_params.update({
        'disturbances': disturbances3,
    })

    # Create a new environment with updated disturbances
    env = create_env(env_params)

    def objective(trial):
        N = trial.suggest_int('N', 5, 20)  # Suggesting an integer range for the prediction horizon
        # R = trial.suggest_loguniform('R', 1e-2, 1e-1)  # Suggesting a range for regularization term using log scale
        R = trial.suggest_uniform('R', 1e-2, 1e-1)  # Suggesting a range for regularization term using linear scale
        try:
            evaluator, data = env.plot_rollout({'PPO': PPO.load(os.path.join(save_dir, "best_model"))}, rollout_number, oracle=True, dist_reward=False, MPC_params={'N': N, 'R': R})
            return np.mean(data['oracle']['r'][0])
        except Exception as e:
            print(f"Failed with N={N}, R={R} due to {str(e)}")
            return float('-inf')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Save Optuna results for the alternative trajectory
    results_path = os.path.join(save_dir, "nmpc_optimization_alt_disturb_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Best trial number: {study.best_trial.number}\n")
        f.write(f"Best parameters: {study.best_params}\n")
        f.write(f"Best observed reward: {study.best_value}\n")

    return study
    
##################################################################################
# Main Execution
##################################################################################

def main():
    # Number of trial evaluations
    n_trials=100

    # Number of rollouts for evaluation
    rollout_number = 3
    
    # Suppress output for cleaner execution
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    # Run studies
    standard_study = run_base_disturb_study(env, save_dir, n_trials, rollout_number)
    alternative_study = run_alternative_disturb_study(env_params, save_dir, n_trials, rollout_number)

    # Re-enable output
    sys.stdout = original_stdout
    sys.stderr = original_stderr

if __name__ == "__main__":
    main()

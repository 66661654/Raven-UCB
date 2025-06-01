# coding=utf-8
"""
Authors:
    Junyi Fang (junyifang001@gmail.com)

Description:
    This script generates Figure 2 from the paper "From Theory to Practice with RAVEN-UCB:
    Addressing Non-Stationarity in Multi-Armed Bandits through Variance Adaptation".

"""

import numpy as np
import optuna
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# ---- Global Settings ----
RANDOM_SEED = 2025  # Random seed for reproducibility of results
np.random.seed(RANDOM_SEED)  # Set the seed for NumPy's random number generator
K = 10  # Number of arms in the multi-armed bandit problem
M = 50  # Number of independent trials per configuration to compute average performance

# ---- Simulation and Evaluation Functions ----

def simulate_single(alpha0: float, beta0: float, epsilon: float, T: int, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Simulate a single trial of UCB1 and RAVEN-UCB (V-UCB) algorithms for a given time horizon T.

    Args:
        alpha0 (float): Initial exploration parameter for RAVEN-UCB.
        beta0 (float): Variance scaling parameter for RAVEN-UCB.
        epsilon (float): Small constant to avoid division by zero in exploration term.
        T (int): Time horizon (number of steps).
        K (int): Number of arms.

    Returns:
        Tuple containing:
        - None: Placeholder for future use (e.g., arm counts).
        - None: Placeholder for future use (e.g., arm means).
        - reg_ucb (np.ndarray): Cumulative regret for UCB1 at each step.
        - reg_vucb (np.ndarray): Cumulative regret for RAVEN-UCB at each step.
        - 0: Placeholder for future metrics.
        - 0: Placeholder for future metrics.
    """
    # Initialize true reward probabilities (thetas) for each arm, drawn from a uniform distribution
    theta0 = np.random.uniform(0.8, 0.95, K)
    best_arm = int(np.argmax(theta0))  # Identify the best arm with the highest reward probability
    best_rate = float(theta0[best_arm])  # Reward probability of the best arm
    thetas = theta0.copy()  # Working copy of reward probabilities

    # Initialize counts and means for UCB1 and RAVEN-UCB
    counts = {'ucb': np.zeros(K), 'vucb': np.zeros(K)}  # Number of pulls for each arm
    means = {'ucb': np.zeros(K), 'vucb': np.zeros(K)}  # Estimated mean reward for each arm
    vars_v = np.zeros(K)  # Estimated variance for RAVEN-UCB

    # Arrays to store cumulative regret for UCB1 and RAVEN-UCB
    reg_ucb = np.zeros(T)
    reg_vucb = np.zeros(T)

    # Simulate for T time steps
    for t in range(T):
        # First K steps: Pull each arm once to initialize
        if t < K:
            arm_u = arm_v = t
        else:
            # UCB1: Select arm with highest upper confidence bound
            bonus = np.sqrt(2 * np.log(t + 1) / (counts['ucb'] + 1))
            arm_u = int(np.argmax(means['ucb'] + bonus))

            # RAVEN-UCB: Select arm with variance-adjusted upper confidence bound
            alpha_t = alpha0 / np.log(t + 1 + 1e-6)  # Decaying exploration coefficient
            b1 = alpha_t * np.sqrt(np.log(t + 1) / (counts['vucb'] + 1))  # Exploration term
            b2 = beta0 * np.sqrt(vars_v / (counts['vucb'] + 1) + 1e-6)  # Variance term
            arm_v = int(np.argmax(means['vucb'] + b1 + b2))

        # UCB1: Pull the selected arm and update statistics
        counts['ucb'][arm_u] += 1
        r_u = float(np.random.rand() < thetas[arm_u])  # Bernoulli reward (0 or 1)
        means['ucb'][arm_u] += (r_u - means['ucb'][arm_u]) / counts['ucb'][arm_u]  # Update mean
        reg_ucb[t] = (reg_ucb[t-1] if t > 0 else 0) + (best_rate - thetas[arm_u])  # Update regret

        # RAVEN-UCB: Pull the selected arm and update statistics
        counts['vucb'][arm_v] += 1
        r_v = float(np.random.rand() < thetas[arm_v])  # Bernoulli reward (0 or 1)
        old = means['vucb'][arm_v]
        means['vucb'][arm_v] += (r_v - old) / counts['vucb'][arm_v]  # Update mean
        if counts['vucb'][arm_v] > 1:
            vars_v[arm_v] += (r_v - old) * (r_v - means['vucb'][arm_v])  # Update variance numerator
            vars_v[arm_v] /= (counts['vucb'][arm_v] - 1)  # Normalize variance
        reg_vucb[t] = (reg_vucb[t-1] if t > 0 else 0) + (best_rate - thetas[arm_v])  # Update regret

    return None, None, reg_ucb, reg_vucb, 0, 0

def evaluate(alpha0: float, beta0: float, epsilon: float, T: int, K: int, M: int, seed: int) -> Dict[str, float]:
    """
    Evaluate the performance of UCB1 and RAVEN-UCB over M trials and compute average regret reduction.

    Args:
        alpha0 (float): Initial exploration parameter for RAVEN-UCB.
        beta0 (float): Variance scaling parameter for RAVEN-UCB.
        epsilon (float): Small constant to avoid division by zero.
        T (int): Time horizon.
        K (int): Number of arms.
        M (int): Number of independent trials.
        seed (int): Random seed for reproducibility.

    Returns:
        Dict containing:
        - 'imp_reg_%': Percentage regret reduction of RAVEN-UCB compared to UCB1.
    """
    all_r_u, all_r_v = [], []  # Store regrets for all trials
    np.random.seed(seed)  # Set seed for reproducibility
    for _ in range(M):
        _, _, r_u, r_v, _, _ = simulate_single(alpha0, beta0, epsilon, T, K)  # Run a single trial
        all_r_u.append(r_u)
        all_r_v.append(r_v)
    mean_r_u = np.stack(all_r_u).mean(axis=0)  # Average regret for UCB1
    mean_r_v = np.stack(all_r_v).mean(axis=0)  # Average regret for RAVEN-UCB
    imp_reg = (mean_r_u[-1] - mean_r_v[-1]) / mean_r_u[-1] * 100  # Compute regret reduction percentage
    return {'imp_reg_%': imp_reg}

def objective(trial: optuna.trial.Trial, T: int) -> float:
    """
    Objective function for Optuna to optimize hyperparameters for maximizing regret reduction.

    Args:
        trial (optuna.trial.Trial): Optuna trial object for suggesting hyperparameters.
        T (int): Time horizon.

    Returns:
        float: Regret reduction percentage to maximize.
    """
    # Suggest hyperparameters within specified ranges
    alpha0 = trial.suggest_loguniform('alpha0', 1e-2, 10.0)
    beta0 = trial.suggest_loguniform('beta0', 1e-2, 10.0)
    epsilon = trial.suggest_uniform('epsilon', 1e-3, 0.5)
    perf = evaluate(alpha0, beta0, epsilon, T, K, M, RANDOM_SEED)  # Evaluate performance
    return perf['imp_reg_%']

# ---- T-varying Hyperparameter Optimization and Collecting Results ----
# Define a range of time horizons T to evaluate the regret reduction
Ts = [100, 200, 250, 500, 1000, 2000, 5000, 5200, 5400, 5500, 6000, 7000, 8000, 9000, 9500, 10000, 10500, 11000, 13000, 13500, 14000, 14500, 15000]
reg_reductions = []  # Store regret reduction percentages for each T

# Optimize hyperparameters for each T and collect regret reduction results
for T in Ts:
    study = optuna.create_study(direction='maximize')  # Create an Optuna study to maximize regret reduction
    study.optimize(lambda tr: objective(tr, T), n_trials=10, timeout=300)  # Run optimization for 10 trials or 300 seconds
    best = study.best_params  # Get the best hyperparameters
    perf = evaluate(best['alpha0'], best['beta0'], best['epsilon'], T, K, M, RANDOM_SEED)  # Evaluate with best parameters
    reg_reductions.append(perf['imp_reg_%'])  # Store the regret reduction
    print(f"Results for T = {T}: Best Parameters = {best}, Imp. Reg. = {perf['imp_reg_%']:.2f}%")  # Print results for this T

# ---- Plotting the Curve ----
# Plot the regret reduction as a function of T
plt.plot(Ts, reg_reductions, marker='o', label='Regret Reduction')  # Plot empirical regret reduction
plt.axhline(y=84, color='red', linestyle='--', label='Theoretical Regret Reduction')  # Add theoretical line at 84%

# Configure plot settings
plt.xscale('log')  # Use logarithmic scale for x-axis (T)
plt.xlabel('T (Number of Steps)')  # Label for x-axis
plt.ylabel('Regret Reduction (%)')  # Label for y-axis
plt.title('Regret Reduction as a Function of Time Horizon T')  # Plot title
plt.grid(True)  # Add grid for better readability
plt.legend(loc='lower right')  # Add legend in the lower right corner
plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('regret_reduction_vs_time_horizon.png')  # Save the plot as a PNG file
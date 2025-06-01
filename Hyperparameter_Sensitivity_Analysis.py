# coding=utf-8
"""
Authors:
    Junyi Fang (junyifang001@gmail.com)

Description:
    This script generates Figure 3 from the paper "From Theory to Practice with RAVEN-UCB:
    Addressing Non-Stationarity in Multi-Armed Bandits through Variance Adaptation".
    The figure titled "Sensitivity Analysis of Hyperparameters α₀ and β₀" includes multiple visualizations:
    - Heatmaps of cumulative regret for different scenarios (Variance Drift, Incremental Drift, Blips/Outliers) and time horizons.
    - Line plots showing regret trends across α₀ values for fixed β₀ values.
    - Box plots of regret distributions for the best parameters per scenario and time horizon.
    - A 3D surface plot for Variance Drift (T=10000) to visualize the regret surface.

"""
import numpy as np
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from mpl_toolkits.mplot3d import Axes3D

# Create output directory for figures
if not os.path.exists("figures"):
    os.makedirs("figures")

# RAVEN-UCB Algorithm Implementation
def raven_ucb(K, T, alpha_0, beta_0, epsilon, scenario, mu_range=(0.3, 0.8), sigma_range=(0.01, 0.09), shift_interval=5000):
    N = np.zeros(K)
    M = np.zeros(K)
    S2 = np.zeros(K)
    total_reward = 0
    regret = 0
    suboptimal_pulls = 0
    mu = np.random.uniform(mu_range[0], mu_range[1], K)
    sigma = np.sqrt(np.random.uniform(sigma_range[0], sigma_range[1], K))
    optimal_arm = np.argmax(mu)

    for t in range(1, T + 1):
        if scenario == "Variance Drift" and t % shift_interval == 0:
            sigma = np.sqrt(np.random.uniform(sigma_range[0], sigma_range[1], K))
        elif scenario == "Incremental Drift" and t % shift_interval == 0:
            mu += np.random.uniform(-0.01, 0.01, K)
            mu = np.clip(mu, mu_range[0], mu_range[1])
            optimal_arm = np.argmax(mu)
        elif scenario == "Blips / Outliers" and t % shift_interval == 0:
            blip_duration = 100
            if t <= T - blip_duration:
                mu_temp = np.random.uniform(mu_range[0], mu_range[1], K)
                sigma_temp = np.sqrt(np.random.uniform(sigma_range[0], sigma_range[1], K))
                for tau in range(t, min(t + blip_duration, T + 1)):
                    if tau == t:
                        mu, sigma = mu_temp, sigma_temp
                    optimal_arm = np.argmax(mu)

        if t <= K:
            k_t = t - 1
        else:
            alpha_t = alpha_0 / np.log(t + epsilon)
            scores = M + alpha_t * np.sqrt(np.log(t + 1) / (N + 1)) + beta_0 * np.sqrt(S2 / (N + 1) + epsilon)
            k_t = np.argmax(scores)

        reward = np.random.normal(mu[k_t], sigma[k_t])
        total_reward += reward
        regret += mu[optimal_arm] - mu[k_t]
        if k_t != optimal_arm:
            suboptimal_pulls += 1

        N[k_t] += 1
        n = N[k_t]
        prev_mean = M[k_t]
        M[k_t] += (reward - M[k_t]) / n
        if n > 1:
            S2[k_t] = ((n - 2) * S2[k_t] + (reward - prev_mean) * (reward - M[k_t])) / (n - 1)
        else:
            S2[k_t] = 0

    return total_reward, regret, suboptimal_pulls

# Parameter search settings
K = 10
T_values = [1000, 10000]
scenarios = ["Variance Drift", "Incremental Drift", "Blips / Outliers"]
alpha_0_values = [0.5, 1.0, 5.0, 10.0]
beta_0_values = [0.5, 1.0, 5.0, 10.0]
epsilon = 0.1
N_trials = 10

# Run parameter search
results = []
for scenario, T in product(scenarios, T_values):
    for alpha_0, beta_0 in product(alpha_0_values, beta_0_values):
        total_rewards = []
        regrets = []
        suboptimal_pulls_list = []
        for _ in range(N_trials):
            total_reward, regret, suboptimal_pulls = raven_ucb(
                K, T, alpha_0, beta_0, epsilon, scenario
            )
            total_rewards.append(total_reward)
            regrets.append(regret)
            suboptimal_pulls_list.append(suboptimal_pulls)

        avg_reward = np.mean(total_rewards)
        avg_regret = np.mean(regrets)
        avg_suboptimal_pulls = np.mean(suboptimal_pulls_list)

        results.append({
            "Scenario": scenario,
            "Time Horizon (T)": T,
            "alpha_0": alpha_0,
            "beta_0": beta_0,
            "Cumulative Reward": avg_reward,
            "Cumulative Regret": avg_regret,
            "Suboptimal Pulls": avg_suboptimal_pulls
        })

# Create results table
results_df = pd.DataFrame(results)
print("\nParameter Search Results Table:")
print(results_df.round(2))

# Find best parameters for each scenario
best_params = results_df.groupby(["Scenario", "Time Horizon (T)"]).apply(
    lambda x: x.loc[x["Cumulative Regret"].idxmin()], include_groups=False
).reset_index()[["Scenario", "Time Horizon (T)", "alpha_0", "beta_0", "Cumulative Reward", "Cumulative Regret", "Suboptimal Pulls"]]

print("\nBest Parameters for Each Scenario:")
print(best_params.round(2))

# Save results to CSV
results_df.to_csv("figures/raven_ucb_param_search_results.csv", index=False)

# Visualization 1: Heatmaps for Cumulative Regret
plt.figure(figsize=(15, 10))
for i, scenario in enumerate(scenarios):
    for j, T in enumerate(T_values):
        subset = results_df[(results_df["Scenario"] == scenario) & (results_df["Time Horizon (T)"] == T)]
        pivot = subset.pivot(index="alpha_0", columns="beta_0", values="Cumulative Regret")
        plt.subplot(len(scenarios), len(T_values), i * len(T_values) + j + 1)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Cumulative Regret'})
        plt.title(f"{scenario} (T={T})")
        plt.xlabel(r"$\beta_0$")
        plt.ylabel(r"$\alpha_0$")
plt.tight_layout()
plt.savefig("figures/heatmap_regret.png")
plt.close()

# Visualization 2: Line Plots for Regret Trends
plt.figure(figsize=(15, 10))
for i, scenario in enumerate(scenarios):
    for j, T in enumerate(T_values):
        subset = results_df[(results_df["Scenario"] == scenario) & (results_df["Time Horizon (T)"] == T)]
        plt.subplot(len(scenarios), len(T_values), i * len(T_values) + j + 1)
        for beta_0 in beta_0_values:
            sub_subset = subset[subset["beta_0"] == beta_0]
            plt.plot(sub_subset["alpha_0"], sub_subset["Cumulative Regret"], marker='o', label=f"β₀={beta_0}")
        plt.title(f"{scenario} (T={T})")
        plt.xlabel(r"$\alpha_0$")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.grid(True)
plt.tight_layout()
plt.savefig("figures/lineplot_regret.png")
plt.close()

# Visualization 3: Box Plots for Best Parameters
best_params_list = best_params.to_dict('records')
plt.figure(figsize=(12, 8))
regret_data = []
labels = []
for param in best_params_list:
    scenario = param["Scenario"]
    T = param["Time Horizon (T)"]
    alpha_0 = param["alpha_0"]
    beta_0 = param["beta_0"]
    # Re-run simulation for best parameters to get per-trial regrets
    regrets = []
    for _ in range(N_trials):
        _, regret, _ = raven_ucb(K, T, alpha_0, beta_0, epsilon, scenario)
        regrets.append(regret)
    regret_data.append(regrets)
    labels.append(f"{scenario}\nT={T}\nα₀={alpha_0}, β₀={beta_0}")
plt.boxplot(regret_data, labels=labels)
plt.title("Regret Distribution for Best Parameters")
plt.ylabel("Cumulative Regret")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/boxplot_best_params.png")
plt.close()

# Visualization 4: 3D Surface Plot for Variance Drift (T=10000)
subset = results_df[(results_df["Scenario"] == "Variance Drift") & (results_df["Time Horizon (T)"] == 10000)]
X = np.array(subset["alpha_0"]).reshape(len(alpha_0_values), len(beta_0_values))
Y = np.array(subset["beta_0"]).reshape(len(alpha_0_values), len(beta_0_values))
Z = np.array(subset["Cumulative Regret"]).reshape(len(alpha_0_values), len(beta_0_values))
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='YlGnBu')])
fig.update_layout(
    title="Cumulative Regret for Variance Drift (T=10000)",
    scene=dict(xaxis_title=r"$\alpha_0$", yaxis_title=r"$\beta_0$", zaxis_title="Cumulative Regret"),
    width=800, height=600
)
fig.write_html("figures/surface_plot_variance_drift.html")

print("Visualizations saved in 'figures' directory.")
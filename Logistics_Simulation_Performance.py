# coding=utf-8
"""
Authors:
    Junyi Fang (junyifang001@gmail.com)
    Yuxun Chen (csu.yuxun.chen@gmail.com)

Description:
    This script generates Figure 4 from the paper "From Theory to Practice with RAVEN-UCB:
    Addressing Non-Stationarity in Multi-Armed Bandits through Variance Adaptation".
    The figure titled "Performance Metrics in Logistics Simulation" includes two subfigures:
    - (a) Cumulative Regret Over Time: Compares the cumulative regret of RAVEN-UCB and other algorithms.
    - (b) Cumulative Reward Over Time: Compares the cumulative reward of RAVEN-UCB and other algorithms.
    Additionally, the script generates a boxplot of final regret distributions, a bar plot of running times,
    and saves statistical significance test results in a results.txt file.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    ttest_rel, norm, expon, lognorm, pareto, beta, poisson,
    geom, gamma, weibull_min, triang, binom, cauchy, chi2,
    t as student_t, logistic, laplace
)
import os
import datetime
import time

# Set random seed for reproducibility
np.random.seed(2025)

# Simulation parameters
K = 100              # Number of arms
T = 50000           # Number of time steps
M = 50               # Number of trials
reset_times = list(range(5000, T, 5000))  # Reset every 5000 steps
reset_num_arms = K // 3  # Reset
reward_distribution = 'normal'

# Algorithm list
algorithms = ['vucb', 'ucb', 'ucbv_orig', 'egreedy', 'thompson',
              'fdsw_min', 'wls_ots', 'ccb', 'ucb_imp']

# Hyperparameters (empirical settings)
# RAVEN-UCB
alpha0 = 0.5
beta0 = 1.0
vucb_eps = 1e-6
# UCB
c_ucb = 2.0
# UCB-V
zeta = 1.0
c = 1.0
# ε-greedy
egreedy_eps = 0.1
# f-DSW Thompson Sampling
gamma_fdsw = 0.99
# WLS + Optimistic TS
lambda_wls = 0.95
n_fdsw = 50

# Mapping algorithm keys to display labels
label_map = {
    'vucb': 'RAVEN-UCB',
    'ucb': 'UCB',
    'ucbv_orig': 'UCB-V',
    'egreedy': '$\\epsilon$-greedy',
    'thompson': 'Thompson Sampling',
    'fdsw_min': 'f-dsw TS (min)',
    'wls_ots': 'WLS + Optimistic TS',
    'ccb': 'CCB',
    'ucb_imp': 'UCB-Imp'
}

# Marker styles for plotting
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', '+']
marker_size = 5
markevery = 2000  # Adjusted for T=50000

# Reward sampling function
def reward_function(rate, variance, distribution=reward_distribution):
    if distribution == 'bernoulli':
        return float(np.random.rand() < rate)
    if distribution == 'normal':
        return norm.rvs(loc=rate, scale=np.sqrt(variance))
    if distribution == 'exponential':
        return expon.rvs(scale=1.0/rate)
    if distribution == 'uniform':
        return np.random.uniform(0, rate)
    if distribution == 'lognormal':
        return lognorm.rvs(s=0.5, scale=np.exp(rate))
    if distribution == 'pareto':
        return pareto.rvs(b=2.0) * rate
    if distribution == 'beta':
        a_p, b_p = rate*5+1, (1-rate)*5+1
        return beta.rvs(a_p, b_p)
    if distribution == 'poisson':
        return poisson.rvs(mu=rate*5)
    if distribution == 'geometric':
        return geom.rvs(p=rate)
    if distribution == 'triangular':
        return triang.rvs(c=0.5, loc=0, scale=rate)
    if distribution == 'logistic':
        return logistic.rvs(loc=rate, scale=0.1)
    if distribution == 'laplace':
        return laplace.rvs(loc=rate, scale=0.1)
    raise ValueError(f"Unknown distribution: {distribution}")

# CCB parameters
def r_i_ccb(t, n_i, T):
    return np.sqrt(T / (n_i + 1e-6))

# UCB-Imp parameters
def log_term(T, delta_m):
    return np.log(T * delta_m**2)

# Prepare results directory
desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
results_folder = os.path.join(desktop, f'results_{timestamp}')
os.makedirs(results_folder, exist_ok=True)

# Container for printed output
output_lines = []

def print_and_save(text):
    print(text)
    output_lines.append(text)

def run_simulation():
    # Initialize arm rates and variances
    initial_rate = np.linspace(0.3, 0.8, K)  # Mean range [0.3, 0.8]
    variances = np.linspace(0.1, 0.3, K) ** 2  # Variance range [0.01, 0.09]
    current_rate = initial_rate.copy()
    current_variances = variances.copy()
    dispatch_count = np.zeros(K, int)
    counts = {alg: np.zeros(K, int) for alg in algorithms}
    means = {alg: np.zeros(K) for alg in ['vucb', 'ucb', 'ucbv_orig', 'egreedy', 'wls_ots', 'ccb', 'ucb_imp']}
    vars_dict = {alg: np.zeros(K) for alg in ['vucb', 'ucbv_orig']}
    a_params = {alg: np.ones(K) for alg in ['thompson', 'fdsw_min']}
    b_params = {alg: np.ones(K) for alg in ['thompson', 'fdsw_min']}
    best0 = initial_rate.max()

    # CCB specific variables
    Delta_m_ccb = 1.0
    m_ccb = 0
    T_m_ccb = 0

    # UCB-Imp specific variables
    Delta_m_ucb_imp = 1.0
    m_ucb_imp = 0
    B_m_ucb_imp = list(range(K))
    n_m_ucb_imp = int(np.ceil(2 * log_term(T, Delta_m_ucb_imp) / Delta_m_ucb_imp**2))

    # WLS + Optimistic TS specific variables
    data_wls = [[] for _ in range(K)]

    succ = {alg: np.zeros(T) for alg in algorithms}
    reg = {alg: np.zeros(T) for alg in algorithms}
    sub = {alg: 0 for alg in algorithms}
    alg_times = {alg: 0 for alg in algorithms}

    # Cache log terms for RAVEN-UCB
    log_t_plus_eps = np.log(np.arange(1, T + 1) + vucb_eps)
    log_t_plus_one = np.log(np.arange(1, T + 1))

    # Main loop over time steps
    for t in range(T):
        if t in reset_times:
            reset_indices = np.random.choice(K, reset_num_arms, replace=False)
            new_means = np.random.uniform(0.3, 0.8, reset_num_arms)
            current_rate[reset_indices] = new_means
            new_variances = np.array([0.1 + 0.2 * np.random.rand() for _ in range(reset_num_arms)]) ** 2
            current_variances[reset_indices] = new_variances
        best_now = int(np.argmax(current_rate))

        for alg in algorithms:
            start_time = time.perf_counter()
            if t < K:
                arm = t
            else:
                if alg == 'ucb':
                    bonus = np.sqrt(c_ucb * log_t_plus_one[t] / (counts[alg] + 1))
                    arm = int(np.argmax(means[alg] + bonus))
                elif alg == 'egreedy':
                    arm = (np.random.randint(K) if np.random.rand() < egreedy_eps
                           else int(np.argmax(means[alg])))
                elif alg == 'thompson':
                    b_params[alg] = np.maximum(b_params[alg], 1e-6)
                    samples = np.random.beta(a_params[alg], b_params[alg])
                    arm = int(np.argmax(samples))
                elif alg == 'ucbv_orig':
                    var_est = np.maximum(vars_dict['ucbv_orig'], 0)
                    bonus_var = np.sqrt(var_est) * np.sqrt(2 * zeta * log_t_plus_one[t] / (counts[alg] + 1))
                    bonus_mean = c * 3 * zeta * log_t_plus_one[t] / (counts[alg] + 1)
                    arm = int(np.argmax(means[alg] + bonus_var + bonus_mean))
                elif alg == 'vucb':
                    exploration = alpha0 / log_t_plus_eps[t] * np.sqrt(log_t_plus_one[t] / (counts[alg] + 1))
                    variance_term = beta0 * np.sqrt(np.maximum(vars_dict['vucb'], 0) / (counts[alg] + 1) + vucb_eps)
                    arm = int(np.argmax(means[alg] + exploration + variance_term))
                elif alg == 'fdsw_min':
                    a_hist = a_params[alg].copy()
                    b_hist = b_params[alg].copy()
                    a_hist *= gamma_fdsw
                    b_hist *= gamma_fdsw
                    theta_hist = np.random.beta(a_hist, b_hist)
                    a_params[alg] *= gamma_fdsw
                    b_params[alg] *= gamma_fdsw
                    theta_hot = np.random.beta(a_params[alg], b_params[alg])
                    theta = np.minimum(theta_hist, theta_hot)
                    arm = int(np.argmax(theta))
                elif alg == 'wls_ots':
                    predictions = []
                    for k in range(K):
                        if len(data_wls[k]) < 2:
                            predictions.append(np.random.rand())
                        else:
                            times, rewards = zip(*data_wls[k])
                            times = np.array(times)
                            rewards = np.array(rewards)
                            weights = lambda_wls ** (t - times)
                            X = np.vstack([np.ones_like(times), times]).T
                            W = np.diag(weights)
                            coef = np.linalg.lstsq(W @ X, W @ rewards, rcond=None)[0]
                            pred = coef[0] + coef[1] * t
                            residuals = rewards - (coef[0] + coef[1] * times)
                            num = np.sum(weights * residuals**2)
                            denom = max(np.sum(weights) - 2, 1)
                            std_est = np.sqrt(max(num / denom, 0))
                            std_est = max(std_est, 1e-6)
                            sample = np.random.normal(pred, std_est)
                            predictions.append(max(pred, sample))
                    arm = int(np.argmax(predictions))
                elif alg == 'ccb':
                    r_i = r_i_ccb(t, counts[alg], T)
                    ucb_i = means[alg] + (Delta_m_ccb / 2) * np.sqrt(r_i)
                    arm = int(np.argmax(ucb_i))
                elif alg == 'ucb_imp':
                    if len(B_m_ucb_imp) > 1:
                        arm = B_m_ucb_imp[np.random.randint(len(B_m_ucb_imp))]
                    else:
                        arm = B_m_ucb_imp[0]
            end_time = time.perf_counter()
            alg_times[alg] += end_time - start_time

            # Common updates
            counts[alg][arm] += 1
            dispatch_count[arm] += 1
            reward = reward_function(current_rate[arm], current_variances[arm])

            # Parameter updates
            start_time = time.perf_counter()
            if alg in ['ucb', 'egreedy', 'vucb', 'ucbv_orig', 'wls_ots', 'ccb', 'ucb_imp']:
                n = counts[alg][arm]
                prev_mean = means[alg][arm]
                means[alg][arm] += (reward - prev_mean) / n
                if alg == 'vucb':
                    diff = reward - means[alg][arm]
                    vars_dict[alg][arm] = ((n - 1) * vars_dict[alg][arm] + (reward - prev_mean) * diff) / max(n - 1, 1)
                elif alg == 'ucbv_orig':
                    diff = reward - means[alg][arm]
                    vars_dict[alg][arm] = ((n - 1) * vars_dict[alg][arm] + (reward - prev_mean) * diff) / max(n - 1, 1)
            elif alg == 'thompson':
                scaled_reward = (reward - min(current_rate)) / (max(current_rate) - min(current_rate) + 1e-6)
                scaled_reward = np.clip(scaled_reward, 0, 1)
                a_params[alg][arm] += scaled_reward
                b_params[alg][arm] += (1 - scaled_reward)
            elif alg == 'fdsw_min':
                scaled_reward = (reward - min(current_rate)) / (max(current_rate) - min(current_rate) + 1e-6)
                scaled_reward = np.clip(scaled_reward, 0, 1)
                a_params[alg][arm] += scaled_reward
                b_params[alg][arm] += (1 - scaled_reward)

            # Update WLS + Optimistic TS data
            if alg == 'wls_ots':
                data_wls[arm].append((t, reward))
                if len(data_wls[arm]) > n_fdsw:
                    data_wls[arm].pop(0)

            # Update CCB specific variables
            if alg == 'ccb':
                if counts[alg][arm] >= T_m_ccb:
                    Delta_m_ccb /= 2
                    m_ccb += 1
                    T_m_ccb += int(np.ceil(2 * log_term(T, Delta_m_ccb) / Delta_m_ccb**2))

            # Update UCB-Imp specific variables
            if alg == 'ucb_imp':
                if counts[alg][arm] >= n_m_ucb_imp:
                    if all(counts[alg][i] >= n_m_ucb_imp for i in B_m_ucb_imp):
                        max_lower = max(means[alg][j] - np.sqrt(log_term(T, Delta_m_ucb_imp) / (2 * n_m_ucb_imp)) for j in B_m_ucb_imp)
                        B_m_ucb_imp = [i for i in B_m_ucb_imp if means[alg][i] + np.sqrt(log_term(T, Delta_m_ucb_imp) / (2 * n_m_ucb_imp)) >= max_lower]
                        Delta_m_ucb_imp /= 2
                        m_ucb_imp += 1
                        n_m_ucb_imp = int(np.ceil(2 * log_term(T, Delta_m_ucb_imp) / Delta_m_ucb_imp**2))
            end_time = time.perf_counter()
            alg_times[alg] += end_time - start_time

            # Record cumulative success and regret
            succ[alg][t] = succ[alg][t - 1] + reward if t > 0 else reward
            reg[alg][t] = reg[alg][t - 1] + (best0 - current_rate[arm]) if t > 0 else (best0 - current_rate[arm])
            if arm != best_now:
                sub[alg] += 1

    return succ, reg, sub, alg_times

# Aggregate results over multiple trials
succ_all = {alg: [] for alg in algorithms}
reg_all = {alg: [] for alg in algorithms}
sub_tot = {alg: 0 for alg in algorithms}
trial_times = {alg: [] for alg in algorithms}

for _ in range(M):
    s, r, so, times = run_simulation()
    for alg in algorithms:
        succ_all[alg].append(s[alg])
        reg_all[alg].append(r[alg])
        sub_tot[alg] += so[alg]
        trial_times[alg].append(times[alg])

avg_s = {alg: np.mean(succ_all[alg], axis=0) for alg in algorithms}
avg_r = {alg: np.mean(reg_all[alg], axis=0) for alg in algorithms}
avg_so = {alg: sub_tot[alg] / M for alg in algorithms}

# Print and save final metrics
print_and_save("=== Final Metrics ===")
for alg in algorithms:
    label = label_map[alg]
    print_and_save(f"{label:12s}: CumSuccess={avg_s[alg][-1]:.1f}, CumRegret={avg_r[alg][-1]:.1f}, SuboptPulls/Trial={avg_so[alg]:.1f}")

# Print and save average running times
print_and_save("\n=== Average Running Time per Trial (seconds) ===")
for alg in algorithms:
    avg_time = np.mean(trial_times[alg])
    print_and_save(f"{label_map[alg]:12s}: {avg_time:.6f}")

# Print per-step average time
print_and_save("\n=== Average Per-Step Time (microseconds) ===")
for alg in algorithms:
    avg_step_time = np.mean(trial_times[alg]) / T * 1e6
    print_and_save(f"{label_map[alg]:12s}: {avg_step_time:.2f}")

with open(os.path.join(results_folder, 'results.txt'), 'w') as f:
    f.write('\n'.join(output_lines))

# Plot cumulative regret
plt.figure(figsize=(12, 7))
for i, alg in enumerate(algorithms):
    plt.plot(avg_r[alg], label=label_map[alg], marker=markers[i], markersize=marker_size, markevery=markevery)
for t in reset_times:
    plt.axvline(t, color='gray', linestyle='--', alpha=0.7)
plt.title('Cumulative Regret Over Time', fontsize=14)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'cumulative_regret.png'), dpi=300)

# Plot cumulative reward
plt.figure(figsize=(12, 7))
for i, alg in enumerate(algorithms):
    plt.plot(avg_s[alg], label=label_map[alg], marker=markers[i], markersize=marker_size, markevery=markevery)
for t in reset_times:
    plt.axvline(t, color='gray', linestyle='--', alpha=0.7)
plt.title('Cumulative Reward Over Time', fontsize=14)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Cumulative Reward', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'cumulative_reward.png'), dpi=300)

# Statistical significance tests on final regret
print_and_save("\n=== Statistical Significance (Final Regret) ===")
final_regrets = {alg: np.array([r[-1] for r in reg_all[alg]]) for alg in algorithms}
for i in range(len(algorithms)):
    for j in range(i + 1, len(algorithms)):
        a1, a2 = algorithms[i], algorithms[j]
        pval = ttest_rel(final_regrets[a1], final_regrets[a2])[1]
        print_and_save(f"{label_map[a1]} vs {label_map[a2]}: p={pval:.4f}")

# Boxplot of final regret distribution
plt.figure(figsize=(12, 6))
plt.boxplot([final_regrets[alg] for alg in algorithms],
            tick_labels=[label_map[alg] for alg in algorithms], showmeans=True)
plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize=8)
plt.title('Distribution of Final Regret', fontsize=14)
plt.ylabel('Final Regret', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'final_regret_boxplot.png'), dpi=300)


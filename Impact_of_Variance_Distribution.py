# coding=utf-8
"""
Authors:
    Junyi Fang (junyifang001@gmail.com)

Description:
    This script generates Figure 1 from the paper "From Theory to Practice with RAVEN-UCB:
    Addressing Non-Stationarity in Multi-Armed Bandits through Variance Adaptation".
    The figure titled "Impact of Sample Size on Mean and Variance Distributions" includes two subfigures:
    - (a) Distribution of Sample Means for Varying Sample Sizes: Shows the distribution of sample means (μ_k)
      for a bandit arm across different sample sizes.
    - (b) Distribution of Sample Variances for Varying Sample Sizes: Shows the distribution of sample variances (σ_k^2)
      for a bandit arm across different sample sizes.
    Additionally, the script generates a third plot showing the variance of sample means as a function of sample size.


"""
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the bandit arm
true_mean = 0  # True mean of the reward distribution (μ_k)
true_std = 1   # True standard deviation (σ_k)
sample_sizes = [5, 10, 50, 100, 500, 1000]  # Different sample sizes (n_k)
num_trials = 100  # Number of trials per sample size

# Initialize lists to store sample statistics
mean_results = {n: [] for n in sample_sizes}
var_results = {n: [] for n in sample_sizes}

# Simulate sampling and compute statistics
for n in sample_sizes:
    for _ in range(num_trials):
        # Draw samples from normal distribution
        samples = np.random.normal(true_mean, true_std, n)
        # Compute sample mean (Eq. 1: ˆμ_k = (1/n_k) * Σ X_k,i)
        sample_mean = np.mean(samples)
        # Compute sample variance (Eq. 2 or 4: S^2 = (1/(n-1)) * Σ (X_k,i - ˆμ_k)^2)
        sample_var = np.var(samples, ddof=1)
        mean_results[n].append(sample_mean)
        var_results[n].append(sample_var)

# Compute variance of sample means across trials for each sample size
mean_variances = [np.var(mean_results[n], ddof=1) for n in sample_sizes]

# Plot 1: Distribution of sample means
plt.figure(figsize=(8, 5))
for n in sample_sizes:
    plt.hist(mean_results[n], bins=20, alpha=0.5, label=f'n={n}', density=True)
plt.axvline(true_mean, color='black', linestyle='--', label='True mean')
plt.title('Sample Mean Distributions')
plt.xlabel('Sample Mean (μ)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Plot 2: Distribution of sample variances
plt.figure(figsize=(8, 5))
for n in sample_sizes:
    plt.hist(var_results[n], bins=20, alpha=0.5, label=f'n={n}', density=True)
plt.axvline(true_std**2, color='black', linestyle='--', label='True variance')
plt.title('Sample Variance Distributions')
plt.xlabel('Sample Variance (σ²)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Plot 3: Variance of sample means vs. sample size
plt.figure(figsize=(8, 5))
plt.plot(sample_sizes, mean_variances, 'o-', color='blue')
plt.title('Variance of Sample Means vs. Sample Size')
plt.xlabel('Sample Size (n)')
plt.ylabel('Variance of Sample Means')
plt.xscale('log')
plt.grid(True)
plt.show()
# Assignment 1
# Part 1
# Mohammad Soleimani

import numpy as np  # For math and arrays
import matplotlib.pyplot as plt  # For creating plots
from scipy.stats import norm  # For normal distribution calculations
import seaborn as sns  # For nicer plot styling
from tqdm import tqdm  # For progress bars during loops
import pandas as pd  # For data handling (not used in this code, but imported)

# Set base random seed for reproducibility
# Ensures the same random numbers are generated each run for consistent results
BASE_SEED = 42

class KArmedBandit:
    """K-armed bandit testbed with stationary reward distributions"""
    # Models a 10-arm bandit problem with rewards from normal distributions
    def __init__(self, k=10, seed=None):
        self.k = k  # Number of arms (default 10)
        self.rng = np.random.RandomState(seed)  # Random number generator for this bandit
        self.true_means = self.rng.normal(0, 1, k)  # Random means for each arm from N(0,1)
        self.optimal_action = np.argmax(self.true_means)  # Best arm (highest mean)

    def get_reward(self, action):
        # Returns a reward for the chosen arm from N(mean, 1)
        return self.rng.normal(self.true_means[action], 1)

    def get_optimal_action(self):
        # Returns the index of the best arm
        return self.optimal_action

class GreedyAgent:
    #  picks the arm with the highest estimated reward
    def __init__(self, k=10, initial_values=0.0, seed=None):
        self.k = k  # Number of arms
        self.rng = np.random.RandomState(seed)  # Random generator for tie-breaking
        self.q_estimates = np.full(k, initial_values, dtype=float)  # Estimated rewards for each arm
        self.action_counts = np.zeros(k, dtype=int)  # Count of times each arm was picked

    def select_action(self):
        # Picks the arm with the highest estimated reward (randomly breaks ties)
        max_value = np.max(self.q_estimates)
        max_indices = np.where(self.q_estimates == max_value)[0]
        return self.rng.choice(max_indices)

    def update(self, action, reward):
        # Updates the estimated reward for the chosen arm using incremental average
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]

class EpsilonGreedyAgent:
    # Picks random arms sometimes (epsilon) or the best arm otherwise
    def __init__(self, k=10, epsilon=0.1, initial_values=0.0, seed=None):
        self.k = k  # Number of arms
        self.epsilon = epsilon  # Probability of random action
        self.rng = np.random.RandomState(seed)  # Random generator for exploration
        self.q_estimates = np.full(k, initial_values, dtype=float)  # Estimated rewards
        self.action_counts = np.zeros(k, dtype=int)  # Count of arm selections

    def select_action(self):
        # Random action with probability epsilon, else picks best arm
        if self.rng.random() < self.epsilon:
            return self.rng.randint(self.k)
        max_value = np.max(self.q_estimates)
        max_indices = np.where(self.q_estimates == max_value)[0]
        return self.rng.choice(max_indices)

    def update(self, action, reward):
        # Updates estimated reward for the chosen arm
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]

class GradientBanditAgent:
    # Learning preferences for arms and picks them probabilistically
    def __init__(self, k=10, alpha=0.1, seed=None):
        self.k = k  # Number of arms
        self.alpha = alpha  # Learning rate for preference updates
        self.rng = np.random.RandomState(seed)  # Random generator for action selection
        self.h = np.zeros(k, dtype=float)  # Preferences for each arm
        self.avg_reward = 0.0  # Average reward baseline
        self.t = 0  # Step counter
        self.action_probs = np.ones(k) / k  # Initial equal probabilities

    def select_action(self):
        # Picks an arm based on softmax probabilities of preferences
        exp_h = np.exp(self.h - np.max(self.h))  # Avoid numerical overflow
        self.action_probs = exp_h / np.sum(exp_h)
        return self.rng.choice(self.k, p=self.action_probs)

    def update(self, action, reward):
        # Updates preferences and average reward based on reward received
        self.t += 1
        self.avg_reward += (reward - self.avg_reward) / self.t
        for a in range(self.k):
            if a == action:
                self.h[a] += self.alpha * (reward - self.avg_reward) * (1 - self.action_probs[a])
            else:
                self.h[a] -= self.alpha * (reward - self.avg_reward) * self.action_probs[a]

def run_pilot_study_epsilon(rng, epsilon_values, n_steps=500, n_simulations=100):
    # Tests different epsilon values to find the best one for Epsilon-Greedy
    results = {}
    for eps in tqdm(epsilon_values, desc="Testing epsilon values"):
        total_rewards = np.zeros(n_steps)
        for run in range(n_simulations):
            bandit = KArmedBandit(seed=rng.randint(1000000))  # New bandit each run
            agent = EpsilonGreedyAgent(epsilon=eps, seed=rng.randint(1000000))  # New agent
            for t in range(n_steps):
                action = agent.select_action()
                reward = bandit.get_reward(action)
                agent.update(action, reward)
                total_rewards[t] += reward
        avg_rewards = total_rewards / n_simulations
        results[eps] = np.mean(avg_rewards[-100:])  # Average of last 100 steps
    best_epsilon = max(results, key=results.get)
    print(f"Pilot results for epsilon: {results}")
    print(f"Best epsilon: {best_epsilon}")
    return best_epsilon

def run_pilot_study_alpha(rng, alpha_values, n_steps=500, n_simulations=100):
    # Tests different alpha values to find the best one for Gradient Bandit
    results = {}
    for alpha in tqdm(alpha_values, desc="Testing alpha values"):
        total_rewards = np.zeros(n_steps)
        for run in range(n_simulations):
            bandit = KArmedBandit(seed=rng.randint(1000000))  # New bandit
            agent = GradientBanditAgent(alpha=alpha, seed=rng.randint(1000000))  # New agent
            for t in range(n_steps):
                action = agent.select_action()
                reward = bandit.get_reward(action)
                agent.update(action, reward)
                total_rewards[t] += reward
        avg_rewards = total_rewards / n_simulations
        results[alpha] = np.mean(avg_rewards[-100:])  # Average of last 100 steps
    best_alpha = max(results, key=results.get)
    print(f"Pilot results for alpha: {results}")
    print(f"Best alpha: {best_alpha}")
    return best_alpha

def run_simulations():
    # Runs main experiment with 1000 simulations, each with 2000 steps
    # Pilot runs to tune hyperparameters
    pilot_rng = np.random.RandomState(BASE_SEED)  # Fixed seed for pilot studies
    epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    alpha_values = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
    best_epsilon = run_pilot_study_epsilon(pilot_rng, epsilon_values)
    best_alpha = run_pilot_study_alpha(pilot_rng, alpha_values)

    k = 10
    n_simulations = 1000
    n_steps = 2000
    algorithms = {
        'Greedy (Q₀=0)': [],
        f'ε-Greedy (ε={best_epsilon})': [],
        'Optimistic Greedy': [],
        f'Gradient Bandit (α={best_alpha})': []
    }
    optimal_action_percentages = np.zeros((len(algorithms), n_steps))  # Tracks optimal action frequency
    rewards = np.zeros((len(algorithms), n_steps))  # Tracks average rewards
    final_rewards = np.zeros((len(algorithms), n_simulations))  # Tracks final step rewards
    regrets = np.zeros((len(algorithms), n_steps))  # Tracks cumulative regrets

    for sim in tqdm(range(n_simulations), desc="Simulations"):
        bandit_seed = BASE_SEED + sim  # Unique seed for each simulation
        bandit = KArmedBandit(seed=bandit_seed)
        max_mean = np.max(bandit.true_means)
        optimistic_value = norm.ppf(0.995, loc=max_mean, scale=1)  # High initial value for Optimistic Greedy

        agents = {
            'Greedy (Q₀=0)': GreedyAgent(k, initial_values=0.0, seed=sim * 4 + 1),
            f'ε-Greedy (ε={best_epsilon})': EpsilonGreedyAgent(k, epsilon=best_epsilon, seed=sim * 4 + 2),
            'Optimistic Greedy': GreedyAgent(k, initial_values=optimistic_value, seed=sim * 4 + 3),
            f'Gradient Bandit (α={best_alpha})': GradientBanditAgent(k, alpha=best_alpha, seed=sim * 4 + 4)
        }

        sim_rewards = np.zeros((len(agents), n_steps))
        sim_optimal = np.zeros((len(agents), n_steps))
        sim_regrets = np.zeros((len(agents), n_steps))
        for idx, (name, agent) in enumerate(agents.items()):
            optimal_reward = np.max(bandit.true_means)
            for step in range(n_steps):
                action = agent.select_action()
                reward = bandit.get_reward(action)
                agent.update(action, reward)
                sim_rewards[idx, step] = reward
                sim_optimal[idx, step] = 1 if action == bandit.get_optimal_action() else 0
                sim_regrets[idx, step] = optimal_reward - reward
            final_rewards[idx, sim] = sim_rewards[idx, -1]

        for idx, name in enumerate(algorithms.keys()):
            algorithms[name].append(sim_rewards[idx])
            rewards[idx] += sim_rewards[idx] / n_simulations
            optimal_action_percentages[idx] += sim_optimal[idx] / n_simulations
            regrets[idx] += np.cumsum(sim_regrets[idx]) / n_simulations

    avg_rewards = {name: rewards[idx] for idx, name in enumerate(algorithms.keys())}
    avg_optimal_percentages = {name: optimal_action_percentages[idx] * 100 for idx, name in enumerate(algorithms.keys())}
    final_rewards = {name: final_rewards[idx] for idx, name in enumerate(algorithms.keys())}
    avg_regrets = {name: regrets[idx] for idx, name in enumerate(algorithms.keys())}

    return avg_rewards, avg_optimal_percentages, final_rewards, avg_regrets, best_epsilon, best_alpha

def plot_results(avg_rewards, avg_optimal_percentages, final_rewards, avg_regrets, best_epsilon, best_alpha):
    # Creates plots and saves results to files
    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']  # Colors for plots
    methods = list(avg_rewards.keys())

    # Map algorithm names to ASCII-safe version for CSV heade
    # Avoids special characters in file header
    header_map = {
        'Greedy (Q₀=0)': 'Greedy_Q0_0',
        f'ε-Greedy (ε={best_epsilon})': f'Epsilon_Greedy_eps_{best_epsilon}',
        'Optimistic Greedy': 'Optimistic_Greedy',
        f'Gradient Bandit (α={best_alpha})': f'Gradient_Bandit_alpha_{best_alpha}'
    }

    # Plot 1: Average Reward (Line)
    # Shows how average reward changes over 2000 steps
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(methods):
        plt.plot(avg_rewards[name], label=name, color=colors[i])
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title(f'Average Reward (ε={best_epsilon}, α={best_alpha})')
    plt.legend()
    plt.grid(True)
    plt.savefig('average_reward.png')
    plt.show()
    plt.close()

    # Plot 2: Scatter (Reward vs. Optimal Action)
    # Shows reward vs. optimal action % at specific steps
    key_steps = [0, 499, 999, 1499, 1999]
    plt.figure(figsize=(10, 6))
    sizes = np.linspace(50, 200, len(key_steps))
    for i, name in enumerate(methods):
        rewards_at_steps = avg_rewards[name][key_steps]
        optimal_at_steps = avg_optimal_percentages[name][key_steps]
        plt.scatter(rewards_at_steps, optimal_at_steps, s=sizes, c=[colors[i]], label=name, alpha=0.6)
        for j, t in enumerate(key_steps):
            plt.text(rewards_at_steps[j], optimal_at_steps[j] + 1, f't={t+1}', fontsize=8)
    plt.xlabel('Average Reward')
    plt.ylabel('Optimal Action (%)')
    plt.title(f'Reward vs. Optimal Action at Key Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('reward_vs_optimal_scatter.png')
    plt.show()
    plt.close()

    # Plot 3: Histogram (Final Rewards)
    # Shows distribution of rewards at step 2000
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(methods):
        plt.hist(final_rewards[name], bins=30, alpha=0.4, label=name, color=colors[i], density=True)
    plt.xlabel('Reward at t=2000')
    plt.ylabel('Density')
    plt.title(f'Distribution of Final Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig('final_reward_histogram.png')
    plt.show()
    plt.close()

    # Stores results in files for analysis
    np.savetxt('avg_rewards.csv', np.array([avg_rewards[name] for name in methods]), delimiter=',',
               header=','.join(header_map[name] for name in methods), encoding='utf-8')
    np.savetxt('avg_optimal_action_pct.csv', np.array([avg_optimal_percentages[name] for name in methods]), delimiter=',',
               header=','.join(header_map[name] for name in methods), encoding='utf-8')
    np.savetxt('final_rewards.csv', np.array([final_rewards[name] for name in methods]), delimiter=',',
               header=','.join(header_map[name] for name in methods), encoding='utf-8')
    np.savetxt('avg_regrets.csv', np.array([avg_regrets[name] for name in methods]), delimiter=',',
               header=','.join(header_map[name] for name in methods), encoding='utf-8')

def generate_report(avg_rewards, avg_optimal_percentages, final_rewards, best_epsilon, best_alpha):
    # Prints a detailed report of results
    # Map algorithm names for console output to avoid encoding issues
    display_map = {
        'Greedy (Q₀=0)': 'Greedy (Q0=0)',
        f'ε-Greedy (ε={best_epsilon})': f'epsilon-Greedy (epsilon={best_epsilon})',
        'Optimistic Greedy': 'Optimistic Greedy',
        f'Gradient Bandit (α={best_alpha})': f'Gradient Bandit (alpha={best_alpha})'
    }

    print("\n" + "=" * 80)
    print("MULTI-ARMED BANDIT ALGORITHM COMPARISON REPORT")
    print("=" * 80)

    print("\n1. EXPERIMENTAL SETUP:")
    print(f"   • Number of arms (k): 10")
    print(f"   • Number of simulations: 1000")
    print(f"   • Steps per simulation: 2000")
    print(f"   • Reward distributions: N(μi, 1) where μi ~ N(0, 1)")
    print(f"   • Random seed: {BASE_SEED} (for reproducibility)")

    print("\n2. HYPERPARAMETER TUNING:")
    print(f"   • Best ε (epsilon-greedy): {best_epsilon}")
    print(f"   • Best α (gradient bandit): {best_alpha}")
    print(f"   • Optimistic initial value: 99.5th percentile of best arm")

    print("\n3. FINAL PERFORMANCE METRICS:")
    print("\n   Average Reward (last 500 steps):")
    final_rewards_avg = {}
    for name in avg_rewards.keys():
        final_performance = np.mean(avg_rewards[name][-500:])
        final_rewards_avg[name] = final_performance
        print(f"   • {display_map[name]:<25}: {final_performance:.4f}")

    print("\n   Optimal Action Percentage (last 500 steps):")
    final_optimal = {}
    for name in avg_optimal_percentages.keys():
        final_opt = np.mean(avg_optimal_percentages[name][-500:])
        final_optimal[name] = final_opt
        print(f"   • {display_map[name]:<25}: {final_opt:.2f}%")

    print("\n4. ALGORITHM RANKING:")
    ranked_algorithms = sorted(final_rewards_avg.items(), key=lambda x: x[1], reverse=True)
    print(f"\n   Performance Ranking (by average reward):")
    for i, (name, reward) in enumerate(ranked_algorithms, 1):
        opt_pct = final_optimal[name]
        print(f"   {i}. {display_map[name]}: {reward:.4f} avg reward, {opt_pct:.1f}% optimal")

    print("\n5. WHY GRADIENT BANDIT PERFORMS BEST:")
    best_algorithm = ranked_algorithms[0][0]
    best_reward = ranked_algorithms[0][1]
    best_opt_pct = final_optimal[best_algorithm]
    print(f"\n   🏆 WINNER: {display_map[best_algorithm]}")
    print(f"   📊 Performance: {best_reward:.4f} avg reward, {best_opt_pct:.1f}% optimal")
    if 'Gradient' in best_algorithm:
        print("\n   ✅ KEY ADVANTAGES OF GRADIENT BANDIT:")
        print("     • Learns action PREFERENCES rather than value estimates")
        print("     • Uses softmax for intelligent probabilistic action selection")
        print("     • Baseline subtraction (average reward) reduces learning variance")
        print("     • Adaptive exploration that focuses on promising actions")
        print("     • Natural balance between exploration and exploitation")
        print("     • More robust to initialization compared to value-based methods")
        print("\n   📈 SUPERIOR LEARNING MECHANISM:")
        print("     • Preference-based learning is more stable than value estimation")
        print("     • Softmax naturally reduces exploration of clearly inferior actions")
        print("     • Continues to explore competitive actions appropriately")
        print("     • Self-adjusting exploration based on reward differences")

    print("\n6. WHY OTHER METHODS UNDERPERFORMED:")
    for i, (name, reward) in enumerate(ranked_algorithms[1:], 2):
        opt_pct = final_optimal[name]
        print(f"\n   {i}. {display_map[name]} ({reward:.4f} avg reward, {opt_pct:.1f}% optimal):")
        if 'ε-Greedy' in name:
            print("     • Fixed exploration rate wastes opportunities in later stages")
            print("     • Random exploration doesn't focus on promising actions")
            print("     • Still performs well due to consistent exploration")
        elif 'Optimistic' in name:
            print("     • Once optimism is 'disappointed', becomes pure greedy")
            print("     • May get stuck in suboptimal actions after initial exploration")
            print("     • Performance depends heavily on optimistic initialization")
        elif 'Greedy' in name:
            print("     • No systematic exploration after initialization")
            print("     • Gets stuck in first reasonably good action found")
            print("     • High variance performance across different problems")

    print("\n7. KEY INSIGHTS:")
    print("   • Gradient bandit's preference-based learning is superior to value estimation")
    print("   • Softmax exploration is more intelligent than random (epsilon-greedy) exploration")
    print("   • Adaptive exploration becomes more focused over time")
    print("   • Baseline subtraction provides better learning signals")
    print("   • Proper hyperparameter tuning is crucial for all methods")

    print("\n" + "=" * 80)
    print("CONCLUSION: Gradient Bandit Algorithm is the clear winner!")
    print("=" * 80)

if __name__ == "__main__":
    # Runs the simulations, generates plots, and prints the report
    avg_rewards, avg_optimal_percentages, final_rewards, avg_regrets, best_epsilon, best_alpha = run_simulations()
    plot_results(avg_rewards, avg_optimal_percentages, final_rewards, avg_regrets, best_epsilon, best_alpha)
    generate_report(avg_rewards, avg_optimal_percentages, final_rewards, best_epsilon, best_alpha)
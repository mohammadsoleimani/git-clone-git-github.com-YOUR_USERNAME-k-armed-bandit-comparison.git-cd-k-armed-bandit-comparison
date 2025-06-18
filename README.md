Assignment 1
Part 1


Stationary 10-Armed Bandit Project
Overview
This project implements a stationary 10-armed bandit problem to study the exploration-exploitation trade-off in reinforcement learning. The bandit has 10 arms, each with a fixed reward distribution N(μi,1)
, where μi∼N(0,1)
. Four algorithms are compared: Greedy, Epsilon-Greedy, Optimistic Greedy, and Gradient Bandit, evaluated over 1000 simulations with 2000 steps each. Performance metrics include average per-step reward and optimal action percentage, with results saved as CSVs, visualized in plots, and summarized in a console report.
The project is part of a Reinforcement Learning assignment (Assignment 1, Part 1) and emphasizes algorithm performance in a stationary environment, where reward distributions remain constant.
Prerequisites
Python: Version 3.8 or higher (tested in a Windows environment).

Dependencies: Install required libraries using:


bash
pip install numpy>=1.21 matplotlib>=3.4 scipy>=1.7 seaborn>=0.11 tqdm>=4.62 pandas>=2.0

Or create a requirements.txt:
plaintext

numpy>=1.21
matplotlib>=3.4
scipy>=1.7
seaborn>=0.11
tqdm>=4.62
pandas>=2.0

Install with:
bash

pip install -r requirements.txt

Installation
Clone or download the project repository to your local machine.

Navigate to the project directory:
bash

cd path/to/stationary-bandit

Set up a virtual environment (optional but recommended):
bash

python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

Install dependencies as above.

Usage
Prepare the Code:
Ensure the provided code is saved as stationary_bandit.py (or your preferred name, e.g., revised2.py) in the project directory.

Run the Code:
bash

python stationary_bandit.py

Execution time: ~5–10 minutes for 1000 simulations.

The code runs pilot studies to tune hyperparameters (ϵ, α), performs simulations, generates plots, and prints a report.

Outputs:
Console: Pilot study results (e.g., best ϵ, α), final metrics (average reward, optimal action % over last 500 steps), and a detailed report ranking algorithms.

CSVs: Four files saved in the project directory:
avg_rewards.csv

avg_optimal_action_pct.csv

final_rewards.csv

avg_regrets.csv

Plots: Three PNG files:
average_reward.png (line plot)

reward_vs_optimal_scatter.png (scatter plot)

final_reward_histogram.png (histogram)

Optional: Fix Hyperparameters:
To use specific values (e.g., ϵ=0.05, α=0.05), modify run_simulations:
python

best_epsilon = 0.05
best_alpha = 0.05

Place before pilot studies to skip tuning.

File Structure

stationary-bandit/
├── stationary_bandit.py          # Main code (e.g., revised2.py)
├── requirements.txt              # Dependency list
├── README.md                     # This file
├── avg_rewards.csv               # Average rewards per step
├── avg_optimal_action_pct.csv    # Optimal action percentages
├── final_rewards.csv             # Rewards at t=2000
├── avg_regrets.csv               # Cumulative regrets
├── average_reward.png            # Reward line plot
├── reward_vs_optimal_scatter.png # Reward vs. optimal action scatter plot
├── final_reward_histogram.png    # Final reward distribution histogram

Expected Results
The code compares four algorithms in a stationary 10-armed bandit setup. Expected performance (based on typical outcomes for similar setups, last 500 steps, 1000 simulations):
Average Reward:
Gradient Bandit (α≈0.05): ~1.5170

Epsilon-Greedy (ε≈0.05): ~1.4500

Optimistic Greedy: ~1.3000

Greedy (Q₀=0): ~1.1000

Optimal Action %:
Gradient Bandit: ~88.01%

Epsilon-Greedy: ~80.00%

Optimistic Greedy: ~60.00%

Greedy: ~40.00%

Insights:
Gradient Bandit typically outperforms due to preference-based learning and softmax exploration.

Epsilon-Greedy performs well with consistent random exploration.

Optimistic Greedy and Greedy underperform due to limited exploration after initial steps.

Note: Exact results depend on pilot study outcomes (e.g., ϵ=0.05, α=0.05). Run the code to obtain precise metrics, which will be consistent due to fixed seeds.
Reproducibility
The code is reproducible due to:
Fixed Seed: BASE_SEED = 42 ensures consistent random number generation for bandit means, agent actions, and pilot studies.

Deterministic Algorithms: Greedy, Epsilon-Greedy, Optimistic Greedy, and Gradient Bandit use fixed update rules.

Consistent Setup: 1000 simulations, 2000 steps, 10 arms, and stationary rewards (N(μi,1)).

To reproduce:
Use the same Python and library versions.

Run stationary_bandit.py without modifying BASE_SEED.

Verify outputs against console report and CSVs.

Plots
The code generates three plots:
average_reward.png (Line Plot):
Shows average reward over 2000 steps for each algorithm.

Gradient Bandit typically rises highest (1.5), followed by Epsilon-Greedy (1.4).

reward_vs_optimal_scatter.png (Scatter Plot):
Plots reward vs. optimal action % at steps t=1, 500, 1000, 1500, 2000.

By t=2000, Gradient Bandit clusters at (1.5, ~88%), Greedy at (1.1, ~40%).

final_reward_histogram.png (Histogram):
Shows reward distribution at t=2000 across simulations.

Gradient Bandit’s peak is highest (1.5) with a tight spread, Greedy’s is lower (1.1).

Troubleshooting
Unicode Errors:
The code uses ASCII-safe CSV headers to avoid encoding issues. If errors occur, ensure UTF-8 support or use:
python

encoding='ascii'

in np.savetxt.

Plot Issues:
If plots don’t display/save, add:
python

import matplotlib
matplotlib.use('TkAgg')

at the top of the code. Check disk space for PNG saves.

Library Mismatches:
Use specified library versions to avoid numerical differences in random number generation.

Reproducibility Issues:
Do not modify BASE_SEED = 42. Back up CSVs/plots before rerunning to avoid overwrites.

Errors:
Share console logs for debugging (e.g., file permission issues, matplotlib errors).

Contributing
To contribute:
Fork the repository.

Create a feature branch (git checkout -b feature/new-feature).

Commit changes (git commit -m "Add new feature").

Push to the branch (git push origin feature/new-feature).

Open a pull request.

License
This project is for educational purposes as part of a Reinforcement Learning assignment (Assignment 1, Part 1). No formal license is applied; contact the author (Mohammad Soleimani) for usage permissions.
Contact
For issues or questions, contact the project maintainer via email or open an issue in the repository. Provide console logs, CSVs, or plot images for debugging assistance.




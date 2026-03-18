"""
Core bandit algorithms and experiment utilities
for adaptive LLM evaluation.
"""

import numpy as np
import math
from scipy import stats


# --- Constants ---

GLOBAL_SEED  = 42
TOTAL_PLAYS  = 20_000   # plays per run
N_TRIALS     = 30       # independent seeds for mean ± SE


# --- Problem definition ---

true_probabilities = [0.45, 0.55]   # arms = prompt categories
K                  = len(true_probabilities)
best_arm_index     = int(np.argmax(true_probabilities))
best_arm_prob      = float(true_probabilities[best_arm_index])


# --- Algorithm implementations ---

def ucb1_select_arm(n_total, arms_data, rng=None):
    """
    UCB1 arm selection.
    If any arm has 0 plays, select it first (initialization).
    """
    for i, a in enumerate(arms_data):
        if a['plays'] == 0:
            return int(i)
    ucb_vals = []
    for a in arms_data:
        avg   = a['rewards'] / a['plays']
        bonus = math.sqrt((2.0 * math.log(max(1, n_total))) / a['plays'])
        ucb_vals.append(avg + bonus)
    return int(np.argmax(ucb_vals))


def epsilon_greedy_select_arm(n_total, arms_data, rng, c=0.1, d=0.1):
    """
    Decaying Epsilon-Greedy: epsilon = min(1, (c * K) / (d^2 * n))
    rng: numpy Generator (required)
    """
    n_arms = len(arms_data)
    epsilon = min(1.0, (c * n_arms) / ((d ** 2) * max(1, n_total)))
    if rng.random() < epsilon:
        return int(rng.integers(0, n_arms))
    avg_rewards = [
        (a['rewards'] / a['plays']) if a['plays'] > 0 else 0.0
        for a in arms_data
    ]
    return int(np.argmax(avg_rewards))


# --- Single trial ---

def run_one_run(
    algorithm_func,
    total_plays=TOTAL_PLAYS,
    seed=None,
    algo_kwargs=None,
    simulated_oracle=None
):
    """
    Run one independent trial.

    algorithm_func signature:
        with rng:    algorithm_func(n_total, arms_data, rng=rng, **algo_kwargs)
        without rng: algorithm_func(n_total, arms_data, **algo_kwargs)

    simulated_oracle (optional):
        function(arm_index, rng) -> (reward: int, text: str)
        If None, uses Bernoulli samples from true_probabilities.

    Returns dict:
        cum_regret      : np.ndarray (total_plays,)
        pulls_history   : np.ndarray (total_plays, K)
        final_pulls     : np.ndarray (K,)
        rewards_history : np.ndarray (total_plays,)

    Note: uses empirical (pseudo) regret based on observed reward,
    not true expected regret. Sufficient for comparative evaluation
    as both algorithms are measured under identical conditions.
    """
    algo_kwargs = {} if algo_kwargs is None else dict(algo_kwargs)
    rng         = np.random.default_rng(seed)
    arms_data   = [{'plays': 0, 'rewards': 0.0} for _ in range(K)]

    cumulative_regret   = 0.0
    cum_regret_history  = np.zeros(total_plays, dtype=float)
    pulls_history       = np.zeros((total_plays, K), dtype=int)
    rewards_history     = np.zeros(total_plays, dtype=int)

    for n in range(1, total_plays + 1):
        # Force each arm to be played once during initialisation
        if n <= K:
            chosen = n - 1
        else:
            try:
                chosen = algorithm_func(n, arms_data, rng=rng, **algo_kwargs)
            except TypeError:
                chosen = algorithm_func(n, arms_data, **algo_kwargs)

        # Reward source
        if simulated_oracle is not None:
            reward, _ = simulated_oracle(chosen, rng)
        else:
            reward = 1 if rng.random() < true_probabilities[chosen] else 0

        arms_data[chosen]['plays']   += 1
        arms_data[chosen]['rewards'] += reward

        cumulative_regret          += (best_arm_prob - reward)
        cum_regret_history[n - 1]   = cumulative_regret
        pulls_history[n - 1, :]     = [a['plays'] for a in arms_data]
        rewards_history[n - 1]      = reward

    return {
        'cum_regret'     : cum_regret_history,
        'pulls_history'  : pulls_history,
        'final_pulls'    : np.array([a['plays'] for a in arms_data]),
        'rewards_history': rewards_history,
    }


# --- Multiple independent trials ---

def average_runs(
    algorithm_func,
    n_runs=N_TRIALS,
    total_plays=TOTAL_PLAYS,
    seed_base=GLOBAL_SEED,
    algo_kwargs=None,
    simulated_oracle=None
):
    """
    Run n_runs independent trials and aggregate results.

    Returns dict:
        mean_regret      : np.ndarray (total_plays,)   — mean cumulative regret
        se_regret        : np.ndarray (total_plays,)   — standard error
        final_regrets    : np.ndarray (n_runs,)        — final regret per trial
        mean_final_pulls : np.ndarray (K,)             — mean pulls per arm
        all_regrets      : np.ndarray (n_runs, total_plays)
    """
    algo_kwargs = {} if algo_kwargs is None else dict(algo_kwargs)

    regs         = np.zeros((n_runs, total_plays))
    final_regrets = np.zeros(n_runs)
    mean_pulls   = np.zeros((n_runs, K), dtype=int)

    for i in range(n_runs):
        seed = seed_base + i * 100
        out  = run_one_run(
            algorithm_func,
            total_plays=total_plays,
            seed=seed,
            algo_kwargs=algo_kwargs,
            simulated_oracle=simulated_oracle
        )
        regs[i]          = out['cum_regret']
        final_regrets[i] = out['cum_regret'][-1]
        mean_pulls[i]    = out['final_pulls']

    return {
        'mean_regret'     : regs.mean(axis=0),
        'se_regret'       : regs.std(axis=0, ddof=1) / math.sqrt(n_runs),
        'final_regrets'   : final_regrets,
        'mean_final_pulls': mean_pulls.mean(axis=0),
        'all_regrets'     : regs,
    }


# --- Statistical test ---

def paired_test(arr1, arr2):
    """
    Paired t-test on two final-regret arrays.
    Returns (t-statistic, p-value).
    """
    tstat, pval = stats.ttest_rel(arr1, arr2)
    return tstat, pval
import numpy as np
import os
import csv
from datetime import datetime

from analysis import (
    ucb1_select_arm,
    epsilon_greedy_select_arm,
    average_runs,
    paired_test,
    true_probabilities
)

CONFIG = {
    "total_plays": 20000,
    "n_trials": 30,
    "seed": 42,
    "epsilon": {"c": 0.1}
}

# derive gap parameter (must match experiment setup)
d_param = max(true_probabilities) - min(true_probabilities)


def main():
    print("Running experiment...")

    # ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # run experiments
    ucb_res = average_runs(
        ucb1_select_arm,
        n_runs=CONFIG["n_trials"],
        total_plays=CONFIG["total_plays"],
        seed_base=CONFIG["seed"]
    )

    eps_res = average_runs(
        epsilon_greedy_select_arm,
        n_runs=CONFIG["n_trials"],
        total_plays=CONFIG["total_plays"],
        seed_base=CONFIG["seed"] + 9999,
        algo_kwargs={
            "c": CONFIG["epsilon"]["c"],
            "d": d_param
        }
    )

    # statistical test
    tstat, pval = paired_test(
        ucb_res["final_regrets"],
        eps_res["final_regrets"]
    )

    # console output
    print("Done.")
    print("\n=== Results ===")

    ucb_mean = float(np.mean(ucb_res["final_regrets"]))
    eps_mean = float(np.mean(eps_res["final_regrets"]))

    print("Final regret (UCB1):", ucb_mean)
    print("Final regret (Epsilon-Greedy):", eps_mean)
    print("p-value (paired t-test):", pval)

    # save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join("results", f"summary_{timestamp}.csv")

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "mean_final_regret"])
        writer.writerow(["UCB1", ucb_mean])
        writer.writerow(["Epsilon-Greedy", eps_mean])

    # optional: save raw arrays (useful for analysis)
    np.save(os.path.join("results", f"ucb_regrets_{timestamp}.npy"),
            ucb_res["final_regrets"])

    np.save(os.path.join("results", f"eps_regrets_{timestamp}.npy"),
            eps_res["final_regrets"])

    print(f"\nSaved results to {summary_path}")


if __name__ == "__main__":
    main()
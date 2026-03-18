import numpy as np
import os
import matplotlib.pyplot as plt


def find_latest_file(prefix, folder="results"):
    files = [f for f in os.listdir(folder) if f.startswith(prefix)]
    if not files:
        raise FileNotFoundError(f"No files found with prefix '{prefix}' in {folder}")
    files.sort()
    return os.path.join(folder, files[-1])


def main():
    print("Loading latest results...")

    # find latest saved arrays
    ucb_path = find_latest_file("ucb_regrets")
    eps_path = find_latest_file("eps_regrets")

    ucb = np.load(ucb_path)
    eps = np.load(eps_path)

    print(f"Loaded:\n  {ucb_path}\n  {eps_path}")

    # simple comparison plot (final regret distribution)
    plt.figure(figsize=(8, 5))

    plt.hist(ucb, bins=15, alpha=0.6, label="UCB1")
    plt.hist(eps, bins=15, alpha=0.6, label="Epsilon-Greedy (decaying)")

    plt.xlabel("Final Regret")
    plt.ylabel("Frequency")
    plt.title("Distribution of Final Regret Across Runs")
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "regret_distribution.png")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
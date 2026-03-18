# Adaptive LLM Evaluation via Multi-Armed Bandits

## Overview

Evaluating large language models is inefficient.

Most evaluation pipelines rely on static test sets or uniform sampling, allocating equal effort across all prompt categories. This leads to wasted evaluation budget, slow failure discovery, and poor coverage of model weaknesses.

This project treats LLM evaluation as a **resource allocation problem**.

Instead of uniformly sampling prompts, it dynamically prioritizes high-risk regions using multi-armed bandit strategies, improving failure discovery efficiency under fixed evaluation budgets.

---

## Approach

We model evaluation as a multi-armed bandit problem:

- Each prompt category → an arm  
- Each evaluation → a pull  
- Reward → failure detection (1 if failure, 0 otherwise)

**Objective:**

> Maximize failure discovery efficiency while minimizing evaluation cost.

---

## Algorithms

- **UCB1 (Upper Confidence Bound)**
- **Epsilon-Greedy (decaying exploration)**

Both are evaluated under identical stochastic conditions.

---

## Experimental Setup

- Arms: 2 (failure probabilities: 0.45, 0.55)  
- Horizon: 20,000 plays per run  
- Trials: 30 independent runs  
- Reproducibility: fixed and incremented random seeds  

### Metrics

- Cumulative regret (mean ± standard error)  
- Fraction of optimal arm selection  
- Statistical validation via paired t-test  

---

## Results

Epsilon-Greedy (decaying) significantly outperforms UCB1.

- UCB1 final regret: ~104.4  
- Epsilon-Greedy final regret: ~12.4  
- p-value (paired t-test): ~0.00116  

This indicates a consistent and repeatable performance advantage of decaying exploration in this setting.

---

## Key Insight

Adaptive exploration concentrates evaluation effort on failure-prone regions, improving sample efficiency relative to uniform or confidence-bound strategies.

This reframes LLM evaluation as a resource allocation problem, where evaluation budget is dynamically directed toward high-risk regions instead of uniformly distributed.

---

## Repository Structure

```text
llm-eval-bandit/
│
├── analysis.py              # Core algorithms and experiment logic
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── analysis.ipynb       # Exploratory analysis and visualization
│
├── scripts/
│   ├── run_experiment.py    # CLI entrypoint (runs experiments)
│   └── plot_results.py      # Generates plots from saved results
│
└── results/                 # Output artifacts (generated)
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run experiment

```bash
python -m scripts.run_experiment
```

**This will generate:**
* **Summary CSV** (mean regret)
* **Raw result arrays** (`.npy`)

### 3. Generate plots

```bash
python -m scripts.plot_results
```

**This produces:**
* **Distribution of final regret** across runs (histogram)

---

## Outputs

> **Note:** Results are not committed to the repository and are generated via scripts for reproducibility.

Results are stored in:
`results/`

**Artifacts include:**
* `.npy` — Raw per-run results (for reproducibility and analysis)
* `.csv` — Summary metrics
* `.png` — Visualizations

*Note: Raw outputs are preserved to enable recomputation of statistics and flexible downstream analysis.*

---

## Limitations

* **Simplified environment:** Restricted to 2 arms.
* **Static rewards:** Fixed reward distributions.
* **No LLM integration:** Lacks real-time LLM API calls.
* **Static prompts:** No dynamic prompt generation logic.

This implementation serves as a controlled baseline for evaluating adaptive sampling strategies.

---

## Future Work

* **API Integration:** Connect with live LLM providers.
* **Multi-category evaluation:** Scale to $K > 2$.
* **Contextual Bandits:** Implement feature-based decision making.
* **Automated Scoring:** Integrated failure scoring pipelines.

---

## Why This Matters

Evaluation cost is a primary bottleneck in production LLM systems. This work demonstrates that **adaptive allocation strategies** can significantly improve failure discovery efficiency without increasing total evaluation volume. This is directly applicable to:

* **Model Auditing:** Identifying edge cases efficiently.
* **Safety Evaluation:** Stress-testing guardrails.
* **Continuous Eval:** Reducing overhead in CI/CD pipelines.

---

## Summary

This project provides a minimal, reproducible evaluation system that:
* Models evaluation as a **multi-armed bandit problem**.
* **Validates results** through statistical rigor.
* **Decouples** computation, execution, and visualization.

It functions as a clean evaluation primitive designed for extension into production-grade workflows.

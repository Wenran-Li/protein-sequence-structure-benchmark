from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

B = Path("/scratch/f008j37/sequence_structure/structure_prediction")
OUT = B / "evaluation_results"

df = pd.read_csv(OUT / "four_model_paired_complete.csv")

models = ["AlphaFold3", "Boltz-2", "ESMFold", "Chai-1"]
metrics = ["tm_score", "lddt_ca"]

rng = np.random.default_rng(42)
results = []

for metric in metrics:
    wide = df.pivot(
        index="target_id",
        columns="model",
        values=metric,
    )[models].dropna()

    metric_results = []

    for a, b in combinations(models, 2):
        x = wide[a].to_numpy()
        y = wide[b].to_numpy()

        diff = x - y

        n = len(diff)
        indices = rng.integers(0, n, size=(10000, n))
        bootstrap = diff[indices].mean(axis=1)

        ci_low, ci_high = np.quantile(
            bootstrap, [0.025, 0.975]
        )

        p = wilcoxon(
            diff,
            alternative="two-sided",
            zero_method="wilcox",
        ).pvalue if np.any(diff != 0) else 1.0

        metric_results.append({
            "metric": metric,
            "model_a": a,
            "model_b": b,
            "n": n,
            "mean_difference_a_minus_b": diff.mean(),
            "median_difference_a_minus_b": np.median(diff),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "wins_a": int((diff > 0).sum()),
            "wins_b": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()),
            "p_raw": p,
        })

    # Holm step-down correction for six pairwise tests
    pvalues = np.array(
        [r["p_raw"] for r in metric_results],
        dtype=float
    )

    order = np.argsort(pvalues)
    m = len(pvalues)

    adjusted_sorted = np.maximum.accumulate(
        (m - np.arange(m)) * pvalues[order]
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    for r, p_adj in zip(metric_results, adjusted):
        r["p_holm"] = p_adj

    results.extend(metric_results)

out = pd.DataFrame(results)

path = OUT / "four_model_paired_statistics.csv"
out.to_csv(path, index=False)

print(out.round(6).to_string(index=False))
print("\nSaved:", path)

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from scipy.stats import spearmanr

ROOT = Path("/scratch/f008j37/sequence_structure")
EVAL = ROOT / "inverse_folding/evaluation"
FIG = ROOT / "inverse_folding/figures"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = {
    "ESM-IF1": "esm_if1_cath_sequences.csv",
    "PiFold": "pifold_cath_sequences.csv",
    "GVP-GNN": "gvp_cath_sequences.csv",
    "ProteinMPNN": "proteinmpnn_cath_sequences.csv",
}

EXPECTED_TARGETS = 883
N_BOOT = 10000
RNG = np.random.default_rng(42)

mapping = pd.read_csv(EVAL / "cath_mpnn_mapping.csv")
expected_ids = set(mapping["target_id"].astype(str))

assert len(mapping) == EXPECTED_TARGETS
assert len(expected_ids) == EXPECTED_TARGETS


def get_recovery(df):
    """Use stored recovery or calculate it from sequences."""
    if "recovery" in df.columns:
        return pd.to_numeric(df["recovery"], errors="raise")

    native_candidates = ["native_sequence", "native_seq"]
    design_candidates = [
        "designed_sequence", "sampled_sequence",
        "generated_sequence", "sequence"
    ]

    native_col = next(
        (c for c in native_candidates if c in df.columns), None
    )
    design_col = next(
        (c for c in design_candidates if c in df.columns), None
    )

    if native_col is None or design_col is None:
        raise ValueError(
            "Cannot find recovery or sequence columns. "
            f"Available columns: {list(df.columns)}"
        )

    def calculate(row):
        native = str(row[native_col])
        design = str(row[design_col])

        if len(native) != len(design) or len(native) == 0:
            raise ValueError("Sequence length mismatch")

        return sum(
            a == b for a, b in zip(native, design)
        ) / len(native)

    return df.apply(calculate, axis=1)


target_scores = {}

for model, filename in MODELS.items():
    path = EVAL / filename
    df = pd.read_csv(path)

    if "target_id" not in df.columns:
        raise ValueError(
            f"{model}: target_id column missing. "
            f"Columns: {list(df.columns)}"
        )

    df["target_id"] = df["target_id"].astype(str)
    df["recovery_checked"] = get_recovery(df)

    assert np.isfinite(df["recovery_checked"]).all()
    assert df["recovery_checked"].between(0, 1).all()

    ids = set(df["target_id"])
    if ids != expected_ids:
        raise ValueError(
            f"{model}: missing={len(expected_ids - ids)}, "
            f"extra={len(ids - expected_ids)}"
        )

    counts = df.groupby("target_id").size()
    expected_samples = 1 if model == "PiFold" else 5

    if not (counts == expected_samples).all():
        raise ValueError(
            f"{model}: expected {expected_samples} "
            "sequences per target; observed counts: "
            f"{counts.value_counts().to_dict()}"
        )

    if "sample_id" in df.columns:
        if df.duplicated(["target_id", "sample_id"]).any():
            raise ValueError(f"{model}: duplicate sample IDs")

    target_scores[model] = (
        df.groupby("target_id")["recovery_checked"].mean()
    )

    print(
        f"{model}: {len(df)} sequences, "
        f"{len(target_scores[model])} targets",
        flush=True
    )

# Paired target-level matrix
matrix = pd.DataFrame(target_scores)
matrix = matrix.reindex(mapping["target_id"].astype(str))

assert matrix.shape == (EXPECTED_TARGETS, 4)
assert not matrix.isna().any().any()

matrix.index.name = "target_id"
matrix.to_csv(EVAL / "inverse_folding_target_matrix.csv")


def bootstrap_mean_ci(values):
    values = np.asarray(values, dtype=float)
    n = len(values)

    estimates = np.empty(N_BOOT)

    for b in range(N_BOOT):
        indices = RNG.integers(0, n, size=n)
        estimates[b] = values[indices].mean()

    return np.quantile(estimates, [0.025, 0.975])


# Overall statistics
overall = []

for model in MODELS:
    values = matrix[model].to_numpy()
    ci = bootstrap_mean_ci(values)

    overall.append({
        "model": model,
        "n_targets": len(values),
        "mean": values.mean(),
        "sd": values.std(ddof=1),
        "median": np.median(values),
        "q25": np.quantile(values, 0.25),
        "q75": np.quantile(values, 0.75),
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    })

overall_df = pd.DataFrame(overall)
overall_df.to_csv(
    EVAL / "inverse_folding_overall_stats.csv",
    index=False
)

print("\n========== OVERALL RECOVERY ==========")
print(overall_df.to_string(index=False, float_format="%.5f"))


# Paired comparisons
paired = []

for model_a, model_b in combinations(MODELS, 2):
    a = matrix[model_a].to_numpy()
    b = matrix[model_b].to_numpy()

    diff = a - b
    ci = bootstrap_mean_ci(diff)

    stat, p = wilcoxon(
        diff,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto"
    )

    paired.append({
        "model_a": model_a,
        "model_b": model_b,
        "mean_difference": diff.mean(),
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "a_better": int((diff > 1e-12).sum()),
        "b_better": int((diff < -1e-12).sum()),
        "ties": int((np.abs(diff) <= 1e-12).sum()),
        "p_raw": p,
    })

paired_df = pd.DataFrame(paired)

# Holm correction
pvals = paired_df["p_raw"].to_numpy()
order = np.argsort(pvals)
m = len(pvals)

adjusted = np.empty(m)
running_max = 0.0

for rank, idx in enumerate(order):
    candidate = (m - rank) * pvals[idx]
    running_max = max(running_max, candidate)
    adjusted[idx] = min(running_max, 1.0)

paired_df["p_holm"] = adjusted

paired_df.to_csv(
    EVAL / "inverse_folding_paired_stats.csv",
    index=False
)

print("\n========== PAIRED COMPARISONS ==========")
print(paired_df.to_string(index=False, float_format="%.6g"))


# Spearman correlation with length
lengths = mapping.set_index("target_id")["length"]
lengths.index = lengths.index.astype(str)
lengths = lengths.reindex(matrix.index)

print("\n========== LENGTH CORRELATIONS ==========")

for model in MODELS:
    rho, p = spearmanr(lengths, matrix[model])
    print(f"{model}: rho={rho:.4f}, p={p:.4g}")


# Figure 3: violin plots + boxplots
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

labels = list(MODELS)
values = [matrix[m].to_numpy() * 100 for m in labels]

fig, ax = plt.subplots(figsize=(7.4, 4.4))

positions = np.arange(1, len(labels) + 1)

violins = ax.violinplot(
    values,
    positions=positions,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)

for body in violins["bodies"]:
    body.set_alpha(0.45)
    body.set_edgecolor("black")
    body.set_linewidth(0.5)

ax.boxplot(
    values,
    positions=positions,
    widths=0.18,
    showfliers=False,
    patch_artist=False,
    medianprops={"linewidth": 1.5},
)

means = [v.mean() for v in values]

ax.scatter(
    positions,
    means,
    marker="D",
    s=28,
    color="black",
    zorder=5,
    label="Mean",
)

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel("Native sequence recovery (%)")
ax.set_ylim(0, 100)
ax.set_xlim(0.5, len(labels) + 0.5)
ax.grid(axis="y", alpha=0.18)
ax.legend(frameon=False, loc="upper right")

fig.tight_layout()

fig.savefig(
    FIG / "inverse_folding_recovery.pdf",
    bbox_inches="tight"
)

fig.savefig(
    FIG / "inverse_folding_recovery.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\n========== OUTPUTS ==========")
print("Figure:", FIG / "inverse_folding_recovery.pdf")
print("Overall stats:", EVAL / "inverse_folding_overall_stats.csv")
print("Paired stats:", EVAL / "inverse_folding_paired_stats.csv")
print("ANALYSIS COMPLETED")

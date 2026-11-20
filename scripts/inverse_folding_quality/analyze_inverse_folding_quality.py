from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

ROOT = Path("/scratch/f008j37/sequence_structure")
EVAL = ROOT / "inverse_folding/evaluation"
OUT = EVAL / "quality_analysis"
OUT.mkdir(parents=True, exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy scale
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5,
    "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5,
    "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8,
    "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

FILES = {
    "ProteinMPNN": (
        "proteinmpnn_cath_sequences.csv",
        "sequence",
        "sequence_recovery",
        5,
    ),
    "ESM-IF1": (
        "esm_if1_cath_sequences.csv",
        "sequence",
        "sequence_recovery",
        5,
    ),
    "GVP-GNN": (
        "gvp_cath_sequences.csv",
        "designed_sequence",
        "recovery",
        5,
    ),
    "PiFold": (
        "pifold_cath_sequences.csv",
        "generated_sequence",
        "sequence_recovery",
        1,
    ),
}

EXPECTED = pd.read_csv(
    EVAL / "inverse_folding_target_matrix.csv"
)

assert len(EXPECTED) == 883
assert EXPECTED.target_id.is_unique

TARGETS = set(EXPECTED.target_id)

def composition(seq):
    counts = np.array(
        [seq.count(a) for a in AA],
        dtype=float
    )
    return counts / len(seq)

def descriptors(seq):
    length = len(seq)
    return {
        "hydropathy": sum(KD[a] for a in seq) / length,
        "charged_fraction": sum(
            seq.count(a) for a in "DEKR"
        ) / length,
        "charge_proxy": (
            seq.count("K") + seq.count("R")
            - seq.count("D") - seq.count("E")
        ) / length,
    }

def bootstrap_mean_ci(values, seed=20261120, n_boot=10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if not len(values):
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        boot[i] = values[idx].mean()

    lo, hi = np.quantile(boot, [0.025, 0.975])
    return values.mean(), lo, hi

# --------------------------------------------------
# 1. Normalize and validate sequence records
# --------------------------------------------------

frames = []

for model, (filename, seq_col, recovery_col, k) in FILES.items():
    raw = pd.read_csv(EVAL / filename)

    required = {
        "target_id", "sample_id",
        "native_sequence", seq_col, recovery_col
    }

    assert required <= set(raw.columns), (
        model, required - set(raw.columns)
    )

    d = pd.DataFrame({
        "target_id": raw["target_id"].astype(str),
        "model": model,
        "sample_id": raw["sample_id"].astype(str),
        "native_sequence": raw["native_sequence"].astype(str).str.upper(),
        "generated_sequence": raw[seq_col].astype(str).str.upper(),
        "reported_recovery": pd.to_numeric(
            raw[recovery_col], errors="raise"
        ),
    })

    assert len(d) == 883 * k, (
        model, len(d)
    )

    assert set(d.target_id) == TARGETS, model

    assert not d.duplicated([
        "target_id", "sample_id"
    ]).any(), model

    counts = d.groupby("target_id").size()
    assert counts.eq(k).all(), (
        model, counts.value_counts().to_dict()
    )

    for row in d.itertuples(index=False):
        native = row.native_sequence
        generated = row.generated_sequence

        assert len(native) == len(generated), (
            model, row.target_id, row.sample_id
        )

        assert len(native) > 0

        assert set(native) <= set(AA), (
            model, row.target_id, "native nonstandard AA"
        )

        assert set(generated) <= set(AA), (
            model, row.target_id, "generated nonstandard AA"
        )

    frames.append(d)

df = pd.concat(frames, ignore_index=True)

assert len(df) == 14128

# All models must have identical native sequences per target.
native_counts = (
    df.groupby("target_id")["native_sequence"]
    .nunique()
)

assert native_counts.eq(1).all(), (
    "Different native sequences across models",
    native_counts[native_counts != 1].to_dict()
)

# Independent recovery verification.
df["calculated_recovery"] = [
    sum(a == b for a, b in zip(n, g)) / len(n)
    for n, g in zip(
        df.native_sequence,
        df.generated_sequence
    )
]

max_recovery_error = (
    df["calculated_recovery"]
    - df["reported_recovery"]
).abs().max()

assert max_recovery_error < 1e-8, (
    "Recovery inconsistency", max_recovery_error
)

print("===== INPUT VALIDATION =====")
print("Targets:", df.target_id.nunique())
print("Total sequences:", len(df))
print("Models:", df.model.nunique())
print("Maximum recovery discrepancy:", max_recovery_error)
print("Sequence lengths: VALID")
print("Native sequence consistency: VALID")

# --------------------------------------------------
# 2. Per-sequence composition and descriptors
# --------------------------------------------------

sample_rows = []
composition_rows = []

for row in df.itertuples(index=False):
    native = row.native_sequence
    generated = row.generated_sequence

    pn = composition(native)
    pg = composition(generated)

    nd = descriptors(native)
    gd = descriptors(generated)

    tv = 0.5 * np.abs(pg - pn).sum()

    sample_rows.append({
        "target_id": row.target_id,
        "model": row.model,
        "sample_id": row.sample_id,
        "length": len(native),
        "recovery": row.calculated_recovery,
        "composition_tv": tv,
        "native_hydropathy": nd["hydropathy"],
        "generated_hydropathy": gd["hydropathy"],
        "hydropathy_delta": (
            gd["hydropathy"] - nd["hydropathy"]
        ),
        "native_charged_fraction": nd["charged_fraction"],
        "generated_charged_fraction": gd["charged_fraction"],
        "charged_fraction_delta": (
            gd["charged_fraction"] - nd["charged_fraction"]
        ),
        "native_charge_proxy": nd["charge_proxy"],
        "generated_charge_proxy": gd["charge_proxy"],
        "charge_proxy_delta": (
            gd["charge_proxy"] - nd["charge_proxy"]
        ),
    })

    for i, aa in enumerate(AA):
        composition_rows.append({
            "target_id": row.target_id,
            "model": row.model,
            "sample_id": row.sample_id,
            "amino_acid": aa,
            "native_fraction": pn[i],
            "generated_fraction": pg[i],
        })

samples = pd.DataFrame(sample_rows)
compositions = pd.DataFrame(composition_rows)

samples.to_csv(
    OUT / "sample_quality_metrics.csv",
    index=False
)

# --------------------------------------------------
# 3. Within-target sequence diversity
# --------------------------------------------------

target_rows = []

metric_cols = [
    "recovery",
    "composition_tv",
    "native_hydropathy",
    "generated_hydropathy",
    "hydropathy_delta",
    "native_charged_fraction",
    "generated_charged_fraction",
    "charged_fraction_delta",
    "native_charge_proxy",
    "generated_charge_proxy",
    "charge_proxy_delta",
]

for (target, model), d in df.groupby(
    ["target_id", "model"], sort=True
):
    seqs = d.generated_sequence.tolist()
    k = len(seqs)

    if k > 1:
        pairwise = [
            sum(a != b for a, b in zip(s1, s2)) / len(s1)
            for s1, s2 in combinations(seqs, 2)
        ]
        unique_fraction = len(set(seqs)) / k
        pairwise_diversity = float(np.mean(pairwise))
    else:
        # Not defined for single deterministic output.
        unique_fraction = np.nan
        pairwise_diversity = np.nan

    row = {
        "target_id": target,
        "model": model,
        "n_samples": k,
        "unique_fraction": unique_fraction,
        "pairwise_diversity": pairwise_diversity,
    }

    sub = samples[
        (samples.target_id == target)
        & (samples.model == model)
    ]

    assert len(sub) == k

    for col in metric_cols:
        row[col] = sub[col].mean()

    target_rows.append(row)

targets = pd.DataFrame(target_rows)

assert len(targets) == 883 * 4

targets.to_csv(
    OUT / "target_quality_metrics.csv",
    index=False
)

# --------------------------------------------------
# 4. Target-balanced amino-acid composition
# --------------------------------------------------

per_target_comp = (
    compositions.groupby(
        ["target_id", "model", "amino_acid"],
        as_index=False
    )
    .agg(
        native_fraction=("native_fraction", "mean"),
        generated_fraction=("generated_fraction", "mean"),
    )
)

profile = (
    per_target_comp.groupby(
        ["model", "amino_acid"],
        as_index=False
    )
    .agg(
        native_fraction=("native_fraction", "mean"),
        generated_fraction=("generated_fraction", "mean")
    )
)

profile["composition_delta"] = (
    profile.generated_fraction
    - profile.native_fraction
)

profile.to_csv(
    OUT / "amino_acid_composition_profile.csv",
    index=False
)

# --------------------------------------------------
# 5. Model summary and bootstrap intervals
# --------------------------------------------------

SUMMARY_METRICS = [
    "recovery",
    "composition_tv",
    "hydropathy_delta",
    "charged_fraction_delta",
    "charge_proxy_delta",
    "unique_fraction",
    "pairwise_diversity",
]

summary_rows = []

for model, d in targets.groupby("model"):
    for metric in SUMMARY_METRICS:
        values = d[metric].dropna().to_numpy(float)

        mean, low, high = bootstrap_mean_ci(values)

        summary_rows.append({
            "model": model,
            "metric": metric,
            "n_targets": len(values),
            "mean": mean,
            "median": (
                float(np.median(values))
                if len(values) else np.nan
            ),
            "sd": (
                float(np.std(values, ddof=1))
                if len(values) > 1 else np.nan
            ),
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
        })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUT / "quality_model_summary.csv",
    index=False
)

# --------------------------------------------------
# 6. Paired model comparisons
# --------------------------------------------------

comparison_rows = []

for metric in [
    "composition_tv",
    "hydropathy_delta",
    "charged_fraction_delta",
    "charge_proxy_delta",
    "pairwise_diversity",
]:
    wide = targets.pivot(
        index="target_id",
        columns="model",
        values=metric
    )

    valid_models = [
        m for m in FILES
        if wide[m].notna().all()
    ]

    for a, b in combinations(valid_models, 2):
        diff = (wide[a] - wide[b]).to_numpy(float)

        mean, low, high = bootstrap_mean_ci(diff)

        comparison_rows.append({
            "metric": metric,
            "model_a": a,
            "model_b": b,
            "n_targets": len(diff),
            "mean_difference_a_minus_b": mean,
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
        })

paired = pd.DataFrame(comparison_rows)

paired.to_csv(
    OUT / "quality_paired_bootstrap.csv",
    index=False
)

# --------------------------------------------------
# 7. Results display
# --------------------------------------------------

print("\n===== QUALITY MODEL SUMMARY =====")

for metric in SUMMARY_METRICS:
    print(f"\n--- {metric} ---")

    d = summary[summary.metric == metric]

    print(
        d[[
            "model", "n_targets", "mean",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high"
        ]].round(5).to_string(index=False)
    )

print("\n===== OUTPUT FILES =====")

for path in sorted(OUT.glob("*.csv")):
    print(path)

print("\nANALYSIS COMPLETE")

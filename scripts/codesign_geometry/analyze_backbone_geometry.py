from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path("/scratch/f008j37/sequence_structure")
DATA = ROOT / "codesign/quality_audit/structure_inventory.csv"
OUT = ROOT / "codesign/quality_audit"

df = pd.read_csv(DATA)

BACKBONE = ("N", "CA", "C", "O")

# Broad, exploratory screening ranges in Angstrom.
RANGES = {
    "N_CA": (1.30, 1.65),
    "CA_C": (1.35, 1.70),
    "C_N": (1.15, 1.50),
    "CA_CA": (3.40, 4.20),
}

def read_pdb(path):
    residues = {}
    order = []

    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break

            if not line.startswith("ATOM  "):
                continue

            atom = line[12:16].strip()

            if atom not in BACKBONE:
                continue

            if line[16] not in (" ", "A"):
                continue

            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()

            key = (chain, resseq, icode)

            if key not in residues:
                residues[key] = {}
                order.append(key)

            if atom in residues[key]:
                continue

            try:
                coord = np.array([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54])
                ], dtype=float)
            except ValueError:
                continue

            if np.isfinite(coord).all():
                residues[key][atom] = coord

    return order, residues

def distance(a, b):
    return float(np.linalg.norm(a - b))

def analyze(path):
    order, residues = read_pdb(path)

    missing = 0
    distances = {name: [] for name in RANGES}

    ca_coords = []
    ca_indices = []
    ca_chains = []

    for idx, key in enumerate(order):
        atoms = residues[key]

        if not all(a in atoms for a in BACKBONE):
            missing += 1

        if "N" in atoms and "CA" in atoms:
            distances["N_CA"].append(
                distance(atoms["N"], atoms["CA"])
            )

        if "CA" in atoms and "C" in atoms:
            distances["CA_C"].append(
                distance(atoms["CA"], atoms["C"])
            )

        if "CA" in atoms:
            ca_coords.append(atoms["CA"])
            ca_indices.append(idx)
            ca_chains.append(key[0])

        if idx == 0:
            continue

        prev_key = order[idx - 1]

        # Do not treat a different chain as a peptide bond.
        if key[0] != prev_key[0]:
            continue

        # Only join sequential residue numbers.
        try:
            prev_num = int(prev_key[1])
            curr_num = int(key[1])
        except ValueError:
            continue

        if curr_num != prev_num + 1:
            continue

        prev = residues[prev_key]

        if "C" in prev and "N" in atoms:
            distances["C_N"].append(
                distance(prev["C"], atoms["N"])
            )

        if "CA" in prev and "CA" in atoms:
            distances["CA_CA"].append(
                distance(prev["CA"], atoms["CA"])
            )

    result = {
        "n_residues": len(order),
        "n_incomplete_backbone_residues": missing,
    }

    for name, values in distances.items():
        arr = np.asarray(values, dtype=float)
        low, high = RANGES[name]

        result[f"{name}_n"] = len(arr)
        result[f"{name}_mean"] = (
            float(arr.mean()) if len(arr) else np.nan
        )
        result[f"{name}_outlier_fraction"] = (
            float(((arr < low) | (arr > high)).mean())
            if len(arr) else np.nan
        )
        result[f"{name}_max_deviation"] = (
            float(np.max(np.maximum(low - arr, arr - high)))
            if len(arr) else np.nan
        )

    # Severe nonlocal CA overlaps.
    clash_pairs = 0

    if len(ca_coords) >= 2:
        coords = np.asarray(ca_coords, dtype=float)
        tree = cKDTree(coords)

        pairs = tree.query_pairs(r=3.0)

        for i, j in pairs:
            # Exclude nearby residues on the same chain.
            if (
                ca_chains[i] == ca_chains[j]
                and abs(ca_indices[i] - ca_indices[j]) < 4
            ):
                continue

            clash_pairs += 1

    result["severe_nonlocal_ca_pairs"] = clash_pairs
    result["has_severe_ca_overlap"] = clash_pairs > 0

    return result

rows = []

for i, r in enumerate(df.itertuples(index=False), 1):
    if not r.generated_exists:
        continue

    path = Path(r.generated_pdb)

    if not path.is_file():
        raise FileNotFoundError(path)

    metrics = analyze(path)

    rows.append({
        "model": r.model,
        "sample_id": r.sample_id,
        "length": int(r.length),
        "scTM": float(r.scTM),
        "high_scTM": float(r.scTM) >= 0.90,
        **metrics
    })

    if i % 250 == 0:
        print(f"Processed {i}/{len(df)}", flush=True)

result = pd.DataFrame(rows)

assert len(result) == 2000

# Evaluate the screened geometry independently of scTM.
outlier_cols = [
    f"{name}_outlier_fraction"
    for name in RANGES
]

result["has_screened_geometry_issue"] = (
    (result["n_incomplete_backbone_residues"] > 0)
    | result["has_severe_ca_overlap"]
    | result[outlier_cols].gt(0).any(axis=1)
)

result.to_csv(
    OUT / "backbone_geometry_per_sample.csv",
    index=False
)

summary = (
    result.groupby(["model", "high_scTM"])
    .agg(
        n=("sample_id", "size"),
        mean_scTM=("scTM", "mean"),
        mean_incomplete_residues=(
            "n_incomplete_backbone_residues", "mean"
        ),
        mean_N_CA_outlier_fraction=(
            "N_CA_outlier_fraction", "mean"
        ),
        mean_CA_C_outlier_fraction=(
            "CA_C_outlier_fraction", "mean"
        ),
        mean_C_N_outlier_fraction=(
            "C_N_outlier_fraction", "mean"
        ),
        mean_CA_CA_outlier_fraction=(
            "CA_CA_outlier_fraction", "mean"
        ),
        structures_with_severe_ca_overlap=(
            "has_severe_ca_overlap", "sum"
        ),
        structures_with_any_screened_issue=(
            "has_screened_geometry_issue", "sum"
        )
    )
    .reset_index()
)

summary.to_csv(
    OUT / "backbone_geometry_summary.csv",
    index=False
)

print("\n===== BACKBONE GEOMETRY SUMMARY =====")
print(summary.round(5).to_string(index=False))

print("\n===== HIGH scTM QUALITY CHECK =====")

high = result[result.high_scTM]

print("High scTM structures:", len(high))
print(
    "With any screened geometry issue:",
    int(high.has_screened_geometry_issue.sum())
)
print(
    "With severe nonlocal CA overlap:",
    int(high.has_severe_ca_overlap.sum())
)

print("\n===== OUTPUT FILES =====")
print(OUT / "backbone_geometry_per_sample.csv")
print(OUT / "backbone_geometry_summary.csv")

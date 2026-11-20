from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path("/scratch/f008j37/sequence_structure")
OUT = ROOT / "codesign/quality_audit"

df = pd.read_csv(OUT / "structure_inventory.csv")
df = df[df["scTM"] >= 0.90].copy()

assert len(df) == 764
assert df["generated_pdb"].notna().all()

def read_backbone(path):
    residues = {}
    order = []

    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break

            if not line.startswith("ATOM  "):
                continue

            atom = line[12:16].strip()

            if atom not in {"N", "CA", "C", "O"}:
                continue

            if line[16] not in (" ", "A"):
                continue

            key = (
                line[21],
                line[22:26].strip(),
                line[26]
            )

            if key not in residues:
                residues[key] = {}
                order.append(key)

            if atom in residues[key]:
                continue

            xyz = np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])

            if np.isfinite(xyz).all():
                residues[key][atom] = xyz

    return order, residues

rows = []
cn_details = []
ca_details = []

for index, r in enumerate(df.itertuples(index=False), 1):
    order, atoms = read_backbone(r.generated_pdb)

    cn = []
    ca = []
    ca_positions = []

    for i, key in enumerate(order):
        if "CA" in atoms[key]:
            ca.append(atoms[key]["CA"])
            ca_positions.append((i, key))

        if i == 0:
            continue

        prev_key = order[i - 1]

        if key[0] != prev_key[0]:
            continue

        try:
            consecutive = (
                int(key[1]) == int(prev_key[1]) + 1
            )
        except ValueError:
            consecutive = False

        if not consecutive:
            continue

        prev = atoms[prev_key]
        curr = atoms[key]

        if "C" in prev and "N" in curr:
            dist = float(
                np.linalg.norm(prev["C"] - curr["N"])
            )
            cn.append(dist)

            if dist < 1.15 or dist > 1.50:
                cn_details.append({
                    "model": r.model,
                    "sample_id": r.sample_id,
                    "residue_index": i,
                    "cn_distance": dist,
                })

    ca_distances = []

    if len(ca) >= 2:
        xyz = np.asarray(ca, dtype=float)
        tree = cKDTree(xyz)

        for a, b in tree.query_pairs(r=3.0):
            ia, key_a = ca_positions[a]
            ib, key_b = ca_positions[b]

            if (
                key_a[0] == key_b[0]
                and abs(ia - ib) < 4
            ):
                continue

            dist = float(
                np.linalg.norm(xyz[a] - xyz[b])
            )
            ca_distances.append(dist)

            ca_details.append({
                "model": r.model,
                "sample_id": r.sample_id,
                "residue_a": ia + 1,
                "residue_b": ib + 1,
                "ca_distance": dist,
            })

    cn = np.asarray(cn, dtype=float)
    ca_distances = np.asarray(ca_distances, dtype=float)

    assert len(cn) == int(r.length) - 1
    assert len(ca) == int(r.length)

    rows.append({
        "model": r.model,
        "sample_id": r.sample_id,
        "length": int(r.length),
        "scTM": float(r.scTM),
        "cn_min": float(cn.min()),
        "cn_max": float(cn.max()),
        "cn_outside_1_15_1_50": int(
            ((cn < 1.15) | (cn > 1.50)).sum()
        ),
        "cn_outside_1_10_1_60": int(
            ((cn < 1.10) | (cn > 1.60)).sum()
        ),
        "ca_pairs_lt3": len(ca_distances),
        "ca_pairs_lt2_5": int(
            (ca_distances < 2.5).sum()
        ),
        "ca_pairs_lt2": int(
            (ca_distances < 2.0).sum()
        ),
        "min_nonlocal_ca_distance": (
            float(ca_distances.min())
            if len(ca_distances) else np.nan
        ),
    })

    if index % 200 == 0:
        print(f"Processed {index}/{len(df)}", flush=True)

result = pd.DataFrame(rows)
cn_detail = pd.DataFrame(cn_details)
ca_detail = pd.DataFrame(ca_details)

result.to_csv(
    OUT / "high_sctm_geometry_severity.csv",
    index=False
)

cn_detail.to_csv(
    OUT / "high_sctm_cn_outliers.csv",
    index=False
)

ca_detail.to_csv(
    OUT / "high_sctm_nonlocal_ca_pairs.csv",
    index=False
)

print("\n===== HIGH scTM GEOMETRY SEVERITY =====")

summary = result.groupby("model").agg(
    n=("sample_id", "size"),
    structures_with_cn_outliers=(
        "cn_outside_1_15_1_50",
        lambda x: int((x > 0).sum())
    ),
    structures_with_large_cn_deviation=(
        "cn_outside_1_10_1_60",
        lambda x: int((x > 0).sum())
    ),
    structures_with_ca_lt3=(
        "ca_pairs_lt3",
        lambda x: int((x > 0).sum())
    ),
    structures_with_ca_lt2_5=(
        "ca_pairs_lt2_5",
        lambda x: int((x > 0).sum())
    ),
    structures_with_ca_lt2=(
        "ca_pairs_lt2",
        lambda x: int((x > 0).sum())
    ),
    minimum_cn_distance=("cn_min", "min"),
    maximum_cn_distance=("cn_max", "max"),
    minimum_nonlocal_ca_distance=(
        "min_nonlocal_ca_distance", "min"
    ),
)

print(summary.round(4).to_string())

print("\n===== TOTAL HIGH scTM COUNTS =====")
for column in [
    "cn_outside_1_15_1_50",
    "cn_outside_1_10_1_60",
    "ca_pairs_lt3",
    "ca_pairs_lt2_5",
    "ca_pairs_lt2",
]:
    print(
        column,
        "structures:",
        int((result[column] > 0).sum()),
        "pairs/bonds:",
        int(result[column].sum())
    )

print("\n===== MOST EXTREME CA CONTACTS =====")
if not ca_detail.empty:
    print(
        ca_detail.nsmallest(
            15, "ca_distance"
        ).round(4).to_string(index=False)
    )

print("\n===== MOST EXTREME C-N DISTANCES =====")
if not cn_detail.empty:
    cn_detail["deviation_from_1_33"] = (
        cn_detail["cn_distance"] - 1.33
    ).abs()

    print(
        cn_detail.nlargest(
            15, "deviation_from_1_33"
        ).round(4).to_string(index=False)
    )

print("\nSaved:",
      OUT / "high_sctm_geometry_severity.csv")

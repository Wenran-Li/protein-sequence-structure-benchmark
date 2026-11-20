from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

ROOT = Path("/scratch/f008j37/sequence_structure")
INPUT = ROOT / "appendix_release/data/codesign_samples.csv"
OUT = ROOT / "codesign/quality_audit"

df = pd.read_csv(INPUT)

required = [
    "model", "sample_id", "length", "status",
    "scTM", "generated_pdb", "esmfold_pdb"
]
assert set(required).issubset(df.columns)

print("===== BENCHMARK MANIFEST =====")
print("Rows:", len(df))
print("Models:", df["model"].value_counts().to_dict())
print("Statuses:", df["status"].value_counts(dropna=False).to_dict())
print("Duplicate model/sample IDs:",
      df.duplicated(["model", "sample_id"]).sum())

def resolve_path(raw):
    if pd.isna(raw) or not str(raw).strip():
        return None

    p = Path(str(raw))

    if p.exists():
        return p

    if not p.is_absolute():
        candidate = ROOT / p
        if candidate.exists():
            return candidate

    return None

def inspect_pdb(path):
    if path is None:
        return {
            "n_atoms": 0,
            "n_ca": 0,
            "n_backbone_atoms": 0,
            "n_sidechain_atoms": 0,
            "n_hydrogen_atoms": 0,
            "n_nonfinite": 0,
            "atom_classes": "",
        }

    atom_counts = Counter()
    n_atoms = 0
    n_ca = 0
    n_backbone = 0
    n_sidechain = 0
    n_hydrogen = 0
    n_nonfinite = 0

    backbone = {"N", "CA", "C", "O", "OXT"}

    with path.open(errors="replace") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break

            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            if len(line) < 54:
                continue

            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue

            atom = line[12:16].strip()
            element = line[76:78].strip().upper() if len(line) >= 78 else ""

            try:
                xyz = np.array([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54])
                ])
            except ValueError:
                n_nonfinite += 1
                continue

            if not np.isfinite(xyz).all():
                n_nonfinite += 1
                continue

            n_atoms += 1
            atom_counts[atom] += 1

            if atom == "CA":
                n_ca += 1

            if element in ("H", "D") or atom.startswith(("H", "D")):
                n_hydrogen += 1
            elif atom in backbone:
                n_backbone += 1
            else:
                n_sidechain += 1

    return {
        "n_atoms": n_atoms,
        "n_ca": n_ca,
        "n_backbone_atoms": n_backbone,
        "n_sidechain_atoms": n_sidechain,
        "n_hydrogen_atoms": n_hydrogen,
        "n_nonfinite": n_nonfinite,
        "atom_classes": ",".join(sorted(atom_counts)),
    }

rows = []

for r in df.itertuples(index=False):
    gen_path = resolve_path(r.generated_pdb)
    fold_path = resolve_path(r.esmfold_pdb)

    info = inspect_pdb(gen_path)

    rows.append({
        "model": r.model,
        "sample_id": r.sample_id,
        "length": r.length,
        "status": r.status,
        "scTM": r.scTM,
        "generated_exists": gen_path is not None,
        "esmfold_exists": fold_path is not None,
        "generated_pdb": str(gen_path) if gen_path else "",
        **info,
    })

result = pd.DataFrame(rows)

result["has_backbone"] = (
    (result.n_ca > 0)
    & (result.n_backbone_atoms >= result.n_ca * 3)
)

result["has_sidechains"] = result.n_sidechain_atoms > 0

print("\n===== STRUCTURE AVAILABILITY =====")

summary = result.groupby("model").agg(
    n=("sample_id", "size"),
    generated_found=("generated_exists", "sum"),
    esmfold_found=("esmfold_exists", "sum"),
    backbone_present=("has_backbone", "sum"),
    sidechains_present=("has_sidechains", "sum"),
    mean_ca_atoms=("n_ca", "mean"),
    nonfinite_coordinate_files=(
        "n_nonfinite", lambda x: int((x > 0).sum())
    )
)

print(summary.to_string())

print("\n===== REPRESENTATIVE STRUCTURE RECORDS =====")
for model, d in result.groupby("model"):
    valid = d[d.generated_exists]
    print("\nMODEL:", model)
    if valid.empty:
        print("No generated structures found")
    else:
        print(valid[[
            "sample_id", "length", "n_atoms", "n_ca",
            "n_sidechain_atoms", "atom_classes"
        ]].head(2).to_string(index=False, max_colwidth=90))

print("\n===== MISSING FILE EXAMPLES =====")
print(
    df.loc[
        ~result.generated_exists.to_numpy(),
        ["model", "sample_id", "generated_pdb"]
    ].head(15).to_string(index=False, max_colwidth=110)
)

output = OUT / "structure_inventory.csv"
result.to_csv(output, index=False)
print("\nSaved:", output)

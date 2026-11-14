from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(
    "/scratch/f008j37/sequence_structure/structure_prediction"
)

OUT = BASE / "evaluation_results"

paired = pd.read_csv(
    OUT / "four_model_paired_preliminary.csv"
)

rmsd = pd.read_csv(
    BASE / "boltz2/boltz2_rmsd.csv"
)

rmsd = rmsd.loc[
    rmsd.status.eq("evaluated"),
    ["target_id", "ca_rmsd_angstrom"]
].rename(columns={
    "ca_rmsd_angstrom": "boltz_rmsd"
})

assert not rmsd.target_id.duplicated().any()

paired = paired.merge(
    rmsd,
    on="target_id",
    how="left",
    validate="many_to_one",
)

mask = paired.model.eq("Boltz-2")

paired.loc[mask, "rmsd"] = paired.loc[
    mask, "boltz_rmsd"
]

paired = paired.drop(columns=["boltz_rmsd"])

models = [
    "AlphaFold3",
    "Boltz-2",
    "ESMFold",
    "Chai-1",
]

assert paired.target_id.nunique() == 124
assert len(paired) == 496

assert not paired.duplicated(
    ["target_id", "model"]
).any()

assert paired[
    ["rmsd", "tm_score", "lddt_ca"]
].notna().all().all(), "Missing metric values"

assert (paired.rmsd >= 0).all()
assert paired.tm_score.between(0, 1).all()
assert paired.lddt_ca.between(0, 1).all()

paired.to_csv(
    OUT / "four_model_paired_complete.csv",
    index=False
)

rows = []

for model in models:

    sub = paired[paired.model == model]

    assert len(sub) == 124

    row = {
        "Model": model,
        "N": len(sub),
    }

    for metric, name in [
        ("rmsd", "RMSD"),
        ("tm_score", "TM-score"),
        ("lddt_ca", "lDDT-Ca"),
    ]:
        values = sub[metric].to_numpy()

        row[f"{name} mean"] = values.mean()
        row[f"{name} SD"] = values.std(ddof=1)
        row[f"{name} median"] = np.median(values)

    rows.append(row)

summary = pd.DataFrame(rows)

summary.to_csv(
    OUT / "table1_four_models.csv",
    index=False
)

print("\n========== TABLE 1 ==========")

for _, row in summary.iterrows():

    print(
        f"{row['Model']:12s} "
        f"N={int(row['N'])}  "
        f"RMSD={row['RMSD mean']:.3f} "
        f"± {row['RMSD SD']:.3f}  "
        f"TM={row['TM-score mean']:.4f} "
        f"± {row['TM-score SD']:.4f}  "
        f"lDDT={row['lDDT-Ca mean']:.4f} "
        f"± {row['lDDT-Ca SD']:.4f}"
    )

# Determine the best mean value for each metric.
best_rmsd = summary["RMSD mean"].min()
best_tm = summary["TM-score mean"].max()
best_lddt = summary["lDDT-Ca mean"].max()


def fmt(mean, sd, best=False, digits=3):
    value = f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"

    if best:
        return "\\textbf{" + value + "}"

    return value


lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Overall structure prediction accuracy on "
    r"124 common CAMEO targets. Values are mean "
    r"$\pm$ standard deviation. Lower RMSD and "
    r"higher TM-score and lDDT-C$\alpha$ indicate "
    r"better agreement with experimental structures.}",
    r"\label{tab:structure_prediction}",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"Model & $N$ & C$\alpha$ RMSD ($\AA$)$\downarrow$ "
    r"& TM-score$\uparrow$ & "
    r"lDDT-C$\alpha$$\uparrow$ \\",
    r"\midrule",
]

for _, row in summary.iterrows():

    rmsd_str = fmt(
        row["RMSD mean"],
        row["RMSD SD"],
        np.isclose(row["RMSD mean"], best_rmsd),
        3,
    )

    tm_str = fmt(
        row["TM-score mean"],
        row["TM-score SD"],
        np.isclose(row["TM-score mean"], best_tm),
        4,
    )

    lddt_str = fmt(
        row["lDDT-Ca mean"],
        row["lDDT-Ca SD"],
        np.isclose(row["lDDT-Ca mean"], best_lddt),
        4,
    )

    lines.append(
        f"{row['Model']} & {int(row['N'])} & "
        f"{rmsd_str} & {tm_str} & {lddt_str} "
        + r"\\"
    )

lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
])

tex = "\n".join(lines) + "\n"

tex_path = OUT / "table1_four_models.tex"
tex_path.write_text(tex)

print("\n========== LATEX ==========")
print(tex)

print("Saved:")
print(OUT / "table1_four_models.csv")
print(OUT / "four_model_paired_complete.csv")
print(tex_path)

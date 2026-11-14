from pathlib import Path
import os
import re
import time
import subprocess
import traceback

import pandas as pd
import torch
import esm
from Bio.PDB import PDBParser

BASE = Path("/scratch/f008j37/sequence_structure/codesign")
MANIFEST = BASE / "results/protpardelle_benchmark_manifest.csv"
OUT = BASE / "results/protpardelle_esmfold_full"
PDB_DIR = OUT / "predictions"
CSV = OUT / "self_consistency.csv"
TMALIGN = "/scratch/f008j37/bin/TMalign"

PDB_DIR.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(MANIFEST).sort_values(
    ["target_length", "sample_id"]
)

if CSV.exists():
    previous = pd.read_csv(CSV)
    records = previous.to_dict("records")
    completed = set(
        previous.loc[
            previous["status"] == "success", "sample_id"
        ].astype(str)
    )
else:
    records = []
    completed = set()

print("Total samples:", len(manifest))
print("Previously completed:", len(completed), flush=True)

torch.set_grad_enabled(False)

print("Loading ESMFold...", flush=True)
model = esm.pretrained.esmfold_v1().eval().cuda()
model.set_chunk_size(32)
print("ESMFold ready.", flush=True)

parser = PDBParser(QUIET=True)

for i, row in enumerate(manifest.itertuples(index=False), 1):
    sample_id = str(row.sample_id)

    if sample_id in completed:
        continue

    generated_pdb = Path(row.pdb_path)
    predicted_pdb = PDB_DIR / f"{sample_id}_esmfold.pdb"
    length = int(row.target_length)

    start = time.time()

    print(
        f"[{i}/{len(manifest)}] {sample_id} L={length}",
        flush=True
    )

    record = {
        "model": "Protpardelle-1c",
        "sample_id": sample_id,
        "length": length,
        "status": "failed",
        "scTM": float("nan"),
        "scRMSD": float("nan"),
        "aligned_length": float("nan"),
        "mean_plddt": float("nan"),
        "esmfold_seconds": float("nan"),
        "total_seconds": float("nan"),
        "generated_pdb": str(generated_pdb),
        "esmfold_pdb": str(predicted_pdb),
        "error": "",
    }

    try:
        assert generated_pdb.exists()

        t0 = time.time()

        if not predicted_pdb.exists():
            with torch.no_grad():
                pdb_string = model.infer_pdb(str(row.sequence))
            predicted_pdb.write_text(pdb_string)

        record["esmfold_seconds"] = time.time() - t0

        structure = parser.get_structure("esmfold", str(predicted_pdb))
        residues = [
            r for r in structure.get_residues()
            if r.id[0] == " "
        ]

        if len(residues) != length:
            raise ValueError(
                f"Predicted length {len(residues)} != {length}"
            )

        plddt = [
            r["CA"].get_bfactor()
            for r in residues
            if "CA" in r
        ]

        if len(plddt) != length:
            raise ValueError("Missing CA atoms in ESMFold output")

        result = subprocess.run(
            [TMALIGN, str(generated_pdb), str(predicted_pdb)],
            capture_output=True,
            text=True,
            check=True,
        )

        output = result.stdout

        tm_match = re.search(
            r"TM-score=\s*([\d.]+).*normalized by length of Chain_1",
            output,
        )

        rmsd_match = re.search(
            r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+)",
            output,
        )

        if not tm_match or not rmsd_match:
            raise ValueError("Unable to parse TM-align output")

        record.update({
            "status": "success",
            "scTM": float(tm_match.group(1)),
            "scRMSD": float(rmsd_match.group(2)),
            "aligned_length": int(rmsd_match.group(1)),
            "mean_plddt": sum(plddt) / len(plddt),
            "total_seconds": time.time() - start,
            "error": "",
        })

        print(
            f"SUCCESS scTM={record['scTM']:.4f} "
            f"pLDDT={record['mean_plddt']:.2f} "
            f"time={record['total_seconds']:.1f}s",
            flush=True,
        )

    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["total_seconds"] = time.time() - start
        print(f"FAILED {record['error']}", flush=True)
        traceback.print_exc()

    records = [
        r for r in records
        if str(r["sample_id"]) != sample_id
    ]
    records.append(record)

    # Atomic CSV replacement: protects completed results on interruption
    temp_csv = CSV.with_suffix(".tmp")
    pd.DataFrame(records).to_csv(temp_csv, index=False)
    os.replace(temp_csv, CSV)

    torch.cuda.empty_cache()

print("\n========== FINAL SUMMARY ==========")
df = pd.read_csv(CSV)
print(df.groupby(["length", "status"]).size())
print("Saved:", CSV)

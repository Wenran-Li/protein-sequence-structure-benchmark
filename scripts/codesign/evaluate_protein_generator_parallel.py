from pathlib import Path
import argparse
import os
import re
import time
import subprocess
import traceback

import pandas as pd
import torch
import esm
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

BASE = Path("/scratch/f008j37/sequence_structure/codesign")
SOURCE = BASE / "results/protein_generator_benchmark_final"
OUT = BASE / "results/protein_generator_esmfold_full"
TMALIGN = "/scratch/f008j37/bin/TMalign"

parser_args = argparse.ArgumentParser()
parser_args.add_argument("--lengths", nargs="+", type=int, required=True)
parser_args.add_argument("--gpu", type=int, required=True)
args = parser_args.parse_args()

tag = "gpu" + str(args.gpu)
OUT.mkdir(parents=True, exist_ok=True)

PDB_DIR = OUT / "predictions" / tag
PDB_DIR.mkdir(parents=True, exist_ok=True)

CSV = OUT / f"self_consistency_{tag}.csv"

parser = PDBParser(QUIET=True)

def get_sequence(pdb_path, expected_length):
    structure = parser.get_structure("generated", str(pdb_path))
    chains = list(structure[0].get_chains())

    if len(chains) != 1:
        raise ValueError(f"Expected one chain, found {len(chains)}")

    residues = [
        r for r in chains[0].get_residues()
        if r.id[0] == " "
    ]

    sequence = "".join(
        seq1(r.resname, custom_map={"UNK": "X"})
        for r in residues
    )

    if len(sequence) != expected_length:
        raise ValueError(
            f"Generated length {len(sequence)} != {expected_length}"
        )

    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", sequence):
        raise ValueError("Non-standard amino acids in generated PDB")

    if any("CA" not in r for r in residues):
        raise ValueError("Missing CA in generated PDB")

    return sequence

tasks = []
for L in args.lengths:
    for i in range(100):
        sample = f"design_{i:06d}"
        sample_id = f"protein_generator_L{L}_{sample}"
        pdb = SOURCE / f"length_{L}" / f"{sample}.pdb"
        tasks.append((sample_id, L, pdb))

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

print("GPU:", args.gpu, flush=True)
print("Lengths:", args.lengths, flush=True)
print("Total tasks:", len(tasks), flush=True)
print("Previously completed:", len(completed), flush=True)

torch.set_grad_enabled(False)

print("Loading ESMFold...", flush=True)
model = esm.pretrained.esmfold_v1().eval().cuda()
model.set_chunk_size(32)
print("ESMFold ready.", flush=True)

for index, (sample_id, L, generated_pdb) in enumerate(tasks, 1):
    if sample_id in completed:
        continue

    predicted_pdb = PDB_DIR / f"{sample_id}_esmfold.pdb"
    start = time.time()

    print(
        f"[{index}/{len(tasks)}] {sample_id} L={L}",
        flush=True
    )

    record = {
        "model": "ProteinGenerator",
        "sample_id": sample_id,
        "length": L,
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
        if not generated_pdb.is_file():
            raise FileNotFoundError(generated_pdb)

        sequence = get_sequence(generated_pdb, L)

        t0 = time.time()

        if not predicted_pdb.exists() or predicted_pdb.stat().st_size == 0:
            with torch.no_grad():
                pdb_string = model.infer_pdb(sequence)

            temp_pdb = predicted_pdb.with_suffix(".tmp")
            temp_pdb.write_text(pdb_string)
            os.replace(temp_pdb, predicted_pdb)

        record["esmfold_seconds"] = time.time() - t0

        structure = parser.get_structure("esmfold", str(predicted_pdb))
        residues = [
            r for r in structure.get_residues()
            if r.id[0] == " "
        ]

        if len(residues) != L:
            raise ValueError(
                f"Predicted length {len(residues)} != {L}"
            )

        plddt = [
            r["CA"].get_bfactor()
            for r in residues
            if "CA" in r
        ]

        if len(plddt) != L:
            raise ValueError("Missing CA in ESMFold output")

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

    temp_csv = CSV.with_suffix(".tmp")
    pd.DataFrame(records).to_csv(temp_csv, index=False)
    os.replace(temp_csv, CSV)

    torch.cuda.empty_cache()

print("\n========== FINAL SUMMARY ==========")
df = pd.read_csv(CSV)
print(df.groupby(["length", "status"]).size())
print("Saved:", CSV)

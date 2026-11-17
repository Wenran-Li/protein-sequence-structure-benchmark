#!/usr/bin/env python3
"""Rebuild per-target inverse folding recovery from the released sequence CSVs.

Validates provided recovery fractions by residue-by-residue sequence comparison.
Uses a per-target arithmetic mean of five samples except PiFold (one sample).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = {
    'ESM-IF1': ('esm_if1_cath_sequences.csv', 'sequence', 'sequence_recovery', 5),
    'GVP-GNN': ('gvp_cath_sequences.csv', 'designed_sequence', 'recovery', 5),
    'ProteinMPNN': ('proteinmpnn_cath_sequences.csv', 'sequence', 'sequence_recovery', 5),
    'PiFold': ('pifold_cath_sequences.csv', 'generated_sequence', 'sequence_recovery', 1),
}

def run(data, out):
    mapping=pd.read_csv(data/'cath_mpnn_mapping.csv')
    if len(mapping)!=883 or not mapping.target_id.is_unique:
        raise ValueError('Mapping must contain 883 unique target IDs')
    result=mapping[['target_id','length']].copy()
    for model,(filename,sequence_col,recovery_col,samples) in MODELS.items():
        df=pd.read_csv(data/'inverse_folding_sequences'/filename)
        if len(df)!=883*samples or df.duplicated(['target_id','sample_id']).any():
            raise ValueError(f'{model}: expected {883*samples} unique target/sample pairs')
        counts=df.groupby('target_id').size()
        if set(counts.index)!=set(mapping.target_id) or not (counts==samples).all():
            raise ValueError(f'{model}: missing or extra target/sample records')
        if df[[sequence_col,'native_sequence']].isna().any().any():
            raise ValueError(f'{model}: missing sequence values')
        seq=df[sequence_col].astype(str)
        native=df.native_sequence.astype(str)
        lengths=seq.str.len().to_numpy()
        native_lengths=native.str.len().to_numpy()
        if (lengths!=native_lengths).any() or (lengths==0).any():
            raise ValueError(f'{model}: invalid sequence lengths')
        calculated=np.fromiter((sum(a==b for a,b in zip(s,n))/len(n)
                      for s,n in zip(seq,native)),dtype=float,count=len(df))
        original=pd.to_numeric(df[recovery_col],errors='raise').to_numpy()
        maxerr=float(np.max(np.abs(calculated-original)))
        if not np.allclose(calculated,original,rtol=0,atol=1e-10):
            raise ValueError(f'{model}: provided recovery differs from sequences; max error={maxerr}')
        target_mean=(pd.DataFrame({'target_id':df.target_id,'value':calculated})
                       .groupby('target_id',as_index=False).value.mean()
                       .rename(columns={'value':model}))
        result=result.merge(target_mean,on='target_id',validate='one_to_one',how='left')
        print(f'{model}: {len(df)} records, max stored-vs-sequence error={maxerr:.3g}')
    if result.isna().any().any():
        raise ValueError('Unexpected missing target-level recovery values')
    out.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(out,index=False,float_format='%.17g')
    print('Wrote',out,'shape',result.shape)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=Path('data'))
    ap.add_argument('--out',type=Path,default=Path('data/inverse_folding_target_recovery.csv'))
    args=ap.parse_args()
    run(args.data,args.out)
if __name__=='__main__':main()

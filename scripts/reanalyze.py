#!/usr/bin/env python3
"""Recompute target-level summary and paired statistics from released CSVs.
Requires numpy, pandas, scipy; no model weights, structures, or cluster paths.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.stats import wilcoxon, spearmanr

MODELS_SP = ['AlphaFold 3', 'Boltz-2', 'ESMFold', 'Chai-1']
MODELS_IF = ['ESM-IF1', 'PiFold', 'GVP-GNN', 'ProteinMPNN']
N_BOOT = 10000
SEED = 42

def norm(s):
    return ''.join(c for c in str(s).lower() if c.isalnum())

def match(df, options, *, required=True):
    columns={norm(c):c for c in df.columns}
    for v in options:
        if norm(v) in columns: return columns[norm(v)]
    if required: raise ValueError(f'Missing one of {options}; available columns: {list(df.columns)}')
    return None

def standard_model(s):
    k=norm(s)
    aliases={'af3':'AlphaFold 3','alphafold3':'AlphaFold 3','boltz2':'Boltz-2',
      'esmfold':'ESMFold','chai1':'Chai-1', 'esmif1':'ESM-IF1',
      'pifold':'PiFold','gvp':'GVP-GNN','gvpgnn':'GVP-GNN',
      'proteinmpnn':'ProteinMPNN','mpnn':'ProteinMPNN'}
    return aliases.get(k,str(s))

def holm(p):
    p=np.asarray(p,float); order=np.argsort(p); adj=np.empty(len(p)); running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(p)-rank)*p[index]); adj[index]=min(running,1.)
    return adj

def boot_mean_ci(x,n=N_BOOT,seed=SEED):
    """Percentile CI, using independent per-contrast RNG initialized with seed 42.
    Existing analysis may have used a different bootstrap random-number schedule.
    """
    x=np.asarray(x,dtype=float); rng=np.random.default_rng(seed)
    vals=np.empty(n)
    for i in range(n): vals[i]=x[rng.integers(0,len(x),len(x))].mean()
    return np.percentile(vals,[2.5,97.5])

def paired(matrix, names, metric, scale=1, tie_tol=1e-10):
    """Paired tests with numerically stable ties in the *scaled* metric.

    E.g., inverse-folding recovery is expressed in percentage points; 1e-10
    percentage points is well below the resolution of one residue difference.
    Differences near zero are set to exact zero for Wilcoxon and tie counts.
    Descriptive mean differences and bootstrap CIs use original differences.
    """
    records=[]
    for a,b in itertools.combinations(names,2):
        diff=(matrix[a]-matrix[b]).to_numpy(float)*scale
        if not np.isfinite(diff).all():
            raise ValueError(f'Invalid paired data in {metric}: {a}/{b}')
        test_diff=diff.copy()
        test_diff[np.abs(test_diff) <= tie_tol]=0.0
        p=float(wilcoxon(diff,alternative='two-sided',zero_method='wilcox',method='auto').pvalue) if np.any(diff!=0) else 1.
        lo,hi=boot_mean_ci(diff)
        records.append({'metric':metric,'model_a':a,'model_b':b,'n_targets':len(diff),
          'mean_difference':diff.mean(),'ci95_low':lo,'ci95_high':hi,
          'p_raw':p,'a_better':int((test_diff>0).sum()),
          'b_better':int((test_diff<0).sum()),'ties':int((test_diff==0).sum())})
    adjusted=holm([r['p_raw'] for r in records])
    for r,q in zip(records,adjusted): r['p_holm']=q
    return records

def unique_matrix(long, target, model, metrics, models):
    subset=long[long[model].isin(models)].copy()
    if subset.duplicated([target,model]).any():
        raise ValueError('Duplicate target-model rows: aggregate/filter before reanalysis')
    out={}
    for label,col in metrics.items():
        p=subset.pivot(index=target,columns=model,values=col)
        missing=[m for m in models if m not in p.columns]
        if missing: raise ValueError(f'{label}: missing models {missing}')
        p=p[models].apply(pd.to_numeric,errors='raise').dropna()
        out[label]=p
    return out

def structure(path, output):
    d=pd.read_csv(path)
    target=match(d,['target_id','target','pdb_id'])
    model=match(d,['model','predictor'])
    d[model]=d[model].map(standard_model)
    fields={'TM-score':match(d,['tm_score','tmscore','tm']),
       'lDDT-Ca':match(d,['lddt_ca','lddtca','lddt']),
       'CA_RMSD_A':match(d,['rmsd','ca_rmsd','rmsd_ca'])}
    mats=unique_matrix(d,target,model,fields,MODELS_SP)
    summary=[]; pairs=[]
    for metric,p in mats.items():
        if len(p)!=124: print(f'WARNING structure {metric}: {len(p)} complete targets; manuscript uses 124')
        for name in MODELS_SP:
            x=p[name]
            summary.append({'metric':metric,'model':name,'n_targets':len(x),'mean':x.mean(),
               'std_sample':x.std(ddof=1),'median':x.median()})
        if metric!='CA_RMSD_A': pairs.extend(paired(p,MODELS_SP,metric))
    pd.DataFrame(summary).to_csv(output/'structure_summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(output/'table_A1_recomputed.csv',index=False)
    print('structure complete target counts:',{k:len(v) for k,v in mats.items()})

def inverse(path, output):
    d=pd.read_csv(path)
    target=match(d,['target_id','target','pdb_id','id'])
    if d[target].duplicated().any(): raise ValueError('Duplicate target rows in inverse-folding matrix')
    columns={}
    for m in MODELS_IF:
        cand=[c for c in d.columns if standard_model(c)==m]
        if not cand:
            # Accept prefixes/suffixes such as esm_if1_recovery
            cand=[c for c in d.columns if norm(m) in norm(c) and ('recovery' in norm(c) or 'rec' in norm(c))]
        if len(cand)!=1: raise ValueError(f'Expected one recovery column for {m}; found {cand}; CSV columns: {list(d.columns)}')
        columns[m]=cand[0]
    p=d.set_index(target)[list(columns.values())].copy()
    p.columns=list(columns)
    p=p.apply(pd.to_numeric,errors='raise').dropna()
    if p.to_numpy().max()>1.0 or p.to_numpy().min()<0:
        raise ValueError('Inverse recovery values must be fractions between 0 and 1, not percentages')
    if len(p)!=883: print(f'WARNING inverse: {len(p)} complete targets; manuscript uses 883')
    summ=[]
    for m in MODELS_IF:
        x=p[m].to_numpy()*100
        lo,hi=boot_mean_ci(x)
        summ.append({'model':m,'n_targets':len(x),'mean_percent':x.mean(),
            'std_sample_percent':x.std(ddof=1),'median_percent':np.median(x),
            'ci95_low_percent':lo,'ci95_high_percent':hi})
    pd.DataFrame(summ).to_csv(output/'table_3_recomputed.csv',index=False)
    pd.DataFrame(paired(p,MODELS_IF,'recovery_pp',scale=100)).to_csv(output/'table_A2_recomputed.csv',index=False)
    length=match(d,['length','sequence_length','seq_len','n_residues'],required=False)
    if length:
        q=d.set_index(target).loc[p.index]
        rows=[]
        for m in MODELS_IF:
            rho,pval=spearmanr(q[length].to_numpy(dtype=float),p[m].to_numpy())
            rows.append({'model':m,'n_targets':len(p),'spearman_rho':rho,'p':pval})
        pd.DataFrame(rows).to_csv(output/'inverse_length_spearman.csv',index=False)
    else: print('NOTE length column unavailable: Figure 3 Spearman correlations not regenerated')
    print('inverse complete target count:',len(p))

def codesign(path, output):
    d=pd.read_csv(path)
    model=match(d,['model']); length=match(d,['length','target_length','sequence_length'])
    score=match(d,['sctm','sc_tm','tm_score','tm'])
    rmsd=match(d,['scrmsd','sc_rmsd','rmsd'],required=False)
    plddt=match(d,['plddt','esmfold_plddt','mean_plddt'],required=False)
    sample=match(d,['sample_id','design_id','id'],required=False)
    status=match(d,['status','evaluation_status'],required=False)
    if status:
        d=d[d[status].astype(str).str.lower().isin(['success','successful','complete','completed','ok'])].copy()
    d[model]=d[model].replace({'Protpardelle':'Protpardelle-1c'})
    if sample:
        dup=d.duplicated([model,length,sample]).sum()
        if dup: print(f'WARNING: removing {dup} duplicate design identifiers')
        d=d.drop_duplicates([model,length,sample])
    d[score]=pd.to_numeric(d[score],errors='raise')
    records=[]
    for (m,l),g in d.groupby([model,length]):
        out={'model':m,'length':l,'n_valid':len(g),'n_designable':int((g[score]>=0.9).sum()),
             'designability_percent':100*(g[score]>=0.9).mean(), 'mean_scTM':g[score].mean()}
        if rmsd: out['mean_scRMSD']=pd.to_numeric(g[rmsd]).mean()
        if plddt: out['mean_ESMFold_pLDDT']=pd.to_numeric(g[plddt]).mean()
        aligned=match(g,['aligned_length','n_aligned'],required=False)
        if aligned: out['mean_aligned_fraction']=(pd.to_numeric(g[aligned])/pd.to_numeric(g[length])).mean()
        records.append(out)
    pd.DataFrame(records).sort_values(["model","length"]).to_csv(output/'table_A3_recomputed.csv',index=False)
    print('codesign valid records:',len(d),'model-length groups:',len(records))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--structure',type=Path,help='Released long-format prediction metrics CSV (496 rows)')
    ap.add_argument('--inverse',type=Path,help='Released target-level inverse recovery matrix (883 rows)')
    ap.add_argument('--codesign',type=Path,help='Released sample-level codesign_samples.csv (2000 rows)')
    ap.add_argument('--out',type=Path,default=Path('results/reanalysis'))
    args=ap.parse_args()
    if not any([args.structure,args.inverse,args.codesign]): ap.error('Supply at least one of --structure, --inverse, --codesign')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.structure: structure(args.structure,args.out)
    if args.inverse: inverse(args.inverse,args.out)
    if args.codesign: codesign(args.codesign,args.out)
    (args.out/'environment.json').write_text(json.dumps({'numpy':np.__version__,'pandas':pd.__version__,
        'scipy':scipy_version,'bootstrap_resamples':N_BOOT,'seed':SEED,'wilcoxon':'two-sided, zero_method=wilcox, abs(diff)<=1e-10 treated as tie (scaled units)',
        'holm':'within metric/family'},indent=2)+'\n')
    print('Wrote:',args.out.resolve())
if __name__=='__main__': main()

"""Paired class-stratified image bootstrap, conditional on the three checkpoints.

Identical image weights are shared across methods AND training seeds. Seeds are
not pooled as independent examples. Mean estimates average seed-specific metrics.
Percentile intervals are marginal, exploratory; they exclude retraining and
patient-cluster uncertainty (patient identifiers are unavailable).
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

METRICS=('acc','auc','f1_macro')

def weighted_metrics(y,p,w):
    pred=p.argmax(1); classes=np.arange(p.shape[1]); total=w.sum(1)
    acc=(w*(pred==y)).sum(1)/total
    f1=[]; auc=[]
    for c in classes:
        pos=y==c; guessed=pred==c
        tp=(w*(pos&guessed)).sum(1)
        denom=(w*pos).sum(1)+(w*guessed).sum(1)
        f1.append(np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0))
        if p.shape[1]==2 and c==0: continue
        order=np.argsort(p[:,c],kind='stable'); values=p[order,c]
        starts=np.r_[0,np.flatnonzero(np.diff(values)!=0)+1]
        positive=np.add.reduceat(w[:,order]*pos[order],starts,axis=1)
        negative=np.add.reduceat(w[:,order]*(~pos[order]),starts,axis=1)
        numerator=(positive*(np.cumsum(negative,axis=1)-negative/2)).sum(1)
        auc.append(numerator/(positive.sum(1)*negative.sum(1)))
    return np.column_stack((acc,np.mean(auc,axis=0),np.mean(f1,axis=0)))

SPECS=[('saved_fp32',32),('saved_fp16',16),('saved_int8_minmax',8),('saved_uniform_full',4),('saved_rotated_uniform',4),('saved_normal_codebook',4),('saved_rotated_normal_codebook',4),('folded_uniform',4),('dfq_cle_bc',4)]
PAIRS=[(6,3),(6,4),(6,5),(8,3),(8,7),(2,0),(1,0)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('study',type=Path);ap.add_argument('--replicates',type=int,default=2000);a=ap.parse_args()
    rng=np.random.default_rng(20260910);rows=[]
    for ds in ('bloodmnist','dermamnist','pathmnist','pneumoniamnist'):
        probs={}; labels=None;indices=None; fingerprint=None
        for seed in (42,123,456):
            for j,(m,b) in enumerate(SPECS):
                with np.load(a.study/'predictions'/f'{ds}_{seed}_{m}_{b}.npz') as z:
                    if labels is None: labels=z['label'];indices=z['index'];fingerprint=str(z['test_sha256'])
                    np.testing.assert_array_equal(z['label'],labels);np.testing.assert_array_equal(z['index'],indices)
                    assert str(z['test_sha256'])==fingerprint
                    probs[seed,j]=z['probs'].copy()
        groups=[np.flatnonzero(labels==c) for c in np.unique(labels)]
        distributions={k:np.empty((a.replicates,3)) for k in probs}
        point={k:weighted_metrics(labels,p,np.ones((1,len(labels))))[0] for k,p in probs.items()}
        for start in range(0,a.replicates,32):
            n=min(32,a.replicates-start);w=np.zeros((n,len(labels)),dtype=np.float64)
            for i in range(n):
                for g in groups:
                    selected=rng.choice(g,len(g),replace=True); w[i]+=np.bincount(selected,minlength=len(labels))
            for k,p in probs.items(): distributions[k][start:start+n]=weighted_metrics(labels,p,w)
        for left,right in PAIRS:
            for seed in (42,123,456,'mean_of_3'):
                seeds=(42,123,456) if seed=='mean_of_3' else (seed,)
                diff=np.mean([distributions[s,left]-distributions[s,right] for s in seeds],axis=0)
                observed=np.mean([point[s,left]-point[s,right] for s in seeds],axis=0)
                low,high=np.quantile(diff,[.025,.975],axis=0)
                for k,metric in enumerate(METRICS):
                    rows.append(dict(dataset=ds,model_seed=seed,left=f'{SPECS[left][0]}_{SPECS[left][1]}',right=f'{SPECS[right][0]}_{SPECS[right][1]}',metric=metric,delta=observed[k],low=low[k],high=high[k],replicates=a.replicates))
        np.savez_compressed(a.study/f'{ds}_bootstrap_replicates.npz',**{f'{s}_{j}':v for (s,j),v in distributions.items()})
        pd.DataFrame(rows).to_csv(a.study/'paired_bootstrap.csv',index=False)
        print('Bootstrap complete',ds,flush=True)
    (a.study/'bootstrap_protocol.json').write_text(json.dumps(dict(seed=20260910,replicates=a.replicates,specs=SPECS,pairs=PAIRS,unit='test image, stratified by class',scope='fixed checkpoints; identical image resamples across seeds and methods; marginal percentile 95% intervals, not adjusted for multiplicity'),indent=2))
if __name__=='__main__':main()

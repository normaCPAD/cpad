"""Empirically estimate the TV divergence of Proposition (self-supervised selection): the bound
|AUC_real - AUC_synth| <= kappa, where kappa is the total-variation distance between the score
distributions of the REAL errors and the SYNTHETIC (value-swap) corruptions, on the same
negatives. We measure kappa on Hospital and check that the AUC gap stays within it."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from _common import DATA
norm=lambda df: df.fillna("").apply(lambda s:s.astype(str).str.strip().str.lower())
dirty=norm(pd.read_csv(DATA+"hospital_dirty.csv",dtype=str)).reset_index(drop=True)
err=np.load(DATA+"hospital_errmask.npy")
cols=list(dirty.columns); n=len(dirty); ci={c:j for j,c in enumerate(cols)}

def fd_cell(df,tau=0.9):
    cell=np.zeros((n,len(cols)))
    for B in cols:
        for A in cols:
            if A==B: continue
            mode=df.groupby(A)[B].transform(lambda s:s.value_counts().idxmax())
            if (df[B].values==mode.values).mean()<tau: continue
            size=df.groupby(A)[B].transform("size").values
            own=df.groupby([A,B])[B].transform("size").values
            cell[:,ci[B]]=np.maximum(cell[:,ci[B]],1.0-own/size)
    return cell

def tv(a,b,bins=20):
    e=np.linspace(0,1,bins+1)
    pa=np.histogram(a,e,density=False)[0]/max(1,len(a))
    pb=np.histogram(b,e,density=False)[0]/max(1,len(b))
    return 0.5*np.abs(pa-pb).sum()

real_cell=fd_cell(dirty); real_pos=real_cell[err]                 # scores of the REAL error cells
clean=~err
kappas, gaps = [], []
for s in range(10):
    rng=np.random.default_rng(s); inj=np.zeros((n,len(cols)),bool); dc=dirty.copy()
    for c in cols:                                                 # inject value-swaps into CLEAN cells only
        idx=np.where((rng.random(n)<0.03)&clean[:,ci[c]])[0]
        repl=rng.permutation(dirty[c].values)[idx]; keep=repl!=dirty[c].values[idx]
        idx,repl=idx[keep],repl[keep]; dc.iloc[idx,ci[c]]=repl; inj[idx,ci[c]]=True
    syn_cell=fd_cell(dc.reset_index(drop=True)); syn_pos=syn_cell[inj]
    neg=syn_cell[clean&~inj]                                       # shared negatives (clean, untouched)
    k=tv(real_pos,syn_pos)
    auc_r=roc_auc_score(np.r_[np.ones(len(real_pos)),np.zeros(len(neg))], np.r_[real_pos,neg])
    auc_s=roc_auc_score(np.r_[np.ones(len(syn_pos)),np.zeros(len(neg))], np.r_[syn_pos,neg])
    kappas.append(k); gaps.append(abs(auc_r-auc_s))
kappas,gaps=np.array(kappas),np.array(gaps)
print(f"Hospital, {len(real_pos)} real-error cells, 10 seeds of value-swap synthetic errors")
print(f"  TV(real, synthetic) score distributions:  kappa = {kappas.mean():.3f} +/- {kappas.std():.3f}")
print(f"  |AUC_real - AUC_synthetic|:                gap   = {gaps.mean():.3f} +/- {gaps.std():.3f}")
print(f"  bound gap <= kappa holds: {bool((gaps.mean()<=kappas.mean()+1e-9))}  (synthetic signal tracks the real one)")

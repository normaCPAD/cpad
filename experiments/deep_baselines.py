"""Modern deep tabular anomaly detectors on the value-swap task. They are distribution/density
learners, so on relational value-swaps (common values in the wrong context) they should ALSO
collapse to chance, like the classical and shallow-deep detectors, confirming the thesis that
the failure is structural to the anomaly class, not the detector."""
import warnings, numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
from _common import DATA as D
from deepod.models import GOAD, ICL, NeuTraL, RCA, RDP, REPEN

norm=lambda df: df.fillna("").apply(lambda s:s.astype(str).str.strip().str.lower())
clean=norm(pd.read_csv(D+"hospital_clean_wide.csv",dtype=str)).reset_index(drop=True)
cols=list(clean.columns); n=len(clean); ci={c:i for i,c in enumerate(cols)}
TARGETS=[c for c in cols if 2<=clean[c].nunique()<=60]; rng=np.random.default_rng(0)

def fd_row(df,tau=0.85):
    cell=np.zeros((n,len(cols)))
    for B in cols:
        for A in cols:
            if A==B: continue
            mode=df.groupby(A)[B].transform(lambda s:s.value_counts().idxmax())
            if (df[B].values==mode.values).mean()<tau: continue
            size=df.groupby(A)[B].transform("size").values
            own=df.groupby([A,B])[B].transform("size").values
            cell[:,ci[B]]=np.maximum(cell[:,ci[B]],1-own/size)
    return cell.max(1)

dc=clean.copy(); y=np.zeros(n,bool)                                # inject value-swaps into governed columns
for c in TARGETS:
    freq=clean[c].value_counts(normalize=True); idx=np.where(rng.random(n)<0.05)[0]
    repl=rng.choice(freq.index.values,len(idx),p=freq.values); keep=repl!=clean[c].values[idx]
    idx,repl=idx[keep],repl[keep]; dc.iloc[idx,ci[c]]=repl; y[idx]=True
y=y if y.ndim==1 else y.any(1)
X=OneHotEncoder(handle_unknown="ignore",max_categories=50).fit_transform(dc).toarray().astype("float32")
print(f"Hospital value-swap detection ({int(y.sum())} corrupted tuples), tuple AUROC:")
print(f"  {'CPAD':<10}{roc_auc_score(y,fd_row(dc)):.3f}   (relational, constraint-based)")
for name,M in [("GOAD",GOAD),("ICL",ICL),("NeuTraL",NeuTraL),("RCA",RCA),("RDP",RDP),("REPEN",REPEN)]:
    try:
        m=M(epochs=30,random_state=0,device='cpu'); m.fit(X); s=m.decision_function(X)
        print(f"  {name:<10}{roc_auc_score(y,s):.3f}   (deep density/SSL)")
    except Exception as e:
        print(f"  {name:<10}fail: {type(e).__name__}: {str(e)[:40]}")

"""Q5: an apples-to-apples 'discovered constraints + calibrated scoring' baseline. We take an
off-the-shelf approximate-FD miner (Desbordante's Pyro) on the dirty table, then score cells
by the SAME violation degree 1-freq(B|A) CPAD uses. This isolates the scoring layer from the
acquisition method: if Pyro+scoring matches CPAD, the gain is in scoring; where CPAD wins, it
is the composite/order/self-supervised acquisition."""
import os, re, sys, tempfile, warnings, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
from desbordante.afd.algorithms import Pyro
RNG=np.random.default_rng(0); CAP=20000; TAU,MG,RATE=0.90,4.0,0.05
DATA=sys.argv[1] if len(sys.argv)>1 else "../data_dq/"

def discover_cpad(df,cols):
    fds=[]
    for B in cols:
        best=None
        for A in cols:
            if A==B: continue
            sub=df[(df[A]!="")&(df[B]!="")]
            if len(sub)<30 or len(sub)/max(1,sub[A].nunique())<MG: continue
            c=sub.groupby(A)[B].transform(lambda s:s.value_counts(normalize=True).max()).mean()
            if c>=TAU and (best is None or c>best[1]): best=(A,c)
        if best: fds.append(([best[0]],B))
    return fds

def discover_pyro(df):
    fp=tempfile.NamedTemporaryFile(suffix=".csv",delete=False,mode="w"); df.to_csv(fp.name,index=False); fp.close()
    p=Pyro(); p.load_data(table=(fp.name,",",True)); p.set_option("error",0.10); p.set_option("max_lhs",2)
    p.execute(); cols=list(df.columns); out=[]
    for fd in p.get_fds():
        s=str(fd); m=re.match(r'\[(.*)\]\s*->\s*(.+)',s)
        if not m: continue
        lhs=[x.strip() for x in m.group(1).replace(',',' ').split() if x.strip()]; rhs=m.group(2).strip()
        if lhs and rhs in cols and all(l in cols for l in lhs): out.append((lhs,rhs))
    os.unlink(fp.name); return out

def score(df,fds,cols,n):
    ci={c:i for i,c in enumerate(cols)}; cell=np.zeros((n,len(cols)))
    for lhs,B in fds:
        try:
            size=df.groupby(lhs)[B].transform("size").values
            own=df.groupby(lhs+[B])[B].transform("size").values
            cell[:,ci[B]]=np.maximum(cell[:,ci[B]],1.0-np.where(size>0,own/size,1.0))
        except Exception: pass
    return cell.max(1)

def run(name,clean):
    clean=clean.astype(str).fillna("").reset_index(drop=True)
    if len(clean)>CAP: clean=clean.sample(CAP,random_state=0).reset_index(drop=True)
    cols=list(clean.columns); ci={c:i for i,c in enumerate(cols)}; n=len(clean)
    fds=discover_cpad(clean,cols)
    if not fds: print(f"{name:<14} no FD"); return
    dirty=clean.copy(); y=np.zeros(n,bool)
    for lhs,B in fds:
        ne=np.where(clean[B].values!="")[0]; sel=ne[RNG.random(len(ne))<RATE]
        fr=clean[B][clean[B]!=""].value_counts(normalize=True); rp=RNG.choice(fr.index.values,len(sel),p=fr.values)
        k=rp!=clean[B].values[sel]; sel,rp=sel[k],rp[k]; dirty.iloc[sel,ci[B]]=rp; y[sel]=True
    cp=roc_auc_score(y,score(dirty,fds,cols,n))
    pf=discover_pyro(dirty); py=roc_auc_score(y,score(dirty,pf,cols,n)) if pf else float('nan')
    print(f"{name:<14}{n:>7}{len(fds):>5}{len(pf):>7}{cp:>9.3f}{py:>9.3f}")

print(f"{'dataset':<14}{'n':>7}{'#cpad':>5}{'#pyro':>7}{'CPAD':>9}{'Pyro+sc':>9}")
print("-"*51)
for nm,p in [("Hospital","hospital_clean_wide.csv"),("Flights","flights/clean.csv"),
             ("Tax","tax/clean.csv"),("Beers","beers/clean.csv")]:
    try: run(nm,pd.read_csv(DATA+p,dtype=str))
    except Exception as e: print(f"{nm}: {type(e).__name__}: {str(e)[:40]}")
try:
    sys.path.insert(0,os.path.dirname(__file__)+"/../benchmarks"); import wikidata_shacl as wd
    t=wd.fetch_entity_table("Q486972",["P17","P131","P30","P421","P37","P1376","P206","P47"],12000)
    run("Wikidata-geo",t[[c for c in t.columns if c.startswith('P') and not c.endswith('_cnt')]])
except Exception as e: print(f"Wikidata: {type(e).__name__}: {str(e)[:50]}")

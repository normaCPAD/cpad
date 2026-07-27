"""
Fast test #3 — do we need CLEAN training data?

We contaminate the training set with a fraction eps of anomalies (rows that
violate the laws) and measure how the null-space detector degrades.
Also test two robust variants:
  - robust covariance (drop the top-residual rows, refit)  [trimming]
  - the "majority clean" assumption pushed to its breaking point.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)

def gen_clean(n):
    region = rng.integers(0, 4, n)
    band   = np.array([10.,20.,30.,40.])[region]
    base   = band + rng.normal(0,0.5,n)
    unit   = 2.0*base + rng.normal(0,0.3,n)          # law: unit = 2*base
    qty    = rng.uniform(1,5,n)
    total  = unit + qty + rng.normal(0,0.3,n)        # law: total = unit + qty
    n1,n2  = rng.normal(0,1,n), rng.normal(0,1,n)
    return np.stack([region.astype(float),base,unit,qty,total,n1,n2],1)

def corrupt_A1(X):
    X = X.copy(); n=len(X)
    col = rng.choice([1,2,3,4]); X[:,col] = X[rng.permutation(n),col]
    return X

def rare_conforming(n):
    region = rng.integers(0,4,n); band=np.array([10.,20.,30.,40.])[region]
    base   = band+rng.normal(0,0.5,n); unit=2.0*base+rng.normal(0,0.3,n)
    qty    = rng.uniform(8,12,n); total=unit+qty+rng.normal(0,0.3,n)
    n1,n2  = rng.normal(0,3,n), rng.normal(0,3,n)
    return np.stack([region.astype(float),base,unit,qty,total,n1,n2],1)

# fixed evaluation pool
N=1000
Xclean_ev = gen_clean(N); Xanom_ev = corrupt_A1(gen_clean(N)); Xrare_ev = rare_conforming(N)
neg_pool = np.vstack([Xclean_ev, Xrare_ev])

def fit_nullspace(Xtr, K=3, trim=0.0):
    """Return scorer. trim>0 => robust: drop top-`trim` residual rows and refit once."""
    mu,sd = Xtr.mean(0), Xtr.std(0); z=lambda X:(X-mu)/sd
    Z = z(Xtr)
    def eig(Z):
        C=np.cov(Z.T); ev,V=np.linalg.eigh(C); return ev[:K], V[:,:K]
    lam,A = eig(Z)
    if trim>0:
        r = np.abs(Z@A)/np.sqrt(np.maximum(lam,1e-6))
        keep = r.max(1) < np.quantile(r.max(1), 1-trim)   # drop worst `trim` fraction
        lam,A = eig(Z[keep])
    lam = np.maximum(lam,1e-6)
    return lambda X: (np.abs(z(X)@A)/np.sqrt(lam)).max(1)

def auc(scorer,pos,neg):
    s=scorer(np.vstack([pos,neg])); y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    return roc_auc_score(y,s)

print("="*82)
print("Contamination sweep — training set = (1-eps) clean + eps anomalies")
print(f"{'eps':>6} | {'plain null-space':^26} | {'robust (trim=eps+5%)':^26}")
print(f"{'':>6} | {'TaskA  Money   FP':^26} | {'TaskA  Money   FP':^26}")
print("-"*82)
for eps in [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40]:
    n=4000; na=int(n*eps)
    Xtr = np.vstack([gen_clean(n-na), corrupt_A1(gen_clean(na))]) if na>0 else gen_clean(n)
    p  = fit_nullspace(Xtr, trim=0.0)
    r  = fit_nullspace(Xtr, trim=min(eps+0.05, 0.5))
    row=lambda f:f"{auc(f,Xanom_ev,Xclean_ev):.3f}  {auc(f,Xanom_ev,neg_pool):.3f}  {auc(f,Xrare_ev,Xclean_ev):.3f}"
    print(f"{eps:>6.2f} | {row(p):^26} | {row(r):^26}")
print("="*82)
print("(TaskA/Money: higher=better. FP: want ~0.500. eps=0 is the clean-data baseline.)")

# ---- worst case: COHERENT contamination (all anomalies break the SAME law) ----
def coherent_anom(n):
    X = gen_clean(n); X[:,4] = X[:,2] + X[:,3] + 8.0    # total = unit+qty+8, systematic
    return X
print("\n" + "="*82)
print("WORST CASE — coherent contamination (all break total=unit+qty the same way)")
print(f"{'eps':>6} | {'plain null-space':^26} | {'robust (trim=eps+5%)':^26}")
print("-"*82)
for eps in [0.0, 0.05, 0.10, 0.20, 0.30]:
    n=4000; na=int(n*eps)
    Xtr = np.vstack([gen_clean(n-na), coherent_anom(na)]) if na>0 else gen_clean(n)
    p = fit_nullspace(Xtr, trim=0.0); r = fit_nullspace(Xtr, trim=min(eps+0.05,0.5))
    row=lambda f:f"{auc(f,Xanom_ev,Xclean_ev):.3f}  {auc(f,Xanom_ev,neg_pool):.3f}  {auc(f,Xrare_ev,Xclean_ev):.3f}"
    print(f"{eps:>6.2f} | {row(p):^26} | {row(r):^26}")
print("="*82)

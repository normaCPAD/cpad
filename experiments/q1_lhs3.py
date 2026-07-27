"""Q1: composite LHS beyond arity 2. Plant a genuine 3-attribute FD (A,B,C)->D (no single or
pair determines D) and let the discrete miner search up to |LHS|=3. Gate/confidence ranking
keeps the search greedy, so it does not enumerate all triples."""
import numpy as np, pandas as pd
from cpad.core.table import Table
from cpad.models.discrete import DiscreteCPAD
from cpad.rules.confidence import fd_confidence
rng=np.random.default_rng(0); n=6000
A,B,C=(rng.integers(0,5,n) for _ in range(3))
Dmap=rng.integers(0,9,125); D=Dmap[A*25+B*5+C]            # D=f(A,B,C), deterministic on the triple only
df=pd.DataFrame({'A':A,'B':B,'C':C,'D':D.astype(int),'N':rng.integers(0,9,n)}).astype(str)
print("single-LHS confidences ->D:", {x:round(fd_confidence(df,[x],'D'),2) for x in 'ABCN'})
print("pair (A,B)->D:", round(fd_confidence(df,['A','B'],'D'),2),
      "| triple (A,B,C)->D:", round(fd_confidence(df,['A','B','C'],'D'),2))
m=DiscreteCPAD(max_lhs=3).fit(Table(df))
for r in m.rules_:
    if r.rhs=='D': print(f"  discovered: ({','.join(r.lhs)}) -> D  conf={r.confidence:.3f}")

"""Q2: runtime/memory profile of the differentiable GATED model as d grows (n fixed). The
per-epoch cost is O(n d^2 d_emb); we measure it and the peak RSS to show where the gated path
is practical and where the discrete scan should be preferred."""
import time, resource, numpy as np, pandas as pd
from cpad.core.table import Table
from cpad.models import GatedCPAD
rng=np.random.default_rng(0); n=4000; EP=30
print(f"GatedCPAD profile, n={n}, {EP} epochs")
print(f"{'d':>5}{'s/epoch':>10}{'total s':>10}{'peak MB':>10}")
for d in [20,50,100,150]:
    data={f'c{i}':rng.integers(0,8,n) for i in range(d)}; data['c1']=data['c0']   # a real FD c0->c1
    df=pd.DataFrame({k:v.astype(str) for k,v in data.items()})
    t=time.time(); GatedCPAD(epochs=EP).fit(Table(df)); dt=time.time()-t
    mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print(f"{d:>5}{dt/EP:>10.2f}{dt:>10.1f}{mb:>10.0f}")

"""Q4: multi-table via a denormalized join. CPAD's single-table machinery applies to a join
view, where it discovers CROSS-table dependencies. customers(city->region) joined with
orders(FK cust_id); errors corrupt region (violating the cross-table FD city->region)."""
import numpy as np, pandas as pd
from cpad.core.table import Table
from cpad.models.discrete import DiscreteCPAD
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(0)
cities=[f"city{i}" for i in range(60)]; city_region={c:f"reg{i%6}" for i,c in enumerate(cities)}
cust_city=rng.choice(cities,2000)
customers=pd.DataFrame({'cust_id':[f"u{i}" for i in range(2000)],'city':cust_city,
                        'region':[city_region[c] for c in cust_city]})
orders=pd.DataFrame({'order_id':[f"o{i}" for i in range(8000)],
                     'cust_id':rng.choice(customers['cust_id'].values,8000),
                     'product':rng.choice([f"p{i}" for i in range(20)],8000)})
joined=orders.merge(customers,on='cust_id').reset_index(drop=True)          # FK join (orders x customers)
y=np.zeros(len(joined),bool); idx=np.where(rng.random(len(joined))<0.05)[0]
repl=rng.choice([f"reg{i}" for i in range(6)],len(idx))
keep=repl!=joined.loc[idx,'region'].values; idx,repl=idx[keep],repl[keep]
joined.loc[idx,'region']=repl; y[idx]=True                                  # cross-table violations
df=joined[['city','region','product']].astype(str); ci={c:i for i,c in enumerate(df.columns)}
m=DiscreteCPAD(max_lhs=2).fit(Table(df))
print("discovered on the JOIN view:", [(tuple(r.lhs),r.rhs,round(r.confidence,2)) for r in m.rules_])
auroc=roc_auc_score(y, m.score(Table(df))[:,ci['region']])
print(f"cross-table FD city->region recovered; detection of injected region violations AUROC={auroc:.3f}")

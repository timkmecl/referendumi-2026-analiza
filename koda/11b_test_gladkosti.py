# Test gladkosti brez modela ozadja (razdelek 4.6): število outlierjev v jedru lambda okoli 1
# v primerjavi s sosednjima pasovoma štirikrat večje skupne širine; gladko ozadje bi bilo v jedru
# sorazmerno širini (pričakovano = sosednji * 1/4), binomski test.
import pandas as pd,numpy as np
from scipy import stats
lam,sig,NUL,res,names=pd.read_pickle('rezultati/lam.pkl'); d=np.abs(lam-1)
for h in [0.1,0.05]:
    core=int(np.sum(d<=h)); side=int(np.sum((d>h)&(d<=5*h)))
    p=stats.binomtest(core,core+side,0.2,alternative='greater').pvalue
    print(f'jedro |λ-1|<={h}: {core};  sosednja pasova {h}<|λ-1|<={5*h}: {side};  pričakovano ob gladkem ozadju {side/4:.1f};  binomski p={p:.1e}')

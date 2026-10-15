# Kakovost napovedi (razdelek 2.4): utežen R^2, uteženi RMSE, mediana absolutnega odstopanja,
# primerjava z binomskim šumom. Uteži: število veljavnih glasov.
import pandas as pd,numpy as np
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); b=pd.read_csv('rezultati/final_table.csv')
K=['PP','RTV','R42','R43']
idx={(o,i):j for j,(o,i) in enumerate(zip(D['Volilni okraj'],D['ID Volišča']))}
out=np.zeros((len(D),4),bool)
for _,x in b.iterrows(): out[idx[(x.okraj,x.id)],K.index(x.ref)]=True
for fe in [False,True]:
    print('=== model', 's FE' if fe else 'brez FE')
    for a,k in enumerate(K):
        p=D['p_'+k].values; ph=R[fe][k]['ph']; n=D['n_'+k].values.astype(float)
        for lab,m in [('brez outlierjev',~out[:,a]),('vse celice',np.ones(len(D),bool))]:
            w=n[m]; e=p[m]-ph[m]; pm=np.average(p[m],weights=w)
            r2=1-np.sum(w*e**2)/np.sum(w*(p[m]-pm)**2); rmse=np.sqrt(np.average(e**2,weights=w))
            mad=np.median(np.abs(e)); binom=np.sqrt(np.average(ph[m]*(1-ph[m])/n[m],weights=w))
            print(f'  {k:4s} {lab:16s} R2={r2:.3f}  RMSE={100*rmse:.2f} o.t.  mediana|odst|={100*mad:.2f} o.t.  binomski šum={100*binom:.2f} o.t.')

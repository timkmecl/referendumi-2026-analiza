# Preverjanje krožnosti čiščenja (razdelek 2.2): nadomestna vrednost brez uporabe ciljnega referenduma.
import pandas as pd, numpy as np
from scipy import stats
src=open('koda/02_model.py').read().split('def run(fe):')[0]
exec(src)
def run2(fe):
    S={(k,j):model(k,[j],D,fe) for k in K for j in K if j!=k}   # single-predictor predictions of k from j
    P1={k:np.median(np.column_stack([S[(k,j)] for j in K if j!=k]),axis=1) for k in K}
    Z1={k:zscore(k,P1[k])[0] for k in K}
    A=np.abs(np.column_stack([Z1[k] for k in K]))
    bad=[(i,A[i].argmax()) for i in np.where(A.max(1)>5)[0]]
    out={}
    for k in K:
        X=D.copy()
        for i,j in bad:
            kj=K[j]
            if kj==k: continue
            # replacement for kj built WITHOUT using target k
            others=[l for l in K if l not in (kj,k)]
            X.loc[i,'p_'+kj]=np.mean([S[(kj,l)][i] for l in others])
        ph=model(k,[j for j in K if j!=k],X,fe); z,zs,c=zscore(k,ph); out[k]=dict(ph=ph,z=z,zs=zs,c=c)
    return out
R2={fe:run2(fe) for fe in [False,True]}
R=pd.read_pickle('rezultati/R.pkl')
def cells(RR,T=5):
    Z=np.column_stack([RR[False][k]['z'] for k in K]); ZF=np.column_stack([RR[True][k]['z'] for k in K])
    ZS=np.column_stack([RR[False][k]['zs'] for k in K]); ZSF=np.column_stack([RR[True][k]['zs'] for k in K])
    m=(np.abs(Z)>T)&(np.abs(ZF)>T); sw=m&(np.abs(ZS)<3)&(np.abs(ZSF)<3)
    return set(zip(*np.where(m))), set(zip(*np.where(sw))), Z
o1,s1,Z1=cells(R); o2,s2,Z2=cells(R2)
print('outlierji: prej',len(o1),'zdaj',len(o2),'skupni',len(o1&o2),'| zamenjave(vkl. Sostro/RTV): prej',len(s1),'zdaj',len(s2),'skupne',len(s1&s2))
print('samo prej:',[(D['Volilni okraj'][i][3:18],D['ID Volišča'][i],K[a],round(Z1[i,a],1),round(Z2[i,a],1)) for i,a in o1-o2])
print('samo zdaj:',[(D['Volilni okraj'][i][3:18],D['ID Volišča'][i],K[a],round(Z1[i,a],1),round(Z2[i,a],1)) for i,a in o2-o1])
print('max |Δz| na vseh celicah:',np.abs(Z1-Z2).max().round(2),' celice z |Δz|>0.5:',np.sum(np.abs(Z1-Z2)>.5))

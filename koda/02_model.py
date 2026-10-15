# Model štirih referendumov (razdelek 2): robustna utežena regresija v logitih, dvostopenjsko čiščenje,
# brez in s fiksnimi učinki okraja (FE). Izpiše c_k in tabelo |z|>3/4/5 (razdelek 3.1). Shrani rezultati/R.pkl.
import pandas as pd, numpy as np
from scipy import stats
exec(open('koda/01_podatki.py').read().split("D=None")[0])   # imports, huber_wls, U,F,K,key
D,_=pd.read_pickle('rezultati/four.pkl')
lg=lambda p: np.log(np.clip(p,.005,.995)/(1-np.clip(p,.005,.995))); il=lambda x:1/(1+np.exp(-x))
FE=pd.get_dummies(D['Volilni okraj'],drop_first=True,dtype=float).values
def model(k, preds, X, fe):
    M=np.column_stack([np.ones(len(D))]+[lg(X['p_'+j]) for j in preds]+([FE] if fe else []))
    y=lg(D['p_'+k]); pc=D['p_'+k].clip(.02,.98); w=D['n_'+k]*pc*(1-pc)
    return il(M@huber_wls(y,M,w))
def zscore(k,ph):
    se=np.sqrt(ph*(1-ph)/D['n_'+k]); raw=(D['p_'+k]-ph)/se
    c=stats.median_abs_deviation(raw,scale='normal'); return raw/c, ((1-D['p_'+k])-ph)/se/c, c
def run(fe):
    # pass 1: robust prediction = median of single-predictor models
    P1={k:np.median(np.column_stack([model(k,[j],D,fe) for j in K if j!=k]),axis=1) for k in K}
    Z1={k:zscore(k,P1[k])[0] for k in K}
    A=np.abs(np.column_stack([Z1[k] for k in K]))
    X=D.copy(); bad=np.zeros_like(A,bool)
    for i in np.where(A.max(1)>5)[0]:
        j=A[i].argmax(); bad[i,j]=True; X.loc[i,'p_'+K[j]]=P1[K[j]][i]
    # pass 2: full model with cleaned predictors
    out={}
    for k in K:
        ph=model(k,[j for j in K if j!=k],X,fe); z,zs,c=zscore(k,ph)
        out[k]=dict(ph=ph,z=z,zs=zs,c=c)
    return out
R={fe:run(fe) for fe in [False,True]}
for fe in R:
    print('\n===','FE' if fe else 'brez FE')
    for k in K:
        z=R[fe][k]['z']; print(k,'c=%.2f'%R[fe][k]['c'],'|z|>3',(abs(z)>3).sum(),'>4',(abs(z)>4).sum(),'>5',(abs(z)>5).sum(),'| Gauss >3: %.1f >4: %.2f'%(len(z)*2*stats.norm.sf(3),len(z)*2*stats.norm.sf(4)),'kurt(<4)',round(stats.kurtosis(z[abs(z)<4]),2))
pd.to_pickle(R,'rezultati/R.pkl')

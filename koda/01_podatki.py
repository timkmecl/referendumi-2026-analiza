# Branje štirih datotek DVK (list "Podatki po voliščih"), združitev po volišču, filter >=10 glasov pri vseh štirih.
# Izpiše nacionalne deleže ZA in korelacije. Shrani rezultati/four.pkl (D = podatki po voliščih; drugi element je
# starejši enostopenjski model, ki ga poročilo ne uporablja). Vsebuje tudi funkcijo huber_wls, ki jo uporabljajo druge skripte.
import pandas as pd, numpy as np, sys
def huber_wls(y,X,w,t=2.0,it=30):
    X=np.asarray(X,float); y=np.asarray(y,float); sw=np.sqrt(np.asarray(w,float))
    Xs=X*sw[:,None]; ys=y*sw; rw=np.ones(len(y))
    for _ in range(it):
        a=np.sqrt(rw); b=np.linalg.lstsq(Xs*a[:,None],ys*a,rcond=None)[0]
        r=ys-Xs@b; s=np.median(np.abs(r))/0.6745; u=np.abs(r)/s
        rw=np.where(u<=t,1.0,t/u)
    return b
from scipy import stats
U='podatki/'
F={'PP':'preiskava.xlsx','RTV':'rtv.xlsx','R42':'sociala.xlsx','R43':'tujci.xlsx'}
K=list(F); key=['Volilni okraj','ID Volišča']
D=None
for k,f in F.items():
    d=pd.read_excel(U+f,sheet_name=1)
    d=d[key+['Volišče','Št. oddanih glasovnic','Glasovi ZA','Glasovi PROTI']].rename(columns={'Št. oddanih glasovnic':'odd_'+k,'Glasovi ZA':'za_'+k,'Glasovi PROTI':'pr_'+k})
    if D is None: D=d
    else: D=D.merge(d.drop(columns='Volišče'),on=key)
for k in K:
    D['n_'+k]=D['za_'+k]+D['pr_'+k]; D['p_'+k]=D['za_'+k]/D['n_'+k]
    print(k,'nacionalno ZA %.1f%%'%(100*D['za_'+k].sum()/D['n_'+k].sum()))
D=D[np.all([D['n_'+k]>=10 for k in K],axis=0)].reset_index(drop=True)
print('volisc',len(D))
print(D[['p_'+k for k in K]].corr().round(3))
lg=lambda p: np.log(np.clip(p,.005,.995)/(1-np.clip(p,.005,.995)))
il=lambda x: 1/(1+np.exp(-x))
def fit(fe, X):
    """X: dataframe of p's used as predictors (possibly cleaned). returns z per k"""
    Z={}; P={}
    for k in K:
        o=[j for j in K if j!=k]
        Xm=pd.DataFrame({j:lg(X['p_'+j]) for j in o})
        if fe: Xm=pd.concat([Xm,pd.get_dummies(D['Volilni okraj'],drop_first=True,dtype=float)],axis=1)
        Xm.insert(0,'const',1.0)
        y=lg(D['p_'+k]); w=D['n_'+k]*D['p_'+k].clip(.02,.98)*(1-D['p_'+k].clip(.02,.98))
        # robust: Huber via RLM on weighted data
        b=huber_wls(y,Xm.values,w)
        ph=il(Xm.values@b)
        se=np.sqrt(ph*(1-ph)/D['n_'+k])
        raw=(D['p_'+k]-ph)/se
        c=stats.median_abs_deviation(raw,scale='normal')
        Z[k]=raw/c; P[k]=ph
        Z[k+'_c']=c
        Z[k+'_swap']=((1-D['p_'+k])-ph)/se/c
    return Z,P
res={}
for fe in [False,True]:
    Z,P=fit(fe,D)
    # pass 2: clean predictors flagged in pass1 -> replace by prediction
    X=D.copy()
    for k in K:
        bad=np.abs(Z[k])>5; X.loc[bad,'p_'+k]=P[k][bad]
    Z,P=fit(fe,X)
    res[fe]=(Z,P)
    print('\n=== FE' if fe else '\n=== brez FE', {k:round(Z[k+'_c'],2) for k in K})
    for k in K:
        z=Z[k]; print(k,'|z|>3',(abs(z)>3).sum(),'>4',(abs(z)>4).sum(),'>5',(abs(z)>5).sum(),'kurt(|z|<5)',round(stats.kurtosis(z[abs(z)<5]),2),'Gauss>3',round(len(z)*2*stats.norm.sf(3),1))
pd.to_pickle((D,res),'rezultati/four.pkl')

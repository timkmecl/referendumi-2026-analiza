# Robustnost mešanega modela na velikost volišča (razdelek 4.6): ujemanje velikosti in "ista volišča".
# Izpiše tudi različici vzorčenja ∝ n in ∝ n², ki v poročilu nista navedeni.
import pandas as pd,numpy as np
from scipy import stats, optimize
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); b=pd.read_csv('rezultati/final_table.csv'); K=['PP','RTV','R42','R43']; r=R[False]
lg=lambda p: np.log(np.clip(p,.005,.995)/(1-np.clip(p,.005,.995))); il=lambda x:1/(1+np.exp(-x))
PH=np.column_stack([r[k]['ph'] for k in K]); C=np.array([r[k]['c'] for k in K]); N=np.column_stack([D['n_'+k] for k in K]).astype(float); P=np.column_stack([D['p_'+k] for k in K])
lam,sig,NUL,res,names=pd.read_pickle('rezultati/lam.pkl')
idx={(o,i):j for j,(o,i) in enumerate(zip(D['Volilni okraj'],D['ID Volišča']))}
I=np.array([idx[(x.okraj,x.id)] for _,x in b.iterrows()]); A=np.array([K.index(x.ref) for _,x in b.iterrows()])
nobs=N[I,A]
rng=np.random.default_rng(42)
def shift(p0,kind,M):
    if kind=='B': return il(lg(p0)+0.7*stats.t.rvs(3,size=M,random_state=rng))
    if kind=='C': return rng.uniform(0,1,M)
    if kind=='T':
        f=np.exp(rng.uniform(-1.5,1.5,M)); d=rng.random(M)<.5; za=p0; pr=1-p0
        za=np.where(d,za*f,za); pr=np.where(d,pr,pr*f); return za/(za+pr)
def sim(kind,weight,M=1500000):
    flatN=N.ravel(); w=flatN**weight; w=w/w.sum()
    j=rng.choice(flatN.size,M,p=w); i=j//4; a=j%4
    ph=PH[i,a]; p0=P[i,a]; n=N[i,a]; p=shift(p0,kind,M)
    se=np.sqrt(ph*(1-ph)/n)*C[a]; m=np.abs((p-ph)/se)>5
    return ((p-ph)/(1-2*ph))[m], (se/np.abs(1-2*ph))[m], n[m]
def matched(l,s,n):
    # reweight to match observed outlier size distribution (log-n histogram)
    bins=np.quantile(np.log(nobs),np.linspace(0,1,7)); bins[0]=-1; bins[-1]=99
    ho=np.histogram(np.log(nobs),bins)[0]/len(nobs); hs=np.histogram(np.log(n),bins)[0]/len(n)
    wt=(ho/np.maximum(hs,1e-9))[np.clip(np.digitize(np.log(n),bins)-1,0,5)]
    return wt/wt.sum()
def fit(l,s,wt):
    sub=rng.choice(len(l),40000,p=wt); kde=stats.gaussian_kde(np.clip(l[sub],-3,4),bw_method=.15)
    h=kde(np.clip(lam,-3,4)); f=stats.norm.pdf(lam,1,np.sqrt(sig**2+.02**2))
    nll=lambda pi:-np.sum(np.log(pi*f+(1-pi)*h))
    o=optimize.minimize_scalar(nll,bounds=(1e-4,1-1e-4),method='bounded'); g=np.linspace(.001,.999,999); ll=np.array([-nll(x) for x in g]); ok=g[ll>=ll.max()-1.92]
    return o.x,ok.min(),ok.max(),2*(nll(1e-4)-o.fun)
rows=[]
for kind in ['B','C','T']:
    for wname,weight in [('enakomerno',0),('∝ n',1),('∝ n²',2)]:
        l,s,n=sim(kind,weight)
        for match in [False,True]:
            wt=matched(l,s,n) if match else np.ones(len(l))/len(l)
            core=np.sum(wt*(np.abs(l-1)<=.1)); band=np.sum(wt*(np.abs(l-1)<2*s))
            pi,lo,hi,lr=fit(l,s,wt)
            rows.append(dict(ozadje=kind,vzorcenje=wname,ujemanje_velikosti=match,med_n=np.median(n) if not match else float(np.exp(np.sum(wt*np.log(n)))),
                E_jedro=53*core,E_pas2s=53*band,pi=pi,IZ=f'{53*lo:.0f}-{53*hi:.0f}',chi2=lr))
T=pd.DataFrame(rows); pd.set_option('display.width',200); print(T.round(2).to_string())
print('opazeno: jedro [0.9,1.1]=',np.sum(np.abs(lam-1)<=.1),' |λ-1|<2σ=',np.sum(np.abs(lam-1)<2*sig),' mediana n outlierjev',np.median(nobs))
# same-stations null: keep exactly the 53 observed cells, draw peculiarity until |z|>5
for kind in ['B','C','T']:
    L=[];S=[]
    for i,a in zip(I,A):
        ph=PH[i,a]; p0=P[i,a]; n=N[i,a]; se=np.sqrt(ph*(1-ph)/n)*C[a]
        # use p0 = predicted (true w/o peculiarity) to avoid using observed
        p=shift(np.full(200000,ph),kind,200000); z=(p-ph)/se; p=p[np.abs(z)>5]
        L.append((p-ph)/(1-2*ph)); S.append(se/abs(1-2*ph))
    Ecore=sum(np.mean(np.abs(x-1)<=.1) for x in L); Eband=sum(np.mean(np.abs(x-1)<2*s) for x,s in zip(L,S))
    # distribution of count via Poisson-binomial
    pc=np.array([np.mean(np.abs(x-1)<=.1) for x in L]); d=np.array([1.]);
    for q in pc: d=np.convolve(d,[1-q,q])
    print(f'isti volišča, ozadje {kind}: E[jedro]={Ecore:.1f}  E[|λ-1|<2σ]={Eband:.1f}  P(jedro>=17)={d[17:].sum():.1e}')

# Nekrožna analiza vseh 53 outlierjev (razdelek 4.6): lambda, štiri ničelna ozadja, mešani model, LR test. Shrani rezultati/lam.pkl.
import pandas as pd,numpy as np
from scipy import stats, optimize
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); b=pd.read_csv('rezultati/final_table.csv'); K=['PP','RTV','R42','R43']; r=R[False]
idx={(o,i):j for j,(o,i) in enumerate(zip(D['Volilni okraj'],D['ID Volišča']))}
lg=lambda p: np.log(np.clip(p,.005,.995)/(1-np.clip(p,.005,.995))); il=lambda x:1/(1+np.exp(-x))
PH=np.column_stack([r[k]['ph'] for k in K]); C=np.array([r[k]['c'] for k in K]); N=np.column_stack([D['n_'+k] for k in K]).astype(float); P=np.column_stack([D['p_'+k] for k in K])
# observed: all 53 outliers (selected only by |z|>5)
lam=[];sig=[]
for _,x in b.iterrows():
    i=idx[(x.okraj,x.id)]; a=K.index(x.ref); ph=PH[i,a]; se=np.sqrt(ph*(1-ph)/N[i,a])*C[a]
    lam.append((P[i,a]-ph)/(1-2*ph)); sig.append(se/abs(1-2*ph))
lam=np.array(lam); sig=np.array(sig)
print('outlierjev',len(lam)); print('lambda kvantili',np.round(np.percentile(lam,[5,25,50,75,95]),2))
print('|lambda-1|<2sig:',np.sum(np.abs(lam-1)<2*sig),' lambda v [0.9,1.1]:',np.sum(np.abs(lam-1)<.1))
rng=np.random.default_rng(11); M=600000
def simnull(kind):
    i=rng.integers(len(D),size=M); a=rng.integers(4,size=M); ph=PH[i,a]; p0=P[i,a]; n=N[i,a]; c=C[a]
    if kind=='B': p=il(lg(p0)+0.7*stats.t.rvs(3,size=M,random_state=rng))
    elif kind=='C': p=rng.uniform(0,1,M)
    elif kind=='P':  # partial misallocation: fraction f of one side's ballots counted to the other side
        f=rng.uniform(0,1,M); d=rng.random(M)<.5
        p=np.where(d,p0*(1-f),p0+f*(1-p0))
    elif kind=='T':  # typo in one count: ZA or PROTI count off by random factor
        za=p0*n; pr=(1-p0)*n; fac=np.exp(rng.uniform(-1.5,1.5,M)); d=rng.random(M)<.5
        za=np.where(d,za*fac,za); pr=np.where(d,pr,pr*fac); p=za/(za+pr)
    se=np.sqrt(ph*(1-ph)/n)*c; z=(p-ph)/se; m=np.abs(z)>5
    return ((p-ph)/(1-2*ph))[m], (se/np.abs(1-2*ph))[m]
NUL={k:simnull(k) for k in ['B','C','P','T']}
names={'B':'lokalni učinek (t3 v logitih)','C':'naključen delež','P':'del glasovnic v napačni kup','T':'napaka v enem številu'}
for k,(l,s) in NUL.items():
    print(f"{names[k]:28s} P(|λ-1|<2σ)={np.mean(np.abs(l-1)<2*s):.3f}  -> pričak. od 53: {53*np.mean(np.abs(l-1)<2*s):.1f}  | P(λ∈[0.9,1.1])={np.mean(np.abs(l-1)<.1):.3f}")
# mixture ML: pi * N(lam;1,sig) + (1-pi)*h(lam), h = KDE of null lambdas (clipped)
res={}
for k,(l,s) in NUL.items():
    lc=np.clip(l,-3,4); kde=stats.gaussian_kde(lc[rng.choice(len(lc),20000)],bw_method=.15)
    h=kde(np.clip(lam,-3,4)); f=stats.norm.pdf(lam,1,np.sqrt(sig**2+0.02**2))
    nll=lambda pi: -np.sum(np.log(pi*f+(1-pi)*h))
    o=optimize.minimize_scalar(nll,bounds=(1e-4,1-1e-4),method='bounded'); pi=o.x
    # profile likelihood 95% CI
    grid=np.linspace(.001,.999,999); ll=np.array([-nll(g) for g in grid]); ok=grid[ll>=ll.max()-1.92]
    lr=2*(nll(1e-4)-o.fun)
    res[k]=(pi,ok.min(),ok.max(),lr)
    print(f"mešanica z ozadjem '{names[k]}': delež zamenjav π={pi:.2f} [{ok.min():.2f},{ok.max():.2f}] -> {53*pi:.1f} [{53*ok.min():.1f},{53*ok.max():.1f}] | LR test π=0: χ²={lr:.0f}")
pd.to_pickle((lam,sig,NUL,res,names),'rezultati/lam.pkl')

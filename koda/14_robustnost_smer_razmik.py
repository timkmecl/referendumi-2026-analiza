# Pragovi |z|>5/7/10: razmik in nepogojni test smeri (štetje, vsota) z enakomernim vzorčenjem in z ujemanjem velikosti
# (razdelka 4.6 in 5.2). V poročilu je uporabljen le prag 5.
import pandas as pd,numpy as np
from scipy import stats, optimize
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); K=['PP','RTV','R42','R43']
lg=lambda p: np.log(np.clip(p,.005,.995)/(1-np.clip(p,.005,.995))); il=lambda x:1/(1+np.exp(-x))
r=R[False]; rf=R[True]
PH=np.column_stack([r[k]['ph'] for k in K]); C=np.array([r[k]['c'] for k in K]); N=np.column_stack([D['n_'+k] for k in K]).astype(float); P=np.column_stack([D['p_'+k] for k in K])
Z=np.column_stack([r[k]['z'] for k in K]); ZF=np.column_stack([rf[k]['z'] for k in K])
SE=np.sqrt(PH*(1-PH)/N)*C[None,:]; SEP=np.abs(1-2*PH)/SE; sgn=np.array([-1,-1,1,1])
rng=np.random.default_rng(7); M=2000000
fi=rng.integers(N.size,size=M); I0=fi//4; A0=fi%4
ph0=PH[I0,A0]; p00=P[I0,A0]; n0=N[I0,A0]; se0=SE[I0,A0]; sep0=SEP[I0,A0]
def shift(kind):
    if kind=='B': return il(lg(p00)+0.7*stats.t.rvs(3,size=M,random_state=rng))
    if kind=='C': return rng.uniform(0,1,M)
    if kind=='T':
        f=np.exp(rng.uniform(-1.5,1.5,M)); d=rng.random(M)<.5; za=np.where(d,p00*f,p00); pr=np.where(d,1-p00,(1-p00)*f); return za/(za+pr)
PS={k:shift(k) for k in 'BCT'}; PS['S']=1-p00
def matchw(nsim,nobs,nb):
    qb=np.quantile(np.log(nobs),np.linspace(0,1,nb+1)); qb[0]=-1; qb[-1]=99
    ho=np.histogram(np.log(nobs),qb)[0]/len(nobs); hs=np.histogram(np.log(nsim),qb)[0]/len(nsim)
    w=(ho/np.maximum(hs,1e-12))[np.clip(np.digitize(np.log(nsim),qb)-1,0,nb-1)]; return w/w.sum()
def mixfit(x,fb,fs):
    nll=lambda pi:-np.sum(np.log(pi*fs+(1-pi)*fb)); o=optimize.minimize_scalar(nll,bounds=(1e-4,1-1e-4),method='bounded')
    g=np.linspace(.001,.999,999); ll=np.array([-nll(v) for v in g]); ok=g[ll>=ll.max()-1.92]
    return o.x,ok.min(),ok.max(),2*(nll(1e-4)-o.fun)
out=[]
for T in [5,7,10]:
    m=(np.abs(Z)>T)&(np.abs(ZF)>T); I,A=np.where(m); n=len(I)
    sobs=SEP[I,A]; nob=N[I,A]; lam=(P[I,A]-PH[I,A])/(1-2*PH[I,A]); e=(P[I,A]-PH[I,A])*N[I,A]*sgn[A]
    nb=6 if n>40 else 4
    print(f'\n===== prag |z|>{T}: outlierjev {n}, λ∈[0.9,1.1]: {np.sum(np.abs(lam-1)<=.1)} ({np.mean(np.abs(lam-1)<=.1):.0%}), razmik<{T}: {np.sum(sobs<T)}')
    print(f'   smer: v škodo zmagovalca {np.sum(e<0)}/{n}, sprememba prednosti {2*e.sum():.0f}')
    sel={}
    for k in 'BCTS':
        z=(PS[k]-ph0)/se0; mm=np.abs(z)>T
        sel[k]=dict(sep=sep0[mm],n=n0[mm],e=((PS[k]-ph0)*n0*sgn[A0])[mm],a=A0[mm])
    for var in ['enakomerno','ujemanje velikosti']:
        W={k:(np.ones(len(sel[k]['n']))/len(sel[k]['n']) if var=='enakomerno' else matchw(sel[k]['n'],nob,nb)) for k in sel}
        # separation test
        sub=lambda k,s=40000: rng.choice(len(sel[k]['sep']),s,p=W[k])
        ks=stats.gaussian_kde(np.log(sel['S']['sep'][sub('S')]))
        res=[]
        for k in 'BCT':
            sb=sel[k]['sep'][sub(k,200000)]
            mw=stats.mannwhitneyu(sobs,sb,alternative='greater').pvalue
            kb=stats.gaussian_kde(np.log(sb[:40000]))
            pi,lo,hi,chi=mixfit(sobs,kb(np.log(sobs)),ks(np.log(sobs)))
            res.append((k,np.median(sb),mw,pi,lo,hi,chi))
        print(f'  [{var}] RAZMIK: opaž. mediana {np.median(sobs):.1f}; ' + ' | '.join(f'{k}: med {mb:.1f} MW p={mw:.3f} π={pi:.2f} [{lo:.2f}-{hi:.2f}] χ²={chi:.1f}' for k,mb,mw,pi,lo,hi,chi in res))
        # direction test: null B vs swap mixture, matched per referendum counts
        cnt=[np.sum(A==a) for a in range(4)]
        pools={k:[sel[k]['e'][sel[k]['a']==a] for a in range(4)] for k in 'BS'}
        wts={k:[W[k][sel[k]['a']==a]/W[k][sel[k]['a']==a].sum() for a in range(4)] for k in 'BS'}
        def simmix(pi,M2=40000):
            s=np.zeros(M2); h=np.zeros(M2)
            for a in range(4):
                c=cnt[a]
                if c==0: continue
                mk=rng.random((M2,c))<pi
                db=rng.choice(pools['B'][a],(M2,c),p=wts['B'][a]); ds=rng.choice(pools['S'][a],(M2,c),p=wts['S'][a])
                d=np.where(mk,ds,db); s+=d.sum(1); h+=(d<0).sum(1)
            return s,h
        grid=np.linspace(0,1,21); okS=[];okH=[];med=[]
        for pi in grid:
            s,h=simmix(pi)
            if np.mean(s<=e.sum())>.025 and np.mean(s>=e.sum())>.025: okS.append(pi)
            if np.mean(h<=np.sum(e<0))>.025 and np.mean(h>=np.sum(e<0))>.025: okH.append(pi)
            med.append((pi,2*np.median(s),np.median(h)))
        s0,h0=simmix(0,200000); s1,h1=simmix(1,200000)
        best=min(med,key=lambda t:abs(t[1]-2*e.sum()))[0]
        print(f'  [{var}] SMER: π=0: E[Δ]={2*s0.mean():.0f} [{2*np.percentile(s0,2.5):.0f},{2*np.percentile(s0,97.5):.0f}] P(≤obs)={np.mean(s0<=e.sum()):.3f}; E[škoda]={h0.mean():.1f} P(≥obs)={np.mean(h0>=np.sum(e<0)):.3f} | π=1: E[Δ]={2*s1.mean():.0f}, E[škoda]={h1.mean():.1f} | π iz Δ≈{best:.2f}, 95% IZ π (Δ): [{min(okS):.2f},{max(okS):.2f}], (škoda): [{min(okH):.2f},{max(okH):.2f}]')

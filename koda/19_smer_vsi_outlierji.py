# Nepogojni test smeri čez vse outlierje (razdelek 5.2) in kontrola na celicah 3<|z|<5. Shrani rezultati/dirall.pkl.
import pandas as pd,numpy as np
from scipy import stats
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); b=pd.read_csv('rezultati/final_table.csv'); K=['PP','RTV','R42','R43']; r=R[False]
lg=lambda p: np.log(np.clip(p,.005,.995)/(1-np.clip(p,.005,.995))); il=lambda x:1/(1+np.exp(-x))
sgn=np.array([-1,-1,1,1])   # winner: PP PROTI, RTV PROTI, R42 ZA, R43 ZA
PH=np.column_stack([r[k]['ph'] for k in K]); C=np.array([r[k]['c'] for k in K]); N=np.column_stack([D['n_'+k] for k in K]).astype(float); P=np.column_stack([D['p_'+k] for k in K])
Z=np.column_stack([r[k]['z'] for k in K])
idx={(o,i):j for j,(o,i) in enumerate(zip(D['Volilni okraj'],D['ID Volišča']))}
cells=[(idx[(x.okraj,x.id)],K.index(x.ref)) for _,x in b.iterrows()]
I=np.array([c[0] for c in cells]); A=np.array([c[1] for c in cells])
e=(P[I,A]-PH[I,A])*N[I,A]*sgn[A]     # published - predicted winner votes; <0 hurts winner
print('outlierjev',len(e),' v škodo zmagovalca:',(e<0).sum(),' vsota učinka na prednost (2e):',round(2*e.sum()))
for a,k in enumerate(K):
    m=A==a; print(f'  {k}: n={m.sum()} škoda={np.sum(e[m]<0)} 2*vsota={2*e[m].sum():.0f}')
# null 1: sign flip
rng=np.random.default_rng(3); M=200000
flip=rng.choice([-1,1],(M,len(e)))
S1=(flip*np.abs(e)).sum(1); print('null sign-flip: P(vsota<=obs)=%.4f'%np.mean(S1<=e.sum()), ' P(škoda>=obs)=%.4f'%stats.binomtest(int((e<0).sum()),len(e),.5,alternative='greater').pvalue)
# null 2: realistic peculiarities (t3 logit shift) on random cells, keep |z|>5, matched per referendum
def null_pool(a,kind='B',M=400000):
    i=rng.integers(len(D),size=M); ph=PH[i,a]; p0=P[i,a]; n=N[i,a]
    if kind=='B': p=il(lg(p0)+0.7*stats.t.rvs(3,size=M,random_state=rng))
    elif kind=='SW': p=1-p0
    se=np.sqrt(ph*(1-ph)/n)*C[a]; z=(p-ph)/se; m=np.abs(z)>5
    return ((p-ph)*n*sgn[a])[m]
for kind in ['B','SW']:
    pools=[null_pool(a,kind) for a in range(4)]
    cnt=[np.sum(A==a) for a in range(4)]
    sims=np.zeros(M); hurts=np.zeros(M)
    for a in range(4):
        d=rng.choice(pools[a],(M,cnt[a])); sims+=d.sum(1); hurts+=(d<0).sum(1)
    print(f'{"null lokalni učinek" if kind=="B" else "100% zamenjave"}: E[2*vsota]={2*sims.mean():.0f} [{2*np.percentile(sims,2.5):.0f},{2*np.percentile(sims,97.5):.0f}]  E[škoda]={hurts.mean():.1f} [{np.percentile(hurts,2.5):.0f},{np.percentile(hurts,97.5):.0f}]  P(vsota<=obs)={np.mean(sims<=e.sum()):.4f} P(škoda>=obs)={np.mean(hurts>=(e<0).sum()):.4f}')
    if kind=='B': SB,HB=sims,hurts
    else: SS,HS=sims,hurts
# estimate pi by moments on count and sum, CI by simulation over pi grid
obsS=e.sum(); obsH=(e<0).sum()
poolsB=[null_pool(a,'B') for a in range(4)]; poolsS=[null_pool(a,'SW') for a in range(4)]
grid=np.linspace(0,1,21); acc=[]
for pi in grid:
    M2=40000; s=np.zeros(M2); h=np.zeros(M2)
    for a in range(4):
        c=np.sum(A==a); ns=rng.binomial(c,pi,M2)
        for j in range(M2):
            pass
    acc.append(pi)
# faster: per-draw mixture via mask
def sim_mix(pi,M2=60000):
    s=np.zeros(M2); h=np.zeros(M2)
    for a in range(4):
        c=np.sum(A==a); mask=rng.random((M2,c))<pi
        d=np.where(mask,rng.choice(poolsS[a],(M2,c)),rng.choice(poolsB[a],(M2,c)))
        s+=d.sum(1); h+=(d<0).sum(1)
    return s,h
rows=[]
for pi in grid:
    s,h=sim_mix(pi); rows.append((pi,np.mean(s<=obsS),np.mean(s>=obsS),np.mean(h<=obsH),np.mean(h>=obsH),2*s.mean(),h.mean()))
T=pd.DataFrame(rows,columns=['pi','PsLE','PsGE','PhLE','PhGE','E2sum','Ehurt'])
ok_s=T[(T.PsLE>.025)&(T.PsGE>.025)].pi; ok_h=T[(T.PhLE>.025)&(T.PhGE>.025)].pi
print(T.round(3).to_string())
print('95% IZ za π iz vsote:',ok_s.min(),ok_s.max(),' iz štetja:',ok_h.min(),ok_h.max())
# control: 3<|z|<5 cells
m=(np.abs(Z)>3)&(np.abs(Z)<5); ii,aa=np.where(m)
ec=(P[ii,aa]-PH[ii,aa])*N[ii,aa]*sgn[aa]
print('kontrola 3<|z|<5: n=',len(ec),' škoda',(ec<0).sum(),'(%.0f%%)'%(100*np.mean(ec<0)),' 2*vsota',round(2*ec.sum()))
pd.to_pickle((e,A,SB,HB,SS,HS,T,ec),'rezultati/dirall.pkl')

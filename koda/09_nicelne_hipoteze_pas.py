# Ničelne hipoteze A/B/C za pas |z_zam|<3 (razdelek 4.2) in razporeditev znotraj pasu (4.3). Shrani rezultati/band.pkl.
import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); b=pd.read_csv('rezultati/final_table.csv'); K=['PP','RTV','R42','R43']
r=R[False]; idx={(o,i):j for j,(o,i) in enumerate(zip(D['Volilni okraj'],D['ID Volišča']))}
lg=lambda p: np.log(np.clip(p,.005,.995)/(1-np.clip(p,.005,.995))); il=lambda x:1/(1+np.exp(-x))
O=[]
for _,x in b.iterrows():
    i=idx[(x.okraj,x.id)]; k=x.ref
    O.append(dict(i=i,k=k,p=D['p_'+k][i],ph=r[k]['ph'][i],z=r[k]['z'][i],zs=r[k]['zs'][i],sw=x.tipt.startswith('zamenjava')))
O=pd.DataFrame(O)
print('opazeno |zs|: 0-1',(O.zs.abs()<1).sum(),'1-2',((O.zs.abs()>=1)&(O.zs.abs()<2)).sum(),'2-3',((O.zs.abs()>=2)&(O.zs.abs()<3)).sum(),'>=3',(O.zs.abs()>=3).sum())
rng=np.random.default_rng(7)
# precompute arrays
PH=np.column_stack([r[k]['ph'] for k in K]); C=np.array([r[k]['c'] for k in K]); N=np.column_stack([D['n_'+k] for k in K]).astype(float)
P=np.column_stack([D['p_'+k] for k in K])
def sim(kind,M=400000):
    i=rng.integers(len(D),size=M); k=rng.integers(4,size=M)
    ph=PH[i,k]; n=N[i,k]; c=C[k]; p0=P[i,k]
    se=np.sqrt(ph*(1-ph)/n)*c
    if kind=='A':   # velikosti odstopanj VSEH 53 outlierjev (o.t.), naključen predznak; brez izbora po pasu, zato ni krožno
        mags=np.abs((O.p-O.ph).values)
        p=np.clip(p0+rng.choice(mags,M)*rng.choice([-1,1],M),0,1)
    elif kind=='B': # real local effect: logit shift, heavy-tailed (t3, scale 0.7)
        p=il(lg(p0)+0.7*stats.t.rvs(3,size=M,random_state=rng))
    elif kind=='C': # uniform random share
        p=rng.uniform(0,1,M)
    z=(p-ph)/se; zs=((1-p)-ph)/se; m=np.abs(z)>5
    return zs[m]
res={}
for kd,name in [('A','napake velikosti vseh outlierjev'),('B','lokalni učinek (t3 v logitih)'),('C','povsem naključen rezultat')]:
    zs=sim(kd); band=np.abs(zs)<3
    q=[np.mean(np.abs(zs)<1),np.mean((np.abs(zs)>=1)&(np.abs(zs)<2)),np.mean((np.abs(zs)>=2)&(np.abs(zs)<3))]
    res[kd]=(name,band.mean(),q,zs)
    print(name,'P(pas)=%.3f'%band.mean(),'pricak od 53: %.1f'%(53*band.mean()),' po celicah 0-1/1-2/2-3: '+'/'.join(f'{53*v:.1f}' for v in q), 'Poisson P(>=27)=%.1e'%stats.poisson.sf(26,53*band.mean()))
nsw=O.sw.sum(); g=[stats.norm.cdf(1)*2-1,(stats.norm.cdf(2)-stats.norm.cdf(1))*2,(stats.norm.cdf(3)-stats.norm.cdf(2))*2]
print('ce so zamenjave prave -> N(0,1): pri',nsw,'pricak 0-1/1-2/2-3:',[round(nsw*v,1) for v in g])
pd.to_pickle((O,res),'rezultati/band.pkl')

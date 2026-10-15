# Slika 15_smer_vsi_outlierji (razdelek 5.2).
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
exec(open('koda/14_robustnost_smer_razmik.py').read().split('out=[]')[0])
RED='#E45756'; GRY='#9AA5B1'; BLUE='#4C78A8'; DG='#555555'
plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False})
T=5; m=(np.abs(Z)>T)&(np.abs(ZF)>T); I,A=np.where(m); n=len(I); nob=N[I,A]; nb=6
e=(P[I,A]-PH[I,A])*N[I,A]*sgn[A]; obs=2*e.sum(); h=int(np.sum(e<0)); cnt=[np.sum(A==a) for a in range(4)]
sel={}
for k in 'BS':
    zz=(PS[k]-ph0)/se0; mm=np.abs(zz)>T; sel[k]=dict(n=n0[mm],e=((PS[k]-ph0)*n0*sgn[A0])[mm],a=A0[mm])
def make(var):
    W={k:(np.ones(len(sel[k]['n'])) if var=='u' else matchw(sel[k]['n'],nob,nb)) for k in sel}
    pools={k:[sel[k]['e'][sel[k]['a']==a] for a in range(4)] for k in 'BS'}
    wts={k:[W[k][sel[k]['a']==a]/W[k][sel[k]['a']==a].sum() for a in range(4)] for k in 'BS'}
    q={k:np.array([np.sum(wts[k][a]*(pools[k][a]<0)) for a in range(4)]) for k in 'BS'}
    def sm(pi,M2=40000):
        s=np.zeros(M2)
        for a in range(4):
            c=cnt[a]; mk=rng.random((M2,c))<pi
            s+=np.where(mk,rng.choice(pools['S'][a],(M2,c),p=wts['S'][a]),rng.choice(pools['B'][a],(M2,c),p=wts['B'][a])).sum(1)
        return 2*s
    def dist(pi):
        d=np.array([1.])
        for a in range(4): d=np.convolve(d,stats.binom.pmf(np.arange(cnt[a]+1),cnt[a],pi*q['S'][a]+(1-pi)*q['B'][a]))
        return d
    return sm,dist
U=make('u'); Mm=make('m'); grid=np.linspace(0,1,41)
fig,axs=plt.subplots(2,2,figsize=(12,8.5),gridspec_kw={'width_ratios':[1.25,1]})
ax=axs[0,0]; bins=np.linspace(-14000,7000,85)
ax.hist(U[0](0,150000),bins=bins,density=True,color=GRY,alpha=.7,label='brez zamenjav (π = 0)')
ax.hist(U[0](1,150000),bins=bins,density=True,color=BLUE,alpha=.45,label='same zamenjave (π = 1)')
for pi,c in [(0,DG),(1,BLUE)]:
    hh,_=np.histogram(Mm[0](pi,150000),bins=bins,density=True); ax.step(bins[:-1],hh,where='post',color=c,ls='--',lw=1.2)
ax.axvline(obs,color=RED,lw=2.5,label=f'opaženo: {obs:,.0f}'.replace(',','.')); ax.axvline(0,color='k',lw=.7)
ax.plot([],[],color=DG,ls='--',label='črtkano: ujemanje velikosti'); ax.legend(frameon=False,fontsize=8.5,loc='upper left'); ax.set_yticks([])
ax.set_xlabel('sprememba prednosti zmagovalca (objavljeno − napoved, glasovi)'); ax.set_title('Vsota: sprememba prednosti zmagovalca',loc='left',weight='bold')
ax=axs[1,0]; x=np.arange(n+1); w=.4
ax.bar(x,U[1](0),1.0,color=GRY,alpha=.7,label='brez zamenjav (π = 0)'); ax.bar(x,U[1](1),1.0,color=BLUE,alpha=.45,label='same zamenjave (π = 1)')
for pi,c in [(0,DG),(1,BLUE)]: ax.step(x,Mm[1](pi),where='mid',color=c,ls='--',lw=1.2)
ax.axvline(h,color=RED,lw=2.5,label=f'opaženo: {h}'); ax.axvline(n/2,color='k',lw=.7,ls=':')
ax.set_xlim(n*.25,n*.9); ax.set_yticks([]); ax.legend(frameon=False,fontsize=8.5,loc='upper left')
ax.set_xlabel('število outlierjev, ki odstopajo v škodo zmagovalca'); ax.set_title('Štetje: smer posameznih outlierjev',loc='left',weight='bold')
for row,(kind) in enumerate(['sum','cnt']):
    ax=axs[row,1]
    for (sm,dist),c,ls,lab in [(U,BLUE,'-','enakomerno'),(Mm,DG,'--','ujemanje velikosti')]:
        if kind=='sum': Q=np.array([np.percentile(sm(pi,20000),[2.5,50,97.5]) for pi in grid])
        else:
            Q=[]
            for pi in grid:
                d=dist(pi); cd=np.cumsum(d); Q.append((np.searchsorted(cd,.025),np.dot(np.arange(n+1),d),np.searchsorted(cd,.975)))
            Q=np.array(Q)
        ax.fill_between(grid,Q[:,0],Q[:,2],color=c,alpha=.12); ax.plot(grid,Q[:,1],color=c,ls=ls,lw=1.8,label=f'pričakovano in 95 %: {lab}')
    ax.axhline(obs if kind=='sum' else h,color=RED,lw=2,label='opaženo')
    ax.axhline(0 if kind=='sum' else n/2,color='k',lw=.7,ls='-' if kind=='sum' else ':')
    ax.set_xlim(0,1); ax.set_ylabel('Δ prednosti zmagovalca' if kind=='sum' else 'št. v škodo zmagovalca')
    if row==0: ax.legend(frameon=False,fontsize=8.5,loc='lower left')
    if row==1: ax.set_xlabel('delež zamenjav med outlierji π')
fig.suptitle('Smer odstopanj vseh 53 outlierjev (|z| > 5, brez razvrščanja v zamenjave)',x=.02,ha='left',fontsize=12.5,weight='bold')
plt.tight_layout(rect=(0,0,1,.96)); plt.savefig('slike/15_smer_vsi_outlierji.png',dpi=140)

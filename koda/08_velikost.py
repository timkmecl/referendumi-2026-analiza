# Velikost volišč (razdelek 3.4, slika 6 = 13_velikost.png): zaznavne celice, MW/KS test, kvintili, hipotezi na celico / na glas.
import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); b=pd.read_csv('rezultati/final_table.csv'); K=['PP','RTV','R42','R43']; r=R[False]
idx={(o,i):j for j,(o,i) in enumerate(zip(D['Volilni okraj'],D['ID Volišča']))}
rows=[]
for a,k in enumerate(K):
    ph=r[k]['ph']; n=D['n_'+k].values; se=np.sqrt(ph*(1-ph)/n)*r[k]['c']
    sep=np.abs(1-2*ph)/se
    for i in range(len(D)): rows.append((i,k,n[i],sep[i],D['ID Volišča'][i]>=900))
C=pd.DataFrame(rows,columns=['i','k','n','sep','pred'])
lab={(idx[(x.okraj,x.id)],x.ref):x.tipt for _,x in b.iterrows()}
C['tip']=[lab.get((i,k),'') for i,k in zip(C.i,C.k)]
C['swap']=C.tip=='zamenjava ZA↔PROTI'; C['other']=(C.tip!='')&~C.tip.str.startswith('zamenjava')
C['det']=C.sep>5
print('celic',len(C),'zaznavnih',C.det.sum(),'zamenjav',C.swap.sum(),'od tega zaznavnih',(C.swap&C.det).sum())
for name,m in [('vse celice',np.ones(len(C),bool)),('zaznavne',C.det),('zamenjave',C.swap),('drugi outlierji',C.other)]:
    x=C.n[m]; print(f'{name:16s} n={m.sum():5d} mediana {x.median():.0f}  kvartili {x.quantile(.25):.0f}-{x.quantile(.75):.0f}  predcasna {C.pred[m].mean():.1%}')
print('MW zamenjave vs zaznavne: p=%.3f'%stats.mannwhitneyu(C.n[C.swap],C.n[C.det&~C.swap]).pvalue)
print('KS: p=%.3f'%stats.ks_2samp(C.n[C.swap],C.n[C.det&~C.swap]).pvalue)
# rate by size bins among detectable
Dd=C[C.det].copy(); Dd['bin']=pd.qcut(Dd.n,5)
g=Dd.groupby('bin',observed=True).agg(cel=('swap','size'),zam=('swap','sum'),nmed=('n','median'),glasov=('n','sum'))
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
g['stopnja_%']=100*g.zam/g.cel; g[['lo','hi']]=[ [100*v for v in wil(k,n)] for k,n in zip(g.zam,g.cel)]
g['na_1000_glasov']=1000*g.zam/g.glasov*1000
print(g.round(3).to_string())
# logistic-ish: expected swaps per bin if probability ∝ 1 per cell vs ∝ n (per vote)
tot=Dd.swap.sum(); e_cell=tot*g.cel/g.cel.sum(); e_vote=tot*g.glasov/g.glasov.sum()
print('pricakovano (na celico):',e_cell.round(1).tolist()); print('pricakovano (na glas):',e_vote.round(1).tolist()); print('opazeno:',g.zam.tolist())
chi_c=stats.chisquare(g.zam,e_cell).pvalue; chi_v=stats.chisquare(g.zam,e_vote).pvalue; print('chi2 p: na celico %.2f, na glas %.2f'%(chi_c,chi_v))
pd.to_pickle((C,g,e_cell,e_vote),'rezultati/size.pkl')
# plot
BLUE='#4C78A8'; RED='#E45756'; ORG='#F2A541'; GRY='#9AA5B1'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,1,figsize=(9,12),gridspec_kw={'height_ratios':[1,1,1]})
ax=axs[0]; xs=np.linspace(0,1500,300)
for name,m,c,ls in [('vse celice',np.ones(len(C),bool),GRY,'-'),('zaznavne celice (razmik > 5σ)',C.det,BLUE,'-'),('zamenjave',C.swap,RED,'-'),('drugi outlierji',C.other,ORG,'--')]:
    v=np.sort(C.n[m]); ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=c,ls=ls,lw=2 if name in('zamenjave','drugi outlierji') else 1.5,label=f'{name} (n={m.sum()})')
ax.set_xlim(0,1500); ax.set_xlabel('velikost volišča (veljavni glasovi)'); ax.set_ylabel('kumulativni delež')
ax.legend(frameon=False,fontsize=9.5,loc='lower right'); ax.set_title('Porazdelitev velikosti volišč (ECDF)',loc='left',weight='bold')
ax=axs[1]
for name,m,c,y in [('zaznavne',C.det&~C.swap&~C.other,BLUE,0),('zamenjave',C.swap,RED,1),('drugi outlierji',C.other,ORG,2)]:
    v=C.n[m].values; jit=np.random.default_rng(1).uniform(-.3,.3,len(v))
    ax.scatter(v,y+jit,s=4 if y==0 else 40,alpha=.15 if y==0 else 1,color=c,ec='white' if y else None,lw=.5)
    ax.plot([np.median(v)]*2,[y-.38,y+.38],color='k',lw=2)
ax.set_yticks([0,1,2]); ax.set_yticklabels(['zaznavne celice','zamenjave','drugi outlierji']); ax.set_xlim(0,1500)
ax.set_xlabel('velikost volišča (veljavni glasovi); črna črta = mediana'); ax.set_title('1D razsevni diagram',loc='left',weight='bold')
ax=axs[2]; x=np.arange(len(g)); w=.27
ax.bar(x-w,g.zam,w,color=RED,label='opaženo')
ax.errorbar(x-w,g.zam,yerr=[g.zam-g.lo*g.cel/100,g.hi*g.cel/100-g.zam],fmt='none',color='k',lw=1,capsize=3)
ax.bar(x,e_cell,w,color=GRY,label='pričakovano: enaka verjetnost na volišče')
ax.bar(x+w,e_vote,w,color='#C9D2DC',label='pričakovano: verjetnost ∝ številu glasov')
ax.set_xticks(x); ax.set_xticklabels([f'{int(iv.left)}–{int(iv.right)}' for iv in g.index],fontsize=9)
ax.set_xlabel('kvintil velikosti med zaznavnimi celicami (glasovi)'); ax.set_ylabel('št. zamenjav')
ax.legend(frameon=False,fontsize=9.5); ax.set_title('Zamenjave po velikosti (95-% interval Wilson)',loc='left',weight='bold')
plt.tight_layout(); plt.savefig('slike/13_velikost.png',dpi=140)

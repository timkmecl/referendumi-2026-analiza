# Zgoščevanje ob zrcalni osi čez vse celice po razredih razmika (razdelek 4.5) in slika 12_zrcalna_os_vsi.png.
import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); K=['PP','RTV','R42','R43']; r=R[False]
Z=np.concatenate([r[k]['z'] for k in K]); ZS=np.concatenate([r[k]['zs'] for k in K])
S=Z-ZS   # signed separation between normal and mirror line, in sigma units
ok=np.isfinite(S)&np.isfinite(Z); Z,ZS,S=Z[ok],ZS[ok],S[ok]
print('celic',len(Z))
w=1.0
# empirična porazdelitev z VSEH celic (brez izločanja zamenjav, da ni krožnosti)
zsort=np.sort(Z); Femp=lambda x: np.searchsorted(zsort,x)/len(zsort)
edges=[0,1,2,3,4,6,10,20,60]
rows=[]
for a,b in zip(edges[:-1],edges[1:]):
    m=(np.abs(S)>=a)&(np.abs(S)<b)
    s=S[m]; O=np.sum(np.abs(ZS[m])<w)
    Eg=np.sum(stats.norm.cdf(s+w)-stats.norm.cdf(s-w))
    pe=Femp(s+w)-Femp(s-w); Ee=np.sum(pe); sd=np.sqrt(np.sum(pe*(1-pe)))
    # sideband estimate: points with 1<|zs|<3 -> per-unit density *2
    side=np.sum((np.abs(ZS[m])>=1)&(np.abs(ZS[m])<3))/4*2
    rows.append(dict(sep=f'{a}–{b}',n=m.sum(),opazeno=O,gauss=round(Eg,2),empir=round(Ee,2),presezek=round(O-Ee,1),sd=round(sd,1),stranski_pas=round(side,1)))
T=pd.DataFrame(rows); print(T.to_string())
m4=np.abs(S)>=4; m24=(np.abs(S)>=2)&(np.abs(S)<4)
for lab,m in [('nad 4 sigma',m4),('2-4 sigma',m24)]:
    pe=Femp(S[m]+w)-Femp(S[m]-w); O=np.sum(np.abs(ZS[m])<w); print(lab,'opaženo',O,'pričakovano %.1f'%pe.sum(),'presežek %.1f ± %.1f'%(O-pe.sum(),np.sqrt(np.sum(pe*(1-pe)))),'Poisson p=%.1e'%stats.poisson.sf(O-1,pe.sum()))
T.to_csv('rezultati/mirror_tab.csv',index=False)
BLUE='#4C78A8'; RED='#E45756'; GRY='#9AA5B1'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,1,figsize=(9,11),gridspec_kw={'height_ratios':[1.3,1]})
ax=axs[0]
near=np.abs(ZS)<w; far=np.abs(S)>=4
ax.scatter(np.clip(S,-60,60),np.clip(Z,-45,45),s=3,color=GRY,alpha=.25,lw=0)
hl=near&far
ax.scatter(S[hl],Z[hl],s=30,color=RED,ec='white',lw=.5,zorder=3,label='|z po zamenjavi| < 1 in razmik ≥ 4σ')
xx=np.array([-60,60]); ax.plot(xx,[0,0],'k',lw=.8,label='normala: z = 0')
ax.plot(xx,xx,color=RED,ls='--',lw=1,label='zrcalna os: z = razmik')
ax.fill_between(xx,xx-w,xx+w,color=RED,alpha=.12)
ax.set_xlim(-60,60); ax.set_ylim(-45,45)
ax.set_xlabel('razmik med normalo in zrcalno osjo  (100−2·napoved)/šum  [σ]'); ax.set_ylabel('z objavljenega rezultata')
ax.legend(frameon=False,loc='upper left',fontsize=9.5)
ax.set_title('Vseh 12.760 celic (volišče × referendum), zrcalna os podaljšana čez vse podatke',loc='left',weight='bold',fontsize=11.5)
ax=axs[1]; x=np.arange(len(T))
ax.bar(x-.27,T.opazeno,.27,color=RED,label='opaženo v pasu |z po zamenjavi| < 1')
ax.bar(x,T.empir,.27,color=GRY,label='pričakovano brez zamenjav (empirična porazdelitev z vseh celic)')
ax.bar(x+.27,T.gauss,.27,color='#C9D2DC',label='pričakovano brez zamenjav (Gauss)')
ax.set_yscale('symlog',linthresh=1); ax.set_ylim(-1.2,2000); ax.set_yticks([0,1,10,100,1000]); ax.set_yticklabels(['0','1','10','100','1000']); ax.set_xticks(x); ax.set_xticklabels(T.sep,rotation=0,fontsize=9)
ax.set_xlabel('razmik med normalo in zrcalno osjo [σ]'); ax.set_ylabel('št. celic (simlog)')
for xi,(o,e) in enumerate(zip(T.opazeno,T.empir)):
    ax.text(xi-.27,o*1.15+.3,str(o),ha='center',fontsize=9,weight='bold')
    ax.text(xi+.05,-0.75,f'{o-e:+.0f}' if abs(o-e)>=0.5 else '0',ha='center',fontsize=9,color=RED)
ax.legend(frameon=False,fontsize=9.5,loc='upper right')
ax.set_title('Zgoščanje ob zrcalni osi glede na oddaljenost od normale',loc='left',weight='bold',fontsize=11.5)
plt.tight_layout(); plt.savefig('slike/12_zrcalna_os_vsi.png',dpi=140)

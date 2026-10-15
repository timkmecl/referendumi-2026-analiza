# Sliki 9_test_pasu in 10_zrcalna_crta (razdelki 4.3, 4.4).
import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); O,res=pd.read_pickle('rezultati/band.pkl'); K=['PP','RTV','R42','R43']; r=R[False]
BLUE='#4C78A8'; RED='#E45756'; ORG='#F2A541'; GRY='#9AA5B1'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
# Fig 9: observed vs null distribution of zs among outliers
fig,axs=plt.subplots(2,1,figsize=(9,9),gridspec_kw={'height_ratios':[1.2,1]})
ax=axs[0]; bins=np.arange(-50,50.1,2)
zsn=np.clip(res['A'][3],-49.9,49.9); w=np.ones_like(zsn)*53/len(zsn)
ax.hist(zsn,bins=bins,weights=w,color=GRY,alpha=.6,label='pričakovano, če bi bila odstopanja naključna (simulacija)')
ax.hist(np.clip(O.zs,-49.9,49.9),bins=bins,histtype='step',lw=2,color=RED,label='opaženih 53 outlierjev')
ax.axvspan(-3,3,color='#FDECEC',zorder=0)
ax.set_xlabel('z po zamenjavi ZA↔PROTI'); ax.set_ylabel('št. outlierjev (|z|>5)')
ax.legend(frameon=False,loc='upper left',fontsize=9.5); ax.set_xlim(-50,50)
ax.text(3.5,ax.get_ylim()[1]*.85 if ax.get_ylim()[1]>0 else 10,'pas\n|z| < 3',color=RED,fontsize=10)
ax.set_title('1) Koliko outlierjev pade v pas zamenjave?',loc='left',weight='bold')
ax.text(.99,.6,'v pasu:\nopaženo 27\nnaključno ≈ 7–9',transform=ax.transAxes,ha='right',fontsize=12,weight='bold',color='#333')
# bottom: bins within band
ax=axs[1]; lab=['0–1','1–2','2–3']
obs=[17,6,4]; nul=[53*v for v in res['A'][2]]
S=23; g=[stats.norm.cdf(1)*2-1,(stats.norm.cdf(2)-stats.norm.cdf(1))*2,(stats.norm.cdf(3)-stats.norm.cdf(2))*2]
bg=[(53-S)*v for v in res['A'][2]]; sw=[S*v for v in g]
x=np.arange(3); wd=.27
ax.bar(x-wd,obs,wd,color=RED,label='opaženo')
ax.bar(x,nul,wd,color=GRY,label='vse naključno')
ax.bar(x+wd,sw,wd,color='#7FA7D6',label=f'{S} pravih zamenjav (N(0,1))'); ax.bar(x+wd,bg,wd,bottom=sw,color='#D5DCE4',label='+ naključno ozadje')
for xi,v in zip(x-wd,obs): ax.text(xi,v+.3,str(v),ha='center',weight='bold')
ax.set_xticks(x); ax.set_xticklabels([f'|z po zamenjavi| {l}' for l in lab]); ax.set_ylabel('št. outlierjev')
ax.legend(frameon=False,fontsize=9.5); ax.set_title('2) Razporeditev znotraj pasu: kopiči se pri 0, kot binomski šum',loc='left',weight='bold')
plt.tight_layout(); plt.savefig('slike/9_test_pasu.png',dpi=140); plt.close()
# Fig 10: deviation vs 1-2*pred
fig,ax=plt.subplots(figsize=(9,7.5))
for k in K:
    ph=r[k]['ph']; dev=(D['p_'+k]-ph)*100; x=(1-2*ph)*100
    ax.scatter(x,dev,s=3,color=GRY,alpha=.15,lw=0)
bb=pd.read_csv('rezultati/final_table.csv'); O['tip']=bb.tipt.values
o=O[~O.sw]; s=O[O.sw&(O.tip!='zamenjava Preiskava↔RTV')]; pr=O[O.tip=='zamenjava Preiskava↔RTV']
ax.scatter((1-2*o.ph)*100,(o.p-o.ph)*100,s=45,color=ORG,ec='white',lw=.5,label='drugi outlierji',zorder=3)
ax.scatter((1-2*s.ph)*100,(s.p-s.ph)*100,s=45,color=RED,ec='white',lw=.5,label='zamenjave ZA↔PROTI',zorder=4)
ax.plot([-80,80],[-80,80],color=RED,lw=1,ls='--',label='zrcalna črta: objavljeno = 100 − napoved'); ax.axhline(0,color='k',lw=.8)
ax.text(33,-4,'normalno: objavljeno = napoved',fontsize=10)
ax.set_xlim(-60,60); ax.set_ylim(-45,45); 
ax.set_xlabel('100 − 2·napoved (razdalja napovedi od 50 %, ×2)  [o.t.]'); ax.set_ylabel('objavljeno − napoved  [o.t.]')
ax.legend(frameon=False,loc='lower right')
ax.set_title('3) Zamenjave ležijo na zrcalni črti, resnične posebnosti bi bile razpršene',loc='left',weight='bold',fontsize=12)
plt.tight_layout(); plt.savefig('slike/10_zrcalna_crta.png',dpi=140); plt.close()

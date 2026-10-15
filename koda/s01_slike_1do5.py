# Slike 1_napoved_vs_objavljeno, 2_diagnostika_zamenjave, 3_prstni_odtisi, 4_qq, 5_vpliv.
import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); b=pd.read_csv('rezultati/final_table.csv')
K=['PP','RTV','R42','R43']; NM={'PP':'Preiskava','RTV':'RTV','R42':'Sociala','R43':'Tujci'}
BLUE='#4C78A8'; RED='#E45756'; ORG='#F2A541'; GRY='#9AA5B1'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
idx={(o,i):j for j,(o,i) in enumerate(zip(D['Volilni okraj'],D['ID Volišča']))}
lab={}
for _,x in b.iterrows(): lab[(idx[(x.okraj,x.id)],x.ref)]=x.tipt
r=R[False]
# 1. observed vs predicted
fig,axs=plt.subplots(2,2,figsize=(10,10))
for ax,k in zip(axs.flat,K):
    p=D['p_'+k]*100; ph=r[k]['ph']*100
    cat=np.array([lab.get((i,k),'') for i in range(len(D))])
    sw=(cat=='zamenjava ZA↔PROTI')|(cat=='morda zamenjava'); pr=(cat=='zamenjava Preiskava↔RTV'); ot=(cat!='')&~sw&~pr
    ax.scatter(ph[~(sw|ot|pr)],p[~(sw|ot|pr)],s=4,color=BLUE,alpha=.25,lw=0)
    ax.scatter(ph[ot],p[ot],s=30,color=ORG,ec='white',lw=.5,label='drugo odstopanje')
    ax.scatter(ph[sw],p[sw],s=40,color=RED,ec='white',lw=.5,label='zamenjava ZA↔PROTI')
    ax.plot([0,100],[0,100],color='k',lw=.8); ax.plot([0,100],[100,0],color=RED,lw=.8,ls='--')
    ax.set_xlim(0,100); ax.set_ylim(0,100); ax.set_aspect('equal'); ax.set_title(NM[k],loc='left',weight='bold')
    ax.set_xlabel('napoved iz ostalih treh (% ZA)'); ax.set_ylabel('objavljeno (% ZA)')
axs[0,0].legend(frameon=False,loc='upper left',fontsize=9)
fig.suptitle('Objavljeni rezultat vs. napoved iz ostalih referendumov\nčrna = popolno ujemanje, rdeča črtkana = zrcalno (ZA↔PROTI)',x=0.02,ha='left',fontsize=13)
plt.tight_layout(); plt.savefig('slike/1_napoved_vs_objavljeno.png',dpi=140); plt.close()
# 2. z vs z after swap
fig,ax=plt.subplots(figsize=(8,7))
for k in K:
    z=r[k]['z']; zs=r[k]['zs']
    ax.scatter(z,zs,s=3,color=GRY,alpha=.15,lw=0)
cols={'zamenjava ZA↔PROTI':RED,'zamenjava Preiskava↔RTV':'#8E5EA2','morda zamenjava':RED,'več referendumov':ORG,'nepojasnjeno':ORG}
for _,x in b.iterrows():
    i=idx[(x.okraj,x.id)]; ax.scatter(r[x.ref]['z'][i],r[x.ref]['zs'][i],s=40,color=cols[x.tipt],ec='white',lw=.5,zorder=3)
ax.axvspan(-5,5,color='#EEF2F6',zorder=0); ax.axhspan(-3,3,color='#FDECEC',zorder=0)
ax.set_xlim(-28,28); ax.set_ylim(-50,50)
ax.set_xlabel('z objavljenega rezultata'); ax.set_ylabel('z, če zamenjaš ZA↔PROTI')
ax.text(-27,-9,'zamenjava\nrazloži odstopanje',color=RED,fontsize=10,va='center')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker='o',ls='',color=RED,mec='white',ms=8,label='zamenjava ZA↔PROTI'),Line2D([],[],marker='o',ls='',color=ORG,mec='white',ms=8,label='drugi outlierji')],frameon=True,framealpha=.9,loc='lower left',fontsize=9)
ax.text(6,40,'zamenjava ne pomaga\n(drugačna napaka ali\nlokalna posebnost)',color='#B4691F',fontsize=10)
ax.text(-4.5,44,'normalna\nvolišča',color='#555',fontsize=10)
ax.set_title('Diagnostika zamenjave: vsaka točka je volišče × referendum',loc='left',weight='bold')
plt.tight_layout(); plt.savefig('slike/2_diagnostika_zamenjave.png',dpi=140); plt.close()
# 3. fingerprints
ex=[('VO 4009 - LJUBLJANA BEŽIGRAD 2',8,'Bežigrad 2 / 8 – OŠ Mirana Jarca (brez odstopanja)'),('VO 2004 - KOPER 1',22,'Koper 1 / 22 – SEPŠ'),('VO 3006 - LJUBLJANA VIČ-RUDNIK 4',31,'Vič / 31 – Obrtna zbornica'),('VO 1008 - ŠKOFJA LOKA 1',6,'Škofja Loka 1 / 6 – ZD'),('VO 4004 - IVANČNA GORICA',4,'Ivančna Gorica / 4 – KD Muljava')]
fig,axs=plt.subplots(len(ex),1,figsize=(8,12.3),sharex=True)
for ax,(o,i0,t) in zip(axs,ex):
    i=idx[(o,i0)]
    for y,k in enumerate(K):
        p=D['p_'+k][i]*100; ph=r[k]['ph'][i]*100; z=r[k]['z'][i]
        se=np.sqrt(ph/100*(1-ph/100)/D['n_'+k][i])*100*r[k]['c']
        ax.barh(y,6*se,left=ph-3*se,height=.5,color='#DDE5EE')
        ax.plot([ph,ph],[y-.25,y+.25],color='#555',lw=1.5)
        ax.plot(100-p,y,marker='o',mfc='none',mec=GRY,ms=7)
        ax.plot(p,y,'o',color=RED if abs(z)>5 else BLUE,ms=9)
    ax.set_yticks(range(4)); ax.set_yticklabels([NM[k] for k in K]); ax.invert_yaxis()
    ax.set_title(t,loc='left',weight='bold',fontsize=11); ax.set_xlim(0,100)
axs[-1].set_xlabel('% ZA')
fig.text(0.01,0.005,'● objavljeno (rdeče = |z|>5)   ○ zrcalna vrednost (100 − objavljeno)   | napoved   ▬ pas ±3σ',fontsize=9.5,color='#444')
fig.suptitle('"Prstni odtisi" volišč: ena napaka izstopa, ostali referendumi jo izdajo',x=0.02,ha='left',fontsize=13,weight='bold')
plt.tight_layout(rect=(0,0.02,1,1)); plt.savefig('slike/3_prstni_odtisi.png',dpi=140); plt.close()
# 4. QQ
fig,ax=plt.subplots(figsize=(7,7))
for k,c in zip(K,[BLUE,RED,'#54A24B',ORG]):
    z=np.sort(r[k]['z']); q=stats.norm.ppf((np.arange(1,len(z)+1)-.5)/len(z))
    ax.plot(q,z,'.',ms=3,color=c,label=NM[k])
ax.plot([-4,4],[-4,4],'k',lw=.8); ax.axhspan(-5,5,color='#F3F5F7',zorder=0)
ax.set_xlabel('teoretični kvantil (normalna)'); ax.set_ylabel('opažen z'); ax.set_ylim(-28,22)
ax.legend(frameon=False); ax.set_title('Q–Q: jedro je normalno, napake so ločen rep',loc='left',weight='bold')
plt.tight_layout(); plt.savefig('slike/4_qq.png',dpi=140); plt.close()
# 5. impact
V=pd.read_csv('rezultati/vpliv.csv')   # iz 18_vpliv.py
data={r.referendum:(r.objavljena,r.samo_zamenjave,r.vse_napake) for r in V.itertuples()}
fig,ax=plt.subplots(figsize=(6,4.5))
v=data['RTV']; ax.bar(['objavljeno','+ zamenjave','+ vse napake\n(ocena)'],v,color=[GRY,RED,ORG])
for i,x in enumerate(v): ax.text(i,x+20,f'{x:,}'.replace(',','.'),ha='center')
ax.set_title('RTV: prednost PROTI (glasovi)',loc='left',weight='bold')
plt.tight_layout(); plt.savefig('slike/5_vpliv.png',dpi=140); plt.close()

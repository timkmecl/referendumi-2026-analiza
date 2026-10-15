# Združena slika: levo Q-Q diagram (vsi štirje referendumi), desno lijakasti prikaz po referendumih (štiri vrstice).
import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); b=pd.read_csv('rezultati/final_table.csv')
K=['PP','RTV','R42','R43']; NM={'PP':'Preiskava','RTV':'RTV','R42':'Sociala','R43':'Tujci'}
BLUE='#4C78A8'; RED='#E45756'; ORG='#F2A541'; QC=['#1F3A5F','#2A9D8F','#7B5EA7','#9C7A4B']
plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False})
idx={(o,i):j for j,(o,i) in enumerate(zip(D['Volilni okraj'],D['ID Volišča']))}
lab={(idx[(x.okraj,x.id)],x.ref):x.tipt for _,x in b.iterrows()}
fig=plt.figure(figsize=(12,8))
gs=fig.add_gridspec(4,2,width_ratios=[1.6,1],hspace=.18,wspace=.22,top=.93,bottom=.15,left=.07,right=.985)
ax=fig.add_subplot(gs[:,0])
for k,c in zip(K,QC):
    z=np.sort(R[False][k]['z']); q=stats.norm.ppf((np.arange(1,len(z)+1)-.5)/len(z))
    ax.plot(q,z,'.',ms=3.5,color=c,label=NM[k])
ax.plot([-4,4],[-4,4],'k',lw=.8); ax.axhspan(-5,5,color='#F3F5F7',zorder=0)
ax.set_xlabel('teoretični kvantil (normalna porazdelitev)'); ax.set_ylabel('opažen z'); ax.set_ylim(-28,22)
ax.legend(frameon=False,loc='upper left',title='referendum',title_fontsize=9.5,alignment='left')
fig.text(ax.get_position().x0,.955,'Q–Q: jedro je normalno, napake so ločen rep',weight='bold',fontsize=12)
nn=np.logspace(1,np.log10(3000),200); axs=[]
for j,k in enumerate(K):
    a=fig.add_subplot(gs[j,1],sharex=axs[0] if axs else None); axs.append(a)
    r=R[False][k]; n=D['n_'+k].values; dev=100*(D['p_'+k].values-r['ph']); c=r['c']
    cat=np.array([lab.get((i,k),'') for i in range(len(D))])
    sw=(cat=='zamenjava ZA↔PROTI')|(cat=='morda zamenjava'); ot=(cat!='')&~sw
    for m,ls in [(5,'-'),(3,'--')]:
        s=100*m*c*np.sqrt(.25/nn); a.plot(nn,s,color='#555' if m==5 else '#999',lw=.9,ls=ls); a.plot(nn,-s,color='#555' if m==5 else '#999',lw=.9,ls=ls)
    a.scatter(n[~(sw|ot)],dev[~(sw|ot)],s=2.5,color=BLUE,alpha=.25,lw=0)
    a.scatter(n[ot],dev[ot],s=18,color=ORG,ec='white',lw=.4,zorder=3,label='drugi outlierji')
    a.scatter(n[sw],dev[sw],s=22,color=RED,ec='white',lw=.4,zorder=4,label='zamenjava ZA↔PROTI')
    a.axhline(0,color='k',lw=.5); a.set_xscale('log'); a.set_xlim(10,3000); a.set_ylim(-45,45); a.set_yticks([-40,-20,0,20,40])
    a.text(0.015,0.95,NM[k],transform=a.transAxes,weight='bold',va='top',bbox=dict(fc='white',ec='none',pad=1.5),zorder=5)
    if j<3: plt.setp(a.get_xticklabels(),visible=False)
axs[-1].set_xlabel('število veljavnih glasov (log)'); axs[1].set_ylabel('objavljeno − napoved [o.t.]',y=-0.15)
axs[0].plot([],[],color='#555',lw=.9,label='±5σ (pri rezultatu 50 %)'); axs[0].plot([],[],color='#999',lw=.9,ls='--',label='±3σ (pri rezultatu 50 %)')
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,frameon=False,fontsize=9,ncol=2,loc='lower left',bbox_to_anchor=(axs[0].get_position().x0-.02,0.0))
fig.text(axs[0].get_position().x0,.955,'Odstopanje glede na velikost volišča',weight='bold',fontsize=12)
plt.savefig('slike/4_qq_lijak.png',dpi=140); plt.close()

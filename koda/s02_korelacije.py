# Sliki 6_korelacije_matrika in 7_korelacije_heatmap (razdelek 2.1).
import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D,_=pd.read_pickle('rezultati/four.pkl'); b=pd.read_csv('rezultati/final_table.csv')
K=['PP','RTV','R42','R43']; NM={'PP':'Preiskava','RTV':'RTV','R42':'Sociala','R43':'Tujci'}
BLUE='#4C78A8'; RED='#E45756'; ORG='#F2A541'
plt.rcParams.update({'font.size':11})
idx={(o,i):j for j,(o,i) in enumerate(zip(D['Volilni okraj'],D['ID Volišča']))}
sw=np.zeros(len(D),bool); ot=np.zeros(len(D),bool); prs=np.zeros(len(D),bool)
for _,x in b.iterrows():
    i=idx[(x.okraj,x.id)]
    if x.tipt=='zamenjava Preiskava↔RTV': prs[i]=True
    elif x.tipt.startswith('zamenjava'): sw[i]=True
    else: ot[i]=True
ot&=~(sw|prs); ok=~(sw|ot|prs)
P=D[['p_'+k for k in K]]*100; w=D[['n_'+k for k in K]].mean(axis=1)
def wcorr(a,b,w):
    a=a-np.average(a,weights=w); b=b-np.average(b,weights=w)
    return np.sum(w*a*b)/np.sqrt(np.sum(w*a*a)*np.sum(w*b*b))
fig,axs=plt.subplots(4,4,figsize=(12,12))
for r,ki in enumerate(K):
    for c,kj in enumerate(K):
        ax=axs[r,c]; x=P['p_'+kj]; y=P['p_'+ki]
        if r==c:
            ax.hist(y,bins=40,weights=w,color=BLUE,alpha=.8); ax.set_yticks([])
            ax.text(.03,.9,NM[ki],transform=ax.transAxes,weight='bold')
        elif r>c:
            ax.scatter(x[ok],y[ok],s=2,color=BLUE,alpha=.2,lw=0)
            ax.scatter(x[ot],y[ot],s=14,color=ORG,lw=0)
            ax.scatter(x[sw],y[sw],s=16,color=RED,lw=0)
            ax.plot([0,100],[0,100],color='k',lw=.6)
            ax.set_xlim(0,100); ax.set_ylim(0,100)
        else:
            ax.axis('off')
            r_all=wcorr(x.values,y.values,w.values); r_ok=wcorr(x[ok].values,y[ok].values,w[ok].values)
            ax.text(.5,.6,f'r = {r_ok:.3f}',ha='center',fontsize=20,weight='bold',color=plt.cm.Blues(0.4+0.6*(r_ok-.9)/.1))
            ax.text(.5,.35,f'z napakami: {r_all:.3f}',ha='center',fontsize=10,color='#666')
        if r==3: ax.set_xlabel(f'{NM[kj]} (% ZA)')
        if c==0 and r>0: ax.set_ylabel(f'{NM[ki]} (% ZA)')
fig.suptitle('Korelacije deležev ZA med referendumi prek 3190 volišč',x=.02,ha='left',fontsize=15,weight='bold')
fig.text(.02,.005,'Modro = volišča; rdeče = volišča z zamenjavo ZA↔PROTI; oranžno = drugo odstopanje. r je utežen z velikostjo volišča (zgoraj brez izstopajočih, sivo z njimi).',fontsize=9.5,color='#444')
plt.tight_layout(rect=(0,.015,1,.97)); plt.savefig('slike/6_korelacije_matrika.png',dpi=130); plt.close()
# heatmaps: raw vs within-okraj
G=D['Volilni okraj']
Pw=P[ok].copy(); ww=w[ok]
dev=Pw-Pw.groupby(G[ok]).transform('mean')
def mat(X):
    return np.array([[wcorr(X['p_'+a].values,X['p_'+b].values,ww.values) for b in K] for a in K])
M1=mat(Pw); M2=mat(dev)
fig,axs=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'wspace':0.45})
for ax,M,t in zip(axs,[M1,M2],['Med volišči (skupaj)','Znotraj okrajev\n(odšteto povprečje okraja)']):
    im=ax.imshow(M,cmap='Blues',vmin=.9,vmax=1)
    for i in range(4):
        for j in range(4): ax.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',color='white' if M[i,j]>.965 else 'black',fontsize=12)
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels([NM[k] for k in K]); ax.set_yticklabels([NM[k] for k in K])
    ax.set_title(t,loc='left',weight='bold')
fig.colorbar(im,ax=axs,shrink=.8,label='utežen r')
fig.text(.02,.01,'Brez izstopajočih volišč. Tudi znotraj istega okraja so referendumi močno povezani, zato napoved iz ostalih treh deluje bolje kot primerjava z okrajem.',fontsize=9.5,color='#444')
plt.savefig('slike/7_korelacije_heatmap.png',dpi=130,bbox_inches='tight'); plt.close()
print(np.round(M1,3)); print(np.round(M2,3))

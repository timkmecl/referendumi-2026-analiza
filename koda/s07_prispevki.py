# Slika 16_prispevki_volisc (razdelek 5.1).
import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
b=pd.read_csv('rezultati/final_table.csv'); s=b[b.tipt=='zamenjava ZA↔PROTI'].copy()
win={'PP':-1,'RTV':-1,'R42':1,'R43':1}; nm={'PP':'Preiskava','RTV':'RTV','R42':'Sociala','R43':'Tujci'}
s['dm']=[-2*x.dza*win[x.ref] for _,x in s.iterrows()]   # change in winner margin after correction
s['lab']=s.ok_s.str.replace(r'^\d+ - ','',regex=True)+' / '+s.id.astype(int).astype(str)
cols={'PP':'#4C78A8','RTV':'#E45756','R42':'#54A24B','R43':'#F2A541'}
fig,axs=plt.subplots(1,4,figsize=(13,6.5),sharex=False)
for ax,k in zip(axs,['PP','RTV','R42','R43']):
    x=s[s.ref==k].sort_values('dm'); y=np.arange(len(x))
    ax.barh(y,x.dm,color=[cols[k] if v>0 else '#BBBBBB' for v in x.dm])
    ax.set_yticks(y); ax.set_yticklabels(x.lab,fontsize=8.5); ax.axvline(0,color='k',lw=.8)
    ax.set_title(f'{nm[k]}\nskupaj {x.dm.sum():+.0f}',loc='left',weight='bold',fontsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlabel('sprememba prednosti\nzmagovalca (glasovi)',fontsize=9)
fig.suptitle('Ilustracija (pogojno na hipotezo zamenjave): kako bi popravek posameznega volišča spremenil prednost zmagovalca\nbarvno = popravek poveča prednost zmagovalca, sivo = zmanjša',x=.01,ha='left',fontsize=11.5)
plt.tight_layout(rect=(0,0,1,.92)); plt.savefig('slike/16_prispevki_volisc.png',dpi=140)
print(s.groupby('ref').dm.sum())

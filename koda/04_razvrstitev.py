# Razvrstitev izstopajočih celic (|z|>5 v OBEH modelih: brez in s FE).
#  - "zamenjava": |z_zam|<3 v obeh modelih; "zamenjava?" (morda): v enem modelu.
#  - "multi": na istem volišču izstopa več celic (v obeh modelih).
#  - Sostro (Moste-Polje 1/25) izstopa pri treh referendumih in je zato med "več referendumov";
#    preverjanje prerazporeditev in napovedi iz sociale je v 20_druge_kontrole.py.
import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
T=pd.read_pickle('rezultati/T.pkl')
a=T[False].set_index(['okraj','id','ref']); f=T[True].set_index(['okraj','id','ref'])
b=a.join(f[['pred','z','zs','tip']],how='inner',rsuffix='_FE').reset_index()
b['zmax']=b[['z','z_FE']].abs().max(axis=1)
b['tipb']=np.where((b.tip=='zamenjava')&(b.tip_FE=='zamenjava'),'zamenjava',np.where((b.tip=='zamenjava')|(b.tip_FE=='zamenjava'),'zamenjava?','drugo'))
cnt=b.groupby(['okraj','id']).ref.transform('size'); b['multi']=cnt>1
b=b.sort_values('zmax',ascending=False).reset_index(drop=True)
b.to_csv('rezultati/both.csv',index=False)
print('celic',len(b),'volišč',b[['okraj','id']].drop_duplicates().shape[0]); print(b.tipb.value_counts().to_dict())
name={'PP':'Preiskava','RTV':'RTV','R42':'Sociala','R43':'Tujci'}
def tip(x):
    if x.tipb=='zamenjava': return 'zamenjava ZA↔PROTI'
    if x.tipb=='zamenjava?': return 'morda zamenjava'
    if x.multi: return 'več referendumov'
    return 'nepojasnjeno'
b['tipt']=b.apply(tip,axis=1); b['zam']=b.tipt.str.startswith('zamenjava')
b['dza_s']=np.where(b.tipt=='zamenjava ZA↔PROTI',b.dza.map(lambda v:f'{v:+.0f}'),b.dza.map(lambda v:f'≈{v:+.0f}'))
b['vol_s']=b.vol.str.split(',').str[0].str.slice(0,32); b['ok_s']=b.okraj.str.slice(3).str.title().str.slice(0,24)
b['refs']=b.ref.map(name); b['pp_s']=b.apply(lambda x:f"{x.p:.0f} ({x.pred:.0f})",axis=1)
b['az']=b[['z','z_FE']].abs().max(axis=1)
order={'zamenjava ZA↔PROTI':0,'zamenjava Preiskava↔RTV':1,'morda zamenjava':2,'več referendumov':3,'nepojasnjeno':4}
b=b.assign(o=b.tipt.map(order)).sort_values(['o','az'],ascending=[True,False])
b.to_csv('rezultati/final_table.csv',index=False)
print(b.tipt.value_counts().to_string())
cols=['ok_s','id','vol_s','refs','n','pp_s','z','z_FE','zs','tipt','kor','dza_s']
hdr=['Volilni okraj','ID','Volišče','Referendum','Glasov','ZA %\n(napoved)','z\nbrez FE','z\nz FE','z po\nzamenjavi','Tip','Koristi','Δ glasov ZA']
W=[.13,.035,.19,.07,.045,.065,.05,.05,.06,.14,.055,.065]
nr=len(b); H=1.6+0.3*nr; top=0.75; bot=0.6
fig=plt.figure(figsize=(16,H)); ax=fig.add_axes([0.01,bot/H,0.98,(H-top-bot)/H]); ax.axis('off')
cells=b[cols].copy()
for c in ['z','z_FE','zs']: cells[c]=cells[c].map(lambda v:f'{v:+.1f}')
cells['n']=cells.n.astype(int).astype(str); cells['id']=cells.id.astype(str)
t=ax.table(cellText=cells.values,colLabels=hdr,cellLoc='center',colWidths=W,bbox=[0,0,1,1])
t.auto_set_font_size(False); t.set_fontsize(9.5)
for j in range(len(hdr)): t[0,j].set_facecolor('#2F3B4C'); t[0,j].set_text_props(color='white',weight='bold')
col={'zamenjava ZA↔PROTI':'#E3EEFB','zamenjava Preiskava↔RTV':'#E3EEFB','morda zamenjava':'#F1F5FA','več referendumov':'#FFF6E0','nepojasnjeno':'#FFFFFF'}
for i,(_,x) in enumerate(b.iterrows(),start=1):
    for j in range(len(cols)): t[i,j].set_facecolor(col[x.tipt]); t[i,j].set_edgecolor('#D0D5DB')
    for j in (0,2): t[i,j]._loc='left'; t[i,j].PAD=0.03
    t[i,9].set_text_props(weight='bold' if x.zam else 'normal',color='#1F5FA8' if x.zam else ('#9A6700' if x.tipt=='več referendumov' else '#666'))
    t[i,10].set_facecolor('#CDEFD5' if x.kor=='ZA' else '#F7D4D4'); t[i,10].set_text_props(weight='bold')
fig.text(0.01,1-0.45/H,'Izstopajoča volišča po modelu štirih referendumov  (|z| > 5 v obeh modelih: 53 primerov na 49 voliščih od 3190)',fontsize=14,weight='bold')
fig.text(0.01,0.12/H,'Napoved: delež ZA iz ostalih treh referendumov na istem volišču (robustna utežena regresija v logitih; "z FE" doda fiksni učinek volilnega okraja). '
 'z po zamenjavi: odstopanje, če ZA↔PROTI zamenjaš.\nModra = zamenjava se ujema z napovedjo (|z| < 3 v obeh modelih). Rumena = na istem volišču odstopa več referendumov. '
 'Nacionalno ZA: preiskava 48,7 %, RTV 50,0 %, sociala 60,5 %, tujci 55,3 %. ≈ = grob presežek ZA glede na napoved.',fontsize=9,color='#444')
plt.savefig('slike/tabela_outlierjev.png',dpi=140)

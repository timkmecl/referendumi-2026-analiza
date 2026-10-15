# Volišča v isti stavbi (razdelek 4.6, "Volišča v isti stavbi").
# Sestrska volišča poiščemo po imenu stavbe ali naslovu (normalizirano; oznake I/II in oklepaji odstranjeni).
# Vsa sestrska volišča so vključena, tudi izstopajoča (OŠ Sostro 25), da izbor kontrol ni prilagojen rezultatu.
import pandas as pd,numpy as np,re
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); b=pd.read_csv('rezultati/final_table.csv')
NM={'PP':'preiskava','RTV':'RTV','R42':'sociala','R43':'tujci'}
def norm(x):
    x=str(x).lower(); x=re.sub(r'\(.*?\)','',x); x=re.sub(r'\b(i|ii|iii|iv|v)\b','',x); x=re.sub(r'[^a-zčšž0-9 ]',' ',x); return ' '.join(x.split())
D['nm']=D['Volišče'].map(lambda v: norm(str(v).split(',')[0])); D['ad']=D['Volišče'].map(lambda v: norm(','.join(str(v).split(',')[1:])))
izstop=set(zip(b.okraj,b.id))
rows=[]
for _,x in b[b.tipt=='zamenjava ZA↔PROTI'].iterrows():
    i=D.index[(D['Volilni okraj']==x.okraj)&(D['ID Volišča']==x.id)][0]; r=D.loc[i]
    sib=D[((D.nm==r.nm)|((D.ad==r.ad)&(r.ad!='')))&(D.index!=i)&(D['ID Volišča']<900)]
    if not len(sib): continue
    k=x.ref; p=100*D['p_'+k][i]; ph=100*R[False][k]['ph'][i]
    ps=100*sib['p_'+k].values; zs=R[False][k]['z'][sib.index]
    rows.append(dict(volisce=f"{x.okraj[3:]} / {x.id}, {str(r['Volišče']).split(',')[0]}",ref=NM[k],objav=round(p),po_zam=round(100-p),napoved=round(ph),
        sestrska=', '.join(f'{v:.0f}' for v in ps),z_sestrskih=', '.join(f'{v:+.1f}' for v in zs),
        razd_objav=f'{np.abs(p-ps).min():.0f}-{np.abs(p-ps).max():.0f}',razd_po_zam=f'{np.abs(100-p-ps).min():.0f}-{np.abs(100-p-ps).max():.0f}'))
T=pd.DataFrame(rows); pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
print(T.to_string(index=False))
allz=np.concatenate([[float(v) for v in s.split(', ')] for s in T.z_sestrskih]); print('največji |z| sestrskih:',np.abs(allz).max().round(1))
T.to_csv('rezultati/sestrska_volisca.csv',index=False)

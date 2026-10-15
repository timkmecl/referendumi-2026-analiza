# Seznam celic z |z|>5 posebej za vsak model (brez FE / s FE). Shrani rezultati/T.pkl.
import pandas as pd,numpy as np
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); K=['PP','RTV','R42','R43']
D['odd_spread']=D[['odd_'+k for k in K]].max(1)-D[['odd_'+k for k in K]].min(1)
def table(fe):
    r=R[fe]; rows=[]
    Zm=np.abs(np.column_stack([r[k]['z'] for k in K]))
    for i in np.where(Zm.max(1)>5)[0]:
        fl=[k for k in K if abs(r[k]['z'][i])>5]
        for k in fl:
            z=r[k]['z'][i]; zs=r[k]['zs'][i]; ph=r[k]['ph'][i]
            typ='zamenjava' if abs(zs)<3 else 'drugo'
            dza=int(D['za_'+k][i]-D['pr_'+k][i]) if typ=='zamenjava' else int(round((D['p_'+k][i]-ph)*D['n_'+k][i]))
            rows.append(dict(okraj=D['Volilni okraj'][i],id=int(D['ID Volišča'][i]),vol=D['Volišče'][i],ref=k,n=int(D['n_'+k][i]),
              p=round(100*D['p_'+k][i]),pred=round(100*ph),z=round(z,1),zs=round(zs,1),tip=typ,kor='ZA' if z>0 else 'PROTI',dza=dza,
              nfl=len(fl),odd=int(D['odd_spread'][i])))
    return pd.DataFrame(rows).sort_values('z',key=abs,ascending=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',28)
T={}
for fe in [False,True]:
    t=table(fe); T[fe]=t
    print('\n=====','FE' if fe else 'brez FE','celic',len(t),'volišč',t[['okraj','id']].drop_duplicates().shape[0],t.tip.value_counts().to_dict(),'več ref/volišče',(t.nfl>1).sum())
    print(t.groupby(['ref','kor']).size().unstack(fill_value=0).to_string())
    print('neto dza po ref', t.groupby('ref').dza.sum().to_dict())
a=T[False].set_index(['okraj','id','ref']); b=T[True].set_index(['okraj','id','ref'])
j=a.join(b[['pred','z','zs','tip']],how='outer',rsuffix='_FE').reset_index()
j['zmax']=j[['z','z_FE']].abs().max(1); j=j.sort_values('zmax',ascending=False)
print(j.drop(columns=['zmax']).to_string())
pd.to_pickle(T,'rezultati/T.pkl'); j.to_csv('rezultati/four_out.csv',index=False)

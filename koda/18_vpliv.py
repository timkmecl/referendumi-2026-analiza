# Vpliv na rezultate (razdelek 5): prednost zmagovalca pred in po popravku.
#  "samo zamenjave": ZA<->PROTI zamenjana na 26 celicah.
#  "vse napake": dodatno je pri ostalih outlierjih delež ZA nadomeščen z napovedjo (model brez FE).
# Pogojna analiza smeri (razdelek 5.2): naključno izbrana zaznavna volišča (razmik > 5 sigma), toliko kot je označenih zamenjav.
import pandas as pd,numpy as np
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); b=pd.read_csv('rezultati/final_table.csv')
K=['PP','RTV','R42','R43']; NM={'PP':'Preiskava','RTV':'RTV','R42':'Sociala','R43':'Tujci'}; win={'PP':-1,'RTV':-1,'R42':1,'R43':1}
F={'PP':'preiskava.xlsx','RTV':'rtv.xlsx','R42':'sociala.xlsx','R43':'tujci.xlsx'}
idx={(o,i):j for j,(o,i) in enumerate(zip(D['Volilni okraj'],D['ID Volišča']))}
tot={}
for k,f in F.items():   # nacionalni seštevki iz vseh volišč (tudi tistih z <10 glasovi)
    d=pd.read_excel('podatki/'+f,sheet_name=1); tot[k]=(d['Glasovi ZA'].sum(),d['Glasovi PROTI'].sum())
def margin(k,dza,dpr): za,pr=tot[k]; return win[k]*((za+dza)-(pr+dpr))
dS={k:[0,0] for k in K}; dA={k:[0,0] for k in K}
for _,x in b.iterrows():
    i=idx[(x.okraj,x.id)]; k=x.ref; za,pr=D['za_'+k][i],D['pr_'+k][i]
    if x.tipt=='zamenjava ZA↔PROTI': d=(pr-za,za-pr)
    else:
        n=za+pr; zn=R[False][k]['ph'][i]*n; d=(zn-za,(n-zn)-pr)
        for c in (0,1): dA[k][c]+=d[c]
        continue
    for c in (0,1): dS[k][c]+=d[c]; dA[k][c]+=d[c]
rows=[]
for k in K:
    m0=margin(k,0,0); m1=margin(k,*dS[k]); m2=margin(k,*dA[k])
    rows.append(dict(referendum=NM[k],objavljena=m0,samo_zamenjave=round(m1),sprememba_zam=round(m1-m0),vse_napake=round(m2),sprememba_vse=round(m2-m0),
                     premik_ZA_ot=round(100*abs((dA[k][0]))/sum(tot[k]),3)))
V=pd.DataFrame(rows); print(V.to_string(index=False)); V.to_csv('rezultati/vpliv.csv',index=False)
# --- pogojna analiza smeri ---
rng=np.random.default_rng(1); M=200000
s=b[b.tipt=='zamenjava ZA↔PROTI']
print('\npogojna analiza (naključna zaznavna volišča):')
tot_h=np.zeros(M)
for k in K:
    ph=R[False][k]['ph']; n=D['n_'+k].values; se=np.sqrt(ph*(1-ph)/n)*R[False][k]['c']; det=np.abs(1-2*ph)/se>5
    za=D['za_'+k].values[det]; pr=D['pr_'+k].values[det]
    # če bi bilo to volišče zamenjano, bi popravek prednost zmagovalca spremenil za 2*(ZA-PROTI)*smer zmagovalca (pravi rezultat = sedanji)
    eff=win[k]*2*(za-pr)
    c=(s.ref==k).sum(); ii=rng.integers(len(eff),size=(M,c)); sim=eff[ii].sum(1); tot_h+=(eff[ii]>0).sum(1); q=np.mean(eff>0)
    ob=s[s.ref==k]; oe=[win[k]*2*(D['pr_'+k][idx[(o,i)]]-D['za_'+k][idx[(o,i)]]) for o,i in zip(ob.okraj,ob.id)]
    print(f'  {NM[k]:9s} zamenjav {c:2d}  v škodo zmag. {sum(e>0 for e in oe)}  sprememba {sum(oe):+.0f}  pričakovano {sim.mean():+.0f} [{np.percentile(sim,2.5):+.0f}, {np.percentile(sim,97.5):+.0f}]  P(>0)={np.mean(sim>0):.2f}  P(vse v škodo)={q**c:.3f}'); Pall=locals().get('Pall',1)*np.mean(sim>0)
print(f'  verjetnost, da popravek poveča prednost pri vseh štirih: {Pall:.2f}')
print(f'  skupaj v škodo zmagovalca: pričakovano {tot_h.mean():.1f} [{np.percentile(tot_h,2.5):.0f}, {np.percentile(tot_h,97.5):.0f}]')

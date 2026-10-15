# Predlog preverjanja (razdelek 6, "Preverljivost"): verjetnost, da je posamezna domnevna zamenjava res zamenjava
# (posteriorna verjetnost iz mešanega modela za lambda), in verjetnosti izidov pregleda treh volišč.
# Konzervativno: spodnja meja 95-% IZ za pi in najmanj ugodno od štirih ozadij.
import pandas as pd,numpy as np
from scipy import stats
lam,sig,NUL,res,names=pd.read_pickle('rezultati/lam.pkl'); b=pd.read_csv('rezultati/final_table.csv')
rng=np.random.default_rng(1)
W={'tocka':[],'konz':[]}
for k in ['B','C','P','T']:
    l,s=NUL[k]; lc=np.clip(l,-3,4); kde=stats.gaussian_kde(lc[rng.choice(len(lc),20000)],bw_method=.15)
    h=kde(np.clip(lam,-3,4)); f=stats.norm.pdf(lam,1,np.sqrt(sig**2+0.02**2))
    for lab,pi in [('tocka',res[k][0]),('konz',res[k][1])]: W[lab].append(pi*f/(pi*f+(1-pi)*h))
b['w_tocka']=np.min(W['tocka'],0); b['w_konz']=np.min(W['konz'],0)
s=b[b.tipt=='zamenjava ZA↔PROTI'].sort_values('w_konz',ascending=False)
pd.set_option('display.width',200)
print(s[['ok_s','id','vol_s','refs','n','p','pred','zs','w_tocka','w_konz']].round(2).to_string(index=False))
print('pričakovano število pravih zamenjav med 26: %.1f (točkovno), %.1f (konzervativno)'%(s.w_tocka.sum(),s.w_konz.sum()))
# v poročilu izbrana volišča: najmočnejši signal (konz. verjetnost >= 0,85), dva referenduma, različni okraji
izbor=s[s.vol_s.isin(['PGD Bizovik','Dom kulture Braslovče II','OŠ Mali Slatnik'])]; top=izbor.w_konz.values
print('\nPregled treh volišč:',', '.join(f'{v} ({w:.3f})' for v,w in zip(izbor.vol_s,top)))
print('  P(nobena napaka | zamenjave obstajajo) = %.4f'%np.prod(1-top))
for q in [0.0025,0.01]:
    print(f'  če zamenjav ni (naključna stopnja {q:.2%} na celico): P(>=1)={1-stats.binom.cdf(0,3,q):.3f}  P(>=2)={1-stats.binom.cdf(1,3,q):.1e}')
s.to_csv('rezultati/posteriorne_verjetnosti.csv',index=False)

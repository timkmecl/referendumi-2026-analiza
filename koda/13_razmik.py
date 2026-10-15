# Preverjanje skladnosti: oddaljenost od 50:50 (razmik) opaženih outlierjev proti ničelnim modelom (razdelek 4.6).
import pandas as pd,numpy as np
from scipy import stats, optimize
exec(open('koda/12_lambda_velikost.py').read().split('rows=[]')[0].replace("def sim(kind,weight,M=1500000)","def _unused(kind,weight,M=1)"))
SE=np.sqrt(PH*(1-PH)/N)*C[None,:]; SEP=np.abs(1-2*PH)/SE
sobs=SEP[I,A]; lam_=lam
print('opazeni outlierji: mediana razmika',np.median(sobs).round(1),' delež razmik>10:',np.mean(sobs>10).round(2),' razmik<5:',np.sum(sobs<5))
def simS(kind,weight,M=1500000):
    flatN=N.ravel(); w=flatN**weight; w=w/w.sum(); j=rng.choice(flatN.size,M,p=w); i=j//4; a=j%4
    ph=PH[i,a]; p0=P[i,a]; n=N[i,a]; p=shift(p0,kind,M); se=SE[i,a]; m=np.abs((p-ph)/se)>5
    return SEP[i,a][m], n[m]
# swap component: cells where swap gives |z|>5, prob ∝ n (per-vote rate)
def swapS(M=1500000):
    flatN=N.ravel(); w=flatN/flatN.sum(); j=rng.choice(flatN.size,M,p=w); i=j//4; a=j%4
    zs=(1-P[i,a]-PH[i,a])/SE[i,a]; m=np.abs(zs)>5
    return SEP[i,a][m]
SW=swapS(); print('zamenjave (sim): mediana razmika',np.median(SW).round(1))
bins=np.array([0,5,10,15,20,30,1e9])
ho=np.histogram(sobs,bins)[0]; print('opazeno po razmiku',bins[:-1],ho)
rows=[]
for kind in ['B','C','T']:
    for wn,wt in [('enakomerno',0),('∝ n',1),('∝ n²',2)]:
        s,n=simS(kind,wt)
        ks=stats.ks_2samp(sobs,s).pvalue; mw=stats.mannwhitneyu(sobs,s,alternative='greater').pvalue
        # mixture on log S
        kb=stats.gaussian_kde(np.log(s[rng.choice(len(s),30000)])); ks_=stats.gaussian_kde(np.log(SW[rng.choice(len(SW),30000)]))
        hb=kb(np.log(sobs)); hs=ks_(np.log(sobs))
        nll=lambda pi:-np.sum(np.log(pi*hs+(1-pi)*hb)); o=optimize.minimize_scalar(nll,bounds=(1e-4,1-1e-4),method='bounded')
        g=np.linspace(.001,.999,999); ll=np.array([-nll(x) for x in g]); ok=g[ll>=ll.max()-1.92]
        he=len(sobs)*np.histogram(s,bins)[0]/len(s)
        rows.append(dict(ozadje=kind,vzorcenje=wn,med_razmik_null=np.median(s),E_razmik_gt20=he[4]+he[5],KS_p=ks,MW_p=mw,pi=o.x,IZ=f'{53*ok.min():.0f}-{53*ok.max():.0f}',chi2=2*(nll(1e-4)-o.fun)))
T=pd.DataFrame(rows); pd.set_option('display.width',220); print(T.round(4).to_string())
print('opazeno razmik>20:',np.sum(sobs>20))
# joint: lambda core AND separation
pd.to_pickle((sobs,SW,T),'rezultati/sepnull.pkl')
print('--- ujemanje velikosti (prevrednotenje na opaženo porazdelitev n) ---')
nob=N[I,A]; qb=np.quantile(np.log(nob),np.linspace(0,1,7)); qb[0]=-1; qb[-1]=99
for kind in ['B','C','T']:
    s,n=simS(kind,0)
    ho_=np.histogram(np.log(nob),qb)[0]/len(nob); hs_=np.histogram(np.log(n),qb)[0]/len(n)
    w=(ho_/np.maximum(hs_,1e-9))[np.clip(np.digitize(np.log(n),qb)-1,0,5)]; w/=w.sum()
    sub=rng.choice(len(s),200000,p=w); ss=s[sub]
    mw=stats.mannwhitneyu(sobs,ss,alternative='greater').pvalue
    kb=stats.gaussian_kde(np.log(ss[:30000])); ks_=stats.gaussian_kde(np.log(SW[rng.choice(len(SW),30000)]))
    hb=kb(np.log(sobs)); hs=ks_(np.log(sobs)); nll=lambda pi:-np.sum(np.log(pi*hs+(1-pi)*hb))
    o=optimize.minimize_scalar(nll,bounds=(1e-4,1-1e-4),method='bounded'); g=np.linspace(.001,.999,999); ll=np.array([-nll(x) for x in g]); ok=g[ll>=ll.max()-1.92]
    print(kind,'med razmik null %.1f'%np.median(ss),'MW p=%.3f'%mw,'pi=%.2f [%.0f-%.0f] chi2=%.1f'%(o.x,53*ok.min(),53*ok.max(),2*(nll(1e-4)-o.fun)))

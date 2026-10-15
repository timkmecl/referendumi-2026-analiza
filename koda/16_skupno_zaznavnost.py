# Ocena skupnega števila zamenjav (razdelek 4.6, "Ocena skupnega števila zamenjav") in
# zaznavnost po referendumih (razdelek 4.7, "Porazdelitev po referendumih").
# Verjetnost zaznave celice: verjetnost, da bi zamenjava ZA<->PROTI na tej celici dala |z|>5
# v obeh modelih. Po zamenjavi je z = razmik - eps, eps ~ N(0,1) (šum pravega rezultata okoli napovedi).
import pandas as pd,numpy as np
from scipy import stats
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); K=['PP','RTV','R42','R43']
lam,sig,NUL,res,names=pd.read_pickle('rezultati/lam.pkl')
N=np.column_stack([D['n_'+k] for k in K]).astype(float)
def sep(fe):
    out=[]
    for k in K:
        ph=R[fe][k]['ph']; se=np.sqrt(ph*(1-ph)/D['n_'+k].values)*R[fe][k]['c']; out.append((1-2*ph)/se)
    return np.column_stack(out)
S0=sep(False); S1=sep(True)
rng=np.random.default_rng(3); M=2000
eps=rng.standard_normal((M,1,1))
det=np.mean((np.abs(S0[None]-eps)>5)&(np.abs(S1[None]-eps)>5),axis=0)   # verjetnost zaznave po celicah
pm=det.mean(); pw=np.average(det,weights=N)
print(f'povprečna verjetnost zaznave {pm:.2f} (utežena z glasovi {pw:.2f}); delež celic s p<0.5: {np.mean(det<0.5):.2f}')
for k_,(pi,lo,hi,chi) in res.items():
    nz,nlo,nhi=53*pi,53*lo,53*hi
    print(f'ozadje {names[k_]:28s}: zaznanih {nz:.1f} [{nlo:.1f}-{nhi:.1f}] -> skupaj {nz/pm:.1f} [{nlo/pm:.1f}-{nhi/pm:.1f}] (uteženo: {nz/pw:.1f})')
pi=res['B'][0]; tot=53*pi/pm
print(f'delež vnosov z zamenjavo {tot/N.size:.4f}; volišča z vsaj eno: {1-np.exp(-tot/len(D)):.3f} (ob neodvisnih napakah); na referendum {tot/4:.1f}')
print(f'na milijon glasov: {tot/N.sum()*1e6:.1f}  (vseh glasov {N.sum():.0f})')
# pričakovano število zaznavnih v pasu razmika 2-4 sigma iz te ocene
s=np.abs(S0); m=(s>=2)&(s<4); rate=tot/N.size
print(f'celic z razmikom 2-4: {m.sum()}, pričakovano zamenjav v njih ob enaki stopnji: {rate*m.sum():.1f}, od tega v pasu |z_zam|<1: {rate*m.sum()*0.6827:.1f}')
print('\nzaznavnost po referendumih:')
pk=det.mean(0); b=pd.read_csv('rezultati/final_table.csv'); obs=b[b.tipt=='zamenjava ZA↔PROTI'].ref.value_counts()
for a,k in enumerate(K): print(f'  {k:4s} povp. verjetnost zaznave {pk[a]:.2f}  pričakovano (od 26, enaka stopnja) {26*pk[a]/pk.sum():.1f}  opaženo {obs.get(k,0)}')
print('  test enakih stopenj (chi2): p=%.2f'%stats.chisquare([obs.get(k,0) for k in K],26*pk/pk.sum()).pvalue)

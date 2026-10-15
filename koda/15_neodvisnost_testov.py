# Medsebojna odvisnost testov (oblika lambda, razmik, štetje, vsota) in razmik brez jedra lambda (razdelka 4.6, 5.2).
import numpy as np
from scipy import stats
exec(open('koda/14_robustnost_smer_razmik.py').read().split('out=[]')[0])
T=5
m=(np.abs(Z)>T)&(np.abs(ZF)>T); I,A=np.where(m); cnt=[np.sum(A==a) for a in range(4)]
lam_o=(P[I,A]-PH[I,A])/(1-2*PH[I,A]); sep_o=SEP[I,A]; e_o=(P[I,A]-PH[I,A])*N[I,A]*sgn[A]
obs=(np.sum(np.abs(lam_o-1)<=.1), np.mean(np.log(sep_o)), np.sum(e_o<0), e_o.sum())
pool={}
for k in 'BCTS':
    zz=(PS[k]-ph0)/se0; mm=np.abs(zz)>T
    pool[k]=[ (((PS[k]-ph0)/(1-2*ph0))[mm][A0[mm]==a], sep0[mm][A0[mm]==a], ((PS[k]-ph0)*n0*sgn[A0])[mm][A0[mm]==a]) for a in range(4)]
def reps(bk,pi,R=20000):
    L=np.zeros((R,4))
    for a in range(4):
        c=cnt[a]; nb=len(pool[bk][a][0]); ns=len(pool['S'][a][0])
        ib=rng.integers(nb,size=(R,c)); is_=rng.integers(ns,size=(R,c)); mk=rng.random((R,c))<pi
        lam=np.where(mk,pool['S'][a][0][is_],pool[bk][a][0][ib]); sep=np.where(mk,pool['S'][a][1][is_],pool[bk][a][1][ib]); e=np.where(mk,pool['S'][a][2][is_],pool[bk][a][2][ib])
        L[:,0]+=np.sum(np.abs(lam-1)<=.1,1); L[:,1]+=np.log(sep).sum(1); L[:,2]+=np.sum(e<0,1); L[:,3]+=e.sum(1)
    L[:,1]/=sum(cnt); return L
names=['jedro λ','log razmik','štetje smeri','vsota smeri']
for bk in 'BCT':
    for pi in [0,0.4]:
        L=reps(bk,pi); rho=stats.spearmanr(L).correlation
        print(f'ozadje {bk}, π={pi}: Spearman  λ–razmik {rho[0,1]:+.2f}  λ–štetje {rho[0,2]:+.2f}  λ–vsota {rho[0,3]:+.2f}  razmik–štetje {rho[1,2]:+.2f}  razmik–vsota {rho[1,3]:+.2f}  štetje–vsota {rho[2,3]:+.2f}')
    L0=reps(bk,0,200000)
    p=[np.mean(L0[:,0]>=obs[0]),np.mean(L0[:,1]>=obs[1]),np.mean(L0[:,2]>=obs[2])]
    # joint empirical: P(all three >= obs simultaneously) vs product
    joint=np.mean((L0[:,1]>=obs[1])&(L0[:,2]>=obs[2]))
    print(f'   p (π=0): jedro {p[0]:.1e}, razmik {p[1]:.3f}, štetje {p[2]:.3f}; P(razmik&štetje)={joint:.4f} vs produkt {p[1]*p[2]:.4f}')
# conditional: are separation & count informative beyond lambda? within null, condition on core count >= observed impossible; instead test excluding core outliers
mc=np.abs(lam_o-1)>.1
print('opaženo brez jedra λ:',mc.sum(),'outlierjev; mean log sep',np.mean(np.log(sep_o[mc])).round(2),'vs z jedrom',np.mean(np.log(sep_o)).round(2),'; v škodo zmag.',np.sum(e_o[mc]<0),'/',mc.sum())
print('--- razmik brez jedra λ (ali signal izvira samo iz istih točk?) ---')
for bk in 'BCT':
    sp=np.concatenate([pool[bk][a][1][np.abs(pool[bk][a][0]-1)>.1] for a in range(4)])
    sp_all=np.concatenate([pool[bk][a][1] for a in range(4)])
    print(bk,'MW brez jedra p=%.3f'%stats.mannwhitneyu(sep_o[mc],sp,alternative='greater').pvalue,'| samo jedro (17) vs ničelni p=%.1e'%stats.mannwhitneyu(sep_o[~mc],sp_all,alternative='greater').pvalue)
print('--- razmik jedrnih točk: opaženo vs ničelne jedrne točke ---')
for bk in 'BCT':
    spc=np.concatenate([pool[bk][a][1][np.abs(pool[bk][a][0]-1)<=.1] for a in range(4)])
    print(bk,'mediana opaž. %.1f vs ničelno-jedro %.1f'%(np.median(sep_o[~mc]),np.median(spc)),' MW p=%.3f'%stats.mannwhitneyu(sep_o[~mc],spc,alternative='two-sided').pvalue)

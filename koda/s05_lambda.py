# Slika 14_lambda_mesanica (razdelek 4.6).
import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
lam,sig,NUL,res,names=pd.read_pickle('rezultati/lam.pkl')
RED='#E45756'; GRY='#9AA5B1'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,1,figsize=(9,9.5),gridspec_kw={'height_ratios':[1.4,1]},sharex=True)
ax=axs[0]; bw=.1; bins=np.arange(-2.5,3.01,bw)
ax.hist(np.clip(lam,-2.49,2.99),bins=bins,color=RED,ec='white',label='vseh 53 outlierjev (izbrani samo po |z| > 5)')
cols={'B':'#4C78A8','C':'#7FB37F','P':'#B07AA1','T':'#F2A541'}
x=np.linspace(-2.5,3,600)
for k,(l,s) in NUL.items():
    kde=stats.gaussian_kde(np.clip(l,-3,4)[np.random.default_rng(0).choice(len(l),20000)],bw_method=.15)
    ax.plot(x,53*kde(x)*bw,color=cols[k],lw=1.5,label=f'brez zamenjav: {names[k]}')
ax.axvline(0,color='k',lw=.8); ax.axvline(1,color=RED,lw=1,ls='--')
ax.text(0.03,ax.get_ylim()[1]*0.95 if False else 15.5,'0 = rezultat\nkot napoved',fontsize=9); ax.text(1.05,15.5,'1 = popolna\nzamenjava ZA↔PROTI',fontsize=9,color=RED)
ax.set_ylim(0,17.5); ax.set_ylabel('št. outlierjev'); ax.legend(frameon=False,fontsize=8.5,loc='upper left',bbox_to_anchor=(0,.88))
pi,lo,hi,lr=res['B']
ax.set_title(f'λ = (objavljeno − napoved)/(100 − 2·napoved)\nmešani model: {53*pi:.0f} zamenjav [95 % IZ {53*lo:.0f}–{53*hi:.0f}], test π = 0: χ² = {lr:.0f}',loc='left',fontsize=11.5,weight='bold')
ax=axs[1]; o=np.argsort(lam); y=np.arange(len(lam))
inn=np.abs(lam-1)<2*sig
ax.errorbar(np.clip(lam[o],-2.5,3),y,xerr=2*sig[o],fmt='o',ms=4,color='k',ecolor=GRY,elinewidth=1,capsize=0)
ax.scatter(np.clip(lam[o],-2.5,3)[inn[o]],y[inn[o]],s=30,color=RED,zorder=3,label=f'λ = 1 znotraj ±2σ ({inn.sum()})')
ax.axvline(1,color=RED,lw=1,ls='--'); ax.axvline(0,color='k',lw=.8)
ax.set_yticks([]); ax.set_xlabel('λ (razdalja od normale proti zrcalni osi; 0 = normala, 1 = zrcalna os)')
ax.legend(frameon=False,fontsize=9.5,loc='upper left'); ax.set_title('Vsak outlier z ±2σ negotovostjo, razvrščeno',loc='left',fontsize=11)
ax.set_xlim(-2.5,3)
plt.tight_layout(); plt.savefig('slike/14_lambda_mesanica.png',dpi=140)

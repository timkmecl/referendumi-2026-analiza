# Slika 11_z_po_zamenjavi_zoom (razdelek 4.3).
import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
O,res=pd.read_pickle('rezultati/band.pkl')
RED='#E45756'; GRY='#9AA5B1'; BLUE='#7FA7D6'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
zs=O.zs.values; inz=zs[np.abs(zs)<5]
nul=res['A'][3]; dens_bg=53*np.mean(np.abs(nul)<5)/10   # per unit z, approx flat
fig,axs=plt.subplots(3,1,figsize=(9,10),gridspec_kw={'height_ratios':[2.2,1,1]},sharex=True)
ax=axs[0]; bw=.5; bins=np.arange(-5,5.01,bw)
ax.hist(inz,bins=bins,color=RED,ec='white',label=f'opaženi outlierji (n = {len(inz)} od 53)')
x=np.linspace(-5,5,400)
ax.plot(x,np.full_like(x,dens_bg*bw),color='#555',ls='--',lw=1.5,label='pričakovano, če bi bilo vse naključno')
S=23; bg=(53-S)/53*dens_bg
ax.plot(x,(S*stats.norm.pdf(x)+bg)*bw,color='#1F5FA8',lw=2,label=f'{S} pravih zamenjav ~ N(0,1) + naključno ozadje')
ax.axvspan(-1,1,color='#E3EEFB',zorder=0)
ax.set_ylabel('št. outlierjev'); ax.legend(frameon=False,fontsize=9.5,loc='upper right')
ax.set_title('z po zamenjavi ZA↔PROTI za outlierje (|z| > 5), izsek |z po zamenjavi| < 5',loc='left',weight='bold')
# strip observed
rng=np.random.default_rng(3)
ax=axs[1]; ax.scatter(inz,rng.uniform(-.35,.35,len(inz)),s=55,color=RED,ec='white',lw=.6)
ax.axvspan(-1,1,color='#E3EEFB',zorder=0); ax.set_yticks([]); ax.set_ylim(-.6,.6); ax.spines['left'].set_visible(False)
ax.set_title(f'Opaženo: {np.sum(np.abs(zs)<1)} točk znotraj ±1',loc='left',fontsize=11)
# strip null example
ax=axs[2]
for j in range(5):
    s=rng.choice(nul,53); s=s[np.abs(s)<5]
    ax.scatter(s,np.full(len(s),j)+rng.uniform(-.25,.25,len(s)),s=30,color=GRY,ec='white',lw=.5)
ax.axvspan(-1,1,color='#E3EEFB',zorder=0); ax.set_yticks(range(5)); ax.set_yticklabels([f'sim. {j+1}' for j in range(5)],fontsize=9)
ax.spines['left'].set_visible(False); ax.set_xlim(-5,5)
ax.set_title('Pet simulacij "vse naključno" (vsakič 53 outlierjev): v povprečju 2,5 točke znotraj ±1',loc='left',fontsize=11)
ax.set_xlabel('z po zamenjavi')
plt.tight_layout(); plt.savefig('slike/11_z_po_zamenjavi_zoom.png',dpi=140)
print(np.sum(np.abs(zs)<1), len(inz))

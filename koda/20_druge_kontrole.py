# Druge kontrole (razdelka 3.2 in 4.7):
#  1. število oddanih glasovnic med referendumi na izstopajočih voliščih,
#  2. odstotki v izvorni datoteki se ujemajo s števili glasov (napaka ni pri branju),
#  3. zgoščenost zamenjav po okrajih,
#  4. primerjava enega referenduma z mediano okraja (brez ostalih referendumov),
#  5. OŠ Sostro 25: vse prerazporeditve parov (ZA, PROTI) med štirimi referendumi.
import pandas as pd,numpy as np,itertools
from scipy import stats
exec(open('koda/01_podatki.py').read().split("D=None")[0])   # huber_wls, F, K
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl'); b=pd.read_csv('rezultati/final_table.csv')
idx={(o,i):j for j,(o,i) in enumerate(zip(D['Volilni okraj'],D['ID Volišča']))}
lg=lambda p: np.log(np.clip(p,.005,.995)/(1-np.clip(p,.005,.995))); il=lambda x:1/(1+np.exp(-x))
sw=b[b.tipt=='zamenjava ZA↔PROTI']
# 1
I=[idx[(o,i)] for o,i in b[['okraj','id']].drop_duplicates().values]
odd=D.loc[I,['odd_'+k for k in K]]; spr=odd.max(axis=1)-odd.min(axis=1)
print('1. izstopajočih volišč:',len(I),' razlika v št. oddanih glasovnic med referendumi:',spr.value_counts().sort_index().to_dict())
# 2
src={k:pd.read_excel('podatki/'+F[k],sheet_name=1).set_index(['Volilni okraj','ID Volišča']) for k in K}
bad=[]
for _,x in sw.iterrows():
    r=src[x.ref].loc[(x.okraj,x.id)]; pz=100*r['Glasovi ZA']/(r['Glasovi ZA']+r['Glasovi PROTI'])
    if abs(pz-100*r['Odstotek glasov ZA'])>0.06: bad.append((x.okraj,x.id,x.ref,pz,r['Odstotek glasov ZA']))
print('2. odstotek ZA v datoteki se ujema s števili pri',len(sw)-len(bad),'od',len(sw),'zamenjav',bad)
# 3
nok=D['Volilni okraj'].nunique(); per=sw.okraj.value_counts()
print(f'3. zamenjave na {len(per)} od {nok} okrajev, največ v enem okraju: {per.max()} ({per.idxmax()})')
rng=np.random.default_rng(0); ok_cells=D['Volilni okraj'].values
# naključna razporeditev 26 zamenjav po zaznavnih celicah, verjetnost sorazmerna s številom glasov
SEP=np.column_stack([np.abs(1-2*R[False][k]['ph'])/(np.sqrt(R[False][k]['ph']*(1-R[False][k]['ph'])/D['n_'+k])*R[False][k]['c']) for k in K])
N=np.column_stack([D['n_'+k] for k in K]).astype(float); w=(N*(SEP>5)).ravel(); w/=w.sum()
mx=[]
for _ in range(20000):
    j=rng.choice(w.size,len(sw),p=w); mx.append(pd.Series(ok_cells[j//4]).value_counts().max())
print(f'   P(največ >= {per.max()} v enem okraju po naključju) = {np.mean(np.array(mx)>=per.max()):.2f}')
# 4
print('4. primerjava z mediano okraja (en referendum):')
hits=[]
for k in K:
    p=D['p_'+k].values; n=D['n_'+k].values.astype(float); med=D.groupby('Volilni okraj')['p_'+k].transform('median').values
    raw=(p-med)/np.sqrt(med*(1-med)/n); c=stats.median_abs_deviation(raw,scale='normal'); z=raw/c
    o=np.abs(z)>5; pr=D['ID Volišča'].values==901
    found=sum(abs(z[idx[(x.okraj,x.id)]])>5 for _,x in sw[sw.ref==k].iterrows()); hits.append(found)
    print(f'   {k:4s} c={c:.2f}  |z|>5: {o.sum()} celic, od tega predčasno glasovanje {np.sum(o&pr)}; najdenih zamenjav {found}')
print(f'   skupaj najdenih {sum(hits)} od {len(sw)} zamenjav ZA<->PROTI')
# 5 Sostro 25: 4! permutacij parov x 2^4 zamenjav znotraj para = 384 prerazporeditev
i=idx[('VO 4005 - LJUBLJANA MOSTE-POLJE 1',25)]
B={}
for a,k in enumerate(K):
    M=np.column_stack([np.ones(len(D))]+[lg(D['p_'+j]) for j in K if j!=k]); pc=D['p_'+k].clip(.02,.98)
    B[k]=huber_wls(lg(D['p_'+k]),M,D['n_'+k]*pc*(1-pc))
pairs=[(D['za_'+k][i],D['pr_'+k][i]) for k in K]; C={k:R[False][k]['c'] for k in K}
res=[]
for perm in itertools.permutations(range(4)):
    for fl in itertools.product([0,1],repeat=4):
        pp=[]
        for a in range(4):
            za,pr=pairs[perm[a]]
            if fl[a]: za,pr=pr,za
            pp.append((za/(za+pr),za+pr))
        z=[]
        for a,k in enumerate(K):
            x=np.r_[1,[lg(pp[j][0]) for j in range(4) if j!=a]]; ph=il(x@B[k])
            z.append((pp[a][0]-ph)/np.sqrt(ph*(1-ph)/pp[a][1])/C[k])
        res.append((np.sum(np.square(z)),perm,fl,np.round(z,1)))
res.sort(key=lambda t:t[0])
print(f'5. Sostro 25: {len(res)} prerazporeditev; objavljeno: vsota z^2={[r for r in res if r[1]==(0,1,2,3) and r[2]==(0,0,0,0)][0][0]:.0f}')
for r in res[:3]: print('   perm',[K[j] for j in r[1]],'zamenjava v paru',r[2],'vsota z^2=%.0f'%r[0],'z=',r[3])
mm=min(res,key=lambda r:np.abs(r[3]).max()); print('   najmanjši največji |z| med vsemi prerazporeditvami: %.1f'%np.abs(mm[3]).max(),'(perm',[K[j] for j in mm[1]],mm[2],')')
print('   prerazporeditev z |z|<3 pri vseh štirih:',sum(np.abs(r[3]).max()<3 for r in res))
zp=[r for r in res if r[1]==(1,0,2,3) and r[2]==(0,0,0,0)][0]; print('   zamenjava preiskava<->RTV: z =',zp[3],' (tujci ostanejo izstopajoči)')
# napoved iz sociale (edini referendum, ki na Sostru 25 ne izstopa), enopredikatorski model
print('   napoved iz sociale (enopredikatorski model):')
for k in ['PP','RTV','R43']:
    M=np.column_stack([np.ones(len(D)),lg(D['p_R42'])]); pc=D['p_'+k].clip(.02,.98); bb=huber_wls(lg(D['p_'+k]),M,D['n_'+k]*pc*(1-pc))
    php=il(M@bb); raw=(D['p_'+k]-php)/np.sqrt(php*(1-php)/D['n_'+k]); c=stats.median_abs_deviation(raw,scale='normal')
    print(f'     {k:4s} napoved {100*php[i]:.1f} %, objavljeno {100*D["p_"+k][i]:.1f} %, z={raw[i]/c:+.1f}')

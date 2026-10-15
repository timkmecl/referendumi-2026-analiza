# Večkratne zamenjave na istem volišču (razdelek 3.3): vseh 16 kombinacij zamenjav, najmanjša vsota z^2.
import pandas as pd,numpy as np,itertools
exec(open('koda/01_podatki.py').read().split("D=None")[0])
D,_=pd.read_pickle('rezultati/four.pkl'); R=pd.read_pickle('rezultati/R.pkl')
lg=lambda p: np.log(np.clip(p,.005,.995)/(1-np.clip(p,.005,.995))); il=lambda x:1/(1+np.exp(-x))
P=np.column_stack([D['p_'+k] for k in K]); N=np.column_stack([D['n_'+k] for k in K]).astype(float)
B={}
for a,k in enumerate(K):
    o=[j for j in range(4) if j!=a]
    M=np.column_stack([np.ones(len(D))]+[lg(P[:,j]) for j in o]); pc=np.clip(P[:,a],.02,.98)
    B[a]=huber_wls(lg(P[:,a]),M,N[:,a]*pc*(1-pc))
C=np.array([R[False][k]['c'] for k in K])
def score(Pc):
    Z=np.zeros_like(Pc)
    for a in range(4):
        o=[j for j in range(4) if j!=a]
        M=np.column_stack([np.ones(len(Pc))]+[lg(Pc[:,j]) for j in o]); ph=il(M@B[a])
        Z[:,a]=(Pc[:,a]-ph)/np.sqrt(ph*(1-ph)/N[:,a])/C[a]
    return Z
configs=list(itertools.product([0,1],repeat=4))
S=[];ZZ=[]
for cf in configs:
    Pc=P.copy()
    for a in range(4):
        if cf[a]: Pc[:,a]=1-Pc[:,a]
    Z=score(Pc); ZZ.append(Z); S.append((Z**2).sum(1))
S=np.column_stack(S); best=S.argmin(1); nsw=np.array([sum(c) for c in configs])
S0=S[:,0]
# best with <=1 swap
m1=np.array([nsw[j]<=1 for j in range(16)])
b1=np.where(m1)[0][S[:,m1].argmin(1)]
rows=[]
for i in range(len(D)):
    cb=configs[best[i]]
    if sum(cb)>=1 and S0[i]-S[i,best[i]]>25:
        rows.append(dict(okraj=D['Volilni okraj'][i][3:],id=D['ID Volišča'][i],vol=str(D['Volišče'][i])[:30],
          swap='+'.join(K[a] for a in range(4) if cb[a]),S0=round(S0[i],1),Sbest=round(S[i,best[i]],1),
          S_best1=round(S[i,b1[i]],1),best1='+'.join(K[a] for a in range(4) if configs[b1[i]][a]) or '-',
          maxz_after=round(np.abs(ZZ[best[i]][i]).max(),1),
          p=' '.join(f'{100*P[i,a]:.0f}' for a in range(4))))
r=pd.DataFrame(rows).sort_values('S0',ascending=False)
pd.set_option('display.width',250)
print(r.to_string())
print('\nnajboljša konfiguracija z 2+ zamenjavama:'); print(r[r.swap.str.count(r'\+')>=1].to_string())

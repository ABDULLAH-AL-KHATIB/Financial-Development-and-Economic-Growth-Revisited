"""Additional diagnostic: two-way FE augmented with country-specific linear trends."""
from pathlib import Path
import pandas as pd, numpy as np, math
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data'; T=ROOT/'tables'
P=pd.read_csv(D/'raw_panel.csv').sort_values(['iso3','year']).reset_index(drop=True)
P['log_gdppc']=np.log((P.rgdpo/P['pop']).where(P.rgdpo/P['pop']>0))
META=pd.read_csv(D/'country_meta.csv'); P=P.merge(META,on='iso3',how='left')
P['fd_z']=(P.fd-P.fd.mean())/P.fd.std()
CTRL=['log_gdppc','hc','trade','govcons','urban','inflation','unemp']
cols=['growth','fd_z']+CTRL
L=P[['iso3','year']+cols].copy(); L['year']+=1; L=L.rename(columns={c:c+'_L' for c in cols})
P=P.merge(L,on=['iso3','year'],how='left',validate='one_to_one')
E=P.dropna(subset=['growth','fd_z_L']+[c+'_L' for c in CTRL]).copy()
uc,ui=pd.factorize(E.iso3); tc,ti=pd.factorize(E.year); NU=len(ui); NT=len(ti); n=len(E)

def dm(v,codes,G):
    s=np.bincount(codes,weights=v,minlength=G); cnt=np.bincount(codes,minlength=G)
    return v-(s/np.maximum(cnt,1))[codes]
def within(v):
    v=np.asarray(v,float).copy()
    for _ in range(5000):
        old=v.copy(); v=dm(v,uc,NU); v=dm(v,tc,NT)
        if np.max(np.abs(v-old))<1e-12:return v
    raise RuntimeError('within transform did not converge')
# Trend columns: country-specific deviations from each country mean year.
year=E.year.to_numpy(float)
country_mean=np.bincount(uc,weights=year,minlength=NU)/np.bincount(uc,minlength=NU)
trel=year-country_mean[uc]
Z=np.zeros((n,NU-1)); keep=uc<NU-1; Z[np.where(keep)[0],uc[keep]]=trel[keep]
# The omitted final-country trend is the reference restriction; year effects absorb the common trend.
cols=['fd_z_L']+[c+'_L' for c in CTRL]
X=np.column_stack([within(E[c].to_numpy(float)) for c in cols]+[within(Z[:,j]) for j in range(NU-1)])
y=within(E.growth.to_numpy(float))
B=np.linalg.pinv(X.T@X); b=B@X.T@y; e=y-X@b
rank=np.linalg.matrix_rank(X,tol=1e-9)
# Covariance by country
scores=np.zeros((NU,X.shape[1])); np.add.at(scores,uc,X*e[:,None])
Vcl=NU/(NU-1)*(n-1)/max(n-(rank+NU+NT-1),1)*B@(scores.T@scores)@B
# DK by year
H=np.zeros((NT,X.shape[1])); np.add.at(H,tc,X*e[:,None]); S=H.T@H
L=max(1,int(math.floor(4*(NT/100)**(2/9))))
for ell in range(1,min(L,NT-1)+1):
    w=1-ell/(L+1); G=H[ell:].T@H[:-ell]; S+=w*(G+G.T)
Vdk=(NT/max(NT-1,1))*((n-1)/max(n-X.shape[1],1))*B@S@B
j=0; beta=b[j]
rows=[]
for name,V,df in [('Country-clustered',Vcl,NU-1),('Driscoll-Kraay',Vdk,NT-1)]:
    se=np.sqrt(max(V[j,j],0)); t=beta/se; p=2*stats.t.sf(abs(t),df)
    rows.append({'specification':'Two-way FE + country-specific linear trends','inference':name,'coef':beta,'se':se,'t':t,'p':p,'ci_low':beta-stats.t.ppf(.975,df)*se,'ci_high':beta+stats.t.ppf(.975,df)*se,'n':n,'countries':NU,'years':NT,'design_rank':rank,'trend_terms':NU-1})
out=pd.DataFrame(rows); out.to_csv(T/'T18_country_trend_robustness.csv',index=False)
print(out.to_string(index=False,float_format=lambda x:f'{x:.4f}'))

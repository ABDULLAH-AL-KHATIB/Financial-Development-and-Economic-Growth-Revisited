import pandas as pd, numpy as np, math
from scipy import stats
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT
P=pd.read_csv(base/'data/raw_panel.csv').sort_values(['iso3','year']).reset_index(drop=True)
P['log_gdppc']=np.log((P.rgdpo/P['pop']).where(P.rgdpo/P['pop']>0)); P['log_cap_pw']=np.log((P.rnna/P.emp).where(P.rnna/P.emp>0))
P['g_growth_pwt']=P.groupby('iso3')['rgdpna'].transform(lambda s:100*np.log(s/s.shift(1)))
def hp(y,l=100):
 n=len(y); D=np.zeros((n-2,n))
 for i in range(n-2):D[i,i:i+3]=[1,-2,1]
 return y-np.linalg.solve(np.eye(n)+l*D.T@D,y)
P['credit_gap']=np.nan
for c,g in P.groupby('iso3'):
 v=g.credit.to_numpy(float)
 if np.isfinite(v).sum()>=8: P.loc[g.index,'credit_gap']=hp(pd.Series(v).interpolate(limit_direction='both').to_numpy())
meta=pd.read_csv(base/'data/country_meta.csv'); P=P.merge(meta,on='iso3',how='left')
IMF=['fd','fi','fm','fid','fia','fie','fmd','fma','fme']
for c in IMF:P[c+'_z']=(P[c]-P[c].mean())/P[c].std()
CTRL=['log_gdppc','hc','trade','govcons','urban','inflation','unemp']
cols=list(dict.fromkeys(['growth','g_growth_pwt','credit','credit_gap','fd']+ [c+'_z' for c in IMF]+['log_gdppc','hc','trade','govcons','urban','inflation','unemp','log_cap_pw','labsh','gcf','school']))
L=P[['iso3','year']+cols].copy(); L['year']+=1; L=L.rename(columns={c:c+'_L' for c in cols})
P=P.merge(L,on=['iso3','year'],how='left',validate='one_to_one')
est=P.dropna(subset=['growth','fd_z_L']+[c+'_L' for c in CTRL]).copy()

def dm(v,codes,G):
 s=np.bincount(codes,weights=v,minlength=G); n=np.bincount(codes,minlength=G); return v-(s/np.maximum(n,1))[codes]
def within(v,uc,tc,NU,NT):
 v=np.asarray(v,float).copy()
 for i in range(5000):
  old=v.copy();v=dm(v,uc,NU);v=dm(v,tc,NT)
  if np.max(np.abs(v-old))<1e-12:return v
 raise Exception('no convergence')
def fit(df,clip=None,trim=None,drop_infl=False):
 d=df.copy()
 if trim is not None:d=d[d.inflation_L<=trim].copy()
 if clip is not None:
  lo,hi=d.inflation_L.quantile(clip[0]),d.inflation_L.quantile(clip[1]);d['inflation_L']=d.inflation_L.clip(lo,hi)
 xcols=['fd_z_L']+[c+'_L' for c in CTRL if not (drop_infl and c=='inflation')]
 if drop_infl:d=d.dropna(subset=['growth','fd_z_L']+[c+'_L' for c in CTRL if c!='inflation'])
 uc,us=pd.factorize(d.iso3);tc,ts=pd.factorize(d.year); NU=len(us);NT=len(ts)
 y=within(d.growth.to_numpy(float),uc,tc,NU,NT)
 X=np.column_stack([within(d[c].to_numpy(float),uc,tc,NU,NT) for c in xcols])
 B=np.linalg.pinv(X.T@X);b=B@X.T@y;e=y-X@b;k=X.shape[1];keff=k+NU+NT-1
 sc=np.zeros((NU,k));np.add.at(sc,uc,X*e[:,None])
 V=NU/max(NU-1,1)*(len(e)-1)/max(len(e)-keff,1)*B@(sc.T@sc)@B
 j=0;se=np.sqrt(max(V[j,j],0));t=b[j]/se;p=2*stats.t.sf(abs(t),NU-1)
 # DK covariance (Bartlett kernel; same lag rule as main code)
 H=np.zeros((NT,k)); np.add.at(H,tc,X*e[:,None]); S=H.T@H
 L=max(1,int(math.floor(4*(NT/100)**(2/9))))
 for ell in range(1,min(L,NT-1)+1):
  w=1-ell/(L+1); G=H[ell:].T@H[:-ell]; S+=w*(G+G.T)
 Vdk=(NT/max(NT-1,1))*((len(d)-1)/max(len(d)-k,1))*B@S@B
 sedk=np.sqrt(max(Vdk[j,j],0));tdk=b[j]/sedk;pdk=2*stats.t.sf(abs(tdk),NT-1)
 return {'specification':None,'beta':b[0],'se_country':se,'p_country':p,'se_DK':sedk,'p_DK':pdk,'n':len(d),'N':NU,'T':NT,'infl_max':d.inflation_L.max(),'infl_p99':d.inflation_L.quantile(.99)}
rows=[]
for nm,kwargs in [
 ('Baseline',{}),
 ('Inflation winsorized 1st/99th percentile',{'clip':(.01,.99)}),
 ('Inflation winsorized 0.5th/99.5th percentile',{'clip':(.005,.995)}),
 ('Exclude inflation > 1,000%',{'trim':1000}),
 ('Exclude inflation > 500%',{'trim':500}),
 ('Drop inflation control',{'drop_infl':True}),
]:
 r=fit(est,**kwargs);r['specification']=nm;rows.append(r)
out=pd.DataFrame(rows);out.to_csv(base/'tables/T16_inflation_outlier_robustness.csv',index=False)
print(out.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('High infl count >500',int((est.inflation_L>500).sum()),'>1000',int((est.inflation_L>1000).sum()))

"""Independently verify the main two-way FE DK covariance with Statsmodels HAC-groupsum."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
ROOT=Path(__file__).resolve().parents[1]
P=pd.read_csv(ROOT/'data/raw_panel.csv').sort_values(['iso3','year']).reset_index(drop=True)
P['log_gdppc']=np.log((P.rgdpo/P['pop']).where(P.rgdpo/P['pop']>0))
P=P.merge(pd.read_csv(ROOT/'data/country_meta.csv'),on='iso3',how='left')
P['fd_z']=(P.fd-P.fd.mean())/P.fd.std()
CTRL=['log_gdppc','hc','trade','govcons','urban','inflation','unemp']
cols=['growth','fd_z']+CTRL
L=P[['iso3','year']+cols].copy(); L['year']+=1; L=L.rename(columns={c:c+'_L' for c in cols})
P=P.merge(L,on=['iso3','year'],how='left',validate='one_to_one')
E=P.dropna(subset=['growth','fd_z_L']+[c+'_L' for c in CTRL]).copy()
uc,us=pd.factorize(E.iso3); tc,ts=pd.factorize(E.year); NU=len(us); NT=len(ts)
def demean(v,codes,G):
    s=np.bincount(codes,weights=v,minlength=G); n=np.bincount(codes,minlength=G)
    return v-(s/np.maximum(n,1))[codes]
def within(v):
    v=np.asarray(v,float).copy()
    for _ in range(5000):
        old=v.copy(); v=demean(v,uc,NU); v=demean(v,tc,NT)
        if np.max(np.abs(v-old))<1e-12: return v
    raise RuntimeError('Two-way within transform did not converge')
y=within(E.growth.to_numpy(float))
X=np.column_stack([within(E[c+'_L'].to_numpy(float)) for c in ['fd_z']+CTRL])
LAGS=max(1,int(np.floor(4*(NT/100)**(2/9))))
fit=sm.OLS(y,X).fit(cov_type='hac-groupsum',cov_kwds={'time':tc,'maxlags':LAGS,'use_correction':'cluster'},use_t=True)
se=float(fit.bse[0])
reported=float(pd.read_csv(ROOT/'tables/T03_primary_inference.csv').query("inference == 'Driscoll-Kraay'").iloc[0].se)
print(f'Statsmodels HAC-groupsum SE: {se:.10f}')
print(f'Replication DK SE:           {reported:.10f}')
print(f'Absolute difference:         {abs(se-reported):.3e}')
assert abs(se-reported)<1e-5, 'DK covariance does not match independent Statsmodels cross-check'

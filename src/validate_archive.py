from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
panel=pd.read_csv(ROOT/'data/raw_panel.csv')
meta=pd.read_csv(ROOT/'data/country_meta.csv')
T03=pd.read_csv(ROOT/'tables/T03_primary_inference.csv')
T18=pd.read_csv(ROOT/'tables/T18_country_trend_robustness.csv')
assert len(panel)==7686 and panel.iso3.nunique()==183 and panel.year.min()==1980 and panel.year.max()==2021
assert not panel.duplicated(['iso3','year']).any() and not meta.duplicated(['iso3']).any()
dk=T03.loc[T03.inference.eq('Driscoll-Kraay')].iloc[0]
trend=T18.loc[T18.inference.eq('Driscoll-Kraay')].iloc[0]
assert 0.39 < dk.se < 0.41 and dk.p < .01 and dk.hi < 0
assert trend.p > .05 and trend.coef < 0
print('Archive integrity checks passed.')
print(f'Baseline DK: beta={dk.coef:.4f}, SE={dk.se:.4f}, p={dk.p:.4g}')
print(f'Country-trend DK: beta={trend.coef:.4f}, SE={trend.se:.4f}, p={trend.p:.4f}')

T04=pd.read_csv(ROOT/'tables/T04_estimators.csv')
T08=pd.read_csv(ROOT/'tables/T08_nonlinearity.csv')
T09=pd.read_csv(ROOT/'tables/T09_diagnostics.csv')
cre=T04.loc[T04.specification.eq('Mundlak CRE (quasi-demeaned GLS)')].iloc[0]
ah=T04.loc[T04.specification.str.startswith('Anderson-Hsiao')].iloc[0]
within=T09.loc[T09.test.eq('Within AR(1): fd_z on fd_z_L (country and year FE)')].iloc[0]
thr=T08.loc[T08.model.str.contains('Hansen threshold')].iloc[0]
assert abs(float(cre.coef)+0.7357319)<1e-5 and abs(float(cre.se)-0.4696750)<1e-5
assert abs(float(cre.p)-0.1195970)<1e-5 and 'Wald chi2(8)' in cre.note
assert pd.isna(ah.coef) and pd.isna(ah.se) and pd.isna(ah.p)
assert abs(float(within.stat)-0.82314042)<1e-7
assert 'boot=1999' in thr.model and abs(float(thr.p)-0.4985)<1e-12
print(f'Mundlak CRE: beta={cre.coef:.4f}, SE={cre.se:.4f}, p={cre.p:.4f}; {cre.note}')
print(f'Within FDI AR(1): {within.stat:.6f}, p={within.p:.4g}')
print(f'Threshold bootstrap: reps=1999, p={thr.p:.4f}')

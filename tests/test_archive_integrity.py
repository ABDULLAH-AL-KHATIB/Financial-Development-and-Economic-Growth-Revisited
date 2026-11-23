from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_frozen_input_schema_and_keys():
    panel = pd.read_csv(ROOT / 'data' / 'raw_panel.csv')
    meta = pd.read_csv(ROOT / 'data' / 'country_meta.csv')
    assert len(panel) == 7686
    assert panel['iso3'].nunique() == 183
    assert panel['year'].min() == 1980 and panel['year'].max() == 2021
    assert not panel.duplicated(['iso3', 'year']).any()
    assert not meta.duplicated(['iso3']).any()

def test_corrected_primary_covariance_table():
    t = pd.read_csv(ROOT / 'tables' / 'T03_primary_inference.csv')
    dk = t.loc[t['inference'].eq('Driscoll-Kraay')].iloc[0]
    assert abs(float(dk['coef']) + 1.3775736934) < 1e-8
    assert 0.39 < float(dk['se']) < 0.41
    assert float(dk['p']) < 0.01
    assert float(dk['lo']) < float(dk['hi']) < 0

def test_country_trend_sensitivity_is_present():
    t = pd.read_csv(ROOT / 'tables' / 'T18_country_trend_robustness.csv')
    dk = t.loc[t['inference'].eq('Driscoll-Kraay')].iloc[0]
    assert float(dk['p']) > 0.05
    assert float(dk['coef']) < 0

def test_corrected_mundlak_cre_and_unidentified_iv_are_reported_safely():
    t = pd.read_csv(ROOT / 'tables' / 'T04_estimators.csv')
    cre = t.loc[t['specification'].eq('Mundlak CRE (quasi-demeaned GLS)')].iloc[0]
    ah = t.loc[t['specification'].str.startswith('Anderson-Hsiao')].iloc[0]
    assert abs(float(cre['coef']) - (-0.7357319)) < 1e-5
    assert abs(float(cre['se']) - 0.4696750) < 1e-5
    assert abs(float(cre['p']) - 0.1195970) < 1e-5
    assert 'Wald chi2(8)' in cre['note']
    assert pd.isna(ah['coef']) and pd.isna(ah['se']) and pd.isna(ah['p'])
    assert 'Not identified' in ah['note'] and 'F=0.0' in ah['note']

def test_within_ar1_is_reported_separately_from_pooled_description():
    t = pd.read_csv(ROOT / 'tables' / 'T09_diagnostics.csv')
    row = t.loc[t['test'].eq('Within AR(1): fd_z on fd_z_L (country and year FE)')].iloc[0]
    assert abs(float(row['stat']) - 0.82314042) < 1e-7
    assert float(row['p']) < 1e-90
    pooled = t.loc[t['test'].eq('Pooled descriptive AR(1): fd_z_L')].iloc[0]
    assert 1.0 < float(pooled['stat']) < 1.01

def test_threshold_refresh_uses_1999_replications():
    t = pd.read_csv(ROOT / 'tables' / 'T08_nonlinearity.csv')
    row = t.loc[t['model'].str.contains('Hansen threshold')].iloc[0]
    assert 'boot=1999' in row['model']
    assert abs(float(row['p']) - 0.4985) < 1e-12

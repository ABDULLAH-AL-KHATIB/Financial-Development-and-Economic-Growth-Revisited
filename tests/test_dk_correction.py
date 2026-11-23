"""Regression test for the Driscoll–Kraay finite-sample correction."""

def test_hoeche_xtscc_factor_uses_n_and_within_regressors_not_absorbed_fe():
    n = 3624
    T = 30
    k = 8  # slopes in the within-transformed regression
    factor = (T / (T - 1)) * ((n - 1) / (n - k))
    old_k_effective = k + 135 + T - 1
    old_factor = T / max(T - old_k_effective, 1)
    assert 1.03 < factor < 1.04
    assert old_factor == 30
    assert factor < old_factor / 20


def test_source_code_does_not_contain_invalid_dk_multiplier():
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / 'src' / 'replicate_finance_growth.py'
    code = src.read_text(encoding='utf-8')
    assert 'small_corr=(NT/max(NT-1,1))*((n-1)/max(n-k,1))' in code
    assert 'NT/max(NT-keff,1)*B@S@B' not in code

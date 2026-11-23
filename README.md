# Replication archive: When Finance Deepens, Does Growth Follow?

This repository accompanies the manuscript and contains the frozen merged panel, source provenance, deterministic analysis code, Colab notebook, software environment, tests, output tables, and figures.

## Empirical design

The source universe contains 7,686 country-year observations for 183 economies over 1980–2021. The complete-case estimation sample contains 3,624 observations for 135 countries over 1992–2021. The baseline model includes country and year fixed effects and relates annual real GDP growth to lagged overall IMF Financial Development Index (FDI) and lagged macroeconomic controls.

## Principal results

- Baseline within estimate: −1.3776 percentage points per standard-deviation increase in lagged FDI.
- Corrected Driscoll–Kraay standard error: 0.3973; p = 0.001662; 95% CI [−2.190, −0.565].
- Country-specific-trend estimate: −1.2091; Driscoll–Kraay p = 0.1199.
- Mundlak country-mean joint test: χ²(8) = 32.4676; p = 0.0000768.
- Within AR(1) for the IMF FDI: 0.8231404 (country-clustered SE 0.0128777).
- Threshold bootstrap: 1,999 replications; 40-point grid; p = 0.4985; no threshold is detected.
- Forecast holdout: 616 observations, 2017–2021; Elastic Net RMSE 5.373 versus 5.423 for the two-way FE benchmark. Dependence-aware tests do not establish predictive superiority.

## Reproduction

1. Open `notebooks/Finance_Growth_Replication_Colab.ipynb` in Google Colab.
2. Upload or mount this repository folder in the runtime.
3. Install the pinned-compatible dependencies listed in `requirements.txt`.
4. Run the notebook from top to bottom. The deterministic script reads only the frozen local data snapshot and does not issue live source-data requests.
5. Inspect `manifest.json`, `run.log`, `tables/`, `figures/`, and the test suite after execution.

## Data and licensing

The frozen panel combines the IMF Financial Development Index, World Development Indicators, and Penn World Table 11.0. See `data/SOURCE_ATTRIBUTION.md` and `data/SNAPSHOT_README.md` for source definitions, provenance, and third-party reuse conditions. The MIT licence applies to the original code only; it does not override licences or terms attached to third-party data.

## Execution record

The archive preserves the completed 1,999-draw run log, which records all nine forecasting models and the 616-observation holdout. The post-patch diagnostic rerun regenerated the updated estimator, Mundlak, within-AR(1), threshold, and diagnostic outputs; its execution log stops during MLP-within tuning. The CRE/within-AR(1) patch did not alter the forecasting code, and the completed forecast outputs from the successful full run are retained. This execution history is stated explicitly to preserve auditability.

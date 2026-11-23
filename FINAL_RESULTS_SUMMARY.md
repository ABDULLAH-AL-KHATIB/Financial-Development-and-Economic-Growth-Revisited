# Replication results accompanying the manuscript

- Frozen source panel: 7,686 country-year rows, 183 economies, 1980–2021.
- Complete-case estimation sample: 3,624 observations, 135 countries, 1992–2021.
- Baseline FE coefficient: −1.3776; corrected Driscoll–Kraay SE: 0.3973; p = 0.001662; 95% CI [−2.190, −0.565].
- Country-specific trend coefficient: −1.2091; DK p = 0.1199.
- Mundlak CRE country-mean test: χ²(8) = 32.4676; p = 0.0000768.
- Within AR(1), IMF FDI: 0.8231404; country-clustered SE = 0.0128777.
- Threshold bootstrap: 1,999 draws; 40-point grid; τ = 0.251638 (grid maximiser only); sup-W = 3.2304; p = 0.4985.
- Forecast holdout: 616 observations, 2017–2021; Elastic Net RMSE = 5.373; FE benchmark RMSE = 5.423.
- Forecast comparisons do not establish dependence-robust predictive superiority.

The archive contains the completed original 1,999-draw nine-model run log and the later post-patch diagnostic rerun log. The latter stops during MLP-within tuning in this execution record; the forecasting code was not changed by the CRE/within-AR(1) diagnostic patch, and the completed forecast outputs from the full run are retained.

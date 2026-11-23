# Frozen data snapshot

`raw_panel.csv` and `country_meta.csv` are the merged, frozen inputs extracted from the user-supplied replication archive. The merged source panel contains 7,686 country-year rows for 183 economies and years 1980–2021. The primary complete-case regression sample contains 3,624 observations from 135 countries in 1992–2021.

The package includes `pwt110.xlsx` as the source PWT 11.0 workbook. The main replication script uses `raw_panel.csv` to guarantee the same merged input across platforms. It does not re-download World Bank or IMF series. Check current source vintages separately before refreshing the analysis.

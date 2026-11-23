# Data sources, provenance and reuse

This package freezes the merged country-year input used by the supplied analysis. The source panel contains 7,686 rows (183 country codes × 42 years, 1980–2021); the complete-case primary estimation sample is unbalanced and contains 3,624 country-years for 135 countries over 1992–2021.

## Sources

1. **IMF Financial Development Index (FDI).** Nine annual series: FD, FI, FM, FID, FIA, FIE, FMD, FMA and FME. The official IMF dataset describes the index as a multidimensional measure of financial-institution and financial-market depth, access and efficiency. Source: https://data.imf.org/Datasets/FDI. Methodology reference: Svirydzenka (2016), IMF Working Paper 16/5, DOI: https://doi.org/10.5089/9781513583709.001. IMF data are redistributed with attribution and remain subject to the IMF's Data Terms of Use: https://www.imf.org/en/About/copyright-and-terms.
2. **World Development Indicators (WDI).** Source: World Bank, https://databank.worldbank.org/source/world-development-indicators and API documentation at https://datahelpdesk.worldbank.org/knowledgebase/articles/898599-indicator-api-queries. World Bank-produced datasets are generally CC BY 4.0 unless the individual dataset states otherwise: https://datacatalog.worldbank.org/public-licenses.
3. **Penn World Table 11.0.** Frozen source workbook: `pwt110.xlsx`. Dataset DOI: https://doi.org/10.34894/FABVLR. PWT 11.0 is documented as covering 185 economies, 1950–2023; the research panel here ends in 2021 because the merged source panel is limited by the IMF-FDI/WDI intersection in the supplied extraction. PWT 11.0 data are CC BY 4.0 according to the dataset provenance record. Cite Feenstra, Inklaar and Timmer (2015), DOI: https://doi.org/10.1257/aer.20130954, and the v11.0 dataset DOI.

## Transformations

The nine IMF index series are standardised to z-scores over the merged raw panel. The primary exposure is the one-year lag of the overall FD z-score. Controls are also lagged one year. Annual GDP growth is the World Bank WDI series `NY.GDP.MKTP.KD.ZG`. The PWT alternative growth series is 100 × the first difference of log `rgdpna`. The HP-filtered credit gap is constructed country by country with lambda = 100 after linear interpolation within a country where at least eight credit observations are present.

The frozen `raw_panel.csv` is a derived merged input, not a raw download of each source. Users must cite the upstream datasets, disclose transformations, and comply with the separate source terms. The MIT licence in the repository applies to the original analysis code only, not to third-party source data.

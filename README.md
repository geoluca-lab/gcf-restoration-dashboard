# GCF Restoration Targeting Dashboard — Chad

Interactive geospatial dashboard developed to support the targeting of pastoral areas with restoration potential under the SCALE / GCF project in Chad.

The application combines biophysical screening results with population estimates and complementary operational information to help identify priority areas for further assessment and intervention planning.

## Live dashboard

**Dashboard:** https://gcf-restoration-dashboard.streamlit.app/

## Project objective

The analysis supports the identification of pastoral areas presenting a potential for restoration in the study area, covering:

- Sila
- Ouaddaï
- Salamat
- Moyen-Chari

The workflow distinguishes between:

1. **Biophysical screening** — identification of areas where criteria related to pastoral land presence and land degradation overlap.
2. **Operational targeting** — aggregation of the screening results into 25 km² hexagons and combination with estimated population.
3. **Final prioritisation** — production of a targeting score and visualisation of complementary information such as transhumance routes and WFP intervention sites.

## Methodological overview

### 1. Pastoral area proxy

Global Pasture Watch (GPW) is used to identify persistent natural / semi-natural grassland as a proxy for pastoral areas.

The proxy is intentionally conservative and does not represent every possible pastoral land-use configuration.

### 2. Restoration potential

The pastoral proxy is combined with indicators related to land degradation, vegetation stress, rainfall deficit, livestock pressure and environmental constraints.

The workflow produces two scenarios:

- **Moderate** — more inclusive;
- **Conservative** — more restrictive.

For the final targeting score, areas identified under the two scenarios are treated together as areas presenting a potential for restoration.

### 3. Operational targeting

Results are aggregated into a grid of **25 km² hexagons**.

For each hexagon, the dashboard includes:

- estimated population;
- share and area of pastoral land presenting restoration potential;
- distance to the nearest transhumance route;
- distance to the nearest WFP intervention site.

The final targeting score combines:

- **population estimate — 3 classes**;
- **restoration potential — 4 classes**.

The resulting score ranges from **1 to 7**. Higher scores correspond to hexagons combining relatively high restoration potential and relatively high population compared with the rest of the study area.

Distances to transhumance routes and WFP sites are displayed as complementary information and are **not included in the score**.

## Main data sources

The analysis currently relies on several global and operational datasets, including:

- Global Pasture Watch (GPW)
- Trends.Earth / SDG 15.3.1
- MODIS vegetation indicators
- CHIRPS rainfall data
- GPW livestock distribution data
- GloFAS flood hazard data
- WorldPop population estimates
- OIM-DTM Transhumance Tracking Tool
- WFP intervention-site data
- ESA WorldCover 2021 for contextual land-cover visualisation

Detailed indicators, thresholds and methodological choices are documented in the methodological annex.

## Repository structure

```text
gcf-restoration-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
│
├── docs/
│   └── Annex_3_Methodologie_Ciblage.pdf
│
├── scripts/
│   └── gcf_rangeland_screening_local_ANNOTATED.py
│
├── notebooks/
│
├── utils/
│
└── data/
    ├── raster/
    └── vector/
```

### Recommended organisation

- `docs/` — current methodological documentation;
- `scripts/` — reproducible processing scripts used to generate analysis outputs;
- `notebooks/` — exploratory analyses and development notebooks;
- `utils/` — reusable helper functions;
- `data/` — dashboard input datasets only.

Temporary or test datasets should preferably be removed from the repository or stored in a clearly identified development/test folder.

## Methodology and reproducibility

The methodological annex should be consulted together with the processing script:

- **Methodological annex:** [`docs/Annex_3_Methodologie_Ciblage.pdf`](docs/Annex_3_Methodologie_Ciblage.pdf)
- **Restoration screening script:** [`scripts/gcf_rangeland_screening_local_ANNOTATED.py`](scripts/gcf_rangeland_screening_local_ANNOTATED.py)

The script documents the geospatial processing used to produce the restoration-potential layer displayed in the dashboard.

Before publishing processing scripts, local absolute paths, credentials, API keys and organisation-specific private paths should be removed or replaced with configurable relative paths.

## Run the dashboard locally

From the repository folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## Dashboard use

Users can:

- switch between contextual basemaps;
- display or hide the analytical layers;
- inspect methodological information for each layer;
- select priority hexagons interactively;
- add comments to selected hexagons;
- export selections for further review;
- reload a previous selection;
- consult the methodological documentation and processing code.

## Notes

This repository supports an operational targeting workflow. The spatial results are intended to guide prioritisation and further assessment rather than replace field validation or programme-level decision making.

---

**IMPACT Initiatives / REACH — Chad**

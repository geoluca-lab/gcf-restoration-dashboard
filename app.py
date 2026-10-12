import streamlit as st
import folium
import geopandas as gpd
import rasterio
import pyogrio
import numpy as np
import json

from pathlib import Path
from affine import Affine
from branca.element import MacroElement
from jinja2 import Template
from streamlit_folium import folium_static

from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling
)


# ==================================================
# CONFIGURATION STREAMLIT
# ==================================================

st.set_page_config(
    page_title="Dashboard GCF",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# PAGE 100%
# ==================================================

st.markdown(
    """
    <style>

    html,
    body {
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    html,
    body,
    [class*="css"] {
        font-family: "Arial Nova", Arial, sans-serif;
    }

    header[data-testid="stHeader"],
    footer,
    #MainMenu,
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    div[data-testid="stAppViewContainer"],
    div[data-testid="stAppViewContainer"] > .main {
        margin: 0 !important;
        padding: 0 !important;
    }

    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }

    iframe {
        width: 100% !important;
        height: 100vh !important;
        border: none !important;
        display: block !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# CHEMINS DES DONNEES
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

gpkg_path = (
    BASE_DIR
    / "data"
    / "vector"
    / "GCF geo.gpkg"
)

restoration_raster_path = (
    BASE_DIR
    / "data"
    / "raster"
    / "TCD_restoration_suitability_moderate_conservative.tif"
)

population_raster_dir = (
    BASE_DIR
    / "data"
    / "raster"
)

population_raster_path = (
    population_raster_dir
    / "pop_zone_AOI_agg3.tif"
)


# ==================================================
# IDENTIFICATION DES COUCHES
# ==================================================

@st.cache_data
def get_available_layers(path):

    layers = pyogrio.list_layers(path)

    return layers[:, 0].tolist()


available_layers = get_available_layers(
    gpkg_path
)


def resolve_layer_name(simple_name):

    candidates = [
        f"main.{simple_name}",
        simple_name
    ]

    for candidate in candidates:

        if candidate in available_layers:
            return candidate

    raise ValueError(
        f"Layer '{simple_name}' non trovato. "
        f"Layer disponibili: {available_layers}"
    )


LAYER_TARGETISATION = resolve_layer_name(
    "hexa_target"
)

LAYER_AOI = resolve_layer_name(
    "AOI"
)

LAYER_SITES_PAM = resolve_layer_name(
    "Sites_PAM"
)

LAYER_TRANSHUMANCE = resolve_layer_name(
    "transhumance"
)


# ==================================================
# CHARGEMENT VECTEURS
# ==================================================

@st.cache_data
def load_vector_layer(
    path,
    layer_name,
    keep_columns=None
):

    gdf = gpd.read_file(
        path,
        layer=layer_name,
        engine="pyogrio"
    )

    if gdf.crs is None:

        raise ValueError(
            f"Il layer {layer_name} non ha un CRS definito."
        )

    gdf = gdf.to_crs(
        epsg=4326
    )

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ]

    if keep_columns is not None:

        existing_columns = [
            col
            for col in keep_columns
            if col in gdf.columns
        ]

        gdf = gdf[
            existing_columns
            + ["geometry"]
        ]

    else:

        gdf = gdf[
            ["geometry"]
        ]

    return gdf


# ==================================================
# RASTER POTENTIEL DE RESTAURATION
# ==================================================

@st.cache_data
def load_suitability_raster(
    path,
    max_size=2200
):

    with rasterio.open(path) as src:

        if src.crs is None:

            raise ValueError(
                "Il raster non ha un CRS definito."
            )

        dst_crs = "EPSG:4326"

        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds
        )

        largest_dimension = max(
            width,
            height
        )

        if largest_dimension > max_size:

            scale = (
                largest_dimension
                / max_size
            )

            new_width = max(
                1,
                int(width / scale)
            )

            new_height = max(
                1,
                int(height / scale)
            )

            transform = (
                transform
                * Affine.scale(
                    width / new_width,
                    height / new_height
                )
            )

            width = new_width
            height = new_height

        data = np.zeros(
            (
                height,
                width
            ),
            dtype=np.uint8
        )

        reproject(
            source=rasterio.band(
                src,
                1
            ),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            dst_nodata=0
        )

        rgba = np.zeros(
            (
                height,
                width,
                4
            ),
            dtype=np.uint8
        )

        # Moderate = bleu foncé
        moderate = (
            data == 1
        )

        rgba[moderate] = [
            43,
            108,
            176,
            220
        ]

        # Conservative = bleu plus foncé
        conservative = (
            data == 2
        )

        rgba[conservative] = [
            8,
            48,
            107,
            230
        ]

        left = transform.c
        top = transform.f

        right = (
            left
            + transform.a * width
        )

        bottom = (
            top
            + transform.e * height
        )

        bounds = [
            [bottom, left],
            [top, right]
        ]

        return (
            rgba,
            bounds
        )


# ==================================================
# RASTER POPULATION
# ==================================================

@st.cache_data
def load_population_raster(
    path,
    max_size=2200
):

    with rasterio.open(path) as src:

        if src.crs is None:

            raise ValueError(
                "Il raster di popolazione non ha un CRS definito."
            )

        dst_crs = "EPSG:4326"

        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds
        )

        largest_dimension = max(
            width,
            height
        )

        if largest_dimension > max_size:

            scale = (
                largest_dimension
                / max_size
            )

            new_width = max(
                1,
                int(width / scale)
            )

            new_height = max(
                1,
                int(height / scale)
            )

            transform = (
                transform
                * Affine.scale(
                    width / new_width,
                    height / new_height
                )
            )

            width = new_width
            height = new_height

        data = np.full(
            (
                height,
                width
            ),
            np.nan,
            dtype=np.float32
        )

        reproject(
            source=rasterio.band(
                src,
                1
            ),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            dst_nodata=np.nan
        )

        valid = (
            np.isfinite(data)
            & (data > 0)
        )

        rgba = np.zeros(
            (
                height,
                width,
                4
            ),
            dtype=np.uint8
        )

        if valid.any():

            log_values = np.log1p(
                data[valid]
            )

            vmin = np.percentile(
                log_values,
                2
            )

            vmax = np.percentile(
                log_values,
                98
            )

            if vmax <= vmin:
                vmax = vmin + 1

            normalized = np.zeros(
                data.shape,
                dtype=np.float32
            )

            normalized[valid] = (
                (
                    np.log1p(
                        data[valid]
                    )
                    - vmin
                )
                /
                (
                    vmax
                    - vmin
                )
            )

            normalized = np.clip(
                normalized,
                0,
                1
            )

            x = normalized[
                valid
            ]

            # Gris clair -> noir
            intensity = (
                240
                - x * 230
            ).astype(
                np.uint8
            )

            rgba[..., 0][valid] = intensity
            rgba[..., 1][valid] = intensity
            rgba[..., 2][valid] = intensity
            rgba[..., 3][valid] = 210

        left = transform.c
        top = transform.f

        right = (
            left
            + transform.a * width
        )

        bottom = (
            top
            + transform.e * height
        )

        bounds = [
            [bottom, left],
            [top, right]
        ]

        return (
            rgba,
            bounds
        )


# ==================================================
# COUCHES VECTORIELLES GENERIQUES
# ==================================================

def add_vector_layer(
    group,
    gdf,
    color,
    weight=2,
    fill_opacity=0.10,
    radius=5
):

    if gdf.empty:
        return

    geometry_types = set(
        gdf.geometry.geom_type
        .dropna()
        .tolist()
    )

    # Points
    if geometry_types.issubset(
        {
            "Point",
            "MultiPoint"
        }
    ):

        folium.GeoJson(
            gdf,
            pane="points_pane",
            marker=folium.CircleMarker(
                radius=radius,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.9
            ),
            interactive=False
        ).add_to(
            group
        )

    # Lignes
    elif geometry_types.issubset(
        {
            "LineString",
            "MultiLineString"
        }
    ):

        folium.GeoJson(
            gdf,
            pane="vector_pane",
            style_function=lambda feature: {
                "color": color,
                "weight": weight,
                "opacity": 0.95
            },
            interactive=False
        ).add_to(
            group
        )

    # Polygones
    else:

        folium.GeoJson(
            gdf,
            pane="vector_pane",
            style_function=lambda feature: {
                "color": color,
                "weight": weight,
                "fillColor": color,
                "fillOpacity": fill_opacity
            },
            interactive=False
        ).add_to(
            group
        )


# ==================================================
# CHARGEMENT DES DONNEES
# ==================================================

aoi = load_vector_layer(
    gpkg_path,
    LAYER_AOI
)

transhumance = load_vector_layer(
    gpkg_path,
    LAYER_TRANSHUMANCE
)

sites_pam = load_vector_layer(
    gpkg_path,
    LAYER_SITES_PAM
)

targetisation = load_vector_layer(
    gpkg_path,
    LAYER_TARGETISATION,
    keep_columns=[
        "GRID_ID",
        "Pop",
        "pop_class_1",
        "pop_label",
        "cand_pct",
        "cand_class",
        "cand_label",
        "target_score",
        "NEAR_DIST",
        "NEAR_DIST_PAM",
        "NEAR_DIST_PAM_SITE"
    ]
)


# ==================================================
# CHARGEMENT DES RASTERS
# ==================================================

population_img, population_bounds = load_population_raster(
    population_raster_path
)

suitability_img, suitability_bounds = load_suitability_raster(
    restoration_raster_path
)


# ==================================================
# EXTENT
# ==================================================

minx, miny, maxx, maxy = aoi.total_bounds

center_lat = (
    miny + maxy
) / 2

center_lon = (
    minx + maxx
) / 2


# ==================================================
# MAP
# ==================================================

m = folium.Map(
    location=[
        center_lat,
        center_lon
    ],
    zoom_start=7,
    tiles=None,
    control_scale=True,
    prefer_canvas=True
)


# ==================================================
# PANES
# ==================================================

folium.map.CustomPane(
    "targetisation_pane",
    z_index=410,
    pointer_events=True
).add_to(m)

folium.map.CustomPane(
    "raster_pane",
    z_index=430,
    pointer_events=False
).add_to(m)

folium.map.CustomPane(
    "vector_pane",
    z_index=450,
    pointer_events=False
).add_to(m)

folium.map.CustomPane(
    "points_pane",
    z_index=460,
    pointer_events=False
).add_to(m)


# ==================================================
# FONDS DE CARTE
# ==================================================

folium.TileLayer(
    tiles="OpenStreetMap",
    name="OpenStreetMap",
    overlay=False,
    control=True,
    show=True
).add_to(m)


folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    attr=(
        "Esri, Maxar, Earthstar Geographics, "
        "and the GIS User Community"
    ),
    name="Imagerie satellite",
    overlay=False,
    control=True,
    show=False
).add_to(m)


# ==================================================
# FEATURE GROUPS
# ==================================================

target_group = folium.FeatureGroup(
    name="Targetisation",
    overlay=True,
    control=True,
    show=True
)

target_group.add_to(m)


population_group = folium.FeatureGroup(
    name="Population",
    overlay=True,
    control=True,
    show=False
)

population_group.add_to(m)


restoration_group = folium.FeatureGroup(
    name="Potentiel de restauration",
    overlay=True,
    control=True,
    show=False
)

restoration_group.add_to(m)


aoi_group = folium.FeatureGroup(
    name="Zone d'étude",
    overlay=True,
    control=True,
    show=False
)

aoi_group.add_to(m)


transhumance_group = folium.FeatureGroup(
    name="Routes de transhumance",
    overlay=True,
    control=True,
    show=False
)

transhumance_group.add_to(m)


sites_group = folium.FeatureGroup(
    name="Sites PAM",
    overlay=True,
    control=True,
    show=False
)

sites_group.add_to(m)


# ==================================================
# POPULATION
# ==================================================

folium.raster_layers.ImageOverlay(
    image=population_img,
    bounds=population_bounds,
    opacity=0.88,
    pane="raster_pane"
).add_to(
    population_group
)


# ==================================================
# POTENTIEL DE RESTAURATION
# ==================================================

folium.raster_layers.ImageOverlay(
    image=suitability_img,
    bounds=suitability_bounds,
    opacity=0.92,
    pane="raster_pane"
).add_to(
    restoration_group
)


# ==================================================
# AOI
# ==================================================

folium.GeoJson(
    aoi,
    pane="vector_pane",
    style_function=lambda feature: {
        "color": "#58595a",
        "weight": 3,
        "fillColor": "#58595a",
        "fillOpacity": 0
    },
    interactive=False
).add_to(
    aoi_group
)


# ==================================================
# ROUTES DE TRANSHUMANCE
# ==================================================

add_vector_layer(
    transhumance_group,
    transhumance,
    color="#E67E22",
    weight=3
)


# ==================================================
# SITES PAM
# ==================================================

add_vector_layer(
    sites_group,
    sites_pam,
    color="#EE5859",
    radius=6,
    fill_opacity=0.9
)


# ==================================================
# TARGETISATION
# ==================================================

pam_distance_field = None

if "NEAR_DIST_PAM" in targetisation.columns:

    pam_distance_field = (
        "NEAR_DIST_PAM"
    )

elif "NEAR_DIST_PAM_SITE" in targetisation.columns:

    pam_distance_field = (
        "NEAR_DIST_PAM_SITE"
    )


# ==================================================
# FORMAT TOOLTIP
# ==================================================

if "Pop" in targetisation.columns:

    targetisation[
        "Population_display"
    ] = (
        targetisation["Pop"]
        .round(0)
        .astype("Int64")
        .astype(str)
    )


if "cand_pct" in targetisation.columns:

    targetisation[
        "Rangeland_display"
    ] = (
        targetisation["cand_pct"]
        .round(1)
        .astype(str)
        + " %"
    )


# ==================================================
# DISTANCE TRANSHUMANCE EN KM
# ==================================================

if "NEAR_DIST" in targetisation.columns:

    targetisation[
        "Transhumance_display"
    ] = (
        (targetisation["NEAR_DIST"] / 1000)
        .round(2)
        .astype(str)
        + " km"
    )

# ==================================================
# DISTANCE SITE PAM EN KM
# ==================================================

if pam_distance_field is not None:

    targetisation[
        "PAM_display"
    ] = (
        (targetisation[pam_distance_field] / 1000)
        .round(2)
        .astype(str)
        + " km"
    )


# ==================================================
# SYMBOLOGIE TARGETISATION
#
# Score possible : 1 à 7
# ==================================================

def target_style(feature):

    score = (
        feature[
            "properties"
        ]
        .get(
            "target_score"
        )
    )

    try:

        score = int(
            float(score)
        )

    except (
        TypeError,
        ValueError
    ):

        score = None


    colors = {

        1: "#FFF7EC",
        2: "#FEE8C8",
        3: "#FDD49E",
        4: "#FDBB84",
        5: "#FC8D59",
        6: "#E34A33",
        7: "#B30000"

    }


    return {

        "fillColor":
            colors.get(
                score,
                "#D9D9D9"
            ),

        "color":
            "#666666",

        "weight":
            0.8,

        "fillOpacity":
            0.72
    }


# ==================================================
# TOOLTIP TARGETISATION
# ==================================================

tooltip_fields = [
    "GRID_ID",
    "target_score",
    "Population_display",
    "Rangeland_display",
    "Transhumance_display"
]


tooltip_aliases = [
    "ID hexagone :",
    "Score de ciblage :",
    "Population dans l'hexagone :",
    "Part de l'hexagone en rangeland à potentiel de restauration :",
    "Distance minimale à la route de transhumance la plus proche :"
]


if pam_distance_field is not None:

    tooltip_fields.append(
        "PAM_display"
    )

    tooltip_aliases.append(
        "Distance minimale au site d'intervention PAM le plus proche :"
    )


valid_fields = []
valid_aliases = []


for field, alias in zip(
    tooltip_fields,
    tooltip_aliases
):

    if field in targetisation.columns:

        valid_fields.append(
            field
        )

        valid_aliases.append(
            alias
        )


# ==================================================
# GEOJSON TARGETISATION
# ==================================================

folium.GeoJson(
    targetisation,
    pane="targetisation_pane",
    style_function=target_style,
    highlight_function=lambda feature: {
        "weight": 2.5,
        "color": "#222222",
        "fillOpacity": 0.90
    },
    tooltip=folium.GeoJsonTooltip(
        fields=valid_fields,
        aliases=valid_aliases,
        sticky=True,
        labels=True,
        localize=True,
        style=(
            "background-color:white;"
            "color:#333333;"
            "font-family:Arial;"
            "font-size:13px;"
            "padding:9px;"
            "border-radius:5px;"
            "box-shadow:0 1px 5px rgba(0,0,0,0.25);"
        )
    )
).add_to(
    target_group
)


# ==================================================
# LAYER CONTROL
# ==================================================

folium.LayerControl(
    position="topright",
    collapsed=False,
    autoZIndex=False
).add_to(
    m
)


# ==================================================
# PANELS
# ==================================================

layer_panels = {


    # ==================================================
    # TARGETISATION
    # ==================================================

    "Targetisation": """
    <div class="info-title">
        Targetisation
    </div>

    <div class="info-text">

        La zone d'étude est divisée en
        <b>hexagones de 25 km²</b>.

        <br><br>

        Le score de ciblage combine deux dimensions :

        <ul>

            <li>
                la population estimée dans chaque hexagone,
                classée en <b>3 catégories</b>.
                En l'absence de population, la classe de
                population prend la valeur <b>0</b> ;
            </li>

            <li>
                la proportion de la superficie de l'hexagone
                correspondant à du rangeland présentant un
                potentiel de restauration, classée en
                <b>4 catégories</b>.
            </li>

        </ul>

        <b>
            target_score =
            classe population +
            classe potentiel de restauration
        </b>

        <br><br>

        Le score varie de
        <b>1 à 7</b>.

        Un score élevé correspond à un hexagone combinant
        une population relativement élevée et une proportion
        importante de rangeland présentant un potentiel
        de restauration.

        <br><br>

        Deux indicateurs de distance sont également présentés
        comme informations complémentaires :
        la distance minimale à la route de transhumance
        la plus proche et la distance minimale au
        site d'intervention PAM le plus proche.

        <br><br>

        Ces distances sont exprimées en
        <b>kilomètres</b> et sont calculées
        <b>à partir du centroïde de chaque hexagone</b>.

        Elles ne sont pas intégrées au score de ciblage.

    </div>


    <div class="info-legend-title">
        Score de ciblage
    </div>


    <div class="legend-row">
        <span class="square"
              style="background:#FFF7EC;">
        </span>
        1
    </div>

    <div class="legend-row">
        <span class="square"
              style="background:#FEE8C8;">
        </span>
        2
    </div>

    <div class="legend-row">
        <span class="square"
              style="background:#FDD49E;">
        </span>
        3
    </div>

    <div class="legend-row">
        <span class="square"
              style="background:#FDBB84;">
        </span>
        4
    </div>

    <div class="legend-row">
        <span class="square"
              style="background:#FC8D59;">
        </span>
        5
    </div>

    <div class="legend-row">
        <span class="square"
              style="background:#E34A33;">
        </span>
        6
    </div>

    <div class="legend-row">
        <span class="square"
              style="background:#B30000;">
        </span>
        7
    </div>
    """,


    # ==================================================
    # POPULATION
    # ==================================================

    "Population": """
    <div class="info-title">
        Population
    </div>

    <div class="info-text">

        Cette couche représente la distribution spatiale
        de la population estimée dans la zone d'étude.

        <br><br>

        La valeur de chaque pixel du raster correspond au
        <b>nombre estimé d'habitants présents dans ce pixel</b>.

        <br><br>

        <span class="source-text">
            Source : <b>WorldPop, 2026</b>.
        </span>

    </div>


    <div class="info-legend-title">
        Population par pixel
    </div>


    <div class="population-gradient">
    </div>


    <div class="gradient-labels">

        <span>
            Faible
        </span>

        <span>
            Élevée
        </span>

    </div>
    """,


    # ==================================================
    # POTENTIEL DE RESTAURATION
    # ==================================================

    "Potentiel de restauration": """
    <div class="info-title">
        Potentiel de restauration
    </div>

    <div class="info-text">

        Cette couche identifie les zones de
        <b>rangeland présentant un potentiel de restauration</b>
        à partir d'un screening combinant plusieurs indicateurs
        environnementaux et pastoraux.

        <br><br>

        L'analyse se concentre notamment sur les zones de
        rangeland persistantes présentant des signes de
        dégradation et prend en compte des informations relatives :

        <ul>

            <li>
                au stress de la végétation ;
            </li>

            <li>
                aux déficits saisonniers de précipitations ;
            </li>

            <li>
                à la présence et à l'intensité du cheptel ;
            </li>

            <li>
                à l'exposition au risque d'inondation.
            </li>

        </ul>

        Deux niveaux de sélection sont représentés :
        <b>Moderate</b>, correspondant à un screening plus
        inclusif, et <b>Conservative</b>, basé sur des critères
        plus stricts.

        <br><br>

        Cette couche constitue un
        <b>outil de screening spatial</b> permettant
        d'identifier des zones potentielles à approfondir.

    </div>


    <div class="info-legend-title">
        Potentiel de restauration
    </div>


    <div class="legend-row">

        <span class="square"
              style="background:#2B6CB0;">
        </span>

        Moderate

    </div>


    <div class="legend-row">

        <span class="square"
              style="background:#08306B;">
        </span>

        Conservative

    </div>
    """,


    # ==================================================
    # AOI
    # ==================================================

    "Zone d'étude": """
    <div class="info-title">
        Zone d'étude
    </div>

    <div class="info-text">

        La zone d'étude couvre les provinces du
        <b>Sila</b>,
        <b>Ouaddaï</b>,
        <b>Salamat</b> et
        <b>Moyen-Chari</b>.

        <br><br>

        Les aires protégées ainsi que les zones situées
        à moins de <b>30 km des frontières internationales
        avec le Soudan et la République centrafricaine</b>
        sont exclues.

        <br><br>

        <span class="source-text">

            Aires protégées :
            <a
                href="https://www.protectedplanet.net/country/TCD"
                target="_blank"
            >
                Protected Planet – Tchad
            </a>

        </span>

    </div>


    <div class="info-legend-title">
        Limite
    </div>


    <div class="legend-row">

        <span class="line-symbol"
              style="border-color:#58595a;">
        </span>

        Zone d'étude

    </div>
    """,


    # ==================================================
    # TRANSHUMANCE
    # ==================================================

    "Routes de transhumance": """
    <div class="info-title">
        Routes de transhumance
    </div>

    <div class="info-text">

        Cette couche représente les principaux
        <b>axes de transhumance</b> utilisés dans l'analyse.

        <br><br>

        Les itinéraires ont été digitalisés à partir
        des informations cartographiques présentées dans
        un rapport du
        <b>Transhumance Tracking Tool (TTT) de l'OIM-DTM</b>.

        <br><br>

        Pour chaque hexagone, la
        <b>distance minimale entre son centroïde
        et la route de transhumance la plus proche</b>
        est calculée.

        La distance est exprimée en
        <b>kilomètres</b> et constitue une information
        complémentaire pour la priorisation.

        <br><br>

        <span class="source-text">

            Source :
            <a
                href="https://dtm.iom.int/dtm_download_track/29721?file=1&type=node&id=22076"
                target="_blank"
            >
                OIM-DTM – Transhumance Tracking Tool (TTT)
            </a>

        </span>

    </div>


    <div class="info-legend-title">
        Route
    </div>


    <div class="legend-row">

        <span class="line-symbol"
              style="border-color:#E67E22;">
        </span>

        Route de transhumance

    </div>
    """,


    # ==================================================
    # SITES PAM
    # ==================================================

    "Sites PAM": """
    <div class="info-title">
        Sites PAM
    </div>

    <div class="info-text">

        Cette couche représente les
        <b>sites d'intervention du Programme alimentaire mondial
        (PAM)</b> considérés dans l'analyse.

        <br><br>

        Pour chaque hexagone, la
        <b>distance minimale entre son centroïde
        et le site d'intervention PAM le plus proche</b>
        est calculée.

        La distance est exprimée en
        <b>kilomètres</b> et constitue une information
        complémentaire pour la priorisation.

        <br><br>

        <span class="source-text">

            Source :
            <a
                href="https://www.arcgis.com/apps/dashboards/0de08bf90dd74c5396da4d4d406c13d2#."
                target="_blank"
            >
                Tableau de bord PAM
            </a>

        </span>

    </div>


    <div class="info-legend-title">
        Site
    </div>


    <div class="legend-row">

        <span class="point-symbol">
        </span>

        Site d'intervention PAM

    </div>
    """
}


# ==================================================
# LEAFLET INFO CONTROL
# ==================================================

class LayerInfoControl(
    MacroElement
):

    def __init__(
        self,
        layer_panels,
        layers,
        initial_layer="Targetisation"
    ):

        super().__init__()

        self._name = (
            "LayerInfoControl"
        )


        self.panels_json = json.dumps(
            layer_panels,
            ensure_ascii=False
        )


        self.initial_layer = json.dumps(
            initial_layer,
            ensure_ascii=False
        )


        lookup_lines = []


        for layer_name, layer_object in layers:

            safe_name = json.dumps(
                layer_name,
                ensure_ascii=False
            )


            lookup_lines.append(
                f"""
                if (
                    layer ===
                    {layer_object.get_name()}
                ) {{
                    return {safe_name};
                }}
                """
            )


        self.lookup_js = "\n".join(
            lookup_lines
        )


        self._template = Template(
            """
            {% macro header(this, kwargs) %}

            <style>

            /* =====================================
               LAYER CONTROL
            ===================================== */

            .leaflet-control-layers {

                border:
                    none !important;

                border-radius:
                    7px !important;

                box-shadow:
                    0 2px 8px
                    rgba(0,0,0,0.25) !important;

                font-family:
                    Arial,
                    sans-serif !important;

            }


            .leaflet-control-layers-expanded {

                background:
                    rgba(255,255,255,0.96)
                    !important;

                padding:
                    12px 14px
                    !important;

                min-width:
                    250px;

            }


            .leaflet-control-layers-base label,
            .leaflet-control-layers-overlays label {

                margin-bottom:
                    6px;

                font-size:
                    13px;

                color:
                    #58595a;

            }


            #map-control-title {

                color:
                    #005A9C;

                font-weight:
                    700;

                font-size:
                    16px;

                line-height:
                    1.3;

                max-width:
                    250px;

                margin-bottom:
                    11px;

                padding-bottom:
                    10px;

                border-bottom:
                    1px solid #d9d9d9;

            }


            #layers-control-title {

                font-weight:
                    700;

                font-size:
                    13px;

                color:
                    #58595a;

                margin-bottom:
                    8px;

            }


            /* =====================================
               INFO CONTROL
            ===================================== */

            .layer-info-control {

                width:
                    390px;

                max-width:
                    calc(100vw - 60px);

                max-height:
                    52vh;

                overflow-y:
                    auto;

                background:
                    rgba(255,255,255,0.96);

                border-left:
                    5px solid #005A9C;

                border-radius:
                    7px;

                padding:
                    14px 17px;

                box-shadow:
                    0 2px 8px
                    rgba(0,0,0,0.25);

                font-family:
                    Arial,
                    sans-serif;

                color:
                    #58595a;

            }


            .info-title {

                color:
                    #005A9C;

                font-size:
                    17px;

                font-weight:
                    700;

                margin-bottom:
                    10px;

            }


            .info-text {

                font-size:
                    13px;

                line-height:
                    1.45;

            }


            .info-text ul {

                padding-left:
                    20px;

                margin-top:
                    6px;

                margin-bottom:
                    8px;

            }


            .info-text a {

                color:
                    #EE5859;

                text-decoration:
                    none;

            }


            .info-text a:hover {

                text-decoration:
                    underline;

            }


            .source-text {

                font-size:
                    11px;

                color:
                    #777777;

            }


            .info-legend-title {

                font-size:
                    12px;

                font-weight:
                    700;

                margin-top:
                    14px;

                margin-bottom:
                    7px;

                padding-top:
                    9px;

                border-top:
                    1px solid #e4e4e4;

            }


            .legend-row {

                display:
                    flex;

                align-items:
                    center;

                font-size:
                    12px;

                margin-top:
                    4px;

            }


            .square {

                width:
                    14px;

                height:
                    14px;

                display:
                    inline-block;

                border:
                    1px solid #cccccc;

                margin-right:
                    7px;

                flex:
                    0 0 14px;

            }


            .line-symbol {

                width:
                    26px;

                height:
                    0;

                border-top:
                    3px solid;

                margin-right:
                    8px;

            }


            .point-symbol {

                width:
                    11px;

                height:
                    11px;

                display:
                    inline-block;

                background:
                    #EE5859;

                border-radius:
                    50%;

                margin-right:
                    8px;

            }


            .population-gradient {

                width:
                    180px;

                height:
                    12px;

                margin-top:
                    5px;

                background:
                    linear-gradient(
                        to right,
                        #F0F0F0,
                        #969696,
                        #000000
                    );

                border:
                    1px solid #cccccc;

            }


            .gradient-labels {

                width:
                    180px;

                display:
                    flex;

                justify-content:
                    space-between;

                font-size:
                    10px;

                margin-top:
                    3px;

            }

            </style>

            {% endmacro %}


            {% macro script(this, kwargs) %}


            const mapObject =
                {{ this._parent.get_name() }};


            const layerPanels =
                {{ this.panels_json }};


            const initialLayer =
                {{ this.initial_layer }};


            /* =====================================
               INFO BOX
            ===================================== */

            const infoControl =
                L.control({

                    position:
                        "bottomright"

                });


            infoControl.onAdd =
                function(map) {

                    const div =
                        L.DomUtil.create(
                            "div",
                            "layer-info-control"
                        );


                    div.id =
                        "layer-info-box";


                    div.innerHTML =
                        layerPanels[
                            initialLayer
                        ];


                    L.DomEvent
                        .disableClickPropagation(
                            div
                        );


                    L.DomEvent
                        .disableScrollPropagation(
                            div
                        );


                    return div;

                };


            infoControl.addTo(
                mapObject
            );


            /* =====================================
               RICONOSCIMENTO LAYER
            ===================================== */

            function getLayerName(
                layer
            ) {

                {{ this.lookup_js }}

                return null;

            }


            /* =====================================
               ORDINE ATTIVO
            ===================================== */

            let activeLayerOrder = [
                initialLayer
            ];


            /* =====================================
               MOSTRA INFO
            ===================================== */

            function showPanel(
                layerName
            ) {

                const box =
                    document.getElementById(
                        "layer-info-box"
                    );


                if (!box) {
                    return;
                }


                if (
                    layerName
                    &&
                    layerPanels[
                        layerName
                    ]
                ) {

                    box.innerHTML =
                        layerPanels[
                            layerName
                        ];


                    box.style.display =
                        "block";

                }

                else {

                    box.style.display =
                        "none";

                }

            }


            /* =====================================
               OVERLAY ADD
            ===================================== */

            mapObject.on(
                "overlayadd",

                function(event) {

                    const layerName =
                        getLayerName(
                            event.layer
                        );


                    if (!layerName) {
                        return;
                    }


                    activeLayerOrder =
                        activeLayerOrder.filter(
                            function(name) {

                                return (
                                    name !==
                                    layerName
                                );

                            }
                        );


                    activeLayerOrder.push(
                        layerName
                    );


                    showPanel(
                        layerName
                    );

                }
            );


            /* =====================================
               OVERLAY REMOVE
            ===================================== */

            mapObject.on(
                "overlayremove",

                function(event) {

                    const layerName =
                        getLayerName(
                            event.layer
                        );


                    if (!layerName) {
                        return;
                    }


                    activeLayerOrder =
                        activeLayerOrder.filter(
                            function(name) {

                                return (
                                    name !==
                                    layerName
                                );

                            }
                        );


                    if (
                        activeLayerOrder.length
                        > 0
                    ) {

                        const lastLayer =
                            activeLayerOrder[
                                activeLayerOrder.length
                                - 1
                            ];


                        showPanel(
                            lastLayer
                        );

                    }

                    else {

                        showPanel(
                            null
                        );

                    }

                }
            );


            /* =====================================
               TITRE + ORDRE VISUEL DU CONTROL
            ===================================== */

            setTimeout(

                function() {

                    const list =
                        document.querySelector(
                            ".leaflet-control-layers-list"
                        );


                    const overlays =
                        document.querySelector(
                            ".leaflet-control-layers-overlays"
                        );


                    if (
                        list
                        &&
                        !document.getElementById(
                            "map-control-title"
                        )
                    ) {

                        const mapTitle =
                            document.createElement(
                                "div"
                            );


                        mapTitle.id =
                            "map-control-title";


                        mapTitle.innerHTML =
                            "Ciblage des zones à potentiel de restauration – SCALE - GCF";


                        const layerTitle =
                            document.createElement(
                                "div"
                            );


                        layerTitle.id =
                            "layers-control-title";


                        layerTitle.innerHTML =
                            "Fond de carte & couches";


                        list.prepend(
                            layerTitle
                        );


                        list.prepend(
                            mapTitle
                        );

                    }


                    if (overlays) {

                        const labels =
                            Array.from(
                                overlays.querySelectorAll(
                                    "label"
                                )
                            );


                        const desiredOrder = [

                            "Zone d'étude",

                            "Sites PAM",

                            "Routes de transhumance",

                            "Potentiel de restauration",

                            "Population",

                            "Targetisation"

                        ];


                        desiredOrder.forEach(
                            function(layerName) {

                                const label =
                                    labels.find(
                                        function(item) {

                                            return (
                                                item.textContent
                                                .trim()
                                                === layerName
                                            );

                                        }
                                    );


                                if (label) {

                                    overlays.appendChild(
                                        label
                                    );

                                }

                            }
                        );

                    }

                },

                300

            );


            /* =====================================
               ETAT INITIAL
            ===================================== */

            showPanel(
                initialLayer
            );


            {% endmacro %}
            """
        )


# ==================================================
# ASSOCIATION INFO <-> LAYERS
# ==================================================

layer_objects = [

    (
        "Zone d'étude",
        aoi_group
    ),

    (
        "Sites PAM",
        sites_group
    ),

    (
        "Routes de transhumance",
        transhumance_group
    ),

    (
        "Potentiel de restauration",
        restoration_group
    ),

    (
        "Population",
        population_group
    ),

    (
        "Targetisation",
        target_group
    )
]


LayerInfoControl(
    layer_panels=layer_panels,
    layers=layer_objects,
    initial_layer="Targetisation"
).add_to(
    m
)


# ==================================================
# ZOOM INITIAL
# ==================================================

m.fit_bounds(
    [
        [
            miny,
            minx
        ],
        [
            maxy,
            maxx
        ]
    ]
)


# ==================================================
# AFFICHAGE
# ==================================================

folium_static(
    m,
    width=2000,
    height=1000
)
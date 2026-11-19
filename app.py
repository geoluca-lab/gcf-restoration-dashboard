import base64
import hashlib

import streamlit as st

import pandas as pd

import folium

import geopandas as gpd

import rasterio

import pyogrio

import numpy as np

import json
from datetime import datetime
from io import BytesIO
from zoneinfo import ZoneInfo



from pathlib import Path

from affine import Affine

from branca.element import MacroElement

from jinja2 import Template

from streamlit_folium import st_folium
from shapely.geometry import Point



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

# Documentation: prefer the reorganised docs/ location, with a fallback to the
# previous data/Annex.pdf path so local and deployed versions remain compatible.
ANNEX_PDF_CANDIDATES = [
    BASE_DIR / "docs" / "Annex_3_Methodologie_Ciblage.pdf",
    BASE_DIR / "docs" / "Annex.pdf",
    BASE_DIR / "data" / "Annex.pdf",
]
ANNEX_PDF_PATH = next((path for path in ANNEX_PDF_CANDIDATES if path.exists()), ANNEX_PDF_CANDIDATES[0])

GITHUB_REPO_URL = "https://github.com/geoluca-lab/gcf-restoration-dashboard"
SCREENING_SCRIPT_URL = (
    GITHUB_REPO_URL
    + "/blob/main/scripts/gcf_rangeland_screening_local_ANNOTATED.py"
)

ANNEX_PDF_DATA_URL = ""
if ANNEX_PDF_PATH.exists():
    ANNEX_PDF_DATA_URL = (
        "data:application/pdf;base64,"
        + base64.b64encode(ANNEX_PDF_PATH.read_bytes()).decode("ascii")
    )


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

    "trans_merge"

)





# ==================================================
# CHARGEMENT VECTEURS
# ==================================================

def load_vector_layer(
    path,
    layer_name,
    keep_columns=None,
    preserve_attributes=False
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

    gdf = gdf.to_crs(epsg=4326)

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
            existing_columns + ["geometry"]
        ]

    elif not preserve_attributes:
        gdf = gdf[["geometry"]]

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

    preserve_attributes=True
)





# ==================================================
# HEXAGON SELECTION - source attributes remain available for Excel export.
# ==================================================


def prepare_hex_selection_data(hexagons):
    """Measure polygon area in local UTM metres, retaining source attributes."""
    if hexagons["GRID_ID"].isna().any() or not hexagons["GRID_ID"].is_unique:
        raise ValueError("hexa_target must have a unique, non-null GRID_ID.")
    result = hexagons.copy()
    result["GRID_ID"] = result["GRID_ID"].astype(str)
    # The current Chad extent lies in UTM zone 34N (EPSG:32634).
    area_crs = result.estimate_utm_crs()
    if area_crs is None:
        raise ValueError("Cannot determine a projected CRS for hexagon areas.")
    result["hex_area_ha"] = result.to_crs(area_crs).geometry.area / 10000
    # Inspection: cand_ha = (VALUE_12 + VALUE_23) / 10000 and
    # cand_pct = round(cand_ha / 2500 * 100, 1). Edge hexagons are clipped,
    # so multiplying their actual area by this rounded percentage understates
    # the source restoration area. Prefer the existing, unrounded cand_ha.
    if "cand_pct" in result:
        result["restoration_potential_pct_estimate_ha"] = (
            result["hex_area_ha"] * result["cand_pct"] / 100
        )
    if "cand_ha" in result:
        result["restoration_potential_ha"] = result["cand_ha"]
    elif "cand_pct" in result:
        result["restoration_potential_ha"] = result[
            "restoration_potential_pct_estimate_ha"
        ]
    return result


def toggle_hex_selection(selected_ids, grid_id):
    """Return a set so repeated IDs can never create duplicate selections."""
    selected_ids = set(selected_ids)
    if grid_id in selected_ids:
        selected_ids.remove(grid_id)
    else:
        selected_ids.add(grid_id)
    return selected_ids


def clicked_hex_id(event, hexagons):
    """Read GRID_ID from the clicked GeoJSON, never from formatted tooltip text."""
    feature = event.get("last_active_drawing") or {}
    grid_id = (feature.get("properties") or {}).get("GRID_ID")
    location = event.get("last_object_clicked")
    if grid_id is None or not location:
        return None
    grid_id = str(grid_id)
    candidates = hexagons.loc[hexagons["GRID_ID"] == grid_id]
    # streamlit-folium retains the previous feature on some non-feature clicks.
    # Verify its geometry against the current click before accepting the ID.
    point = Point(location["lng"], location["lat"])
    if not candidates.empty and candidates.geometry.covers(point).any():
        return grid_id
    return None


selection_data = prepare_hex_selection_data(targetisation)
st.session_state.setdefault("selected_hex_ids", set())
st.session_state.setdefault("hex_comments", {})
st.session_state.setdefault("last_restored_excel_signature", None)
st.session_state.selected_hex_ids = set(st.session_state.selected_hex_ids) & set(
    selection_data["GRID_ID"]
)


def handle_hex_click():
    """Component callback runs before rerendering the summary and highlight."""
    event = st.session_state.get("hex_selection_map", {})
    count = event.get("last_object_clicked_count")
    if not count or count == st.session_state.get("hex_selection_click_count"):
        return
    st.session_state.hex_selection_click_count = count
    grid_id = clicked_hex_id(event, selection_data)
    if grid_id is not None:
        st.session_state.selected_hex_ids = toggle_hex_selection(
            st.session_state.selected_hex_ids, grid_id
        )


def clear_hex_selection():
    st.session_state.selected_hex_ids = set()


def normalize_hex_id_value(value):
    """Normalize Excel values so numeric GRID_IDs do not become strings such as '1.0'."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def uploaded_file_signature(uploaded_file):
    """Stable signature so an uploaded workbook/sheet is processed only once per version."""
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def get_excel_sheet_names(uploaded_file):
    """Return workbook sheet names without altering the uploaded-file cursor."""
    try:
        workbook = pd.ExcelFile(BytesIO(uploaded_file.getvalue()), engine="openpyxl")
        return workbook.sheet_names
    except Exception as exc:
        raise ValueError(f"Le fichier Excel est invalide : {exc}") from exc


def restore_hex_selection_from_excel(uploaded_file, sheet_name):
    """Restore a selection and comments from one selected Excel sheet using GRID_ID."""
    if uploaded_file is None:
        return None
    try:
        df = pd.read_excel(
            BytesIO(uploaded_file.getvalue()),
            sheet_name=sheet_name,
            engine="openpyxl",
        )
    except Exception as exc:
        raise ValueError(f"Impossible de lire la feuille '{sheet_name}' : {exc}") from exc

    grid_id_columns = []
    for column in df.columns:
        normalized = "".join(ch for ch in str(column).lower() if ch.isalnum())
        if normalized in {"gridid", "idhexagone", "hexagonid"}:
            grid_id_columns.append(column)

    comment_columns = []
    for column in df.columns:
        normalized = "".join(ch for ch in str(column).lower() if ch.isalnum())
        if normalized in {"commentaire", "comment", "commentaries", "notes", "note", "remarks", "remark"}:
            comment_columns.append(column)

    if not grid_id_columns:
        raise ValueError(
            "Aucune colonne de GRID_ID n'a été trouvée dans la feuille sélectionnée. "
            "Le fichier doit contenir l'identifiant de l'hexagone."
        )

    valid_ids = set(selection_data["GRID_ID"].astype(str))
    ids = set()
    restored_comments = {}
    for _, row in df.iterrows():
        for column in grid_id_columns:
            grid_id = normalize_hex_id_value(row.get(column))
            if not grid_id or grid_id not in valid_ids:
                continue
            ids.add(grid_id)
            comment_value = ""
            for comment_column in comment_columns:
                candidate = row.get(comment_column)
                if pd.notna(candidate) and str(candidate).strip():
                    comment_value = str(candidate).strip()
                    break
            if comment_value:
                restored_comments[grid_id] = comment_value
            break

    if not ids:
        raise ValueError(
            "Aucun identifiant d'hexagone valide n'a été trouvé dans la feuille sélectionnée."
        )

    st.session_state.selected_hex_ids = set(ids)
    st.session_state.hex_comments = restored_comments
    return set(ids)


def dataframe_to_excel_bytes(df, sheet_name):
    """Create an in-memory XLSX workbook for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
        worksheet = writer.book[sheet_name[:31]]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for column_cells in worksheet.columns:
            values = [str(cell.value) if cell.value is not None else "" for cell in column_cells]
            max_length = min(max((len(value) for value in values), default=0) + 2, 45)
            worksheet.column_dimensions[column_cells[0].column_letter].width = max(10, max_length)
    return output.getvalue()


class HexSelectionStyle(MacroElement):
    """Update existing paths through st_folium's dynamic feature-group channel.

    Keeping selection out of the base map script preserves the component key,
    viewport, open controls and layer visibility through Streamlit reruns.
    """

    _template = Template("""
        {% macro script(this, kwargs) %}
        (function () {
            const hexLayer = {{ this.hex_layer.get_name() }};
            hexLayer.selectedHexIds = new Set({{ this.selected_ids | tojson }});
            if (!hexLayer.selectionResetStyle) {
                const originalReset = hexLayer.resetStyle;
                hexLayer.resetStyle = function (layer) {
                    originalReset.call(this, layer);
                    const applySelection = (path) => {
                        if (path.feature && this.selectedHexIds.has(
                                String(path.feature.properties.GRID_ID))) {
                            path.setStyle({color: '#00A6C7', weight: 3.5,
                                           opacity: 1, fillOpacity: 0.90});
                        }
                    };
                    if (layer) { applySelection(layer); }
                    else { this.eachLayer(applySelection); }
                    return this;
                };
                hexLayer.selectionResetStyle = true;
            }
            hexLayer.resetStyle();
        })();
        {% endmacro %}
    """)

    def __init__(self, hex_layer, selected_ids):
        super().__init__()
        self._name = "HexSelectionStyle"
        self.hex_layer = hex_layer
        self.selected_ids = sorted(selected_ids)


def render_selection_summary(hexagons, detail_data, detail_fields, detail_aliases):
    selected = hexagons.loc[
        hexagons["GRID_ID"].isin(st.session_state.selected_hex_ids)
    ]
    # Keep tooltip labels and formatting, adding both component scores before
    # the combined targeting score in the Excel exports only.
    export_fields = list(detail_fields)
    export_labels = {
        field: alias.rstrip(" :")
        for field, alias in zip(detail_fields, detail_aliases)
    }
    score_position = export_fields.index("target_score")
    for field, label in (
        ("pop_class_1", "Score de Population"),
        ("cand_class", "Score potentiel de restauration"),
    ):
        if field in detail_data.columns:
            export_fields.insert(score_position, field)
            export_labels[field] = label
            score_position += 1
    export_labels = {
        **export_labels,
        "centroid_x": "X du centroïde",
        "centroid_y": "Y du centroïde",
        "Commentaire": "Commentaire",
    }

    def build_export_table(rows):
        """Build the same Excel table for a selection or for all hexagons."""
        row_ids = rows["GRID_ID"].astype(str)
        export_df = detail_data.loc[
            detail_data["GRID_ID"].astype(str).isin(row_ids),
            export_fields,
        ].copy()
        export_df["GRID_ID"] = export_df["GRID_ID"].astype(str)
        export_df["Commentaire"] = export_df["GRID_ID"].map(
            st.session_state.get("hex_comments", {})
        ).fillna("")
        centroid_frame = rows[["GRID_ID"]].copy()
        centroid_frame["GRID_ID"] = centroid_frame["GRID_ID"].astype(str)
        centroid_frame["centroid_x"] = rows.geometry.centroid.x
        centroid_frame["centroid_y"] = rows.geometry.centroid.y
        export_df = export_df.merge(centroid_frame, on="GRID_ID", how="left")
        return export_df.rename(columns=export_labels)

    details = build_export_table(selected)
    all_details = build_export_table(hexagons)
    export_date = datetime.now(ZoneInfo("Africa/Ndjamena")).strftime("%Y-%m-%d")

    selected_excel = dataframe_to_excel_bytes(details, "Selection")
    all_excel = dataframe_to_excel_bytes(all_details, "Hexagones")

    with st.container(key="hex_selection_panel"):
        with st.expander("Sélection des hexagones", expanded=False):
            import_col, totals_col, actions_col = st.columns(
                [2.2, 3.8, 2.5],
                vertical_alignment="center",
                gap="medium",
            )

            with import_col:
                with st.container(key="hex_restore_import_block"):
                    st.markdown(
                        "<div class='selection-section-title'>Recharger une sélection précédente</div>",
                        unsafe_allow_html=True,
                    )
                    uploaded_selection = st.file_uploader(
                        "Importer un fichier Excel de sélection",
                        type=["xlsx"],
                        key="hex_selection_restore_upload",
                        label_visibility="collapsed",
                        help="Recharge un fichier Excel exporté depuis ce dashboard afin de restaurer la sélection et les commentaires associés.",
                    )

                    if uploaded_selection is not None:
                        try:
                            sheet_names = get_excel_sheet_names(uploaded_selection)
                            selected_sheet = st.selectbox(
                                "Feuille à charger",
                                options=sheet_names,
                                key="hex_selection_restore_sheet",
                                help="Choisissez la feuille Excel contenant les GRID_ID à restaurer.",
                            )
                            load_sheet = st.button(
                                "Charger cette feuille",
                                key="hex_selection_restore_sheet_button",
                                use_container_width=True,
                                help="Restaure les hexagones et les commentaires présents dans la feuille choisie.",
                            )
                            if load_sheet:
                                signature = (
                                    uploaded_file_signature(uploaded_selection)
                                    + "::"
                                    + selected_sheet
                                )
                                if signature != st.session_state.get("last_restored_excel_signature"):
                                    restored_ids = restore_hex_selection_from_excel(
                                        uploaded_selection,
                                        selected_sheet,
                                    )
                                    if restored_ids is not None:
                                        st.session_state.last_restored_excel_signature = signature
                                        st.session_state.restore_selection_toast = (
                                            f"Sélection restaurée : {len(restored_ids)} hexagones "
                                            f"depuis la feuille « {selected_sheet} »."
                                        )
                                        # The map GeoJSON and its tooltip are built earlier in the run.
                                        # Rerun once so restored comments are injected before map creation.
                                        st.rerun()
                        except ValueError as exc:
                            st.toast(str(exc), icon="⚠️")
                    else:
                        # Allow the same workbook/sheet to be uploaded again after clearing the uploader.
                        st.session_state.last_restored_excel_signature = None

                    restore_toast = st.session_state.pop("restore_selection_toast", None)
                    if restore_toast:
                        st.toast(restore_toast, icon="✅")

            with totals_col:
                population = selected["Pop"].sum() if "Pop" in selected else 0
                restoration = selected["restoration_potential_ha"].sum()
                total_area = selected["hex_area_ha"].sum()
                st.markdown(
                    '<div class="hex-selection-totals">'
                    '<div class="selection-section-title">Sélection actuelle</div>'
                    '<table>'
                    '<tr>'
                    '<td class="metric-label">Hexagones sélectionnés</td>'
                    f'<td class="metric-value"><strong>{len(selected):,}</strong></td>'
                    '<td class="metric-label">Surface totale</td>'
                    f'<td class="metric-value"><strong>{total_area:,.1f} ha</strong></td>'
                    '</tr>'
                    '<tr>'
                    '<td class="metric-label">Population estimée</td>'
                    f'<td class="metric-value"><strong>{population:,.0f}</strong></td>'
                    '<td class="metric-label">Potentiel de restauration</td>'
                    f'<td class="metric-value"><strong>{restoration:,.1f} ha</strong></td>'
                    '</tr>'
                    '</table>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                if not selected.empty:
                    missing = selected[[c for c in ("Pop", "restoration_potential_ha")
                                        if c in selected]].isna().sum()
                    if missing.any():
                        st.caption("Valeurs manquantes exclues des totaux : " + ", ".join(
                            f"{name} ({count})" for name, count in missing.items() if count
                        ))

            with actions_col:
                st.markdown(
                    "<div class='selection-section-title'>Actions</div>",
                    unsafe_allow_html=True,
                )
                clear_col, download_selected_col = st.columns(
                    [1.0, 1.25],
                    vertical_alignment="center",
                    gap="small",
                )

                with clear_col:
                    st.button(
                        "Effacer la sélection",
                        on_click=clear_hex_selection,
                        use_container_width=True,
                        help="Retire tous les hexagones actuellement sélectionnés sur la carte.",
                    )

                with download_selected_col:
                    st.download_button(
                        "Télécharger la sélection",
                        selected_excel,
                        file_name=f"{export_date}_selected_hexagons.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        disabled=selected.empty,
                        use_container_width=True,
                        help="Télécharge les hexagones sélectionnés dans un fichier Excel avec leurs attributs, centroïdes et commentaires.",
                    )

                with st.container(key="download_all_hexagons_link"):
                    st.download_button(
                        "Télécharger l’ensemble des hexagones",
                        all_excel,
                        file_name=f"{export_date}_all_hexagons.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=False,
                        help="Télécharge tous les hexagones dans un fichier Excel avec les mêmes attributs et les commentaires disponibles.",
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

    zoom_start=8,

    tiles=None,

    control_scale=True,

    prefer_canvas=True

)

# st_folium supplies a pixel height, while the dashboard iframe fills the
# viewport. Keep Leaflet's own container at that same visible height so its
# bottom-right descriptions and bottom-left scale cannot fall below the frame.
m.get_root().header.add_child(folium.Element("""
<style>
html, body { margin: 0; height: 100%; overflow: hidden; }
#map_div { width: 100% !important; height: 100vh !important; height: 100dvh !important; }
.leaflet-control-zoom { margin-top: 6px !important; margin-right: 6px !important; margin-left: 0 !important; display: block !important; }
.leaflet-top.leaflet-left .leaflet-control-layers { margin-top: 4px !important; margin-left: 4px !important; }
.hex-tooltip table { max-width: 430px; }
.hex-tooltip table tr:last-child { display: block; margin-top: 6px; padding-top: 6px; border-top: 1px solid #dddddd; }
.hex-tooltip table tr:last-child th { display: none !important; }
.hex-tooltip table tr:last-child td { display: block !important; width: 100% !important; max-width: 430px; white-space: normal !important; overflow-wrap: anywhere; word-break: break-word; line-height: 1.35; }
.leaflet-control-scale { margin-left: 12px !important; margin-bottom: 10px !important; display: block !important; }
.leaflet-control-scale-line { background: rgba(255,255,255,0.9) !important; border: 1px solid rgba(0,0,0,0.25) !important; border-top: none !important; color: #222 !important; }
</style>
"""))


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



worldcover_layer = folium.WmsTileLayer(

    url="https://mapproxy.terrascope.be/mapproxy/service",

    layers="esa-worldcover-map-10m-2021-v2_map",

    name="Couverture du sol (2021)",

    fmt="image/png",

    transparent=False,

    overlay=False,

    control=True,

    show=False,

    opacity=1.0,

    attr="ESA WorldCover 2021 v2"

).add_to(m)





# ==================================================

# FEATURE GROUPS

# ==================================================



target_group = folium.FeatureGroup(

    name="Score final de ciblage",

    overlay=True,

    control=True,

    show=True

)



target_group.add_to(m)





population_group = folium.FeatureGroup(

    name="Population estimée (2026)",

    overlay=True,

    control=True,

    show=False

)



population_group.add_to(m)





restoration_group = folium.FeatureGroup(

    name="Parcours pastoraux à potentiel de restauration (2 scénarios)",

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

    name="Sites d’intervention du PAM",

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





if "cand_ha" in targetisation.columns:

    targetisation["Degraded_rangeland_display"] = (
        targetisation["cand_ha"].round(1).astype(str) + " ha"
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



# A missing OSM name means the locality is not recorded in this dataset.
if "name" in targetisation.columns:
    targetisation["Localite_display"] = targetisation["name"].fillna(
        "-"
    )

comments_by_grid_id = st.session_state.get("hex_comments", {})
targetisation["Commentaire"] = targetisation["GRID_ID"].map(
    comments_by_grid_id
).fillna("")

tooltip_fields = [

    "GRID_ID",

    "adm1_name",

    "adm2_name",

    "Localite_display",

    "target_score",

    "Population_display",

    "Rangeland_display",

    "Degraded_rangeland_display",

    "Transhumance_display"

]





tooltip_aliases = [

    "ID hexagone :",

    "Province de l'hexagone (centroïde) :",

    "Département de l'hexagone (centroïde) :",

    "Localité dans l'hexagone (OSM) :",

    "Score de ciblage :",

    "Population dans l'hexagone :",

    "Part de l'hexagone en parcours pastoraux à potentiel de restauration :",

    "Superficie des parcours pastoraux à potentiel de restauration :",

    "Distance minimale à la route de transhumance la plus proche :"

]





if pam_distance_field is not None:



    tooltip_fields.append(

        "PAM_display"

    )



    tooltip_aliases.append(

        "Distance minimale au site d'intervention PAM le plus proche :"

    )


# Commentaire : toujours affiché en dernière ligne du tooltip, sans libellé,
# afin de laisser toute la largeur disponible aux commentaires longs.
tooltip_fields.append("Commentaire")
tooltip_aliases.append("")


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



target_hex_layer = folium.GeoJson(

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

        class_name="hex-tooltip",

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

    position="topleft",

    collapsed=True,

    autoZIndex=False

).add_to(

    m

)





# ==================================================

# PANELS

# ==================================================



layer_panels = {

    "Zone d'étude": """
    <div class="info-title">Zone d'étude</div>
    <div class="info-text">
        La zone d'étude couvre le <b>Sila, le Ouaddaï, le Salamat et le Moyen-Chari</b>.
        Les aires protégées et les zones situées à moins de <b>30 km des frontières
        avec le Soudan et la République centrafricaine</b> sont exclues du périmètre d'analyse.
        <br><br>
        <span class="source-text">Aires protégées : Protected Planet – Tchad.</span>
    </div>
    <div class="info-legend-title">Limite</div>
    <div class="legend-row">
        <span class="line-symbol" style="border-color:#58595a;"></span>
        Zone d'étude
    </div>
    <div class="methodological-note-footer">
        <b>Référence méthodologique :</b> Annexe 3 – Étape 2, « Délimitation de la zone d'étude ».
        Voir la note méthodologique téléchargeable dans ce panneau.
    </div>
    """,

    "Couverture du sol (2021)": """
    <div class="info-title">Couverture du sol (2021)</div>
    <div class="info-legend-title" style="margin-top:0; padding-top:0; border-top:none;">Légende</div>
    <div class="legend-row"><span class="square" style="background:#006400;"></span>Couvert arboré</div>
    <div class="legend-row"><span class="square" style="background:#FFBB22;"></span>Arbustes</div>
    <div class="legend-row"><span class="square" style="background:#FFFF4C;"></span>Prairies / formations herbacées</div>
    <div class="legend-row"><span class="square" style="background:#F096FF;"></span>Terres cultivées</div>
    <div class="legend-row"><span class="square" style="background:#FA0000;"></span>Zones bâties</div>
    <div class="legend-row"><span class="square" style="background:#B4B4B4;"></span>Sol nu / végétation clairsemée</div>
    <div class="legend-row"><span class="square" style="background:#F0F0F0;"></span>Neige et glace</div>
    <div class="legend-row"><span class="square" style="background:#0064C8;"></span>Plans d'eau permanents</div>
    <div class="legend-row"><span class="square" style="background:#0096A0;"></span>Zones humides herbacées</div>
    <div class="legend-row"><span class="square" style="background:#00CF75;"></span>Mangroves</div>
    <div class="legend-row"><span class="square" style="background:#FAE6A0;"></span>Mousses et lichens</div>
    <div class="info-text" style="margin-top:10px;">
        La <b>couverture du sol</b> décrit ce qui est physiquement observable à la surface terrestre.
        ESA WorldCover 2021 fournit ici un fond de contexte à <b>10 m</b>.
        Il ne remplace pas le proxy Global Pasture Watch utilisé pour identifier les parcours pastoraux.
        <br><br>
        <span class="source-text">Source : ESA WorldCover 2021 v2.</span>
    </div>
    <div class="methodological-note-footer">
        <b>Référence méthodologique :</b> Annexe 3 – Étape 1, « Cadre conceptuel : couverture du sol,
        utilisation du sol et parcours pastoraux », puis Étape 3 pour le proxy GPW.
    </div>
    """,

    "Parcours pastoraux à potentiel de restauration (2 scénarios)": f"""
    <div class="info-title">Parcours pastoraux à potentiel de restauration (2 scénarios)</div>
    <div class="info-text">
        Cette couche correspond au résultat du <b>screening biophysique</b> : un proxy des parcours
        pastoraux est croisé avec les signes de dégradation, puis avec des indicateurs complémentaires
        de stress, de pression et de contrainte environnementale.
        <br><br>
        Deux scénarios sont représentés : <b>Moderate</b> et <b>Conservative</b>.
    </div>
    <div class="info-legend-title">Potentiel de restauration</div>
    <div class="legend-row"><span class="square" style="background:#2B6CB0;"></span>Moderate</div>
    <div class="legend-row"><span class="square" style="background:#08306B;"></span>Conservative</div>
    <div class="methodological-note-footer">
        <b>Référence méthodologique :</b> Annexe 3 – Étapes 3 à 6, « Screening biophysique ».
        Les indicateurs, seuils et règles des deux scénarios sont détaillés dans la note.
        <br><br>
        <a href="{SCREENING_SCRIPT_URL}" target="_blank" rel="noopener noreferrer">
            Voir le script de calcul du potentiel de restauration (GitHub)
        </a>
    </div>
    """,

    "Population estimée (2026)": """
    <div class="info-title">Population estimée (2026)</div>
    <div class="info-text">
        Cette couche représente la distribution spatiale de la <b>population estimée</b> dans la zone d'étude.
        Elle est mobilisée dans la phase de ciblage opérationnel pour mettre en relation le potentiel de
        restauration avec la présence de population.
        <br><br>
        <span class="source-text">Source : WorldPop, 2026.</span>
    </div>
    <div class="info-legend-title">Population par pixel</div>
    <div class="population-gradient"></div>
    <div class="gradient-labels"><span>Faible</span><span>Élevée</span></div>
    <div class="methodological-note-footer">
        <b>Référence méthodologique :</b> Annexe 3 – Étape 7, « Intégration de la densité de population ».
    </div>
    """,

    "Routes de transhumance": """
    <div class="info-title">Routes de transhumance</div>
    <div class="info-text">
        Cette couche représente les principaux <b>axes de transhumance</b> utilisés comme information
        complémentaire de ciblage. Pour chaque hexagone, le dashboard indique la distance à l'axe le plus proche.
        <br><br>
        <span class="source-text">Source : OIM-DTM – Transhumance Tracking Tool (TTT).</span>
    </div>
    <div class="info-legend-title">Route</div>
    <div class="legend-row"><span class="line-symbol" style="border-color:#E67E22;"></span>Route de transhumance</div>
    <div class="methodological-note-footer">
        <b>Référence méthodologique :</b> Annexe 3 – Étape 8, « Données de transhumance et des interventions du PAM ».
    </div>
    """,

    "Sites d’intervention du PAM": """
    <div class="info-title">Sites d’intervention du PAM</div>
    <div class="info-text">
        Cette couche représente les <b>sites d'intervention du PAM</b> considérés dans l'analyse.
        Pour chaque hexagone, le dashboard indique la distance au site le plus proche comme information
        complémentaire au ciblage.
        <br><br>
        <span class="source-text">Source : Programme alimentaire mondial (PAM).</span>
    </div>
    <div class="info-legend-title">Site</div>
    <div class="legend-row"><span class="point-symbol"></span>Site d'intervention PAM</div>
    <div class="methodological-note-footer">
        <b>Référence méthodologique :</b> Annexe 3 – Étape 8, « Données de transhumance et des interventions du PAM ».
    </div>
    """,

    "Score final de ciblage": """
    <div class="info-title">Score final de ciblage</div>
    <div class="info-text">
        Les résultats sont agrégés dans une grille d’<b>hexagones de 25 km²</b>. Le score combine la population estimée
        (<b>3 classes</b>) et la proportion de superficie à potentiel de restauration (<b>4 classes</b>).
        Pour cette dernière composante, les scénarios <b>Moderate</b> et <b>Conservative</b> sont traités
        <b>indistinctement</b> : leurs superficies sont comptabilisées ensemble, sans pondération différente entre les deux scénarios.
        Le score varie de <b>1 à 7</b> : plus le score est élevé, plus ces deux dimensions sont importantes
        relativement au reste de la zone d'étude.
        <br><br>
        Les distances aux routes de transhumance et aux sites du PAM sont affichées comme informations
        complémentaires et <b>ne sont pas intégrées au score</b>.
    </div>
    <div class="info-legend-title">Score de ciblage</div>
    <div class="score-legend-compact">
        <span class="score-legend-item"><span class="square" style="background:#FFF7EC;"></span>1</span>
        <span class="score-legend-item"><span class="square" style="background:#FEE8C8;"></span>2</span>
        <span class="score-legend-item"><span class="square" style="background:#FDD49E;"></span>3</span>
        <span class="score-legend-item"><span class="square" style="background:#FDBB84;"></span>4</span>
        <span class="score-legend-item"><span class="square" style="background:#FC8D59;"></span>5</span>
        <span class="score-legend-item"><span class="square" style="background:#E34A33;"></span>6</span>
        <span class="score-legend-item"><span class="square" style="background:#B30000;"></span>7</span>
    </div>
    <div class="methodological-note-footer">
        <b>Référence méthodologique :</b> Annexe 3 – Étape 9, « Construction du score final de ciblage ».
        L'utilisation de ce score dans le dashboard est décrite à l'Étape 10.
    </div>
    """
}



layer_panels = {
    layer_name: layer_html
    for layer_name, layer_html in layer_panels.items()
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

        initial_layer="Score final de ciblage"

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



        self.methodology_url = json.dumps(

            ANNEX_PDF_DATA_URL,

            ensure_ascii=False

        )

        self.github_repo_url = json.dumps(

            GITHUB_REPO_URL,

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

                overflow:

                    visible;



            }





            .leaflet-control-layers-base label,

            .leaflet-control-layers-overlays label {



                margin-bottom:

                    6px;



                font-size:

                    13px;



                color:

                    #58595a;

                position:

                    relative;

                display:

                    flex;

                align-items:

                    center;

                gap:

                    6px;

            }



            .layer-info-trigger {

                position:

                    relative;

                z-index:

                    2;

                width:

                    16px;

                height:

                    16px;

                border:

                    none;

                border-radius:

                    50%;

                background:

                    #005A9C;

                color:

                    white;

                font-size:

                    10px;

                line-height:

                    1;

                font-weight:

                    700;

                cursor:

                    pointer;

                display:

                    inline-flex;

                align-items:

                    center;

                justify-content:

                    center;

                padding:

                    0;

                margin-left:

                    auto;

                box-shadow:

                    0 1px 3px rgba(0,0,0,0.18);

            }





            .layer-info-trigger:hover {

                background:

                    #003F73;

            }



            .layer-info-popover {

                position:

                    fixed;

                width:

                    min(360px, calc(100vw - 32px));

                max-height:

                    none;

                overflow:

                    visible;

                background:

                    rgba(255,255,255,0.98);

                border:

                    1px solid #d9d9d9;

                border-left:

                    4px solid #005A9C;

                border-radius:

                    7px;

                box-shadow:

                    0 3px 12px rgba(0,0,0,0.2);

                padding:

                    10px 12px;

                z-index:

                    2000;

                display:

                    none;

                color:

                    #58595a;

                font-size:

                    12px;

                line-height:

                    1.4;

                pointer-events:

                    auto;

                box-sizing:

                    border-box;

            }



            .layer-info-popover.visible {

                display:

                    block;

            }



            .layer-info-popover .info-title {

                font-size:

                    14px;

                margin-bottom:

                    6px;

            }



            .layer-info-popover .info-text {

                font-size:

                    12px;

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

                    none;



                overflow:

                    visible;



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



            #methodological-note-control-footer {

                margin-top:

                    12px;

                padding-top:

                    10px;

                border-top:

                    1px solid #d9d9d9;

                font-size:

                    12px;

                font-weight:

                    700;

            }



            #methodological-note-control-footer a {

                color:

                    #005A9C;

                text-decoration:

                    none;

            }



            #methodological-note-control-footer a:hover {

                text-decoration:

                    underline;

            }



            .methodological-note-footer {

                margin-top:

                    14px;

                padding-top:

                    10px;

                border-top:

                    1px solid #e4e4e4;

                font-size:

                    12px;

                font-weight:

                    700;

            }



            .methodological-note-footer a {

                color:

                    #005A9C;

                text-decoration:

                    none;

            }



            .methodological-note-footer a:hover {

                text-decoration:

                    underline;

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





            .score-legend-compact {
                display: flex;
                align-items: center;
                gap: 7px;
                flex-wrap: nowrap;
                margin-top: 3px;
                margin-bottom: 2px;
            }

            .score-legend-item {
                display: inline-flex;
                align-items: center;
                gap: 3px;
                font-size: 11px;
                white-space: nowrap;
            }

            .score-legend-compact .square {
                width: 11px;
                height: 11px;
                flex: 0 0 11px;
                margin-right: 0;
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

            if (mapObject.zoomControl) {
                mapObject.zoomControl.setPosition("topright");
            }





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

                        "bottomleft"



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

                        "";

                    div.style.display =

                        "none";

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



            function closeLayerInfoPopovers() {

                document.querySelectorAll(

                    ".layer-info-popover"
                ).forEach(

                    function(popover) {

                        popover.classList.remove(

                            "visible"
                        );

                    }
                );

            }



            function normalizeLayerLabelText(labelText) {
                const rawText =
                    (labelText || "").replace(/\s+/g, " ").trim();
                if (!rawText) {
                    return "";
                }
                return rawText
                    .replace(/\s*i\s*$/i, "")
                    .replace(/\s+/g, " ")
                    .trim();
            }

            function resolveLayerKey(labelText) {

                const rawText = normalizeLayerLabelText(labelText);

                if (!rawText) {
                    return null;
                }

                if (layerPanels[rawText]) {
                    return rawText;
                }

                const normalized = rawText.toLowerCase();
                const lowercaseMatch = Object.keys(layerPanels).find(function(key) {
                    return key.toLowerCase() === normalized;
                });

                if (lowercaseMatch) {
                    return lowercaseMatch;
                }

                const flatText = rawText.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
                const accentMatch = Object.keys(layerPanels).find(function(key) {
                    return key.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase() === flatText;
                });

                return accentMatch || null;
            }

            function attachLayerInfoTriggers() {

                const labels =

                    document.querySelectorAll(

                        ".leaflet-control-layers-base label, .leaflet-control-layers-overlays label"
                    );



                labels.forEach(

                    function(label) {

                        if (label.dataset.layerInfoBound === "true") {
                            return;
                        }

                        const layerName =

                            resolveLayerKey(label.textContent);


                        if (!layerName || !layerPanels[layerName]) {

                            return;

                        }

                        label.dataset.layerInfoBound = "true";
                        label.style.position = "relative";

                        const trigger =

                            document.createElement(

                                "button"
                            );



                        trigger.type = "button";

                        trigger.className = "layer-info-trigger";

                        trigger.innerHTML = "i";

                        trigger.title = "Description";

                        trigger.setAttribute(

                            "aria-label",

                            `Afficher la description de ${layerName}`
                        );



                        trigger.addEventListener(

                            "pointerdown",

                            function(event) {

                                event.preventDefault();

                                event.stopPropagation();

                            },

                            true
                        );



                        trigger.addEventListener(

                            "click",

                            function(event) {

                                event.preventDefault();

                                event.stopPropagation();

                                event.stopImmediatePropagation();

                                let popover =

                                    document.getElementById(

                                        "layer-info-floating-popover"

                                    );

                                if (!popover) {

                                    popover = document.createElement("div");

                                    popover.id = "layer-info-floating-popover";

                                    popover.className = "layer-info-popover";

                                    popover.addEventListener("click", function(popoverEvent) {

                                        popoverEvent.stopPropagation();

                                    });

                                    document.body.appendChild(popover);

                                }

                                const isVisible =

                                    popover.classList.contains("visible")

                                    && popover.dataset.layerName === layerName;

                                closeLayerInfoPopovers();

                                if (!isVisible) {

                                    popover.innerHTML = layerPanels[layerName];

                                    popover.dataset.layerName = layerName;

                                    popover.classList.add("visible");

                                    const triggerRect = trigger.getBoundingClientRect();

                                    const margin = 8;

                                    popover.style.left = (triggerRect.right + margin) + "px";

                                    popover.style.top = triggerRect.top + "px";

                                    requestAnimationFrame(function() {

                                        const popupRect = popover.getBoundingClientRect();

                                        let left = triggerRect.right + margin;

                                        let top = triggerRect.top;

                                        if (left + popupRect.width > window.innerWidth - margin) {

                                            left = triggerRect.left - popupRect.width - margin;

                                        }

                                        // Reserve the lower part of the viewport for the hexagon-selection panel.
                                        // This keeps information popovers readable on single-screen setups.
                                        const bottomReserve = 150;
                                        const availableBottom = window.innerHeight - bottomReserve;

                                        if (top + popupRect.height > availableBottom) {

                                            top = availableBottom - popupRect.height;

                                        }

                                        popover.style.left = Math.max(margin, left) + "px";

                                        popover.style.top = Math.max(margin, top) + "px";

                                    });

                                    const box = document.getElementById("layer-info-box");

                                    if (box) {

                                        box.style.display = "none";

                                    }

                                }

                            }
                        );



                        label.appendChild(trigger);

                    }
                );

            }



            document.addEventListener(

                "click",

                function(event) {

                    const clickedTrigger =

                        event.target.closest(

                            ".layer-info-trigger"
                        );

                    const clickedPopover =

                        event.target.closest(

                            ".layer-info-popover"
                        );



                    if (!clickedTrigger && !clickedPopover) {
                        closeLayerInfoPopovers();
                    }

                }
            );

            const layerListObserver = new MutationObserver(function() {
                attachLayerInfoTriggers();
            });

            const layerListNode = document.querySelector(".leaflet-control-layers-list");
            if (layerListNode) {
                layerListObserver.observe(layerListNode, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    attributeFilter: ["class", "style"]
                });
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

                        null
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



                        showPanel(

                        null
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


                    const bases =

                        document.querySelector(

                            ".leaflet-control-layers-base"

                        );





                    attachLayerInfoTriggers();

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

                            "Ciblage des parcours pastoraux à potentiel de restauration – SCALE - GCF";





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





                    if (

                        list

                        &&

                        !document.getElementById(

                            "methodological-note-control-footer"

                        )

                    ) {

                        const methodologicalFooter =

                            document.createElement(

                                "div"

                            );



                        methodologicalFooter.id =

                            "methodological-note-control-footer";



                        methodologicalFooter.innerHTML =

                            '<a href="' + {{ this.methodology_url }} + '" download="Annex_3_Methodologie_Ciblage.pdf" target="_blank" rel="noopener noreferrer">Télécharger la note méthodologique (Annexe 3)</a>' +
                            '<br><a href="' + {{ this.github_repo_url }} + '" target="_blank" rel="noopener noreferrer">Accéder au dépôt GitHub du projet</a>';



                        list.appendChild(

                            methodologicalFooter

                        );

                    }

                    if (bases) {

                        const baseLabels = Array.from(bases.querySelectorAll("label"));
                        const desiredBaseOrder = [
                            "Couverture du sol (2021)",
                            "OpenStreetMap",
                            "Imagerie satellite"
                        ];

                        const normalizedBaseLabels = baseLabels.map(function(item) {
                            return {item: item, normalized: normalizeLayerLabelText(item.textContent)};
                        });

                        const orderedBaseLabels = [];
                        desiredBaseOrder.forEach(function(layerName) {
                            const match = normalizedBaseLabels.find(function(entry) {
                                return entry.normalized.toLowerCase() === layerName.toLowerCase();
                            });
                            if (match && !orderedBaseLabels.includes(match.item)) {
                                orderedBaseLabels.push(match.item);
                            }
                        });
                        normalizedBaseLabels
                            .filter(function(entry) { return !orderedBaseLabels.includes(entry.item); })
                            .forEach(function(entry) { orderedBaseLabels.push(entry.item); });
                        orderedBaseLabels.forEach(function(label) { bases.appendChild(label); });
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

                            "Parcours pastoraux à potentiel de restauration (2 scénarios)",

                            "Population estimée (2026)",

                            "Routes de transhumance",

                            "Sites d’intervention du PAM",

                            "Score final de ciblage"

                        ];

                        const normalizedLabels = labels.map(function(item) {
                            return {
                                item: item,
                                normalized: normalizeLayerLabelText(item.textContent)
                            };
                        });

                        const orderedLabels = [];

                        desiredOrder.forEach(function(layerName) {
                            const match = normalizedLabels.find(function(entry) {
                                return entry.normalized.toLowerCase() === layerName.toLowerCase();
                            });
                            if (match && !orderedLabels.includes(match.item)) {
                                orderedLabels.push(match.item);
                            }
                        });

                        normalizedLabels
                            .filter(function(entry) {
                                return !orderedLabels.includes(entry.item);
                            })
                            .map(function(entry) {
                                return entry.item;
                            })
                            .forEach(function(item) {
                                orderedLabels.push(item);
                            });

                        orderedLabels.forEach(function(label) {
                            overlays.appendChild(label);
                        });


                    }



                },



                300



            );





            /* =====================================

               ETAT INITIAL

            ===================================== */



            showPanel(

                null

            );





            {% endmacro %}

            """

        )





# ==================================================

# ASSOCIATION INFO <-> LAYERS

# ==================================================



layer_objects = [

    (
        "Couverture du sol (2021)",
        worldcover_layer
    ),

    (
        "Zone d'étude",
        aoi_group
    ),

    (
        "Parcours pastoraux à potentiel de restauration (2 scénarios)",
        restoration_group
    ),

    (
        "Population estimée (2026)",
        population_group
    ),

    (
        "Routes de transhumance",
        transhumance_group
    ),

    (
        "Sites d’intervention du PAM",
        sites_group
    ),

    (
        "Score final de ciblage",
        target_group
    )
]



LayerInfoControl(

    layer_panels=layer_panels,

    layers=layer_objects,

    initial_layer="Score final de ciblage"

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

    ],

    max_zoom=8

)





# ==================================================

# AFFICHAGE

# ==================================================



# Compact, collapsible selection panel for single-screen use.
st.markdown("""
<style>
.st-key-hex_selection_panel {
    position: fixed; bottom: 12px; left: 12px; z-index: 1000;
    width: min(1320px, calc(100vw - 24px));
    background: transparent !important; color: #333;
    padding: 0 !important; margin: 0 !important;
    box-sizing: border-box;
    font: 12px Arial, sans-serif;
}

/* When closed, keep only a small tab visible. Chrome/Edge support :has(). */
.st-key-hex_selection_panel:has(details:not([open])) {
    width: 245px !important;
}

.st-key-hex_selection_panel [data-testid="stExpander"] {
    width: 100% !important;
    margin: 0 !important;
}
.st-key-hex_selection_panel [data-testid="stExpander"] details {
    background: rgba(255,255,255,0.97) !important;
    border: 1px solid #d7d7d7 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.22) !important;
    overflow: visible !important;
}
.st-key-hex_selection_panel [data-testid="stExpander"] summary {
    min-height: 34px !important;
    padding: 5px 10px !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #444 !important;
}
.st-key-hex_selection_panel [data-testid="stExpander"] details[open] summary {
    border-bottom: 1px solid #e5e5e5 !important;
}
.st-key-hex_selection_panel [data-testid="stExpanderDetails"] {
    padding: 8px 10px 9px !important;
}

.st-key-hex_selection_panel [data-testid="stHorizontalBlock"] {
    min-width: 0 !important;
    height: auto !important;
    margin-top: 0 !important;
    align-items: center !important;
}
.st-key-hex_selection_panel [data-testid="stElementContainer"] {
    min-width: 0 !important;
}

.st-key-hex_restore_import_block {
    background: rgba(49, 51, 63, 0.06) !important;
    border: 1px solid rgba(49, 51, 63, 0.10) !important;
    border-radius: 0.45rem !important;
    padding: 6px 8px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
}
.st-key-hex_restore_import_block [data-testid="stFileUploader"] {
    margin: 2px 0 4px !important;
}
.st-key-hex_restore_import_block [data-testid="stFileUploader"] section {
    min-height: 32px !important;
    padding: 4px 8px !important;
    background: rgba(255,255,255,0.78) !important;
    border: 1px solid rgba(49, 51, 63, 0.12) !important;
    border-radius: 0.4rem !important;
}
.st-key-hex_restore_import_block [data-testid="stFileUploaderFileName"] {
    font-size: 10px !important;
}
.st-key-hex_restore_import_block [data-testid="stSelectbox"] {
    margin-top: 2px !important;
}
.st-key-hex_restore_import_block [data-testid="stSelectbox"] > div > div {
    min-height: 30px !important;
    font-size: 10.5px !important;
}

.st-key-hex_selection_panel .selection-section-title {
    font-size: 11px;
    font-weight: 700;
    color: #444;
    line-height: 1.2;
    margin: 0 0 3px 0;
}

.st-key-download_all_hexagons_link {
    margin-top: 3px !important;
    text-align: center !important;
}
.st-key-download_all_hexagons_link [data-testid="stDownloadButton"] {
    display: flex !important;
    justify-content: center !important;
}
.st-key-download_all_hexagons_link [data-testid="stDownloadButton"] button {
    min-height: auto !important;
    height: auto !important;
    width: auto !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #005A9C !important;
    font-size: 10.5px !important;
    font-weight: 400 !important;
    text-decoration: underline !important;
}
.st-key-download_all_hexagons_link [data-testid="stDownloadButton"] button:hover {
    color: #003F73 !important;
    background: transparent !important;
    border: none !important;
}

.st-key-hex_selection_panel .hex-selection-totals {
    display: block;
    width: 100%;
    font-size: 11.5px;
    line-height: 1.25;
}
.st-key-hex_selection_panel .hex-selection-totals table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    margin-top: 2px;
}
.st-key-hex_selection_panel .hex-selection-totals td {
    padding: 2px 6px 2px 0;
    vertical-align: middle;
}
.st-key-hex_selection_panel .hex-selection-totals .metric-label {
    width: 30%;
    color: #555;
    white-space: normal;
}
.st-key-hex_selection_panel .hex-selection-totals .metric-value {
    width: 20%;
    text-align: right;
    white-space: nowrap;
    padding-right: 12px;
}
.st-key-hex_selection_panel button {
    font-size: 10.5px;
    min-height: 34px;
    padding: 0.3rem 0.45rem !important;
    white-space: normal !important;
    line-height: 1.15 !important;
}

/* Give the map maximum room on narrower / single-screen setups. */
@media (max-width: 1200px) {
    .st-key-hex_selection_panel {
        width: calc(100vw - 16px) !important;
        left: 8px !important;
        bottom: 8px !important;
    }
    .st-key-hex_selection_panel:has(details:not([open])) {
        width: 230px !important;
    }
}
</style>
""", unsafe_allow_html=True)

render_selection_summary(
    selection_data, targetisation, valid_fields, valid_aliases
)
selection_updates = folium.FeatureGroup(control=False)
HexSelectionStyle(target_hex_layer, st.session_state.selected_hex_ids).add_to(
    selection_updates
)
st_folium(
    m,
    key="hex_selection_map",
    width=2000,
    height=1000,
    use_container_width=True,
    returned_objects=["last_active_drawing", "last_object_clicked",
                      "last_object_clicked_count"],
    feature_group_to_add=selection_updates,
    on_change=handle_hex_click,
)

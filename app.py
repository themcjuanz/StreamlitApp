# app_mpacolombia.py
"""
Aplicación Streamlit reorganizada para:
Modelos predictivos de la adopción de vehículos eléctricos e híbridos en Colombia

Estructura:
- Imports
- Constantes / configuración
- Helpers (funciones reutilizables)
- Carga de datos
- UI (secciones / tabs)
"""

# -------------------------
# IMPORTS
# -------------------------
import os
import glob
import json
import unicodedata
import streamlit.components.v1 as components
import base64
from typing import Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from datetime import datetime, timedelta

# -------------------------
# CONSTANTS / CONFIG
# -------------------------
PAGE_TITLE = "mpacolombia"
PAGE_ICON = "📈"

COLUMNA_FECHA = "ds"
COLUMNA_VALOR = "y"

# Archivos / paths
CSV_PIB = "pib_forecast_prophet.csv"
CSV_MENSUAL = "mensual.csv"
CSV_ORIGINAL = "Numero_de_Veh_culos_El_ctricos_-_Hibridos_20250804.csv"
CSV_CLEAN = "cleaned_ds.csv"
CSV_FORECAST_MAIN = "forecast_72m.csv"  # fallback local forecast
FORECAST_FOLDER = "forecasts"
GEOJSON_PATH = "colombia_departments.json"

# Visual / parámetros
PREDICTED_YEAR_THRESHOLD = 2023
ZERO_TOL_FACTOR = 0.01
MIN_ZERO_TOL = 1

# Slider range (extiende mucho más allá de los datos históricos para previsiones)
START_PERIOD = pd.Period("2010-01", freq="M")
END_PERIOD = pd.Period("2028-12", freq="M")

# -------------------------
# PAGE CONFIG & CSS
# -------------------------
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
/* Fondo general oscuro */
.report-background { background-color: #0b0f0c; color: #0b0f0c; }

/* Encabezado principal (más pequeño, color verde claro) */
.main-header { 
    background: #0d1f17;         
    padding: 1rem; 
    border-radius: 0px; 
    color: #a8f5c0;              
    text-align: center; 
    margin-bottom: 1.5rem; 
    border: 1px solid #1e5a3a;   
    box-shadow: 0 3px 8px rgba(0,0,0,0.5); 
}
.main-header h1 {
    font-size: 1.9rem;    
    font-weight: 600;    
    margin: 0;
    letter-spacing: 0.5px;
}

/* Cajas de presentación e insights */
.presentation-box, .insight-box { 
    background: #1a1f1d;   
    padding: 1rem; 
    border-radius: 0px; 
    margin: 1rem 0; 
    color: #ccffcc;        
}

/* Encabezados de sección (verde más oscuro, igual que objetivo) */
.section-header { 
    background: #1a1f1d; 
    padding: 1rem; 
    border-radius: 0px; 
    color: #ccffcc; 
    text-align: center; 
    margin: 1rem 0; 
}

/* Fondo general de la app */
[data-testid="stAppViewContainer"] > div { 
    background: #0b0f0f; 
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# HELPERS
# -------------------------
def normalize_name(s: Optional[str]) -> str:
    """Normaliza nombres para comparación insensible a tildes / mayúsculas / espacios / caracteres no alfanuméricos."""
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = "".join(ch for ch in s if ch.isalnum())
    return s


def count_to_rgb(c: float, vmin: float = 0, vmax: float = 300, alpha: int = 100) -> List[int]:
    """Mapea un conteo a RGBA interpolando entre dos tonos de verde/naranja."""
    denom = (vmax - vmin) if (vmax is not None and vmax > vmin) else 1.0
    t = max(0.0, min(1.0, (c - vmin) / denom))
    dark_green = (0, 72, 46)
    bright_green = (255, 200, 80)
    r = int(dark_green[0] + t * (bright_green[0] - dark_green[0]))
    g = int(dark_green[1] + t * (bright_green[1] - dark_green[1]))
    b = int(dark_green[2] + t * (bright_green[2] - dark_green[2]))
    a = int(alpha)
    return [r, g, b, a]


def load_csv_safe(path: str, **kwargs) -> Optional[pd.DataFrame]:
    """Carga CSV devolviendo None si falla (en lugar de lanzar excepción)."""
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError:
        return None
    except Exception as ex:
        st.warning(f"Error leyendo {path}: {ex}")
        return None


def load_forecast_files(folder: str) -> Dict[str, pd.DataFrame]:
    """Carga archivos de forecast del folder que terminen en '_forecast.csv' y devuelve un dict normalizado."""
    files = glob.glob(os.path.join(folder, "*_forecast.csv"))
    dept_dfs: Dict[str, pd.DataFrame] = {}
    for fp in files:
        fname = os.path.basename(fp)
        if not fname.upper().endswith("_FORECAST.CSV"):
            continue
        dept_raw = fname[:-len("_forecast.csv")]
        dept_norm = normalize_name(dept_raw)
        try:
            df = pd.read_csv(fp)
            df.columns = [c.lower() for c in df.columns]
            if "ds" not in df.columns:
                st.warning(f"{fname} no tiene columna 'ds' — se omite.")
                continue
            df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
            df = df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
            if not df.empty:
                dept_dfs[dept_norm] = df
        except Exception as e:
            st.warning(f"Error leyendo {fname}: {e}")
            continue
    return dept_dfs


def compute_vmin_vmax(values: List[float]) -> Tuple[float, float]:
    """Calcula vmin/vmax robustos para el mapeo de colores usando percentiles."""
    if values:
        vmin = float(np.percentile(values, 5))
        vmax = float(np.percentile(values, 97))
        if abs(vmax - vmin) < 1e-6:
            vmin -= 1
            vmax += 1
    else:
        vmin, vmax = 0.0, 2.0
    return vmin, vmax


def prepare_geojson_props(
    geojson: dict,
    dept_yhat: Dict[str, Optional[int]],
    dept_yhat_date: Dict[str, Optional[str]],
    is_predicted: bool,
    vmin: float,
    vmax: float,
    zero_tol: float,
) -> None:
    """Añade propiedades 'count', 'color', 'line_color', 'yhat_display' a cada feature del geojson."""
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        nombre = props.get("NOMBRE_DPT") or props.get("NOMBRE") or ""
        nombre_norm = normalize_name(nombre)

        count_val = dept_yhat.get(nombre_norm)
        count_date = dept_yhat_date.get(nombre_norm)

        transparent = False
        if count_val is None:
            transparent = True
        else:
            try:
                if count_val <= 0 or abs(count_val) <= zero_tol:
                    transparent = True
            except Exception:
                transparent = True

        if transparent:
            color = [0, 0, 0, 0]
            line_color = [0, 0, 0, 0]
            count_prop = None
            display_text = None
        else:
            color = count_to_rgb(count_val, vmin=vmin, vmax=vmax, alpha=200)
            line_color = [80, 80, 80, 255]
            count_prop = int(count_val)
            display_text = f"{count_prop}"
            if count_date:
                display_text = f"{display_text} (fecha: {count_date})"
            if is_predicted:
                display_text = display_text + " (predicho)"

        feat["properties"]["count"] = count_prop if count_prop is not None else None
        feat["properties"]["color"] = color
        feat["properties"]["line_color"] = line_color
        feat["properties"]["yhat_display"] = display_text


# -------------------------
# CARGA DE DATOS
# -------------------------
# Carga principal (mensual)
df_datos = load_csv_safe(CSV_MENSUAL)
df_datosp = load_csv_safe(CSV_ORIGINAL)
df_datosc = load_csv_safe(CSV_CLEAN)

# Procesamiento del dataframe mensual base (df_ejemplo)
df_ejemplo = None
if df_datos is None:
    st.error(f"No se encontró {CSV_MENSUAL} en el directorio.")
    st.stop()
else:
    try:
        # Mantener seguridad al renombrar columnas
        if COLUMNA_FECHA in df_datos.columns and COLUMNA_VALOR in df_datos.columns:
            df_ejemplo = df_datos[[COLUMNA_FECHA, COLUMNA_VALOR]].copy()
        else:
            # Intento flexible: busca columnas similares
            cols_lower = {c.lower(): c for c in df_datos.columns}
            if "ds" in cols_lower and "y" in cols_lower:
                df_ejemplo = df_datos[[cols_lower["ds"], cols_lower["y"]]].copy()
                df_ejemplo.columns = [COLUMNA_FECHA, COLUMNA_VALOR]
            else:
                raise ValueError("El CSV mensual no contiene columnas 'ds' y 'y' reconocibles.")
        df_ejemplo.columns = ["fecha", "valor"]
        df_ejemplo["fecha"] = pd.to_datetime(df_ejemplo["fecha"], errors="coerce")
        df_ejemplo = df_ejemplo.dropna(subset=["fecha", "valor"]).sort_values("fecha").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")
        st.stop()

if df_ejemplo is None or df_ejemplo.empty:
    st.error("El dataset está vacío después del preprocesamiento.")
    st.stop()

# -------------------------
# HEADER UI
# -------------------------
st.markdown(
    """
<div class="main-header">
    <h1>Modelos predictivos de la adopción de vehículos eléctricos e híbridos en Colombia</h1>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------
# NAV / TABS
# -------------------------
sections = ["PRESENTACIÓN", "FORECAST", "ARBOLES DE DECISION", "RANDOM FOREST", "GRADIENT BOOSTING", "KNN", "REGRESION LOGISTICA", "CONCLUSIONES"]
tabs = st.tabs(sections)

# -------------------------
# PRESENTACIÓN TAB
# -------------------------
with tabs[0]:
    st.markdown(
        """
    <div class="presentation-box">
    <h2>Contexto y objetivo del proyecto</h2>

    <p>La transición hacia la movilidad sostenible en Colombia enfrenta retos importantes, tanto sociales como económicos, que condicionan la adopción de vehículos eléctricos e híbridos. Factores como el alto costo de adquisición, la limitada infraestructura de carga en regiones fuera de las principales ciudades y la percepción del consumidor sobre autonomía y confiabilidad han generado una adopción desigual de estas tecnologías en el país. Esta situación refleja un mercado fragmentado, donde las barreras de acceso y las diferencias regionales limitan la masificación de los vehículos electrificados, a pesar de los esfuerzos en incentivos gubernamentales y campañas de concienciación ambiental.</p>

    <p>En este contexto, el presente proyecto tiene como objetivo analizar y predecir la adopción de vehículos electrificados a nivel departamental en Colombia, mediante el desarrollo de modelos de clasificación que permitan identificar el tipo de combustible utilizado en los registros vehiculares (eléctrico, híbrido gasolina, híbrido diésel). Para ello, se trabaja con un conjunto de datos que combina variables categóricas y numéricas, evaluando su significancia y aplicando técnicas de codificación y preprocesamiento. El uso de modelos predictivos permite detectar patrones, evaluar la influencia de factores regionales y técnicos, y generar insumos que faciliten la toma de decisiones tanto para autoridades públicas como para actores del sector automotor en el marco de la transición energética.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.subheader("Vista previa del dataset original")
    if df_datosp is not None:
        st.dataframe(df_datosp.head(4))
    else:
        st.info(f"No se encontró {CSV_ORIGINAL} — se omitirá la vista previa.")

    st.subheader("Valores nulos")
    st.image("antes.png", use_container_width=True)

    st.subheader("Análisis descriptivo de la base de datos")
    # Incrustar Power BI público (si se desea)
    iframe_code = """
    <iframe title="elec" 
        width="100%" height="600" 
        src="https://app.powerbi.com/view?r=eyJrIjoiZTFhNDExOWYtOGVlNC00MWU2LThlZmItMGQ5NjFmOTE0Yjc0IiwidCI6IjU3N2ZjMWQ4LTA5MjItNDU4ZS04N2JmLWVjNGY0NTVlYjYwMCIsImMiOjR9" 
        frameborder="0" allowFullScreen="true">
    </iframe>
    """
    components.html(iframe_code, height=650, scrolling=True)

    st.markdown(
        """
    <div class="presentation-box">
    <h2>Selección y limpieza de variables</h2>

    <p>Antes de visualizar valores faltantes y ceros, se decidió excluir variables que no aportan información relevante para la predicción, tras consultar con el equipo docente.</p>

    <ul>
        <li><strong>AÑO_REGISTRO, FECHA_REGISTRO:</strong> Variables temporales que no se usan para el entrenamiento.</li>
        <li><strong>MARCA, LÍNEA:</strong> Cardinalidad muy alta (→ cientos de clases); convertirlas a dummies introduciría >500 columnas, contraproducente para el modelo.</li>
        <li><strong>MODALIDAD:</strong> Aplicable principalmente a vehículos grandes y con muchos valores faltantes.</li>
        <li><strong>CAPACIDAD_CARGA, CAPACIDAD_PASAJEROS:</strong> Exclusivas de buses/camiones y con numerosos registros incompletos.</li>
        <li><strong>CANTIDAD:</strong> Siempre igual a 1 en los registros, por lo que no aporta.</li>
        <li><strong>ORGANISMO_TRANSITO, MUNICIPIO:</strong> Redundantes con la variable <em>DEPARTAMENTO</em>.</li>
        <li><strong>CILINDRAJE:</strong> Fuga de información (los vehículos eléctricos aparecen con 0), por lo que se omite.</li>
    </ul>

    <p>Con el dataset ya preparado, el siguiente paso fue limpiar: se eliminaron los registros con variables nulas o con valores iguales a cero para evitar sesgos y problemas durante el entrenamiento de los modelos.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.subheader("Valores nulos")
    st.image("limpieza.png", use_container_width=True)

    st.subheader("Vista previa del dataset limpio")
    if df_datosc is not None:
        st.dataframe(df_datosc.head(4))
    else:
        st.info(f"No se encontró {CSV_CLEAN} — se omitirá la vista previa.")


# -------------------------
# FORECAST TAB
# -------------------------
with tabs[1]:
    st.markdown(
        """
    <div class="presentation-box">
        <h2>Forecast con Prophet</h2>
        <p>En este análisis se utilizó el Producto Interno Bruto (PIB) como regresor dentro de un modelo de series de tiempo con el fin de capturar la relación existente entre la dinámica económica nacional y la adopción de vehículos eléctricos e híbridos en Colombia, para 
        ello se necesitaba de igual forma una proyección del PIB colombiano en el mismo periodo de tiempo al que se planeaba realizar la predicción de registros de vehículos eléctricos e híbridos:<p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.write("""
[Fuente: DANE](https://www.dane.gov.co)
""")

    # ------------------------------
# BLOQUE: cargar y mostrar CSV PIB
# ------------------------------
    df_pib = load_csv_safe(CSV_PIB)  # CSV_PIB = "pib_hist_forecast.csv" o como lo tengas

    # permitir subir si no existe localmente
    if df_pib is None:
        uploaded_pib = st.sidebar.file_uploader("Sube el CSV de PIB (histórico + predicción)", type=["csv"], key="pib_file")
        if uploaded_pib is not None:
            df_pib = pd.read_csv(uploaded_pib)

    if df_pib is None:
        st.info("No hay archivo de PIB cargado.")
    else:
        # Normalizar nombres de columnas (flexible)
        cols_lower = {c.lower(): c for c in df_pib.columns}

        # Detectar la columna de valor: 'pib' preferido, si no, aceptar 'yhat'
        value_col = None
        if "pib" in cols_lower:
            value_col = cols_lower["pib"]
        elif "yhat" in cols_lower:
            value_col = cols_lower["yhat"]
        else:
            st.warning("El archivo de PIB debe contener una columna 'pib' o 'yhat' con los valores del PIB.")
            value_col = None

        if value_col is not None:
            # detectar posibles columnas de intervalo de confianza
            lower_col = None
            upper_col = None
            if "pib_lower" in cols_lower and "pib_upper" in cols_lower:
                lower_col = cols_lower["pib_lower"]
                upper_col = cols_lower["pib_upper"]
            elif "yhat_lower" in cols_lower and "yhat_upper" in cols_lower:
                lower_col = cols_lower["yhat_lower"]
                upper_col = cols_lower["yhat_upper"]
            else:
                # intento heurístico: buscar columnas que contengan 'lower'/'upper' junto a 'pib' o 'yhat'
                lower_candidates = [k for k in cols_lower.keys() if ("lower" in k or "low" in k) and ("pib" in k or "yhat" in k)]
                upper_candidates = [k for k in cols_lower.keys() if ("upper" in k or "high" in k) and ("pib" in k or "yhat" in k)]
                if lower_candidates and upper_candidates:
                    lower_col = cols_lower[lower_candidates[0]]
                    upper_col = cols_lower[upper_candidates[0]]

            # renombrar internamente a columnas estándar: ds, pib, pib_lower?, pib_upper?
            rename_map = {}
            # ds
            if "ds" in cols_lower:
                rename_map[cols_lower["ds"]] = "ds"
            else:
                st.warning("El archivo de PIB debe tener columna de fecha 'ds'. Intentando inferir...")
                # si no hay ds, intentar columnas de fecha comunes
                for cand in ["date", "fecha"]:
                    if cand in cols_lower:
                        rename_map[cols_lower[cand]] = "ds"
                        break

            rename_map[value_col] = "pib"
            if lower_col:
                rename_map[lower_col] = "pib_lower"
            if upper_col:
                rename_map[upper_col] = "pib_upper"

            df_pib = df_pib.rename(columns=rename_map)

            # parsear fecha y limpiar
            df_pib["ds"] = pd.to_datetime(df_pib["ds"], errors="coerce")
            required = ["ds", "pib"]
            if all(c in df_pib.columns for c in required):
                if ("pib_lower" in df_pib.columns) and ("pib_upper" in df_pib.columns):
                    df_pib = df_pib.dropna(subset=["ds", "pib", "pib_lower", "pib_upper"]).sort_values("ds")
                    has_ci = True
                else:
                    df_pib = df_pib.dropna(subset=["ds", "pib"]).sort_values("ds")
                    has_ci = False
            else:
                st.error("No se pudieron normalizar las columnas requeridas del CSV de PIB.")
                has_ci = False

            # obtener última fecha histórica si existe df_ejemplo (igual que en tu código)
            try:
                ultima_fecha_historica = df_ejemplo["fecha"].max()
            except Exception:
                ultima_fecha_historica = df_pib["ds"].max()

            # --- Construir chart Altair (banda opcional + línea + puntos + regla) ---
            layers = []

            if has_ci:
                band = (
                    alt.Chart(df_pib)
                    .mark_area(opacity=0.25, interpolate="monotone")
                    .encode(x="ds:T", y="pib_lower:Q", y2="pib_upper:Q")
                )
                layers.append(band)

            pib_line = (
                alt.Chart(df_pib)
                .mark_line(color="#00bfff", strokeWidth=2)  # cyan suave
                .encode(x="ds:T", y="pib:Q", tooltip=["ds:T", "pib:Q"])
            )
            layers.append(pib_line)

            pib_points = (
                alt.Chart(df_pib)
                .mark_point(size=40)
                .encode(x="ds:T", y="pib:Q")
            )
            layers.append(pib_points)

            if pd.notna(ultima_fecha_historica):
                rule = (
                    alt.Chart(pd.DataFrame({"fecha": [ultima_fecha_historica]}))
                    .mark_rule(color="orange", strokeDash=[5, 5], strokeWidth=2)
                    .encode(x="fecha:T")
                )
                layers.append(rule)

            chart = (
                alt.layer(*layers)
                .properties(title="PIB (histórico + predicción)", height=420, background="#0a0f0a")
                .configure_axis(labelColor="white", titleColor="white")
                .configure_title(color="white")
            )

            st.altair_chart(chart, use_container_width=True)


    # Cargar archivo de forecast principal o permitir subir
    df_forecast = load_csv_safe(CSV_FORECAST_MAIN)
    if df_forecast is None:
        uploaded_f = st.sidebar.file_uploader("Sube forecast_72m.csv (o similar)", type=["csv"], key="forecast")
        if uploaded_f is not None:
            df_forecast = pd.read_csv(uploaded_f)

    if df_forecast is None:
        st.info("No hay archivo de forecast cargado.")
    else:
        # Normalizar columnas esperadas
        cols_lower = {c.lower(): c for c in df_forecast.columns}
        required_lower = {"ds", "yhat", "yhat_lower", "yhat_upper"}
        if not required_lower.issubset(set(cols_lower.keys())):
            st.warning(f"El forecast debe contener {required_lower}. Encontradas: {list(df_forecast.columns)}")
        else:
            df_forecast = df_forecast.rename(
                columns={
                    cols_lower["ds"]: "ds",
                    cols_lower["yhat"]: "yhat",
                    cols_lower["yhat_lower"]: "yhat_lower",
                    cols_lower["yhat_upper"]: "yhat_upper",
                }
            )
            df_forecast["ds"] = pd.to_datetime(df_forecast["ds"], errors="coerce")
            df_forecast = df_forecast.dropna(subset=["ds", "yhat"]).sort_values("ds")

            ultima_fecha_historica = df_ejemplo["fecha"].max()

            # Banda de confianza, línea y puntos (Altair)
            band = (
                alt.Chart(df_forecast)
                .mark_area(opacity=0.25, color="#00cc66")
                .encode(x="ds:T", y="yhat_lower:Q", y2="yhat_upper:Q")
            )

            line = (
                alt.Chart(df_forecast)
                .mark_line(color="#00ff99", strokeWidth=2)
                .encode(x="ds:T", y="yhat:Q")
            )

            points = (
                alt.Chart(df_forecast)
                .mark_point(size=40)
                .encode(x="ds:T", y="yhat:Q")
            )

            rule = (
                alt.Chart(pd.DataFrame({"fecha": [ultima_fecha_historica]}))
                .mark_rule(color="orange", strokeDash=[5, 5], strokeWidth=2)
                .encode(x="fecha:T")
            )

            st.write("""
[Fuente: DATOS ABIERTOS](https://www.datos.gov.co/Transporte/Numero-de-Veh-culos-El-ctricos-Hibridos/7qfh-tkr3/about_data)
""")

            chart = (
                alt.layer(band, line, points, rule)
                .properties(title="Registros (Electricos & Hibridos)", height=450, background="#0a0f0a")
                .configure_axis(labelColor="white", titleColor="white")
                .configure_title(color="white")
            )

            st.altair_chart(chart, use_container_width=True)

            # ---------------------------
            # Mapa predictivo por departamentos
            # ---------------------------


            st.markdown(
                """
            <div class="presentation-box">
                <h2>Mapa de Registros Departamentales</h2>
                <p>Para este mapa se hizo un precesamiento parecido al anterior solo que filtrando el conteo de registros por departamento y luego haciendo una predicción basada en esos datos historicos mensuales.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Cargar geojson
            geojson = {}
            try:
                with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
                    geojson = json.load(f)
            except Exception as e:
                st.error(f"No se pudo leer {GEOJSON_PATH}: {e}")
                st.stop()

            # Cargar forecasts por departamento desde carpeta
            dept_dfs = load_forecast_files(FORECAST_FOLDER)
            if not dept_dfs:
                st.error(f"No se encontraron archivos '*_forecast.csv' en '{FORECAST_FOLDER}'")
                st.stop()

            # Slider de meses
            months_extended = [p.strftime("%Y-%m") for p in pd.period_range(start=START_PERIOD, end=END_PERIOD, freq="M")]

            all_dates_from_files = [df["ds"].max() for df in dept_dfs.values() if not df.empty]
            if all_dates_from_files:
                last_date_available = max(all_dates_from_files)
                default_period = pd.Period(last_date_available, freq="M")
                default_value = default_period.strftime("%Y-%m")
            else:
                default_value = months_extended[-1]

            selected_month = st.select_slider("Año/Mes", options=months_extended, value=default_value)
            selected_period = pd.Period(selected_month, freq="M")
            selected_year = selected_period.year
            is_predicted = selected_year >= PREDICTED_YEAR_THRESHOLD
            
            # --- Obtener valores por departamento (histórico o forecast según el caso) ---
            dept_yhat: Dict[str, Optional[int]] = {}
            dept_yhat_date: Dict[str, Optional[str]] = {}
            yhat_values: List[float] = []

            historical_cutoff = pd.Period("2010-01", freq="M").to_timestamp()
            if selected_period.to_timestamp() <= historical_cutoff:
                # Usar df_datosc para históricos (si está disponible)
                if df_datosc is None:
                    st.warning("No hay dataframe limpio (cleaned_dataframe.csv) — no se pueden mostrar datos históricos para este mes.")
                else:
                    df_hist_filtered = df_datosc[pd.to_datetime(df_datosc["Fecha"]).dt.to_period("M") == selected_period].copy()
                    df_hist_filtered["Departamento_Norm"] = df_hist_filtered["Departamento"].apply(normalize_name)
                    df_hist_sum = df_hist_filtered.groupby("Departamento_Norm")["Cantidad"].sum().reset_index()
                    for _, row in df_hist_sum.iterrows():
                        dept_norm = row["Departamento_Norm"]
                        yi = int(row["Cantidad"])
                        dept_yhat[dept_norm] = yi
                        dept_yhat_date[dept_norm] = selected_month
                        yhat_values.append(yi)
            else:
                # Valores desde forecast files
                for dept_norm, df in dept_dfs.items():
                    cutoff = selected_period.to_timestamp(how="end")
                    df_le = df[df["ds"] <= cutoff]
                    if not df_le.empty:
                        row = df_le.iloc[-1]
                    else:
                        row = None

                    if row is None:
                        dept_yhat[dept_norm] = None
                        dept_yhat_date[dept_norm] = None
                    else:
                        yhat = row.get("yhat")
                        try:
                            if pd.isna(yhat):
                                dept_yhat[dept_norm] = None
                                dept_yhat_date[dept_norm] = None
                            else:
                                yi = int(round(float(yhat)))
                                dept_yhat[dept_norm] = yi
                                dept_yhat_date[dept_norm] = row["ds"].strftime("%Y-%m-%d")
                                if yi != 0:
                                    yhat_values.append(yi)
                        except Exception:
                            dept_yhat[dept_norm] = None
                            dept_yhat_date[dept_norm] = None

            # Calcular vmin/vmax y tolerancia a cero
            vmin, vmax = compute_vmin_vmax(yhat_values)
            zero_tol = max(MIN_ZERO_TOL, ZERO_TOL_FACTOR * (abs(vmax - vmin)))

            # Preparar propiedades geojson para pydeck
            prepare_geojson_props(geojson, dept_yhat, dept_yhat_date, is_predicted, vmin, vmax, zero_tol)

            # Crear capa y mapa pydeck
            polygon_layer = pdk.Layer(
                "GeoJsonLayer",
                geojson,
                stroked=True,
                filled=True,
                extruded=False,
                pickable=True,
                auto_highlight=True,
                get_fill_color="properties.color",
                get_line_color="properties.line_color",
            )

            view_state = pdk.ViewState(latitude=4.5, longitude=-74.0, zoom=5)
            tooltip = {
                "html": "<b>{NOMBRE_DPT}</b><br/>Código: {DPTO}<br/>yhat: {yhat_display}",
                "style": {"backgroundColor": "white", "color": "black", "fontSize": "12px"},
            }

            deck = pdk.Deck(layers=[polygon_layer], initial_view_state=view_state, tooltip=tooltip)
            st.pydeck_chart(deck)

            # Tabla con valores yhat por departamento
            rows_out = []
            for feat in geojson.get("features", []):
                props = feat.get("properties", {})
                rows_out.append(
                    {
                        "Departamento": props.get("NOMBRE_DPT") or props.get("NOMBRE") or "",
                        "yhat": props.get("count"),
                    }
                )

            st.markdown("Mes Actual {}".format(selected_period.strftime("%B %Y")))

            df_rows = pd.DataFrame(rows_out)
            df_rows["yhat"] = pd.to_numeric(df_rows["yhat"], errors="coerce").fillna(0).astype(int)
            df_rows = df_rows.sort_values("yhat", ascending=False).reset_index(drop=True)
            st.markdown("### Yhat por Departamento")
            st.dataframe(df_rows, height=300)


# -------------------------
# ÁRBOLES DE DECISIÓN TAB
# -------------------------
with tabs[2]:
    st.markdown(
        """
    <div class="presentation-box">
        <h2>Arboles de Decisión</h2>
        <p>Se utilizó un modelo basado en árboles de decisión a varias profundidades debido a su transparencia y facilidad de interpretación.
           Mediante GridSearch optimizamos los hiperparámetros y se obtuvo un modelo robusto.</p>
        <ul>
            <li>Modelo: Árboles de Decisión</li>
            <li>Hiperparámetros: Optimizados mediante GridSearch</li>
            <li>Profundidad: 10, 17 (Mejor profundidad, aunque poca diferencia)</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    df_chi2 = load_csv_safe("chi2.csv", dtype=str)
    df_anova = load_csv_safe("anova_results.csv", dtype=str)
    df_cramers = load_csv_safe("cramers_results.csv", dtype=str)
    df_vif = load_csv_safe("vif.csv", dtype=str)

    st.subheader("Análisis de significancia en variables predictoras categóricas (chi²)")
    if df_chi2 is not None:
        st.dataframe(df_chi2.head(4))
    else:
        st.info("chi2.csv no encontrado.")

    st.subheader("Análisis de significancia en variables predictoras numéricas usando ANOVA")
    if df_anova is not None:
        st.dataframe(df_anova.head(4))
    else:
        st.info("anova_results.csv no encontrado.")

    st.subheader("Análisis de correlación entre variables categóricas usando Cramér's V")
    st.markdown("""
**Se evidencia una gran correlación entre algunas variables categóricas.**  
Este análisis se hace a partir de la medición de **Cramér's V**, ya que el test chi² y su valor *p* pueden verse muy afectados en datasets grandes como el de este proyecto y también por la cantidad de categorías que tiene cada variable.  

Con esta medición identificamos las siguientes correlaciones fuertes:

- **CLASIFICACIÓN vs CLASE** → V = **1.0** → Correlación perfecta (son redundantes).  
- **CLASE vs CARROCERÍA** → V = **0.725** → Muy fuerte (probablemente porque el tipo de vehículo determina la carrocería).  
- **SERVICIO vs CARROCERÍA** → V = **0.674** → Fuerte (por ejemplo, los vehículos públicos tienen carrocerías específicas).

Se decidió eliminar algunas variables redundantes, específicamente CLASE y SERVICIO, ya que presentan una alta correlación con otras variables y aportan poca significancia para clasificar la variable dependiente COMBUSTIBLE en comparación con el resto de variables predictoras.
""")

    if df_cramers is not None:
        st.dataframe(df_cramers)
    else:
        st.info("cramers_results.csv no encontrado.")

    st.subheader("Matriz de Confusión")
    st.write("A partir del análisis de la matriz de correlación entre las variables numéricas, se observa que no existen relaciones lineales fuertes (todas las correlaciones son menores a 0.7), lo cual indica que no hay un problema de multicolinealidad considerable en el conjunto de datos. En particular, la relación entre peso y potencia presenta una correlación moderadas. Por lo tanto, se concluye que cada variable numérica aporta información diferenciada al modelo, razón por la cual ninguna debe ser eliminada en esta etapa del análisis.")
    st.image("confusion.png", width=325)  # ancho en píxeles

    st.subheader("Correlación de Spearman")
    st.write("Los resultados de la correlación de Spearman muestran que existe una relación positiva. En particular, se observa que el peso y la potencia presentan una correlación fuerte (ρ = 0.73), lo cual tiene sentido desde el punto de vista técnico: a medida que los vehículos poseen mayor peso, requieren también de mayor potencia para su desplazamiento eficiente Por tal motivo se desea terminar de verificar la existencia de correlación entre las varaibles númericas empleando la evaluación del vif entre ellas.")
    st.image("spearman.png", width=325)  # ancho en píxeles

    st.subheader("Factor de Inflación de Varianza (VIF)")
    st.write("El estudio de correlaciones mediante Pearson y Spearman evidenció relaciones positivas entre las variables numéricas, especialmente entre PESO y POTENCIA, lo cual resulta coherente desde el punto de vista técnico, ya que a mayor peso del vehículo suele requerirse una mayor potencia. Sin embargo, el análisis de multicolinealidad mediante VIF confirmó que estas correlaciones no representan un problema estadístico, ya que los valores obtenidos se encuentran en rangos aceptables. En conjunto, los resultados permiten concluir que las variables analizadas pueden emplearse en el modelo sin riesgo de redundancia significativa.")
    if df_vif is not None:
        st.dataframe(df_vif.head(10))
    else:
        st.info("vif.csv no encontrado.")

    st.subheader("Grafica del Arbol")
    st.write("Profundidad en la grafica = 4")

    st.image("Arbol.png", use_container_width=True)

    st.markdown(
    """
<div class="presentation-box">
    <h2>Resultados</h2>
    <p style="display:flex; gap:9px; align-items:center; white-space:nowrap; overflow-x:auto; padding-bottom:3px;">
        <span><strong>F1 Score:</strong> Training: 0.9940 — Testing: 0.9950</span>
        <span><strong>Accuracy:</strong> Training: 0.9967 — Testing: 0.9974</span>
        <span><strong>Recall:</strong> Training: 0.9984 — Testing: 0.9973</span>
        <span><strong>Precision:</strong> Training: 0.9898 — Testing: 0.9927</span>
    </p>
</div>
""",
    unsafe_allow_html=True,
)

with tabs[3]:
    st.markdown(
        """
        <div class="presentation-box">
        <h2>Arboles Aleatorios</h2>
        <p>Se utilizó un modelo basado en árboles aleatorios a varias profundidades debido a su transparencia y facilidad de interpretación.
           Mediante GridSearch optimizamos los hiperparámetros y se obtuvo un modelo robusto.</p>
        <ul>
            <li>Modelo: Árboles Aleatorios</li>
            <li>Hiperparámetros: Optimizados mediante GridSearch</li>
            <li>Estimators: 1, 22 (Mejor número de estimadores)</li>
            <li>Class Weight: balanced</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.subheader("Frontera")
    st.markdown(r"Se redujo cada muestra $\mathbb{R}^d$ a un vector de 2 coordenadas (componentes principales).")

    st.image("frontera.png", width=500)  # ancho en píxeles

    st.subheader("Resultados del modelo")

    st.markdown(
    f"""
    <div class="presentation-box">
        <h3>Métricas de desempeño</h3>
        <p style="display:flex; gap:9px; align-items:center; white-space:nowrap; overflow-x:auto; padding-bottom:3px;">
            <span><strong>F1 Score:</strong> Training: {0.9954:.4f} — Testing: {0.9899:.4f}</span>
            <span><strong>Accuracy:</strong> Training: {0.9990:.4f} — Testing: {0.9978:.4f}</span>
            <span><strong>Recall:</strong> Training: {0.9952:.4f} — Testing: {0.9884:.4f}</span>
            <span><strong>Precision:</strong> Training: {0.9955:.4f} — Testing: {0.9915:.4f}</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with tabs[4]:
    st.markdown(
        """
        <div class="presentation-box">
        <h2>Gradient Boosting</h2>
        <p>Se utilizó un modelo de Gradient Boosting con <code>learning_rate=0.1</code>. 
           Este algoritmo combina múltiples clasificadores débiles (árboles de decisión) 
           de manera secuencial, donde cada nuevo árbol corrige los errores del anterior. 
           Es potente para capturar relaciones no lineales en los datos.</p>
        <ul>
            <li>Modelo: Gradient Boosting</li>
            <li>Learning Rate: 0.1</li>
            <li>Base Learners: Árboles de decisión</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
    """
    <div class="presentation-box">
        <h3>Resultados</h3>
        <p style="display:flex; gap:9px; align-items:center; white-space:nowrap; overflow-x:auto; padding-bottom:3px;">
            <span><strong>F1 Score:</strong> Training: 0.9986 — Testing: 0.9990</span>
            <span><strong>Accuracy:</strong> Training: 0.9997 — Testing: 0.9997</span>
            <span><strong>Recall:</strong> Training: 0.9996 — Testing: 0.9993</span>
            <span><strong>Precision:</strong> Training: 0.9977 — Testing: 0.9987</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with tabs[5]:
    st.markdown(
    """
    <div class="presentation-box">
        <h2>K-Nearest Neighbors (KNN)</h2>
        <p>Se utilizó un modelo KNN con <code>n_neighbors=10</code>. KNN clasifica según la mayoría de vecinos más cercanos y es sensible a la escala de las características.</p>
        <ul>
            <li>Modelo: K-Nearest Neighbors (KNN)</li>
            <li>n_neighbors: 10, 1 (Mejor Valor)</li>
            <li>Recomendación: escalar características</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
    )

    st.subheader("One-Hot Encoding")
    st.write("pd.get_dummies convierte las columnas categóricas listadas en variables binarias (una columna 0/1 por cada categoría), produciendo df_one_hot_encoding listo para modelos que no aceptan texto; luego x_dummies obtiene las características eliminando la columna objetivo COMBUSTIBLE, y y_label toma la serie con las etiquetas numéricas ya codificadas (Label Encoding) que se usará como variable dependiente. Ten en cuenta que para asegurar columnas idénticas en train/test y manejar categorías desconocidas en producción es preferible usar OneHotEncoder de sklearn (con handle_unknown=\"ignore\") o un ColumnTransformer.")
    df_dummies = pd.read_csv("x_dummies.csv")
    st.dataframe(df_dummies.head(4))

    st.markdown(
    """
    <div class="presentation-box">
        <h3>Resultados</h3>
        <p style="display:flex; gap:9px; align-items:center; white-space:nowrap; overflow-x:auto; padding-bottom:3px;">
            <span><strong>F1 Score:</strong> Training: 0.9988 — Testing: 0.9991</span>
            <span><strong>Accuracy:</strong> Training: 0.9998 — Testing: 0.9998</span>
            <span><strong>Recall:</strong> Training: 0.9999 — Testing: 0.9997</span>
            <span><strong>Precision:</strong> Training: 0.9978 — Testing: 0.9985</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
    )
    
with tabs[6]:
    st.markdown(
    """
    <div class="presentation-box">
        <h2>Regresión Logística</h2>
        <p>Se utilizó un modelo de Regresión Logística con <code>multi_class='multinomial'</code> y <code>solver='lbfgs'</code>. 
           Este algoritmo estima probabilidades para cada clase y es ampliamente usado en clasificación multiclase.</p>
        <ul>
            <li>Modelo: Regresión Logística</li>
            <li>multi_class: multinomial</li>
            <li>solver: lbfgs</li>
            <li>max_iter: 10000</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
    st.markdown(
    """
    <div class="presentation-box">
        <h3>Resultados</h3>
        <p style="display:flex; gap:9px; align-items:center; white-space:nowrap; overflow-x:auto; padding-bottom:3px;">
            <span><strong>F1 Score:</strong> Training: 0.8207 — Testing: 0.8126</span>
            <span><strong>Accuracy:</strong> Training: 0.9426 — Testing: 0.9425</span>
            <span><strong>Recall:</strong> Training: 0.7683 — Testing: 0.7648</span>
            <span><strong>Precision:</strong> Training: 0.8931 — Testing: 0.8771</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
    )

with tabs[7]:
    st.markdown(
        """
    <div class="presentation-box">
        <h2>Conclusiones</h2>
        <p>En el desarrollo de los modelos de clasificación para predecir el tipo de combustible ("COMBUSTIBLE": ELÉCTRICO, HÍBRIDO GASOLINA, HÍBRIDO DIESEL), se evaluaron Decision Tree, Random Forest, Gradient Boosting, KNN, MLPClassifier (Red Neuronal) y Logistic Regression. Todos los modelos, excepto Logistic Regression, exhibieron excelentes resultados tanto en el conjunto de entrenamiento como en el de prueba, con F1-scores superiores al 99% en la mayoría de los casos. Este desempeño excepcional se atribuye a una selección cuidadosa de variables predictoras, que incluyó el análisis de significancia mediante chi-cuadrado para categóricas y ANOVA para numéricas, eliminando redundancias (como CLASIFICACION y CARROCERIA por altas correlaciones en Cramér's V) y manejando la alta cardinalidad (agrupando categorías raras en "OTROS"). Estas variables, en conjunto, capturan patrones robustos, permitiendo una predicción precisa incluso con desbalance de clases, lo que demuestra la efectividad del preprocesamiento y la elección de modelos no lineales.

El único modelo que no alcanzó un buen desempeño, con F1-scores inferiores (tanto en entrenamiento como en prueba), fue Logistic Regression, a pesar de usar variables categóricas transformadas en dummies (One-Hot Encoding). La razón probable es que, como modelo lineal, asume relaciones lineales entre las features y la variable objetivo, lo que no se adapta bien a patrones no lineales complejos en los datos (ej. interacciones entre categóricas de alta cardinalidad y numéricas como CILINDRAJE=0 para eléctricos). Además, la alta dimensionalidad generada por dummies puede causar problemas de convergencia o multicolinealidad residual, a diferencia de modelos basados en árboles (Decision Tree, Random Forest, Gradient Boosting) o KNN, que manejan estas complejidades de manera más flexible. El F1-score, utilizado como método de evaluación principal, es la media armónica de precisión y recall, diferenciándose de métricas como accuracy (que ignora desbalance) o precisión/recall individuales al equilibrar falsos positivos y negativos. Elegimos `average='macro'` para promediar el F1 por clase sin ponderar por soporte, tratando todas las clases por igual y mitigando el desbalance, aunque también evaluamos accuracy, precisión y recall, obteniendo resultados excelentes en todas. Dado el alto rendimiento general con leves diferencias entre los modelos (excepto Logistic Regression), no nos decantamos por uno en particular, cumpliendo el objetivo de crear un sistema robusto para clasificar combustibles de vehículos en Colombia.
<p>
    </div>
    """,
        unsafe_allow_html=True,
    )


import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

# -------------------------
# Configuración de la página
# -------------------------
st.set_page_config(
    page_title="mpacolombia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CSS minimalista
# -------------------------
st.markdown("""
<style>
/* Fondo general oscuro */
.report-background { background-color: #0b0f0c; color: #0b0f0c; }

/* Encabezado principal (más pequeño, color verde claro) */
.main-header { 
    background: #0d1f17;        /* verde oscuro sólido */
    padding: 1rem; 
    border-radius: 0px; 
    color: #a8f5c0;             /* verde claro elegante */
    text-align: center; 
    margin-bottom: 1.5rem; 
    border: 1px solid #1e5a3a;  /* borde verde profundo */
    box-shadow: 0 3px 8px rgba(0,0,0,0.5); /* sombra sutil */
}
.main-header h1 {
    font-size: 1.9rem;    
    font-weight: 600;     
    margin: 0;
    letter-spacing: 0.5px;
}

/* Cajas de presentación e insights */
.presentation-box, .insight-box { 
    background: #1a1f1d;   /* gris oscuro */
    padding: 1rem; 
    border-radius: 0px; 
    margin: 1rem 0; 
    color: #ccffcc;        /* texto verde claro */
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
""", unsafe_allow_html=True)

# -------------------------
# CARGA DE DATOS
# -------------------------
COLUMNA_FECHA = 'ds'
COLUMNA_VALOR = 'y'


df_datos = pd.read_csv("mensual.csv")
df_datosp = pd.read_csv("Numero_de_Veh_culos_El_ctricos_-_Hibridos_20250804.csv")
df_datosc = pd.read_csv("cleaned_dataframe.csv")

# Procesamiento
try:
    df_ejemplo = df_datos[[COLUMNA_FECHA, COLUMNA_VALOR]].copy()
    df_ejemplo.columns = ['fecha', 'valor']
    df_ejemplo['fecha'] = pd.to_datetime(df_ejemplo['fecha'], errors='coerce')
    df_ejemplo = df_ejemplo.dropna(subset=['fecha', 'valor']).sort_values('fecha').reset_index(drop=True)
except Exception as e:
    st.error(f"Error al procesar los datos: {e}")
    st.stop()

if df_ejemplo.empty:
    st.error("El dataset está vacío después del preprocesamiento.")
    st.stop()

# -------------------------
# HEADER
# -------------------------
st.markdown("""
<div class="main-header">
    <h1>Modelos predictivos de la adopción de vehículos eléctricos e híbridos en Colombia</h1>
</div>
""", unsafe_allow_html=True)

# -------------------------
# NAVEGACIÓN
# -------------------------
sections = ["PRESENTACIÓN", "FORECAST", "ARBOLES DE DECISION"]
tabs = st.tabs(sections)

# --- PRESENTACIÓN ---
with tabs[0]:
    st.markdown(f"""
    <div class="presentation-box">
        <h2>Objetivo</h2>
        <p>Se busca analizar la adopcion de vehiculos electrificados en los diferentes departamentos de colombia y proponer estrategias para su fomento mediante el uso de modelos de series temporales y modelos de clasificacion.</p>
        <p>Analisis previo usando los registros mensuales:<p>
        <ul>
            <li><strong>Registros procesados:</strong> {len(df_datosp):,}</li>
            <li><strong>Período:</strong> {df_ejemplo['fecha'].min().strftime('%Y-%m-%d')} a {df_ejemplo['fecha'].max().strftime('%Y-%m-%d')}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # --- Vista previa del dataset completo ---
    st.subheader("Vista previa del dataset original")
    st.dataframe(df_datosp.head(4))  # muestra primeras filas

    st.markdown("""
    <div class="presentation-box">
        <p>Decidimos prescindir de algunas columnas y filas que contenían datos nulos</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Vista previa del dataset limpio ---
    st.subheader("Vista previa del dataset limpio")
    st.dataframe(df_datosc.head(4))  # muestra primeras filas

# --- FORECAST ---
with tabs[1]:
    st.markdown(f"""
    <div class="presentation-box">
        <h2>Forecast con Prophet</h2>
        <p>Usando el conteo mensual extraido del dataset original se hizo una pequeña prediccion:<p>
    </div>
    """, unsafe_allow_html=True)

    df_forecast = None
    try:
        df_forecast = pd.read_csv("forecast_24m.csv")
    except FileNotFoundError:
        uploaded_f = st.sidebar.file_uploader("Sube forecast_24m.csv", type=["csv"], key="forecast")
        if uploaded_f is not None:
            df_forecast = pd.read_csv(uploaded_f)

    if df_forecast is None:
        st.info("No hay archivo de forecast cargado.")
    else:
        # Normalizar columnas
        cols_lower = {c.lower(): c for c in df_forecast.columns}
        required_lower = {'ds', 'yhat', 'yhat_lower', 'yhat_upper'}
        if not required_lower.issubset(set(cols_lower.keys())):
            st.warning(f"El forecast debe contener {required_lower}. Encontradas: {list(df_forecast.columns)}")
        else:
            df_forecast = df_forecast.rename(columns={cols_lower['ds']: 'ds', cols_lower['yhat']: 'yhat',
                                                      cols_lower['yhat_lower']: 'yhat_lower', cols_lower['yhat_upper']: 'yhat_upper'})
            df_forecast['ds'] = pd.to_datetime(df_forecast['ds'], errors='coerce')
            df_forecast = df_forecast.dropna(subset=['ds', 'yhat']).sort_values('ds')

            ultima_fecha_historica = df_ejemplo['fecha'].max()

            # Banda de confianza
            band = alt.Chart(df_forecast).mark_area(
                opacity=0.25, 
                color="#00cc66"   # verde oscuro translúcido
            ).encode(
                x='ds:T',
                y='yhat_lower:Q',
                y2='yhat_upper:Q'
            )

            # Línea de forecast (verde brillante)
            line = alt.Chart(df_forecast).mark_line(
                color="#00ff99", 
                strokeWidth=2
            ).encode(
                x='ds:T', 
                y='yhat:Q'
            )

            # Puntos del forecast (verde brillante)
            points = alt.Chart(df_forecast).mark_point(
                color="#00ff99", 
                size=40
            ).encode(
                x='ds:T', 
                y='yhat:Q'
            )

            # Línea divisoria histórico vs forecast
            rule = alt.Chart(pd.DataFrame({'fecha': [ultima_fecha_historica]})).mark_rule(
                color='orange', 
                strokeDash=[5,5], 
                strokeWidth=2
            ).encode(x='fecha:T')

            # Combinar todo
            chart = alt.layer(band, line, points, rule).properties(
                title="Forecast con intervalo de confianza",
                height=450,
                background="#0a0f0a"  # fondo oscuro igual al resto
            ).configure_axis(
                labelColor="white",
                titleColor="white"
            ).configure_title(
                color="white"
            )

            st.altair_chart(chart, use_container_width=True)

                    # Example dataimport streamlit as st
            import os
            import glob
            import json
            import unicodedata

            import pandas as pd
            import numpy as np

            import streamlit as st
            import pydeck as pdk

            # ---------------- Config ----------------
            GEOJSON_PATH = "colombia_departments.json"
            FORECAST_FOLDER = "forecasts"   # carpeta con archivos DEPARTAMENTO_forecast.csv
            PREDICTED_YEAR_THRESHOLD = 2023
            # ----------------------------------------

            st.title("Mapa predictivo por departamentos (basado en yhat)")
            st.markdown("""
            <div class="presentation-box">
                <h2>Mapa predictivo por departamentos</h2>
                <p>Se colorea cada departamento según <code>yhat</code> (última fila del año seleccionado)</p>
            </div>
            """, unsafe_allow_html=True)

            def normalize_name(s: str):
                if s is None:
                    return ""
                s = s.upper().strip()
                s = unicodedata.normalize("NFD", s)
                s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
                s = "".join(ch for ch in s if ch.isalnum())
                return s

            # paleta original (interpolación linear)
            def count_to_rgb(c, vmin=0, vmax=200):
                if c is None:
                    c = 0
                # normaliza entre 0 y 1
                denom = (vmax - vmin) if (vmax is not None and vmax > vmin) else 1.0
                t = max(0.0, min(1.0, (c - vmin) / denom))
                dark_green = (0, 204, 102)   # #00cc66
                bright_green = (0, 255, 153) # #00ff99
                r = int(dark_green[0] + t * (bright_green[0] - dark_green[0]))
                g = int(dark_green[1] + t * (bright_green[1] - dark_green[1]))
                b = int(dark_green[2] + t * (bright_green[2] - dark_green[2]))
                a = 180
                return [r, g, b, a]

            # --- Cargar geojson base ---
            with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
                geojson = json.load(f)

            # --- Leer forecasts ---
            files = glob.glob(os.path.join(FORECAST_FOLDER, "*_forecast.csv"))
            if not files:
                st.error(f"No se encontraron archivos '*_forecast.csv' en '{FORECAST_FOLDER}'")
                st.stop()

            dept_dfs = {}
            years = []
            for fp in files:
                fname = os.path.basename(fp)
                if not fname.upper().endswith("_FORECAST.CSV"):
                    continue
                dept_raw = fname[:-len("_forecast.csv")]
                dept_norm = normalize_name(dept_raw)
                try:
                    df = pd.read_csv(fp)
                    # normalizar columnas a minúsculas
                    df.columns = [c.lower() for c in df.columns]
                    if "ds" not in df.columns:
                        st.warning(f"{fname} no tiene columna 'ds' — se omite.")
                        continue
                    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
                    df = df.dropna(subset=["ds"])
                    if df.empty:
                        continue
                    dept_dfs[dept_norm] = df.reset_index(drop=True)
                    years.extend(df["ds"].dt.year.dropna().astype(int).tolist())
                except Exception as e:
                    st.warning(f"Error leyendo {fname}: {e}")
                    continue

            if not dept_dfs:
                st.error("No se pudo leer ningún forecast válido.")
                st.stop()

            min_year = int(min(years))
            max_year = int(max(years))

            # Slider arriba
            selected_year = st.slider("Selecciona año", min_year, max_year, value=max_year, format="%d")
            is_predicted = selected_year >= PREDICTED_YEAR_THRESHOLD
            if is_predicted:
                st.markdown(f"**Año seleccionado:** {selected_year} — **(predecido)**")
            else:
                st.markdown(f"**Año seleccionado:** {selected_year}")

            # --- Extraer solo yhat (entero) para el año seleccionado ---
            dept_yhat = {}   # key: dept_norm or code -> int or None
            yhat_values = []
            for dept_norm, df in dept_dfs.items():
                df_year = df[df["ds"].dt.year == int(selected_year)]
                if not df_year.empty:
                    row = df_year.iloc[-1]
                else:
                    df_before = df[df["ds"].dt.year < int(selected_year)]
                    row = df_before.iloc[-1] if not df_before.empty else None
                if row is None:
                    dept_yhat[dept_norm] = None
                else:
                    yhat = None
                    for cand in ("yhat","yhat","yhat"):  # column already lowercased
                        if cand in row.index:
                            yhat = row[cand]
                            break
                    try:
                        if pd.isna(yhat):
                            dept_yhat[dept_norm] = None
                        else:
                            yi = int(round(float(yhat)))
                            dept_yhat[dept_norm] = yi
                            if yi != 0:
                                yhat_values.append(yi)
                    except Exception:
                        dept_yhat[dept_norm] = None

            # vmin/vmax para la paleta (basado en yhat_values)
            if yhat_values:
                vmin = float(np.percentile(yhat_values, 5))
                vmax = float(np.percentile(yhat_values, 95))
                if abs(vmax - vmin) < 1e-6:
                    vmin -= 1
                    vmax += 1
            else:
                vmin, vmax = 0.0, 1.0

            # --- Añade propiedades count y color por feature (reemplaza counts) ---
            for feat in geojson["features"]:
                props = feat.get("properties", {})
                # intentamos código DPTO primero
                code = props.get("DPTO") or props.get("DPTO_") or props.get("DPTO_COD") or props.get("DPTO_CODE")
                count_val = None
                if code is not None:
                    # code puede ser numérico o string; buscamos como string y sin normalizar
                    cstr = str(code)
                    if cstr in dept_yhat:
                        count_val = dept_yhat[cstr]
                if count_val is None:
                    # intentar por nombre normalizado
                    nombre = props.get("NOMBRE_DPT") or props.get("NOMBRE") or ""
                    nombre_norm = normalize_name(nombre)
                    count_val = dept_yhat.get(nombre_norm)
                # si sigue None o es 0 -> pintar gris apagado
                if count_val is None or count_val == 0:
                    color = [200, 200, 200, 120]
                    count_int = 0 if count_val is None else int(count_val)
                else:
                    color = count_to_rgb(count_val, vmin=vmin, vmax=vmax)
                    count_int = int(count_val)
                # set properties
                feat["properties"]["count"] = count_int
                feat["properties"]["color"] = color
                # guardamos yhat para tooltip
                feat["properties"]["yhat_display"] = f"{count_int}{' (predecido)' if is_predicted and (count_val is not None) else ''}" if (count_val is not None) else None

            # Capa GeoJson — usamos propiedades.color para el fill
            polygon_layer = pdk.Layer(
                "GeoJsonLayer",
                geojson,
                stroked=True,
                filled=True,
                extruded=False,
                pickable=True,
                auto_highlight=True,                 # <-- resalta al hover
                get_fill_color="properties.color",
                get_line_color=[80, 80, 80],
            )

            # Centra vista en Colombia
            view_state = pdk.ViewState(latitude=4.5, longitude=-74.0, zoom=5)

            tooltip = {
                "html": "<b>{NOMBRE_DPT}</b><br/>Código: {DPTO}<br/>yhat: {yhat_display}",
                "style": {"backgroundColor":"white","color":"black","fontSize":"12px"}
            }

            deck = pdk.Deck(layers=[polygon_layer], initial_view_state=view_state, tooltip=tooltip)
            st.pydeck_chart(deck)

            # Mostrar tabla con valores usados
            rows = []
            for feat in geojson["features"]:
                props = feat.get("properties", {})
                rows.append({
                    "departamento": props.get("NOMBRE_DPT") or props.get("NOMBRE") or "",
                    "DPTO": props.get("DPTO") or props.get("DPTO_") or props.get("DPTO_COD") or "",
                    "yhat": props.get("count")
                })

            st.markdown("### Valores yhat usados (por departamento)")
            st.dataframe(pd.DataFrame(rows).sort_values("departamento"), height=300)

with tabs[2]:
    st.markdown(f"""
    <div class="presentation-box">
        <h2>Arboles de Decisión</h2>
        <p>Se utilizo un modelo basado en arboles de decision a varias profundidades debido a su transparencia y facilidad de interpretacion
                mediante un gridsearch optimizamos los hiperparametros y se obtuvo un modelo robusto.</p>
    </div>
    """, unsafe_allow_html=True)

    df_chi2 = pd.read_csv("chi2.csv", dtype=str)


    st.subheader("Valores de Chi-cuadrado")
    st.dataframe(df_chi2.head(4))
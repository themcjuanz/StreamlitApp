import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import streamlit.components.v1 as components

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

    st.set_page_config(layout="wide")

    st.subheader("Analisis descriptivo de la base de datos")

    # Incrustar Power BI público
    iframe_code = """
    <iframe title="elec" 
        width="100%" height="600" 
        src="https://app.powerbi.com/view?r=eyJrIjoiZTFhNDExOWYtOGVlNC00MWU2LThlZmItMGQ5NjFmOTE0Yjc0IiwidCI6IjU3N2ZjMWQ4LTA5MjItNDU4ZS04N2JmLWVjNGY0NTVlYjYwMCIsImMiOjR9" 
        frameborder="0" allowFullScreen="true">
    </iframe>
    """

    components.html(iframe_code, height=650, scrolling=True)

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
            from datetime import datetime

            import pandas as pd
            import numpy as np

            import streamlit as st
            import pydeck as pdk

            # ---------------- Config ----------------
            GEOJSON_PATH = "colombia_departments.json"
            FORECAST_FOLDER = "forecasts"   # carpeta con archivos DEPARTAMENTO_forecast.csv
            PREDICTED_YEAR_THRESHOLD = 2023
            ZERO_TOL_FACTOR = 0.03   # tol relativo (3% del rango v95-v5); se usa max(MIN_ZERO_TOL,...)
            MIN_ZERO_TOL = 1         # mínimo absoluto
            # ----------------------------------------

            st.markdown("""
            <div class="presentation-box">
                <h2>Mapa predictivo por departamentos</h2>
                <p>Selecciona un mes (si no hay datos para ese mes se usa la última fecha anterior disponible).</p>
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
            def count_to_rgb(c, vmin=0, vmax=200, alpha=200):
                denom = (vmax - vmin) if (vmax is not None and vmax > vmin) else 1.0
                t = max(0.0, min(1.0, (c - vmin) / denom))
                dark_green = (0, 204, 102)   # #00cc66
                bright_green = (100, 255, 180) # #00ff99
                r = int(dark_green[0] + t * (bright_green[0] - dark_green[0]))
                g = int(dark_green[1] + t * (bright_green[1] - dark_green[1]))
                b = int(dark_green[2] + t * (bright_green[2] - dark_green[2]))
                a = int(alpha)
                return [r, g, b, a]

            # --- Cargar geojson ---
            with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
                geojson = json.load(f)

            # --- Leer forecasts ---
            files = glob.glob(os.path.join(FORECAST_FOLDER, "*_forecast.csv"))
            if not files:
                st.error(f"No se encontraron archivos '*_forecast.csv' en '{FORECAST_FOLDER}'")
                st.stop()

            dept_dfs = {}
            all_months = set()
            years = []
            for fp in files:
                fname = os.path.basename(fp)
                if not fname.upper().endswith("_FORECAST.CSV"):
                    continue
                dept_raw = fname[:-len("_forecast.csv")]
                dept_norm = normalize_name(dept_raw)
                try:
                    df = pd.read_csv(fp)
                    # normalizar y ordenar por fecha (registros mensuales)
                    df.columns = [c.lower() for c in df.columns]
                    if "ds" not in df.columns:
                        st.warning(f"{fname} no tiene columna 'ds' — se omite.")
                        continue
                    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
                    df = df.dropna(subset=["ds"])
                    if df.empty:
                        continue
                    df = df.sort_values("ds").reset_index(drop=True)
                    dept_dfs[dept_norm] = df
                    years.extend(df["ds"].dt.year.dropna().astype(int).tolist())
                    # recolectar meses disponibles en formato "YYYY-MM"
                    months = df["ds"].dt.to_period("M").astype(str).unique().tolist()
                    all_months.update(months)
                except Exception as e:
                    st.warning(f"Error leyendo {fname}: {e}")
                    continue

            if not dept_dfs:
                st.error("No se pudo leer ningún forecast válido.")
                st.stop()

            # construir lista ordenada de meses (strings "YYYY-MM")
            months_sorted = sorted(list(all_months), key=lambda s: datetime.strptime(s, "%Y-%m"))
            if not months_sorted:
                st.error("No se encontraron meses en los archivos de forecasts.")
                st.stop()

            # Slider de meses (select_slider para mostrar un slider con las opciones)
            selected_month = st.select_slider("Selecciona mes (YYYY-MM)", options=months_sorted, value=months_sorted[-1])
            # parsear a year,month y fecha límite = último día del mes
            selected_period = pd.Period(selected_month, freq="M")
            selected_year = selected_period.year
            selected_month_int = selected_period.month
            # predicted if year >= threshold
            is_predicted = selected_year >= PREDICTED_YEAR_THRESHOLD
            if is_predicted:
                st.markdown(f"**Mes seleccionado:** {selected_month} — **(predecido)**")
            else:
                st.markdown(f"**Mes seleccionado:** {selected_month}")

            # --- Extraer solo yhat (entero) y la fecha asociada para el mes seleccionado (fallback: última fila <= mes) ---
            dept_yhat = {}        # key: dept_norm or code -> int or None
            dept_yhat_date = {}   # key -> "YYYY-MM-DD" string or None
            yhat_values = []

            for dept_norm, df in dept_dfs.items():
                # buscamos la última fila con ds <= último día del mes seleccionado
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

            # vmin/vmax para la paleta (basado en yhat_values)
            if yhat_values:
                vmin = float(np.percentile(yhat_values, 5))
                vmax = float(np.percentile(yhat_values, 95))
                if abs(vmax - vmin) < 1e-6:
                    vmin -= 1
                    vmax += 1
            else:
                vmin, vmax = 0.0, 1.0

            # calcular umbral "cerca a 0"
            zero_tol = max(MIN_ZERO_TOL, ZERO_TOL_FACTOR * (abs(vmax - vmin)))

            # --- Añade propiedades color y line_color por feature (transparencia total si corresponde) ---
            for feat in geojson["features"]:
                props = feat.get("properties", {})
                # intentar por código DPTO
                code = props.get("DPTO") or props.get("DPTO_") or props.get("DPTO_COD") or props.get("DPTO_CODE")
                count_val = None
                count_date = None
                if code is not None:
                    cstr = str(code)
                    if cstr in dept_yhat:
                        count_val = dept_yhat[cstr]
                        count_date = dept_yhat_date.get(cstr)
                if count_val is None:
                    nombre = props.get("NOMBRE_DPT") or props.get("NOMBRE") or ""
                    nombre_norm = normalize_name(nombre)
                    count_val = dept_yhat.get(nombre_norm)
                    count_date = dept_yhat_date.get(nombre_norm)

                # Si sin dato, 0 o cerca a 0 => TRANSPARENTE TOTAL (fill y borde)
                transparent = False
                if count_val is None:
                    transparent = True
                else:
                    try:
                        if count_val <= 0 or abs(count_val) <= zero_tol:   # <- nota el <= 0
                            transparent = True
                    except Exception:
                        transparent = True

                if transparent:
                    color = [0, 0, 0, 0]        # fill totalmente transparente
                    line_color = [0, 0, 0, 0]   # borde totalmente transparente
                    count_prop = None
                    display_text = None
                else:
                    color = count_to_rgb(count_val, vmin=vmin, vmax=vmax, alpha=200)
                    line_color = [80, 80, 80, 255]
                    count_prop = int(count_val)
                    # construir texto para tooltip con fecha asociada
                    if count_date:
                        display_text = f"{count_prop} (fecha: {count_date})"
                    else:
                        display_text = f"{count_prop}"

                    if is_predicted:
                        display_text = display_text + " (predecido)"

                feat["properties"]["count"] = count_prop if count_prop is not None else None
                feat["properties"]["color"] = color
                feat["properties"]["line_color"] = line_color
                feat["properties"]["yhat_display"] = display_text

            # Capa GeoJson — usamos propiedades.color y propiedades.line_color
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

            # Vista
            view_state = pdk.ViewState(latitude=4.5, longitude=-74.0, zoom=5)

            tooltip = {
                "html": "<b>{NOMBRE_DPT}</b><br/>Código: {DPTO}<br/>yhat: {yhat_display}",
                "style": {"backgroundColor":"white","color":"black","fontSize":"12px"}
            }

            deck = pdk.Deck(layers=[polygon_layer], initial_view_state=view_state, tooltip=tooltip)
            st.pydeck_chart(deck)

            # Tabla con valores usados
            rows = []
            for feat in geojson["features"]:
                props = feat.get("properties", {})
                rows.append({
                    "departamento": props.get("NOMBRE_DPT") or props.get("NOMBRE") or "",
                    "DPTO": props.get("DPTO") or props.get("DPTO_") or props.get("DPTO_COD") or "",
                    "yhat": props.get("count"),
                    "yhat_date": props.get("yhat_display")
                })

            st.markdown("### Valores yhat usados (por departamento)")
            rows = []
            for feat in geojson["features"]:
                props = feat.get("properties", {})
                yhat_val = props.get("count")
                rows.append({
                    "departamento": props.get("NOMBRE_DPT") or props.get("NOMBRE") or "",
                    "DPTO": props.get("DPTO") or props.get("DPTO_") or props.get("DPTO_COD") or "",
                    # si hay None lo dejamos como None por ahora; lo convertiremos después a 0
                    "yhat": yhat_val,
                    "yhat_date": props.get("yhat_display")
                })

            df_rows = pd.DataFrame(rows)

            # Convertir None/NaN a 0 en la columna yhat y asegurarnos tipo entero
            df_rows["yhat"] = pd.to_numeric(df_rows["yhat"], errors="coerce").fillna(0).astype(int)

            # Ordenar de mayor a menor por yhat
            df_rows = df_rows.sort_values("yhat", ascending=False).reset_index(drop=True)
            st.dataframe(df_rows, height=300)



with tabs[2]:
    st.markdown(f"""
    <div class="presentation-box">
        <h2>Arboles de Decisión</h2>
        <p>Se utilizo un modelo basado en arboles de decision a varias profundidades debido a su transparencia y facilidad de interpretacion
                mediante un gridsearch optimizamos los hiperparametros y se obtuvo un modelo robusto.</p>
        <ul>
            <li>Modelo: Arboles de Decisión</li>
            <li>Hiperparametros: Optimizados mediante GridSearch</li>
            <li>Profundidad: 10, 17 (Mejor profundidad, aunque poca diferencia)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    df_chi2 = pd.read_csv("chi2.csv", dtype=str)
    df_anova = pd.read_csv("anova_results.csv", dtype=str)
    df_cramers = pd.read_csv("cramers_results.csv", dtype=str)

    st.subheader("Análisis de significancia en variables predictoras categoricas chi²")

    st.dataframe(df_chi2.head(4))

    st.subheader("Análisis de significancia en variables predictoras numéricas usando ANOVA")

    st.dataframe(df_anova.head(4))

    st.subheader("Análisis de correlación entre variables categóricas usando Cramér's V")

    st.dataframe(df_cramers.head(10))
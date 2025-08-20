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
sections = ["PRESENTACIÓN", "FORECAST"]
tabs = st.tabs(sections)

# --- PRESENTACIÓN ---
with tabs[0]:
    st.markdown(f"""
    <div class="presentation-box">
        <h2>Objetivo</h2>
        <p>Se busca analizar la adopcion de vehiculos electrificados en los diferentes departamentos de colombia y proponer estrategias para su fomento mediante el uso de modelos de series temporales y modelos de clasificacion..</p>
        <p>Analisis previo usando los registros mensuales:<p>
        <ul>
            <li><strong>Registros procesados:</strong> {len(df_ejemplo):,}</li>
            <li><strong>Período:</strong> {df_ejemplo['fecha'].min().strftime('%Y-%m-%d')} a {df_ejemplo['fecha'].max().strftime('%Y-%m-%d')}</li>
            <li><strong>Frecuencia:</strong> Mensual</li>
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
    st.markdown('<div class="section-header"><h3>Forecast con Prophet</h3><p>Generación de pronósticos para los próximos 24 meses usando un '
    'conteo mensual extraido del dataset original</p></div>', unsafe_allow_html=True)

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
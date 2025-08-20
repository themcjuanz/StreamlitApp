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
.report-background { background-color: #0b0f14; color: #e6eef6; }
.main-header { background: linear-gradient(90deg, #2b3150 0%, #1f2438 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1rem; }
.presentation-box, .insight-box { background: #111827; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; color: #e6eef6; }
.section-header { background: #1f2937; padding: 1rem; border-radius: 8px; color: #e6eef6; text-align: center; margin: 1rem 0; }
[data-testid="stAppViewContainer"] > div { background: #071016; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# CARGA DE DATOS
# -------------------------
COLUMNA_FECHA = 'ds'
COLUMNA_VALOR = 'y'


df_datos = pd.read_csv("mensual.csv")
df_datosp = pd.read_csv("cleaned_dataframe.csv")

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
    st.dataframe(df_datosp.head(10))  # muestra primeras filas

    # --- Valores únicos de columnas clave ---
    st.subheader("Valores únicos en variables categóricas")

    if "DEPARTAMENTO" in df_datosp.columns:
        st.markdown("**Departamentos registrados:**")
        st.write(sorted(df_datosp["DEPARTAMENTO"].dropna().unique().tolist()))

    if "CLASE" in df_datosp.columns:
        st.markdown("**Clases de vehículos:**")
        st.write(sorted(df_datosp["CLASE"].dropna().unique().tolist()))

    if "MARCA" in df_datosp.columns:
        st.markdown("**Marcas de vehículos:**")
        st.write(sorted(df_datosp["MARCA"].dropna().unique().tolist()))

# --- FORECAST ---
with tabs[1]:
    st.markdown('<div class="section-header"><h3>FORECAST</h3></div>', unsafe_allow_html=True)

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
            band = alt.Chart(df_forecast).mark_area(opacity=0.25).encode(
                x='ds:T',
                y='yhat_lower:Q',
                y2='yhat_upper:Q'
            )
            line = alt.Chart(df_forecast).mark_line(color='cyan').encode(x='ds:T', y='yhat:Q')
            points = alt.Chart(df_forecast).mark_point(color='cyan').encode(x='ds:T', y='yhat:Q')

            # Línea que separa histórico y predicción
            rule = alt.Chart(pd.DataFrame({'fecha': [ultima_fecha_historica]})).mark_rule(color='orange', strokeDash=[5,5]).encode(x='fecha:T')

            chart = alt.layer(band, line, points, rule).properties(
                title="Forecast con intervalo de confianza",
                height=450
            )

            st.altair_chart(chart, use_container_width=True)

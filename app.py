import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

# -------------------------
# Configuración de la página
# -------------------------
st.set_page_config(
    page_title="Modelos Predictivos de la Adopcion de Autos Electricos y Hibridos en Colombia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CSS (mantén tu estilo)
# -------------------------
st.markdown("""
<style>
/* Copia aquí todo tu CSS original si quieres mantenerlo idéntico */
.report-background { background-color: #0b0f14; color: #e6eef6; }
.main-header { background: linear-gradient(90deg, #2b3150 0%, #1f2438 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.6); }
.metric-container { background: linear-gradient(135deg, #0f1722 0%, #111827 100%); padding: 1.5rem; border-radius: 12px; color: #f7fafc; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.6); margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.03); }
.insight-box { background: linear-gradient(180deg, #090b0d 0%, #0f1722 100%); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #16a34a; margin: 1rem 0; box-shadow: 0 6px 24px rgba(2,6,23,0.7); color: #e6eef6; }
.presentation-box { background: linear-gradient(135deg, #0f1722 0%, #1a202c 100%); padding: 2rem; border-radius: 15px; border: 2px solid rgba(56, 178, 172, 0.3); margin: 1.5rem 0; box-shadow: 0 8px 32px rgba(0,0,0,0.4); color: #e6eef6; }
.section-header { background: linear-gradient(90deg, #0b1220 0%, #131824 100%); padding: 1rem; border-radius: 8px; color: #e6eef6; text-align: center; margin: 1.5rem 0 1rem 0; border: 1px solid rgba(255,255,255,0.02); }
.stDataFrame, .dataframe { background: transparent !important; color: #e6eef6 !important; }
[data-testid="stAppViewContainer"] > div { background: #071016; }
.sidebar-title { color: #38b2ac; font-size: 1.2rem; font-weight: bold; margin-bottom: 1rem; padding: 0.5rem; border-bottom: 2px solid rgba(56, 178, 172, 0.3); }
</style>
""", unsafe_allow_html=True)

# -------------------------
# CARGA DE DATOS
# -------------------------
COLUMNA_FECHA = 'ds'
COLUMNA_VALOR = 'y'

df_datos = None
try:
    df_datos = pd.read_csv("mensual.csv")
except FileNotFoundError:
    uploaded = st.sidebar.file_uploader("Sube mensual.csv (si no existe en el workspace)", type=["csv"])
    if uploaded is not None:
        df_datos = pd.read_csv(uploaded)
    else:
        st.sidebar.info("No se encontró 'mensual.csv'. Sube el archivo o colócalo en el workspace.")
        st.stop()
except Exception as e:
    st.error(f"Error al leer mensual.csv: {e}")
    st.stop()

# -------------------------
# PROCESAMIENTO DE DATOS
# -------------------------
try:
    df_ejemplo = df_datos[[COLUMNA_FECHA, COLUMNA_VALOR]].copy()
    df_ejemplo.columns = ['fecha', 'valor']
    df_ejemplo['fecha'] = pd.to_datetime(df_ejemplo['fecha'], errors='coerce')
    df_ejemplo = df_ejemplo.dropna(subset=['fecha', 'valor']).sort_values('fecha').reset_index(drop=True)
except KeyError as e:
    st.error(f"Error: no se encontró la columna {e}. Asegúrate de que las columnas sean '{COLUMNA_FECHA}' y '{COLUMNA_VALOR}'.")
    st.stop()
except Exception as e:
    st.error(f"Error al procesar los datos: {e}")
    st.stop()

if df_ejemplo.empty:
    st.error("El dataset está vacío después del preprocesamiento.")
    st.stop()

# Estadísticas defensivas
if len(df_ejemplo) >= 2:
    valor_actual = float(df_ejemplo['valor'].iloc[-1])
    valor_anterior = float(df_ejemplo['valor'].iloc[-2])
    cambio_absoluto = valor_actual - valor_anterior
    cambio_porcentual = (cambio_absoluto / valor_anterior) * 100 if valor_anterior != 0 else np.nan
else:
    valor_actual = float(df_ejemplo['valor'].iloc[-1])
    valor_anterior = np.nan
    cambio_absoluto = np.nan
    cambio_porcentual = np.nan

promedio_total = float(df_ejemplo['valor'].mean())
maximo = float(df_ejemplo['valor'].max())
minimo = float(df_ejemplo['valor'].min())
desviacion = float(df_ejemplo['valor'].std(ddof=0))

# -------------------------
# HEADER
# -------------------------
st.markdown("""
<div class="main-header">
    <h1>ANÁLISIS DE SERIE DE TIEMPO</h1>
    <h3>DASHBOARD DE VISUALIZACIÓN Y ANÁLISIS ESTADÍSTICO</h3>
</div>
""", unsafe_allow_html=True)

# -------------------------
# NAVEGACIÓN SUPERIOR: BARRA (USANDO TABS)
# -------------------------
sections = [
    "PRESENTACIÓN",
    "MÉTRICAS PRINCIPALES",
    "VISUALIZACIÓN PRINCIPAL",
    "ESTADÍSTICAS DESCRIPTIVAS",
    "ANÁLISIS DE TENDENCIA",
    "ANÁLISIS TEMPORAL",
    "INSIGHTS AUTOMÁTICOS",
    "PREDICCIONES (FORECAST)"
]

tabs = st.tabs(sections)

# ========================================
# Contenido por pestaña
# ========================================

# --- PRESENTACIÓN ---
with tabs[0]:
    st.markdown("""
    <div class="presentation-box">
        <h2>PRESENTACIÓN DEL PROYECTO</h2>
        <h3>OBJETIVO</h3>
        <p>Este dashboard ha sido desarrollado para realizar un análisis integral de series temporales, proporcionando herramientas de visualización y análisis estadístico.</p>
        <h3>DATOS ACTUALES</h3>
        <ul>
            <li><strong>Registros procesados:</strong> {}</li>
            <li><strong>Período de análisis:</strong> {} a {}</li>
            <li><strong>Frecuencia de datos:</strong> Mensual</li>
            <li><strong>Calidad de datos:</strong> {}% completos</li>
        </ul>
    </div>
    """.format(
        f"{len(df_ejemplo):,}",
        df_ejemplo['fecha'].min().strftime('%Y-%m-%d'),
        df_ejemplo['fecha'].max().strftime('%Y-%m-%d'),
        round((len(df_ejemplo) / len(df_datos)) * 100, 1)
    ), unsafe_allow_html=True)

# --- MÉTRICAS PRINCIPALES ---
with tabs[1]:
    st.markdown('<div class="section-header"><h2>MÉTRICAS CLAVE</h2></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-box">
        <h4>DATASET CARGADO</h4>
        <p><strong>Registros procesados:</strong> {len(df_ejemplo):,}</p>
        <p><strong>Período:</strong> {df_ejemplo['fecha'].min().strftime('%Y-%m-%d')} a {df_ejemplo['fecha'].max().strftime('%Y-%m-%d')}</p>
        <p><strong>Columna fecha:</strong> {COLUMNA_FECHA}</p>
        <p><strong>Columna valores:</strong> {COLUMNA_VALOR}</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-container"><h3>VALOR ACTUAL</h3><h2>{valor_actual:.2f}</h2><p>Último registro</p></div>', unsafe_allow_html=True)
    with col2:
        tendencia_class = "trend-positive" if (not np.isnan(cambio_absoluto) and cambio_absoluto > 0) else "trend-negative"
        cambio_text = f"{cambio_absoluto:+.2f}" if not np.isnan(cambio_absoluto) else "N/A"
        cambio_pct_text = f"{cambio_porcentual:+.1f}%" if not np.isnan(cambio_porcentual) else "N/A"
        st.markdown(f'<div class="metric-container"><h3>CAMBIO PERÍODO</h3><h2 class="{tendencia_class}">{cambio_text}</h2><p>{cambio_pct_text}</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-container"><h3>PROMEDIO</h3><h2>{promedio_total:.2f}</h2><p>Serie completa</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-container"><h3>VOLATILIDAD</h3><h2>{desviacion:.2f}</h2><p>Desv. estándar</p></div>', unsafe_allow_html=True)

    with st.expander("VISTA PREVIA DE LOS DATOS"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**PRIMEROS 5 REGISTROS:**")
            st.dataframe(df_ejemplo.head())
        with c2:
            st.write("**ÚLTIMOS 5 REGISTROS:**")
            st.dataframe(df_ejemplo.tail())

# --- VISUALIZACIÓN PRINCIPAL ---
with tabs[2]:
    st.markdown('<div class="section-header"><h2>VISUALIZACIÓN PRINCIPAL</h2></div>', unsafe_allow_html=True)
    base = alt.Chart(df_ejemplo).encode(
        x=alt.X('fecha:T', title='FECHA'),
        y=alt.Y('valor:Q', title='VALOR'),
        tooltip=[alt.Tooltip('fecha:T', title='Fecha'), alt.Tooltip('valor:Q', title='Valor')]
    )
    line = base.mark_line(point=True).properties(height=500)
    avg_rule = alt.Chart(pd.DataFrame({'y': [promedio_total]})).mark_rule(strokeDash=[6,4]).encode(y='y:Q')
    avg_text = alt.Chart(pd.DataFrame({'y':[promedio_total], 'label':[f'Promedio: {promedio_total:.2f}']})).mark_text(align='left', dx=5, dy=-5).encode(y='y:Q', text='label:N')
    st.altair_chart((line + avg_rule + avg_text).interactive(), use_container_width=True)

# --- ESTADÍSTICAS DESCRIPTIVAS ---
with tabs[3]:
    st.markdown('<div class="section-header"><h3>ESTADÍSTICAS DESCRIPTIVAS</h3></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="insight-box">
            <h4>ESTADÍSTICAS BÁSICAS</h4>
            <p><strong>Valor Mínimo:</strong> {minimo:.2f}</p>
            <p><strong>Valor Máximo:</strong> {maximo:.2f}</p>
            <p><strong>Rango:</strong> {maximo - minimo:.2f}</p>
            <p><strong>Media:</strong> {promedio_total:.2f}</p>
            <p><strong>Mediana:</strong> {df_ejemplo['valor'].median():.2f}</p>
            <p><strong>Desviación Estándar:</strong> {desviacion:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        hist = alt.Chart(df_ejemplo).mark_bar().encode(
            alt.X('valor:Q', bin=alt.Bin(maxbins=20), title='VALOR'),
            alt.Y('count()', title='FRECUENCIA'),
            tooltip=[alt.Tooltip('count()', title='Frecuencia')]
        ).properties(title="DISTRIBUCIÓN DE VALORES", height=400)
        st.altair_chart(hist, use_container_width=True)

# --- ANÁLISIS DE TENDENCIA ---
with tabs[4]:
    st.markdown('<div class="section-header"><h3>ANÁLISIS DE TENDENCIA</h3></div>', unsafe_allow_html=True)
    x_numeric = np.arange(len(df_ejemplo))
    y = df_ejemplo['valor'].values
    if len(x_numeric) < 2 or np.allclose(np.std(y), 0):
        slope = 0.0
        intercept = float(np.mean(y)) if len(y) > 0 else 0.0
        r_value = 0.0
    else:
        slope, intercept = np.polyfit(x_numeric, y, 1)
        try:
            r_value = np.corrcoef(x_numeric, y)[0, 1]
            if np.isnan(r_value):
                r_value = 0.0
        except Exception:
            r_value = 0.0
    df_trend = df_ejemplo.copy()
    df_trend['trend'] = slope * x_numeric + intercept
    chart_orig = alt.Chart(df_trend).mark_line().encode(x='fecha:T', y='valor:Q', tooltip=[alt.Tooltip('fecha:T', title='Fecha'), alt.Tooltip('valor:Q', title='Valor')])
    chart_trend = alt.Chart(df_trend).mark_line(strokeDash=[6,4]).encode(x='fecha:T', y='trend:Q', tooltip=[alt.Tooltip('trend:Q', title='Trend')])
    st.altair_chart((chart_orig + chart_trend).properties(height=400).interactive(), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="stat-highlight"><h4>PENDIENTE DE TENDENCIA</h4><h2>{slope:.4f}</h2><p>Cambio por período</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-highlight"><h4>CORRELACIÓN (R²)</h4><h2>{r_value**2:.4f}</h2><p>Ajuste del modelo</p></div>', unsafe_allow_html=True)

# --- ANÁLISIS TEMPORAL ---
with tabs[5]:
    st.markdown('<div class="section-header"><h3>ANÁLISIS TEMPORAL</h3></div>', unsafe_allow_html=True)
    df_tiempo = df_ejemplo.copy()
    df_tiempo['año'] = df_tiempo['fecha'].dt.year
    df_tiempo['mes'] = df_tiempo['fecha'].dt.month
    df_tiempo['trimestre'] = df_tiempo['fecha'].dt.quarter
    c1, c2 = st.columns(2)
    with c1:
        df_anual = df_tiempo.groupby('año')['valor'].mean().reset_index()
        fig_anual = alt.Chart(df_anual).mark_bar().encode(
            x=alt.X('año:O', title='AÑO'),
            y=alt.Y('valor:Q', title='PROMEDIO'),
            tooltip=[alt.Tooltip('año:O', title='Año'), alt.Tooltip('valor:Q', title='Promedio')],
            color=alt.Color('valor:Q', scale=alt.Scale(scheme='blues'), legend=None)
        ).properties(height=350, title="PROMEDIO ANUAL")
        st.altair_chart(fig_anual, use_container_width=True)
    with c2:
        df_mensual = df_tiempo.groupby('mes')['valor'].mean().reset_index()
        fig_mensual = alt.Chart(df_mensual).mark_line(point=True).encode(
            x=alt.X('mes:O', title='MES', sort=list(range(1,13))),
            y=alt.Y('valor:Q', title='VALOR'),
            tooltip=[alt.Tooltip('mes:O', title='Mes'), alt.Tooltip('valor:Q', title='Valor')]
        ).properties(height=350, title="PATRÓN ESTACIONAL (MENSUAL)")
        st.altair_chart(fig_mensual, use_container_width=True)

# --- INSIGHTS AUTOMÁTICOS ---
with tabs[6]:
    st.markdown('<div class="section-header"><h3>INSIGHTS AUTOMÁTICOS</h3></div>', unsafe_allow_html=True)
    crecimiento_total = ((valor_actual / df_ejemplo['valor'].iloc[0]) - 1) * 100 if df_ejemplo['valor'].iloc[0] != 0 else np.nan
    volatilidad_relativa = (desviacion / promedio_total) * 100 if promedio_total != 0 else np.nan
    mejor_idx = df_ejemplo['valor'].idxmax()
    peor_idx = df_ejemplo['valor'].idxmin()
    mejor_fecha = df_ejemplo.loc[mejor_idx, 'fecha'].strftime('%B %Y')
    peor_fecha = df_ejemplo.loc[peor_idx, 'fecha'].strftime('%B %Y')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="insight-box">
            <h4>INSIGHTS CLAVE</h4>
            <p><strong>Crecimiento Total:</strong> {crecimiento_total:+.1f}% desde el inicio</p>
            <p><strong>Volatilidad Relativa:</strong> {volatilidad_relativa:.1f}%</p>
            <p><strong>Mejor Período:</strong> {mejor_fecha} ({maximo:.2f})</p>
            <p><strong>Peor Período:</strong> {peor_fecha} ({minimo:.2f})</p>
            <p><strong>Períodos Analizados:</strong> {len(df_ejemplo)}</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if len(df_ejemplo) >= 2 and not np.allclose(np.std(df_ejemplo['valor'].values), 0):
            slope, intercept = np.polyfit(np.arange(len(df_ejemplo)), df_ejemplo['valor'].values, 1)
            try:
                r_value = np.corrcoef(np.arange(len(df_ejemplo)), df_ejemplo['valor'].values)[0,1]
                if np.isnan(r_value):
                    r_value = 0.0
            except:
                r_value = 0.0
        else:
            slope = 0.0
            r_value = 0.0
        if slope > 0:
            interpretacion = "TENDENCIA POSITIVA: La serie muestra crecimiento a lo largo del tiempo"
            color_tendencia = "#4CAF50"
        elif slope < 0:
            interpretacion = "TENDENCIA NEGATIVA: La serie muestra decrecimiento a lo largo del tiempo"
            color_tendencia = "#f44336"
        else:
            interpretacion = "TENDENCIA ESTABLE: La serie se mantiene relativamente constante"
            color_tendencia = "#ff9800"
        if np.isnan(volatilidad_relativa):
            vol_desc = "No disponible"
        elif volatilidad_relativa < 10:
            vol_desc = "Baja volatilidad"
        elif volatilidad_relativa < 25:
            vol_desc = "Volatilidad moderada"
        else:
            vol_desc = "Alta volatilidad"
        st.markdown(f"""
        <div class="insight-box">
            <h4>INTERPRETACIÓN</h4>
            <p style="color: {color_tendencia};">{interpretacion}</p>
            <p><strong>Característica:</strong> {vol_desc} ({volatilidad_relativa:.1f}%)</p>
            <p><strong>R² de tendencia:</strong> {r_value**2:.3f} ({'Fuerte' if r_value**2 > 0.7 else 'Moderado' if r_value**2 > 0.3 else 'Débil'} ajuste lineal)</p>
        </div>
        """, unsafe_allow_html=True)

# --- PREDICCIONES (FORECAST) ---
with tabs[7]:
    st.markdown('<div class="section-header"><h3>PREDICCIONES (FORECAST)</h3></div>', unsafe_allow_html=True)

    # Intentamos cargar el forecast desde disco; si no existe, usamos uploader en sidebar
    df_forecast = None
    try:
        df_forecast = pd.read_csv("forecast_24m.csv")
    except FileNotFoundError:
        uploaded_f = st.sidebar.file_uploader("Sube forecast_24m.csv (opcional)", type=["csv"], key="forecast")
        if uploaded_f is not None:
            try:
                df_forecast = pd.read_csv(uploaded_f)
            except Exception as e:
                st.warning(f"No se pudo leer el CSV subido: {e}")
    except Exception as e:
        st.warning(f"No se pudo leer forecast_24m.csv: {e}")

    if df_forecast is None:
        st.info("No hay archivo de forecast cargado. Sube 'forecast_24m.csv' si deseas ver predicciones.")
    else:
        # Validar columnas necesarias (tolerante a mayúsculas/minúsculas)
        cols_lower = {c.lower(): c for c in df_forecast.columns}
        required_lower = {'ds', 'yhat', 'yhat_lower', 'yhat_upper'}
        if not required_lower.issubset(set(cols_lower.keys())):
            st.warning(f"El forecast debe contener las columnas {required_lower} (no importa mayúsculas/minúsculas). Columnas encontradas: {list(df_forecast.columns)}")
        else:
            # Normalizar nombres de columnas al formato esperado
            df_forecast = df_forecast.rename(columns={cols_lower['ds']: 'ds', cols_lower['yhat']: 'yhat', cols_lower['yhat_lower']: 'yhat_lower', cols_lower['yhat_upper']: 'yhat_upper'})

            # Asegurar tipos numéricos
            for c in ['yhat', 'yhat_lower', 'yhat_upper']:
                df_forecast[c] = pd.to_numeric(df_forecast[c], errors='coerce')
            df_forecast['ds'] = pd.to_datetime(df_forecast['ds'], errors='coerce')
            df_forecast = df_forecast.dropna(subset=['ds', 'yhat'])

            if df_forecast.empty:
                st.warning("El forecast está vacío o inválido después del parseo de fechas/valores.")
            else:
                # Ordenar por fecha
                df_forecast = df_forecast.sort_values('ds').reset_index(drop=True)
                ultima_fecha_historica = df_ejemplo['fecha'].max() if not df_ejemplo['fecha'].isna().all() else None

                # Preparar banda y línea con Altair (manejo defensivo)
                try:
                    # Registrar tema oscuro (si no existe)
                    try:
                        alt.themes.register('dark_theme', lambda: {
                            'config': {
                                'background': 'black',
                                'title': {'color': 'white'},
                                'axis': {'domainColor': 'white', 'gridColor': '#444', 'labelColor': 'white', 'titleColor': 'white'},
                                'legend': {'labelColor': 'white', 'titleColor': 'white'}
                            }
                        })
                        alt.themes.enable('dark_theme')
                    except Exception:
                        pass

                    band = alt.Chart(df_forecast).mark_area(opacity=0.25).encode(
                        x=alt.X('ds:T', title='FECHA'),
                        y=alt.Y('yhat_lower:Q', title='Predicción', axis=alt.Axis(format=',', labelAngle=0)),
                        y2='yhat_upper:Q',
                        tooltip=[
                            alt.Tooltip('ds:T', title='Fecha'),
                            alt.Tooltip('yhat_lower:Q', title='Límite Inferior'),
                            alt.Tooltip('yhat_upper:Q', title='Límite Superior')
                        ]
                    )

                    line = alt.Chart(df_forecast).mark_line().encode(
                        x='ds:T',
                        y=alt.Y('yhat:Q'),
                        tooltip=[alt.Tooltip('ds:T', title='Fecha'), alt.Tooltip('yhat:Q', title='Predicción')]
                    )

                    points = alt.Chart(df_forecast).mark_point().encode(
                        x='ds:T',
                        y='yhat:Q',
                        tooltip=[alt.Tooltip('ds:T', title='Fecha'), alt.Tooltip('yhat:Q', title='Predicción')]
                    )

                    # regla vertical y etiqueta (si tenemos fecha histórica)
                    layer_items = [band, line, points]
                    if ultima_fecha_historica is not None:
                        prediction_line = alt.Chart(pd.DataFrame({'fecha': [ultima_fecha_historica]})).mark_rule(
                            color='orange', strokeDash=[5,5], strokeWidth=2
                        ).encode(x='fecha:T')
                        # colocar texto algo por encima del máximo de yhat para que no se superponga
                        max_yhat = df_forecast['yhat'].max()
                        text_y = (max_yhat * 0.9) if pd.notna(max_yhat) else df_forecast['yhat'].mean()
                        prediction_text = alt.Chart(pd.DataFrame({
                            'fecha': [ultima_fecha_historica],
                            'valor': [text_y],
                            'texto': ['INICIO PREDICCIÓN']
                        })).mark_text(align='center', dx=0, dy=-10, color='orange', fontSize=12, fontWeight='bold').encode(
                            x='fecha:T', y='valor:Q', text='texto:N'
                        )
                        layer_items += [prediction_line, prediction_text]

                    # usar alt.layer para mantener una sola escala Y compartida (evita ejes duplicados)
                    chart = alt.layer(*layer_items).properties(
                        title='FORECAST CON INTERVALO DE CONFIANZA',
                        width=800,
                        height=450
                    )

                    st.altair_chart(chart, use_container_width=True)

                except Exception as e:
                    st.error(f"Ocurrió un error al construir la visualización del forecast: {e}")

                # Información adicional del forecast
                try:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>INFORMACIÓN DEL FORECAST</h4>
                            <p><strong>Períodos de predicción:</strong> {len(df_forecast)} registros</p>
                            <p><strong>Fecha inicio predicción (histórica):</strong> {ultima_fecha_historica.strftime('%Y-%m-%d') if ultima_fecha_historica is not None else 'N/A'}</p>
                            <p><strong>Valor promedio predicho:</strong> {df_forecast['yhat'].mean():.2f}</p>
                            <p><strong>Rango de predicción:</strong> {df_forecast['yhat'].min():.2f} - {df_forecast['yhat'].max():.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>INTERVALOS DE CONFIANZA</h4>
                            <p><strong>Límite inferior promedio:</strong> {df_forecast['yhat_lower'].mean():.2f}</p>
                            <p><strong>Límite superior promedio:</strong> {df_forecast['yhat_upper'].mean():.2f}</p>
                            <p><strong>Amplitud promedio:</strong> {(df_forecast['yhat_upper'] - df_forecast['yhat_lower']).mean():.2f}</p>
                            <p><strong>Última predicción:</strong> {df_forecast['yhat'].iloc[-1]:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"No se pudo mostrar la información adicional del forecast: {e}")

# Fin del script

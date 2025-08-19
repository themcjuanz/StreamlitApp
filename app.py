import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Serie de Tiempo",
    page_icon="📈",
    layout="wide"
)

# Estilo personalizado
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}

.metric-container {
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    margin-bottom: 1rem;
}

.insight-box {
    background: rgba(255, 255, 255, 0.9);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 5px solid #4CAF50;
    margin: 1rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.trend-positive {
    color: #4CAF50;
    font-weight: bold;
}

.trend-negative {
    color: #f44336;
    font-weight: bold;
}

.stat-highlight {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    padding: 1rem;
    border-radius: 8px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.section-header {
    background: linear-gradient(90deg, #74b9ff 0%, #0984e3 100%);
    padding: 1rem;
    border-radius: 8px;
    color: white;
    text-align: center;
    margin: 1.5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>📈 Análisis de Serie de Tiempo</h1>
    <h3>Dashboard de Visualización y Análisis Estadístico</h3>
</div>
""", unsafe_allow_html=True)

# ========================================
# CONFIGURACIÓN DE DATOS - MODIFICAR AQUÍ
# ========================================

# Asignar tu dataset aquí
# Ejemplo: df_datos = pd.read_csv('tu_archivo.csv')
df_datos = pd.read_csv("mensual.csv") # Reemplaza con tu DataFrame

# Especificar nombres de columnas
COLUMNA_FECHA = 'ds'      # Nombre de tu columna de fecha
COLUMNA_VALOR = 'y'      # Nombre de tu columna de valores

# ========================================

# Verificar que se haya asignado un dataset
if df_datos is None:
    st.markdown("""
    <div class="insight-box">
        <h4>⚠️ Configuración Requerida</h4>
        <p><strong>Para usar este dashboard:</strong></p>
        <ol>
            <li>Asigna tu DataFrame a la variable <code>df_datos</code></li>
            <li>Especifica el nombre de tu columna de fecha en <code>COLUMNA_FECHA</code></li>
            <li>Especifica el nombre de tu columna de valores en <code>COLUMNA_VALOR</code></li>
        </ol>
        <p><strong>Ejemplo:</strong></p>
        <pre>
df_datos = pd.read_csv('mi_serie_tiempo.csv')
COLUMNA_FECHA = 'fecha'
COLUMNA_VALOR = 'ventas'
        </pre>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    # Preparar los datos
    df_ejemplo = df_datos[[COLUMNA_FECHA, COLUMNA_VALOR]].copy()
    df_ejemplo.columns = ['fecha', 'valor']
    df_ejemplo['fecha'] = pd.to_datetime(df_ejemplo['fecha'])
    df_ejemplo = df_ejemplo.dropna().sort_values('fecha').reset_index(drop=True)
    
    st.markdown(f"""
    <div class="insight-box">
        <h4>✅ Dataset Cargado</h4>
        <p><strong>Registros procesados:</strong> {len(df_ejemplo):,}</p>
        <p><strong>Período:</strong> {df_ejemplo['fecha'].min().strftime('%Y-%m-%d')} a {df_ejemplo['fecha'].max().strftime('%Y-%m-%d')}</p>
        <p><strong>Columna fecha:</strong> {COLUMNA_FECHA}</p>
        <p><strong>Columna valores:</strong> {COLUMNA_VALOR}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar vista previa
    with st.expander("👀 Vista previa de los datos"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Primeros 5 registros:**")
            st.dataframe(df_ejemplo.head())
        with col2:
            st.write("**Últimos 5 registros:**")
            st.dataframe(df_ejemplo.tail())
    
except KeyError as e:
    st.error(f"❌ Error: No se encontró la columna {e}. Verifica los nombres de las columnas.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error al procesar los datos: {str(e)}")
    st.stop()

# Calcular estadísticas principales
valor_actual = df_ejemplo['valor'].iloc[-1]
valor_anterior = df_ejemplo['valor'].iloc[-2]
cambio_absoluto = valor_actual - valor_anterior
cambio_porcentual = (cambio_absoluto / valor_anterior) * 100

promedio_total = df_ejemplo['valor'].mean()
maximo = df_ejemplo['valor'].max()
minimo = df_ejemplo['valor'].min()
desviacion = df_ejemplo['valor'].std()

# Métricas principales
st.markdown('<div class="section-header"><h2>📊 Métricas Clave</h2></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-container">
        <h3>Valor Actual</h3>
        <h2>{valor_actual:.2f}</h2>
        <p>Último registro</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    tendencia_class = "trend-positive" if cambio_absoluto > 0 else "trend-negative"
    st.markdown(f"""
    <div class="metric-container">
        <h3>Cambio Período</h3>
        <h2 class="{tendencia_class}">{cambio_absoluto:+.2f}</h2>
        <p>{cambio_porcentual:+.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-container">
        <h3>Promedio</h3>
        <h2>{promedio_total:.2f}</h2>
        <p>Serie completa</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-container">
        <h3>Volatilidad</h3>
        <h2>{desviacion:.2f}</h2>
        <p>Desv. estándar</p>
    </div>
    """, unsafe_allow_html=True)

# Gráfico principal
st.markdown('<div class="section-header"><h2>📈 Visualización Principal</h2></div>', unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_ejemplo['fecha'],
    y=df_ejemplo['valor'],
    mode='lines+markers',
    name='Serie de Tiempo',
    line=dict(color='#667eea', width=3),
    marker=dict(size=6, color='#764ba2')
))

# Línea de promedio
fig.add_hline(y=promedio_total, 
              line_dash="dash", 
              line_color="red", 
              annotation_text=f"Promedio: {promedio_total:.2f}")

fig.update_layout(
    title="Evolución Temporal de la Serie",
    xaxis_title="Fecha",
    yaxis_title="Valor",
    hovermode='x unified',
    height=500,
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

# Análisis detallado en pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Estadísticas Descriptivas", 
    "📈 Análisis de Tendencia", 
    "📅 Análisis Temporal", 
    "🔍 Insights Automáticos"
])

with tab1:
    st.markdown('<div class="section-header"><h3>📊 Estadísticas Descriptivas</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <h4>📈 Estadísticas Básicas</h4>
            <p><strong>Valor Mínimo:</strong> {minimo:.2f}</p>
            <p><strong>Valor Máximo:</strong> {maximo:.2f}</p>
            <p><strong>Rango:</strong> {maximo - minimo:.2f}</p>
            <p><strong>Media:</strong> {promedio_total:.2f}</p>
            <p><strong>Mediana:</strong> {df_ejemplo['valor'].median():.2f}</p>
            <p><strong>Desviación Estándar:</strong> {desviacion:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Histograma de distribución
        fig_hist = px.histogram(
            df_ejemplo, 
            x='valor', 
            nbins=20,
            title="Distribución de Valores",
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.markdown('<div class="section-header"><h3>📈 Análisis de Tendencia</h3></div>', unsafe_allow_html=True)
    
# Importar scipy solo si hay datos
if df_ejemplo is not None:
    from scipy import stats
    x_numeric = np.arange(len(df_ejemplo))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, df_ejemplo['valor'])
    
    # Gráfico con línea de tendencia
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=df_ejemplo['fecha'],
        y=df_ejemplo['valor'],
        mode='lines+markers',
        name='Serie Original',
        line=dict(color='#667eea', width=2),
        marker=dict(size=4)
    ))
    
    # Línea de tendencia
    trend_line = slope * x_numeric + intercept
    fig_trend.add_trace(go.Scatter(
        x=df_ejemplo['fecha'],
        y=trend_line,
        mode='lines',
        name='Línea de Tendencia',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    fig_trend.update_layout(
        title="Serie de Tiempo con Línea de Tendencia",
        xaxis_title="Fecha",
        yaxis_title="Valor",
        height=400
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="stat-highlight">
            <h4>Pendiente de Tendencia</h4>
            <h2>{slope:.4f}</h2>
            <p>Cambio por período</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-highlight">
            <h4>Correlación (R²)</h4>
            <h2>{r_value**2:.4f}</h2>
            <p>Ajuste del modelo</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-header"><h3>📅 Análisis Temporal</h3></div>', unsafe_allow_html=True)
    
    # Preparar datos por año y mes
    df_tiempo = df_ejemplo.copy()
    df_tiempo['año'] = df_tiempo['fecha'].dt.year
    df_tiempo['mes'] = df_tiempo['fecha'].dt.month
    df_tiempo['trimestre'] = df_tiempo['fecha'].dt.quarter
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Análisis por año
        df_anual = df_tiempo.groupby('año')['valor'].mean().reset_index()
        
        fig_anual = px.bar(
            df_anual,
            x='año',
            y='valor',
            title="Promedio Anual",
            color='valor',
            color_continuous_scale='Blues'
        )
        fig_anual.update_layout(height=350)
        st.plotly_chart(fig_anual, use_container_width=True)
    
    with col2:
        # Análisis por mes
        df_mensual = df_tiempo.groupby('mes')['valor'].mean().reset_index()
        
        fig_mensual = px.line(
            df_mensual,
            x='mes',
            y='valor',
            title="Patrón Estacional (Mensual)",
            markers=True
        )
        fig_mensual.update_layout(height=350)
        fig_mensual.update_traces(line_color='#764ba2', line_width=3, marker_size=8)
        st.plotly_chart(fig_mensual, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header"><h3>🔍 Insights Automáticos</h3></div>', unsafe_allow_html=True)
    
    # Generar insights automáticos
    crecimiento_total = ((valor_actual / df_ejemplo['valor'].iloc[0]) - 1) * 100
    volatilidad_relativa = (desviacion / promedio_total) * 100
    
    # Detectar el mejor y peor período
    mejor_idx = df_ejemplo['valor'].idxmax()
    peor_idx = df_ejemplo['valor'].idxmin()
    
    mejor_fecha = df_ejemplo.loc[mejor_idx, 'fecha'].strftime('%B %Y')
    peor_fecha = df_ejemplo.loc[peor_idx, 'fecha'].strftime('%B %Y')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <h4>🎯 Insights Clave</h4>
            <p><strong>Crecimiento Total:</strong> {crecimiento_total:+.1f}% desde el inicio</p>
            <p><strong>Volatilidad Relativa:</strong> {volatilidad_relativa:.1f}%</p>
            <p><strong>Mejor Período:</strong> {mejor_fecha} ({maximo:.2f})</p>
            <p><strong>Peor Período:</strong> {peor_fecha} ({minimo:.2f})</p>
            <p><strong>Períodos Analizados:</strong> {len(df_ejemplo)} registros</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Interpretación de la tendencia
        if slope > 0:
            interpretacion = "📈 **Tendencia POSITIVA**: La serie muestra crecimiento a lo largo del tiempo"
            color_tendencia = "#4CAF50"
        elif slope < 0:
            interpretacion = "📉 **Tendencia NEGATIVA**: La serie muestra decrecimiento a lo largo del tiempo"
            color_tendencia = "#f44336"
        else:
            interpretacion = "📊 **Tendencia ESTABLE**: La serie se mantiene relativamente constante"
            color_tendencia = "#ff9800"
        
        if volatilidad_relativa < 10:
            vol_desc = "Baja volatilidad"
        elif volatilidad_relativa < 25:
            vol_desc = "Volatilidad moderada"
        else:
            vol_desc = "Alta volatilidad"
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 Interpretación</h4>
            <p style="color: {color_tendencia};">{interpretacion}</p>
            <p><strong>Característica:</strong> {vol_desc} ({volatilidad_relativa:.1f}%)</p>
            <p><strong>R² de tendencia:</strong> {r_value**2:.3f} ({'Fuerte' if r_value**2 > 0.7 else 'Moderado' if r_value**2 > 0.3 else 'Débil'} ajuste lineal)</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>📈 <strong>Sistema de Análisis de Series de Tiempo</strong></p>
    <p>Dashboard automático para visualización y análisis estadístico avanzado</p>
</div>
""", unsafe_allow_html=True)
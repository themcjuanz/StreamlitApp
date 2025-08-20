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
            import math

            import numpy as np
            import pandas as pd
            from shapely.geometry import shape, mapping
            from shapely.affinity import scale

            import streamlit as st
            import pydeck as pdk

            # ---------- Config ----------
            GEOJSON_PATH = "colombia_departments.json"
            FORECAST_FOLDER = "forecasts"
            INNER_MIN_SCALE = 0.30     # escala del anillo interior (30% del área original)
            CONTRAST_FACTOR = 8.0      # <- aumentado para exagerar aún más
            TEXT_SIZE = 16             # tamaño base del texto en el mapa
            PREDICTED_YEAR_THRESHOLD = 2023
            # ---------------------------

            st.title("Mapa predictivo — exagerado al máximo, yhat entero y etiqueta 'predecido' desde 2023")
            st.caption("Slider arriba. Anillo exterior = yhat_lower, medio = yhat, interior = yhat_upper. Deptos con yhat==0 no muestran anillos.")

            def normalize_name(s: str):
                if s is None:
                    return ""
                s = s.upper().strip()
                s = unicodedata.normalize("NFD", s)
                s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
                s = "".join(ch for ch in s if ch.isalnum())
                return s

            def color_from_t(t):
                dark_green = (0, 204, 102)   # #00cc66
                bright_green = (0, 255, 153) # #00ff99
                r = int(dark_green[0] + t * (bright_green[0] - dark_green[0]))
                g = int(dark_green[1] + t * (bright_green[1] - dark_green[1]))
                b = int(dark_green[2] + t * (bright_green[2] - dark_green[2]))
                a = 220
                return [r, g, b, a]

            def count_to_rgb_exaggerated(val, vmin, vmax, contrast=CONTRAST_FACTOR):
                """
                - val None o 0 => color apagado (gris).
                - Normaliza val entre vmin/vmax, aplica contraste grande y una ligera transformación no lineal
                para exagerar aún más los extremos.
                """
                if val is None:
                    return [180, 180, 180, 120]
                if val == 0:
                    return [180, 180, 180, 120]
                # seguridad
                if vmin is None or vmax is None or vmax == vmin:
                    t = 0.5
                else:
                    t = (val - vmin) / (vmax - vmin)
                t = max(0.0, min(1.0, t))
                # amplificar simétricamente respecto a 0.5
                t = 0.5 + contrast * (t - 0.5)
                # recortar
                t = max(0.0, min(1.0, t))
                # aplicar pequeña potenciación para estirar los extremos (más contraste visual)
                # si t>0.5 lo hacemos más cercano a 1, si t<0.5 lo hacemos más cercano a 0
                if t >= 0.5:
                    t = 0.5 + ( (t - 0.5) ** 0.8 ) * 0.5
                else:
                    t = 0.5 - ( (0.5 - t) ** 0.8 ) * 0.5
                t = max(0.0, min(1.0, t))
                return color_from_t(t)

            # --- Cargar geojson base ---
            with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
                base_geo = json.load(f)

            # --- Cargar forecasts (dataframes por depto normalizado) ---
            dept_dfs = {}
            files = glob.glob(os.path.join(FORECAST_FOLDER, "*_forecast.csv"))
            if not files:
                st.error(f"No se encontraron archivos '*_forecast.csv' en '{FORECAST_FOLDER}'")
                st.stop()

            years = []
            for fp in files:
                fname = os.path.basename(fp)
                if not fname.upper().endswith("_FORECAST.CSV"):
                    continue
                dept_raw = fname[: -len("_forecast.csv")]
                dept_norm = normalize_name(dept_raw)
                try:
                    df = pd.read_csv(fp)
                    cols_lower = {c:c.lower() for c in df.columns}
                    df = df.rename(columns=cols_lower)
                    if "ds" not in df.columns:
                        st.warning(f"{fp} no tiene columna 'ds' (fecha). Se omite.")
                        continue
                    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
                    df = df.dropna(subset=["ds"])
                    if df.empty:
                        continue
                    dept_dfs[dept_norm] = df.reset_index(drop=True)
                    years.extend(df["ds"].dt.year.dropna().astype(int).tolist())
                except Exception as e:
                    st.warning(f"Error leyendo {fp}: {e}")
                    continue

            if not dept_dfs:
                st.error("No se pudo leer ningún forecast válido.")
                st.stop()

            min_year = int(min(years))
            max_year = int(max(years))

            # -------------------------
            # Slider ARRIBA del mapa
            # -------------------------
            selected_year = st.slider("Selecciona año", min_year, max_year, value=max_year, format="%d")
            # Mostrar etiqueta especial si estamos en periodo predecido
            is_predicted = selected_year >= PREDICTED_YEAR_THRESHOLD
            if is_predicted:
                st.markdown(f"**Año seleccionado:** {selected_year} — **(predecido)**")
            else:
                st.markdown(f"**Año seleccionado:** {selected_year}")

            # --- Extraer valores redondeados (ENTEROS) para el año seleccionado ---
            dept_values = {}  # dept_norm -> dict yhat_lower,yhat,yhat_upper (INT or None)
            vals_for_year = []
            for dept_norm, df in dept_dfs.items():
                df_year = df[df["ds"].dt.year == int(selected_year)]
                if not df_year.empty:
                    row = df_year.iloc[-1]
                else:
                    df_before = df[df["ds"].dt.year < int(selected_year)]
                    row = df_before.iloc[-1] if not df_before.empty else None
                if row is None:
                    dept_values[dept_norm] = None
                else:
                    def getval(r, names):
                        for n in names:
                            if n in r.index:
                                return r[n]
                        return None
                    yhat = getval(row, ["yhat", "YHAT", "yHat"])
                    yhat_lower = getval(row, ["yhat_lower", "YHAT_LOWER", "yhatLower", "yhat-lower"])
                    yhat_upper = getval(row, ["yhat_upper", "YHAT_UPPER", "yhatUpper", "yhat-upper"])
                    def to_int_or_none(x):
                        try:
                            if x is None or (isinstance(x, float) and np.isnan(x)):
                                return None
                            xi = int(round(float(x)))
                            return xi
                        except Exception:
                            return None
                    yi = to_int_or_none(yhat)
                    yi_l = to_int_or_none(yhat_lower)
                    yi_u = to_int_or_none(yhat_upper)
                    if yi is None and yi_l is None and yi_u is None:
                        dept_values[dept_norm] = None
                    else:
                        dept_values[dept_norm] = {"yhat_lower": yi_l, "yhat": yi, "yhat_upper": yi_u}
                        # recolectar para percentiles, EXCLUYENDO None y excluyendo ceros
                        for v in (yi_l, yi, yi_u):
                            if v is not None and v != 0:
                                vals_for_year.append(v)

            # fallback global si necesario (excluyendo ceros)
            if not vals_for_year:
                vals_global = []
                for df in dept_dfs.values():
                    for candidate in ["yhat_lower","yhat","yhat_upper","YHAT_LOWER","YHAT","YHAT_UPPER"]:
                        if candidate in df.columns:
                            vals = pd.to_numeric(df[candidate], errors="coerce").dropna().tolist()
                            for v in vals:
                                try:
                                    vi = int(round(float(v)))
                                    if vi != 0:
                                        vals_global.append(vi)
                                except Exception:
                                    continue
                vals_for_year = vals_global

            if not vals_for_year:
                vmin, vmax = 0, 1
            else:
                p5, p95 = np.percentile(vals_for_year, [5, 95])
                rng = p95 - p5
                if rng < 1e-6:
                    p5 -= 1
                    p95 += 1
                else:
                    p5 -= 0.05 * rng
                    p95 += 0.05 * rng
                vmin, vmax = float(p5), float(p95)

            st.markdown(f"Escala usada (excluyendo ceros/nulos): **vmin={vmin:.1f}**, **vmax={vmax:.1f}** (percentiles 5-95)")

            # --- Construir features: solo crear anillos si yhat != 0 (yhat None o 0 -> no anillos) ---
            new_features = []
            labels = []  # para TextLayer

            for feat in base_geo["features"]:
                props = feat.get("properties", {})
                nombre = props.get("NOMBRE_DPT") or props.get("NOMBRE") or ""
                nombre_norm = normalize_name(nombre)

                # match por código o nombre normalizado
                matched = None
                for key in ("DPTO", "DPTO_", "DPTO_COD", "DPTO_CODE"):
                    code = props.get(key)
                    if code is not None:
                        cstr = str(code)
                        if cstr in dept_values:
                            matched = cstr
                            break
                if matched is None and nombre_norm in dept_values:
                    matched = nombre_norm

                geom = shape(feat["geometry"])
                centroid = geom.centroid

                # obtener yhat entero para etiqueta
                yhat_val = None
                if matched is not None and dept_values.get(matched) is not None:
                    yhat_val = dept_values[matched].get("yhat")

                # agregar etiqueta solo si yhat_val is not None -> mostrar entero + "(predecido)" si corresponde
                if yhat_val is not None:
                    label_text = f"{int(yhat_val)}"
                    if is_predicted:
                        label_text = f"{label_text} (predecido)"
                    labels.append({"lon": centroid.x, "lat": centroid.y, "text": label_text})

                # decidir si dibujar anillos: SOLO si yhat_val is not None AND yhat_val != 0
                draw_rings = (yhat_val is not None) and (yhat_val != 0)

                if (matched is None) or (dept_values.get(matched) is None) or (not draw_rings):
                    # sin anillos -> agregamos la geometría completa en gris tenue
                    new_features.append({
                        "type": "Feature",
                        "geometry": mapping(geom),
                        "properties": {
                            **props,
                            "ring": -1,
                            "value": None,
                            "color": [200, 200, 200, 110],
                            "label": props.get("NOMBRE_DPT", nombre)
                        }
                    })
                    continue

                vals = dept_values[matched]
                outer_val = vals.get("yhat_lower")
                mid_val = vals.get("yhat")
                inner_val = vals.get("yhat_upper")
                scales = [1.0, 0.65, INNER_MIN_SCALE]
                vals_ring = [outer_val, mid_val, inner_val]

                parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
                for part in parts:
                    c = part.centroid
                    for i, scale_factor in enumerate(scales):
                        try:
                            scaled = scale(part, xfact=scale_factor, yfact=scale_factor, origin=(c.x, c.y))
                            if scaled.is_empty:
                                continue
                        except Exception:
                            scaled = part
                        val = vals_ring[i]
                        if val is None or val == 0:
                            color = [190, 190, 190, 120]
                        else:
                            color = count_to_rgb_exaggerated(val, vmin=vmin, vmax=vmax, contrast=CONTRAST_FACTOR)
                        new_features.append({
                            "type": "Feature",
                            "geometry": mapping(scaled),
                            "properties": {
                                **props,
                                "ring": i,
                                "value": int(val) if (val is not None) else None,
                                "color": color,
                                "label": props.get("NOMBRE_DPT", nombre)
                            }
                        })

            # FeatureCollection final
            new_geojson = {"type": "FeatureCollection", "features": new_features}
            with open("colombia_forecast_3rings_exagerado_max.geojson", "w", encoding="utf-8") as f:
                json.dump(new_geojson, f, ensure_ascii=False)

            # Capa de anillos (NO pickable)
            polygon_layer = pdk.Layer(
                "GeoJsonLayer",
                new_geojson,
                stroked=False,
                filled=True,
                extruded=False,
                pickable=False,
                auto_highlight=False,
                get_fill_color="properties.color",
                get_line_color=[60, 60, 60],
            )

            # Capa de texto con yhat encima de cada departamento (NO pickable)
            if labels:
                labels_df = pd.DataFrame(labels)
            else:
                labels_df = pd.DataFrame(columns=["lon","lat","text"])

            text_layer = pdk.Layer(
                "TextLayer",
                data=labels_df,
                pickable=False,
                get_position=["lon", "lat"],
                get_text="text",
                get_size=TEXT_SIZE,
                get_color=[0, 0, 0],
                get_alignment_baseline="bottom"
            )

            view_state = pdk.ViewState(latitude=4.5, longitude=-74.0, zoom=5)
            deck = pdk.Deck(layers=[polygon_layer, text_layer], initial_view_state=view_state)
            st.pydeck_chart(deck)

            # Tabla de referencia (enteros)
            tab = []
            for k, v in dept_values.items():
                tab.append({
                    "departamento_norm": k,
                    "yhat_lower": (v["yhat_lower"] if v else None),
                    "yhat": (v["yhat"] if v else None),
                    "yhat_upper": (v["yhat_upper"] if v else None)
                })
            st.markdown("### Valores usados (enteros, por departamento normalizado)")
            st.dataframe(pd.DataFrame(tab).sort_values("departamento_norm"), height=300)

            st.success("He exagerado aun más el contraste. Cuando el slider entra en 2023+ las etiquetas muestran '(predecido)' y el yhat aparece en cifras enteras encima de cada departamento.")


"""
Uso:
  python forecast_registrations.py --csv ruta.csv [--horizon 5] [--year-col COL] [--date-col COL]

Salida:
  - forecast_registros_por_anio.csv  (pronóstico)
  - forecast_plot.png                 (gráfico histórico + pronóstico)
"""

import argparse, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv_auto(path):
    """Intenta leer el CSV infiriendo separador y codificación comunes."""
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8", low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
        except Exception:
            try:
                return pd.read_csv(path, sep=",", encoding="utf-8", low_memory=False)
            except Exception:
                # Último intento con latin-1
                try:
                    return pd.read_csv(path, sep=None, engine="python", encoding="latin-1", low_memory=False)
                except Exception:
                    return pd.read_csv(path, sep=",", encoding="latin-1", low_memory=False)

def find_year_series(frame: pd.DataFrame, year_col=None, date_col=None) -> pd.Series:
    """
    Devuelve una Serie con el año:
    - Si year_col está, lo convierte a numérico.
    - Si date_col está, extrae año.
    - Si no, heurística: busca columnas con 'año/fecha/registro' o extrae año por regex.
    """
    if year_col and year_col in frame.columns:
        s = pd.to_numeric(frame[year_col], errors="coerce")
        return s.astype("Int64")

    if date_col and date_col in frame.columns:
        s_dt = pd.to_datetime(frame[date_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
        return s_dt.dt.year.astype("Int64")

    cols = list(frame.columns)
    lower_map = {c: c.lower() for c in cols}
    year_like = []
    for c in cols:
        lc = lower_map[c]
        if any(k in lc for k in ["año", "anio", "ano", "year", "fecha", "matric", "registro"]):
            year_like.append(c)

    # Deduplicar
    seen = set()
    year_like = [x for x in year_like if not (x in seen or seen.add(x))]

    # 1) Numérico tipo año
    for c in year_like + cols:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s):
            s2 = pd.to_numeric(s, errors="coerce")
            frac_yearish = np.mean((s2 >= 1980) & (s2 <= 2100))
            if frac_yearish > 0.5:
                return s2.astype("Int64")

    # 2) Datetime
    for c in year_like + cols:
        try:
            s_dt = pd.to_datetime(frame[c], errors="coerce", dayfirst=True, infer_datetime_format=True)
            if s_dt.notna().mean() > 0.5:
                return s_dt.dt.year.astype("Int64")
        except Exception:
            pass

    # 3) Regex 4 dígitos
    for c in year_like + cols:
        try:
            s_str = frame[c].astype(str)
            y = pd.to_numeric(s_str.str.extract(r"(\d{4})", expand=False), errors="coerce")
            frac_yearish = np.mean((y >= 1980) & (y <= 2100))
            if frac_yearish > 0.5:
                return y.astype("Int64")
        except Exception:
            pass

    raise RuntimeError("No pude identificar una columna de año/fecha. Use --year-col o --date-col.")

def fit_and_forecast(by_year_df: pd.DataFrame, horizon: int = 5):
    """
    Selecciona el mejor grado polinómico (1..3) por RMSE en validación temporal (últimos 2 años),
    y pronostica 'horizon' años hacia adelante. Si no hay scikit-learn, cae a recta (numpy.polyfit).
    """
    X = by_year_df["_year_"].values.reshape(-1, 1).astype(float)
    y = by_year_df["registros"].values.astype(float)
    years = by_year_df["_year_"].values.astype(int)

    # Intentar con scikit-learn
    try:
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error

        best_deg = 1
        best_rmse = float("inf")
        best_model = None

        # Split: últimos 2 años como test (si hay)
        split_n = max(0, len(X) - 2)
        X_train, y_train = X[:split_n], y[:split_n]
        X_test, y_test = X[split_n:], y[split_n:]

        for deg in [1, 2, 3]:
            pipe = Pipeline([
                ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
                ("lin", LinearRegression())
            ])
            if len(X_train) >= 2:
                pipe.fit(X_train, y_train)
                if len(X_test) > 0:
                    pred = pipe.predict(X_test)
                    rmse = math.sqrt(mean_squared_error(y_test, pred))
                else:
                    pred = pipe.predict(X)
                    rmse = math.sqrt(mean_squared_error(y, pred))
            else:
                pipe.fit(X, y)
                pred = pipe.predict(X)
                rmse = float(np.sqrt(((pred - y) ** 2).mean()))

            if rmse < best_rmse:
                best_rmse = rmse
                best_deg = deg
                best_model = pipe

        model = best_model
        model.fit(X, y)

        last_year = int(years.max())
        future_years = np.arange(last_year + 1, last_year + 1 + horizon)
        futX = future_years.reshape(-1, 1).astype(float)
        fut_pred = model.predict(futX)
        fut_pred = np.maximum(0, np.round(fut_pred, 0).astype(int))  # no negativos, redondeo

        in_pred = model.predict(X)
        in_rmse = float(np.sqrt(((in_pred - y) ** 2).mean()))

        return {
            "best_deg": best_deg,
            "in_rmse": in_rmse,
            "future_years": future_years.tolist(),
            "future_pred": fut_pred.tolist(),
        }

    except Exception:
        # Fallback: regresión lineal simple con numpy
        coefs = np.polyfit(X.flatten(), y, 1)
        p = np.poly1d(coefs)
        last_year = int(years.max())
        future_years = np.arange(last_year + 1, last_year + 1 + horizon)
        fut_pred = p(future_years)
        fut_pred = np.maximum(0, np.round(fut_pred, 0).astype(int))
        in_pred = p(X.flatten())
        in_rmse = float(np.sqrt(((in_pred - y) ** 2).mean()))
        return {
            "best_deg": 1,
            "in_rmse": in_rmse,
            "future_years": future_years.tolist(),
            "future_pred": fut_pred.tolist(),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Ruta del CSV de registros")
    ap.add_argument("--horizon", type=int, default=5, help="Años a pronosticar (default: 5)")
    ap.add_argument("--year-col", default=None, help="Nombre de la columna con el año (opcional)")
    ap.add_argument("--date-col", default=None, help="Nombre de la columna fecha para extraer año (opcional)")
    args = ap.parse_args()

    # 1) Leer y detectar año
    df = read_csv_auto(args.csv)
    year_s = find_year_series(df, year_col=args.year_col, date_col=args.date_col)
    df["_year_"] = year_s

    # 2) Agregar por año
    by_year = (df
               .dropna(subset=["_year_"])
               .groupby("_year_")
               .size()
               .reset_index(name="registros")) \
               .sort_values("_year_").reset_index(drop=True)

    if by_year.empty:
        raise RuntimeError("No hay filas con año válido tras la limpieza.")

    # 3) Ajustar y pronosticar
    res = fit_and_forecast(by_year, horizon=args.horizon)

    # 4) Guardar pronóstico
    out = pd.DataFrame({
        "year": res["future_years"],
        "pred_registros": res["future_pred"]
    })
    out_path = "forecast_registros_por_anio.csv"
    out.to_csv(out_path, index=False)
    print(f"Mejor grado polinómico: {res['best_deg']} (RMSE in-sample: {res['in_rmse']:.2f})")
    print("Pronóstico:")
    print(out.to_string(index=False))
    print(f"\nArchivo guardado: {out_path}")

    # 5) Gráfico histórico + pronóstico
    plt.figure()
    plt.plot(by_year["_year_"], by_year["registros"], marker="o", label="Histórico")
    plt.plot(out["year"], out["pred_registros"], marker="x", linestyle="--", label="Pronóstico")
    plt.title("Registros por año (histórico y pronóstico)")
    plt.xlabel("Año")
    plt.ylabel("Número de registros")
    plt.legend()
    plt.savefig("forecast_plot.png", bbox_inches="tight")
    print("Gráfico guardado: forecast_plot.png")

if __name__ == "__main__":
    main()

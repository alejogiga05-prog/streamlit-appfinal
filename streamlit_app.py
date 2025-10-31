# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import os
from dotenv import load_dotenv
import plotly.express as px
from influxdb_client import InfluxDBClient

load_dotenv()  # carga .env en desarrollo local

st.set_page_config(page_title="Monitoreo Industrial", layout="wide")
st.title("📊 Tablero: Monitorización y Predicción")

# --- Config desde entorno (o .env local) ---
INFLUX_URL = os.getenv("INFLUX_URL", "")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "")
INFLUX_ORG = os.getenv("INFLUX_ORG", "")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "")

# --- Utilidades ---
def gen_synthetic(start, stop, freq='1T'):
    idx = pd.date_range(start, stop, freq=freq)
    n = len(idx)
    t = np.arange(n)
    temp = 50 + 5*np.sin(2*np.pi*t/1440) + 0.5*np.random.randn(n)
    hum  = 30 + 10*np.sin(2*np.pi*t/2880) + 1.2*np.random.randn(n)
    vib  = 0.2 + 0.05*np.sin(2*np.pi*t/60) + 0.02*np.random.randn(n)
    df = pd.DataFrame({'timestamp': idx, 'temperature': temp, 'humidity': hum, 'vibration': vib})
    df = df.set_index('timestamp')
    return df

def read_influx(start, stop, measurement='sensors'):
    # Construye y ejecuta una consulta Flux (ajusta measurement/fields según tu esquema)
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()
    flux = f'''
    from(bucket:"{INFLUX_BUCKET}")
      |> range(start: {start.isoformat()}, stop: {stop.isoformat()})
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time","temperature","humidity","vibration"])
    '''
    tables = query_api.query_data_frame(flux)
    if isinstance(tables, list):
        df = pd.concat(tables, ignore_index=True)
    else:
        df = tables
    df['_time'] = pd.to_datetime(df['_time'])
    df = df.set_index('_time')
    df = df[['temperature','humidity','vibration']]
    return df

def moving_average(series, window=10):
    return series.rolling(window=window, min_periods=1).mean()

def exponential_smoothing(series, alpha=0.2):
    return series.ewm(alpha=alpha, adjust=False).mean()

def linear_regression_forecast(series, n_forecast=60):
    s = series.dropna()
    if len(s) < 5:
        return pd.Series(dtype=float)
    X = (s.index.astype('int64') // 10**9).values.reshape(-1,1)
    y = s.values
    model = LinearRegression().fit(X,y)
    last_ts = s.index[-1]
    freq = s.index.to_series().diff().median()
    future_idx = [last_ts + (i+1)*freq for i in range(n_forecast)]
    Xf = (pd.Series(future_idx).astype('datetime64[ns]').astype('int64') // 10**9).values.reshape(-1,1)
    yf = model.predict(Xf)
    return pd.Series(yf, index=future_idx)

def detect_anomalies_zscore(series, threshold=3.0):
    s = series.dropna()
    mu = s.mean(); sigma = s.std()
    if sigma == 0:
        return pd.Series(False, index=s.index)
    z = (s - mu) / sigma
    return z.abs() > threshold

# --- Sidebar / Controles ---
st.sidebar.header("Controles")
now = datetime.utcnow()
period = st.sidebar.selectbox("Rango de tiempo", ["Última 1h","Últimas 6h","Últimas 24h","Personalizado"])
if period == "Personalizado":
    start_date = st.sidebar.date_input("Fecha inicio", value=(now - timedelta(hours=1)).date())
    start_time = st.sidebar.time_input("Hora inicio", value=(now - timedelta(hours=1)).time())
    end_date = st.sidebar.date_input("Fecha fin", value=now.date())
    end_time = st.sidebar.time_input("Hora fin", value=now.time())
    start_dt = datetime.combine(start_date, start_time)
    stop_dt  = datetime.combine(end_date, end_time)
else:
    if period == "Última 1h":
        start_dt, stop_dt = now - timedelta(hours=1), now
    elif period == "Últimas 6h":
        start_dt, stop_dt = now - timedelta(hours=6), now
    else:
        start_dt, stop_dt = now - timedelta(hours=24), now

variable = st.sidebar.selectbox("Variable", ["temperature","humidity","vibration"])
method = st.sidebar.selectbox("Método predictivo", ["Ninguno","Promedio móvil","Suavizado exponencial","Regresión lineal"])
n_forecast = st.sidebar.slider("Minutos a predecir", 5, 180, 30)
run = st.sidebar.button("Cargar datos")

# --- Cargar datos (Influx o sintético) ---
df = pd.DataFrame()
use_synthetic = False
try:
    if INFLUX_TOKEN == "" or INFLUX_BUCKET == "" or INFLUX_URL == "":
        raise Exception("Credenciales Influx no configuradas")
    df = read_influx(start_dt, stop_dt)
    st.sidebar.success("Conectado a InfluxDB")
except Exception as e:
    use_synthetic = True
    st.sidebar.warning(f"No se pudo conectar a InfluxDB ({e}). Usando datos sintéticos.")
    df = gen_synthetic(start_dt, stop_dt, freq='1T')

# --- Indicadores ---
if not df.empty:
    latest = df[variable].iloc[-1]
    promedio = df[variable].mean()
    minimo = df[variable].min()
    maximo = df[variable].max()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual", f"{latest:.3f}")
    c2.metric("Promedio", f"{promedio:.3f}")
    c3.metric("Mínimo", f"{minimo:.3f}")
    c4.metric("Máximo", f"{maximo:.3f}")

# --- Gráfica principal ---
st.subheader(f"{variable} en el tiempo")
hist = df[variable].rename(variable).reset_index()
fig = px.line(hist, x='timestamp', y=variable, title=f"{variable} vs tiempo")
fig.update_xaxes(title="Tiempo")
fig.update_yaxes(title=variable)
st.plotly_chart(fig, use_container_width=True)

# --- Predicción ---
forecast = pd.Series(dtype=float)
if method == "Promedio móvil":
    sma = moving_average(df[variable], window=10)
    st.plotly_chart(px.line(pd.concat([df[variable].rename('valor'), sma.rename('sma')], axis=1).reset_index(), x='timestamp', y=['valor','sma']), use_container_width=True)
elif method == "Suavizado exponencial":
    ses = exponential_smoothing(df[variable], alpha=0.2)
    st.plotly_chart(px.line(pd.concat([df[variable].rename('valor'), ses.rename('exp')], axis=1).reset_index(), x='timestamp', y=['valor','exp']), use_container_width=True)
elif method == "Regresión lineal":
    forecast = linear_regression_forecast(df[variable].dropna(), n_forecast=n_forecast)
    if not forecast.empty:
        combined = pd.concat([df[variable], forecast.rename('forecast')], axis=0)
        combined = combined.reset_index().rename(columns={'index':'timestamp'})
        st.plotly_chart(px.line(combined, x='timestamp', y=combined.columns[1:]), use_container_width=True)
else:
    st.caption("Sin método predictivo seleccionado")

# --- Anomalías ---
anoms = detect_anomalies_zscore(df[variable], threshold=3.0)
if anoms.any():
    st.warning(f"Se detectaron {anoms.sum()} anomalías")
    st.dataframe(df[variable][anoms])

# --- Tabla y descarga ---
st.subheader("Muestra de datos")
st.dataframe(df.reset_index().tail(200))

csv = df.reset_index().to_csv(index=False).encode('utf-8')
st.download_button("Descargar CSV", data=csv, file_name='datos.csv', mime='text/csv')

}


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Ecopack - Monitoreo de Sensores", layout="wide")

st.title("📊 Monitoreo y Predicción de Variables Industriales - Ecopack")

# Simular datos (puedes conectar InfluxDB después)
fechas = pd.date_range(datetime.now() - timedelta(hours=24), datetime.now(), freq="10min")
datos = np.random.normal(loc=25, scale=2, size=len(fechas))
df = pd.DataFrame({"Fecha": fechas, "Temperatura (°C)": datos})

# Mostrar tabla y gráfico
st.subheader("Datos recientes")
st.dataframe(df.tail(10))

st.subheader("Tendencia de temperatura")
fig, ax = plt.subplots()
ax.plot(df["Fecha"], df["Temperatura (°C)"], label="Temperatura", color="orange")
ax.set_xlabel("Hora")
ax.set_ylabel("°C")
st.pyplot(fig)

# Predicción simple
st.subheader("Predicción próxima hora")
media = df["Temperatura (°C)"].mean()
pred = np.random.normal(media, 0.5, 6)
st.line_chart(pred)

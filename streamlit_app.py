import streamlit as st
import pandas as pd
import mysql.connector
import plotly.express as px

# -----------------------------
# CONFIGURACIÓN DE PÁGINA
# -----------------------------
st.set_page_config(page_title="Monitoreo Ecopack", layout="wide")

# -----------------------------
# CONEXIÓN A LA BASE DE DATOS
# -----------------------------
DB_CONFIG = {
    "host": "ecopack.cmyllmaytgv1.us-east-1.rds.amazonaws.com",
    "user": "flaskuser",
    "password": "tu_contraseña_aquí",
    "database": "EXTREME MANUFACTURING"
}

def obtener_datos():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT fecha, temperatura FROM sensores ORDER BY fecha DESC LIMIT 10;")
    datos = cursor.fetchall()
    cursor.close()
    conn.close()

    df = pd.DataFrame(datos, columns=["Fecha", "Temperatura (°C)"])
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df

# -----------------------------
# INTERFAZ STREAMLIT
# -----------------------------
st.markdown("<h1>📊 Monitoreo y Predicción de Variables Industriales - Ecopack</h1>", unsafe_allow_html=True)
st.subheader("Datos recientes")

df = obtener_datos()
st.dataframe(df)

# -----------------------------
# GRÁFICO
# -----------------------------
st.subheader("Tendencia de temperatura")
fig = px.line(df.sort_values("Fecha"), x="Fecha", y="Temperatura (°C)", markers=True, title="Temperatura en el tiempo")
st.plotly_chart(fig, use_container_width=True)

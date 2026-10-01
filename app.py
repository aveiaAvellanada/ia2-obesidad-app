"""App web interactiva del proyecto IA2 (Estimation of Obesity Levels, UCI 544).

Ejecutar desde la raíz del proyecto:   streamlit run app.py
Las etapas y los pasos tienen la misma numeración y títulos que estudio_etapas.html.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import streamlit as st

from src import salida
from webapp import etapa01, etapa02, etapa03, etapa04, etapa05
from webapp.navegacion import ETAPAS, PASOS

# Sin ventanas ni archivos: elige el backend Agg y aplica el mismo estilo que el CLI
salida.configurar(mostrar=False, guardar=False, graficar_tablas=False)
st.set_page_config(page_title="Proyecto IA2 — Obesidad", layout="wide")

RENDER = {"01": etapa01.RENDER, "02": etapa02.RENDER, "03": etapa03.RENDER,
          "04": etapa04.RENDER, "05": etapa05.RENDER}

st.sidebar.title("Proyecto IA2")
etapa = st.sidebar.radio("Etapa", list(ETAPAS), key="etapa",
                         format_func=lambda e: f"{e} · {ETAPAS[e]}")
titulos = dict(PASOS[etapa])
paso = st.sidebar.radio("Paso", list(titulos), key=f"paso_{etapa}",
                        format_func=lambda i: f"{i} · {titulos[i]}")

st.header(f"Etapa {etapa} — {ETAPAS[etapa]}")
st.subheader(f"{paso} · {titulos[paso]}")

render = RENDER[etapa].get(paso)
if render is None:
    st.error("Paso no implementado.")
else:
    render()

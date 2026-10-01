"""Salida de la app: figuras y tablas en Streamlit."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def figura(fig):
    st.pyplot(fig)
    plt.close(fig)


def subtitulo(texto):
    st.markdown(f"##### {texto}")


def tabla(df, titulo=None, decimales=4):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.round(decimales).copy()
    for c in df.columns:               # evita columnas de tipos mezclados (Arrow)
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    if titulo:
        subtitulo(titulo)
    st.dataframe(df)

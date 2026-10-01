"""Etapa 01 — EDA y preprocesamiento."""
import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE
from etapas import e01_eda as e01
from src.data import TARGET_CLF, TARGET_REG
from src.preprocessing import CLASS_ORDER, build_datasets
from webapp import receptor, ui


def paso_1():
    df = e01.df_raw()
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", df.shape[0])
    c2.metric("Columnas", df.shape[1])
    c3.metric("Filas duplicadas", int(df.duplicated().sum()))
    n = st.slider("Filas a mostrar", 5, 50, 5, key="e01_head")
    ui.tabla(df.head(n), "Primeras filas", 3)
    ui.tabla(df.describe().T, "Estadísticos de las columnas numéricas", 3)
    ui.tabla(pd.DataFrame({"tipo": df.dtypes.astype(str), "faltantes": df.isna().sum()}),
             "Tipos y valores faltantes por columna")
    buf = io.StringIO()
    df.info(buf=buf)
    with st.expander("df.info()"):
        st.code(buf.getvalue())
    cat = list(df.select_dtypes(exclude="number").columns)
    col = st.selectbox("Columna categórica", cat, key="e01_cat")
    ui.tabla(df[col].value_counts().rename("n"), f"Valores de {col}")


def paso_2_2():
    df = e01.df_raw()
    bins = st.slider("Número de bins del histograma", 10, 100, 40, key="e01_bins")
    ui.tabla(df[TARGET_REG].describe().round(2), "Weight (kg)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[TARGET_REG], bins=bins, kde=True, ax=axes[0])
    axes[0].set_title("Histograma de Weight (kg)")
    sns.boxplot(data=df, x=TARGET_CLF, y=TARGET_REG, order=CLASS_ORDER, ax=axes[1])
    axes[1].set_title("Weight por clase de NObeyesdad")
    axes[1].tick_params(axis="x", rotation=45)
    ui.figura(fig)


def paso_2_3():
    receptor.ejecutar(e01.paso_2_3_fuga)
    d = e01.df_bmi()
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    filas = {str(cols): cross_val_score(DecisionTreeClassifier(random_state=RANDOM_STATE),
                                        d[cols], d[TARGET_CLF], cv=cv).mean()
             for cols in (["BMI"], ["Weight", "Height"], ["Weight"], ["Height"])}
    ui.tabla(pd.Series(filas, name="accuracy (CV 5)"),
             "Accuracy de un árbol usando sólo estas columnas")


def paso_2_4():
    metodo = st.selectbox("Método de correlación", ["pearson", "spearman", "kendall"],
                          key="e01_corr")
    num = e01.df_raw().select_dtypes("number")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(method=metodo), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"Correlación ({metodo}) entre numéricas")
    ui.figura(fig)


def paso_3():
    df_clf, df_reg = build_datasets(e01.df_raw())
    c1, c2 = st.columns(2)
    c1.metric("clf.csv (filas × columnas)", f"{df_clf.shape[0]} × {df_clf.shape[1]}")
    c2.metric("reg.csv (filas × columnas)", f"{df_reg.shape[0]} × {df_reg.shape[1]}")
    ui.tabla(df_clf.head(), "clf.csv — primeras filas", 3)
    ui.tabla(df_reg.head(), "reg.csv — primeras filas", 3)


RENDER = {
    "1": paso_1,
    "2.1": lambda: receptor.ejecutar(e01.paso_2_1_clases),
    "2.2": paso_2_2,
    "2.3": paso_2_3,
    "2.4": paso_2_4,
    "3": paso_3,
}

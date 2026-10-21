"""ETAPA 01 — EDA y preprocesamiento
=====================================

Dataset: *Estimation of Obesity Levels Based on Eating Habits and Physical
Condition* (UCI id 544). 2111 filas, 16 atributos + la clase `NObeyesdad`.

Qué hace esta etapa:
  1.   Carga el dataset y hace un EDA mínimo (describe, info, faltantes, duplicados).
  2.1  Distribución de las 7 clases de `NObeyesdad`.
  2.2  Distribución de `Weight` (target de regresión).
  2.3  Chequeo de FUGA DE DATOS: `NObeyesdad` ≈ f(Weight, Height) vía IMC.
  2.4  Correlaciones entre numéricas.
  3.   Preprocesamiento (encoding, duplicados) -> genera data/clf.csv y data/reg.csv,
       que usan las etapas 02, 03 y 04.

Cómo ejecutar:
  python etapas/e01_eda.py             -> menú de pasos
  python etapas/e01_eda.py --todo      -> todos los pasos
  python etapas/e01_eda.py --paso 2.3  -> sólo el paso 2.3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # raíz del proyecto

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE
from src.data import TARGET_CLF, TARGET_REG, CLF_CSV, REG_CSV, load_raw
from src.pasos import Etapa, cli
from src.preprocessing import CLASS_ORDER, build_datasets
from src.salida import subtitulo, tabla

etapa = Etapa("01", "EDA y preprocesamiento", "eda",
              "Exploración del dataset, chequeo de fuga de datos y generación de clf.csv / reg.csv",
              globales=("RANDOM_STATE",))

# ----------------------------------------------------------------------------
# Estado compartido entre pasos (se carga una sola vez por sesión)
# ----------------------------------------------------------------------------
_estado = {}


def df_raw() -> pd.DataFrame:
    if "df" not in _estado:
        _estado["df"] = load_raw()
    return _estado["df"]


def df_bmi() -> pd.DataFrame:
    """Dataset crudo + columna IMC = Weight / Height²."""
    if "df_bmi" not in _estado:
        df = df_raw()
        _estado["df_bmi"] = df.assign(BMI=df[TARGET_REG] / df["Height"] ** 2)
    return _estado["df_bmi"]


# ----------------------------------------------------------------------------
# 1. Carga y EDA mínimo
# ----------------------------------------------------------------------------
@etapa.paso("1", "Carga del dataset y EDA mínimo (describe, info, faltantes, duplicados)")
def paso_1_carga():
    df = df_raw()
    print("Dimensiones (filas, columnas):", df.shape)
    tabla(df.head(), "Primeras 5 filas")
    tabla(df.describe().T, "Estadísticos de las columnas numéricas", decimales=3)

    subtitulo("df.info()")
    df.info()

    subtitulo("Calidad de datos")
    print("Valores faltantes:        ", df.isna().sum().sum())
    print("Filas duplicadas exactas: ", df.duplicated().sum())

    subtitulo("Valores de cada columna categórica")
    for col in df.select_dtypes(exclude="number").columns:
        print(f"{col}: {df[col].value_counts().to_dict()}")


# ----------------------------------------------------------------------------
# 2.1 Distribución del target de clasificación
# ----------------------------------------------------------------------------
@etapa.paso("2.1", "Distribución de las 7 clases (NObeyesdad)", figuras=1)
def paso_2_1_clases():
    df = df_raw()
    counts = df[TARGET_CLF].value_counts().reindex(CLASS_ORDER)
    tabla(counts.rename("n"), "Filas por clase")
    tabla((counts / counts.sum()).rename("proporción").round(3), "Proporción por clase")

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="steelblue")
    ax.set_title("Distribución de NObeyesdad")
    ax.set_ylabel("n")
    ax.tick_params(axis="x", rotation=45)
    etapa.figura(fig, "2.1_distribucion_clases")


# ----------------------------------------------------------------------------
# 2.2 Distribución del target de regresión
# ----------------------------------------------------------------------------
@etapa.paso("2.2", "Distribución de Weight (target de regresión)", figuras=1)
def paso_2_2_weight():
    df = df_raw()
    tabla(df[TARGET_REG].describe().round(2), "Weight (kg)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[TARGET_REG], bins=40, kde=True, ax=axes[0])
    axes[0].set_title("Histograma de Weight (kg)")
    sns.boxplot(data=df, x=TARGET_CLF, y=TARGET_REG, order=CLASS_ORDER, ax=axes[1])
    axes[1].set_title("Weight por clase de NObeyesdad")
    axes[1].tick_params(axis="x", rotation=45)
    etapa.figura(fig, "2.2_distribucion_weight")


# ----------------------------------------------------------------------------
# 2.3 Chequeo de fuga de datos
# ----------------------------------------------------------------------------
@etapa.paso("2.3", "Fuga de datos: Weight, Height y NObeyesdad (IMC vs umbrales OMS)", figuras=1)
def paso_2_3_fuga():
    d = df_bmi()
    tabla(d.groupby(TARGET_CLF, observed=True)["BMI"].describe().loc[CLASS_ORDER],
          "IMC por clase", decimales=2)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=d, x=TARGET_CLF, y="BMI", order=CLASS_ORDER, ax=ax)
    for umbral in [18.5, 25, 30, 35, 40]:
        ax.axhline(umbral, ls="--", c="gray", lw=0.8)
    ax.set_title("IMC por clase — líneas: umbrales OMS (18.5 / 25 / 30 / 35 / 40)")
    ax.tick_params(axis="x", rotation=45)
    etapa.figura(fig, "2.3_imc_por_clase")

    # ¿Cuánto acierta un árbol usando SÓLO peso y/o estatura?
    subtitulo("Accuracy (5-fold CV) de un árbol de decisión usando sólo estas columnas")
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    for cols in [["BMI"], ["Weight", "Height"], ["Weight"], ["Height"]]:
        acc = cross_val_score(DecisionTreeClassifier(random_state=RANDOM_STATE),
                              d[cols], d[TARGET_CLF], cv=cv).mean()
        print(f"   sólo {cols!s:<22}: {acc:.3f}")

    tabla(d[["Weight", "Height", "BMI", "Age"]].corr(), "Correlación Weight / Height / BMI / Age", 3)


# ----------------------------------------------------------------------------
# 2.4 Correlaciones
# ----------------------------------------------------------------------------
@etapa.paso("2.4", "Correlaciones (Pearson) entre variables numéricas", figuras=1)
def paso_2_4_correlaciones():
    num = df_raw().select_dtypes("number")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlación (Pearson) entre numéricas")
    etapa.figura(fig, "2.4_correlaciones")


# ----------------------------------------------------------------------------
# 3. Preprocesamiento -> clf.csv y reg.csv
# ----------------------------------------------------------------------------
@etapa.paso("3", "Preprocesamiento: encoding, duplicados y generación de clf.csv / reg.csv",
            parametros=("build_datasets",))
def paso_3_preprocesamiento():
    df_clf, df_reg = build_datasets(df_raw())
    print("Clasificación:", df_clf.shape, "| target:", TARGET_CLF)
    print("Regresión:    ", df_reg.shape, "| target:", TARGET_REG,
          "| contiene NObeyesdad:", TARGET_CLF in df_reg.columns)
    tabla(df_clf.head(), "clf.csv — primeras filas")
    tabla(df_reg.head(), "reg.csv — primeras filas")

    df_clf.to_csv(CLF_CSV, index=False)
    df_reg.to_csv(REG_CSV, index=False)
    print(f"Guardados {CLF_CSV.name} y {REG_CSV.name} en {CLF_CSV.parent}")
    print("(las etapas 02-04 los leen; si no existen, los regeneran solas)")


if __name__ == "__main__":
    sys.exit(cli(etapa))

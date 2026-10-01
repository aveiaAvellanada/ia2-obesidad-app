"""Etapa 05 — Comparación final de todos los casos con los parámetros activos."""
import pandas as pd
import streamlit as st

from etapas import e05_comparacion as e05
from webapp import clustering as CL
from webapp import controles as C
from webapp import modelos as M
from webapp import receptor


def _repeticiones():
    """Repeticiones de la CV de filas reales vigentes en el paso 8 de la etapa 02."""
    return int(C.leer("reales", C.DEF_REALES)["repeticiones"])


def _ruido(spc):
    """Tabla de e05._ruido (limpio vs NIVEL_RUIDO, variante B) con el split activo de la 02.
    Con el split por defecto da las mismas cifras que el CLI y no escribe en cache/."""
    df = M.barrido_ruido(spc["test_size"], spc["seed"], (0.0, e05.NIVEL_RUIDO))
    nivel = int(round(e05.NIVEL_RUIDO * 100))
    limpio = df[df["nivel"] == 0].set_index("modelo")
    ruido = df[df["nivel"] == nivel].set_index("modelo")
    return pd.DataFrame({"test_accuracy_limpio": limpio["accuracy"],
                         "test_accuracy": ruido["accuracy"],
                         "test_f1_macro": ruido["f1_macro"]}).rename_axis(None)


def _resultados():
    """Dict con el formato que espera etapas/e05_comparacion.py (resultados())."""
    spc, hpc = C.params_split("clf"), C.params_hp("clf")
    spr, hpr = C.params_split("reg"), C.params_hp("reg")
    pc = C.params_clu()
    todas = tuple(M.features_clf())
    sin_wh = tuple(f for f in todas if f not in M.DROP_WH)

    _, clf_A, _ = M.tabla_clf(hpc, todas, spc["test_size"], spc["seed"])
    _, clf_B, _ = M.tabla_clf(hpc, sin_wh, spc["test_size"], spc["seed"])
    ruido = _ruido(spc)
    _, reg, _ = M.tabla_reg(hpr, tuple(M.features_reg()), spr["test_size"], spr["seed"])

    def metricas(variante):
        km = CL.ajustar(variante, 7, pc["n_init"], pc["init"], pc["seed"])
        return CL.analizar(variante, km.labels_)["metricas"]

    clu = pd.DataFrame({
        "Sin clase (20 features)": metricas("base"),
        "Sólo numéricas (8)": metricas("num"),
        "Con clase ordinal": metricas("ord"),
        "Con clase one-hot": metricas("oh"),
    }).T[["silueta", "ARI", "NMI", "pureza"]].astype(float)

    rep = _repeticiones()
    reales = {"B7": M.evaluar_reales(False, 7, rep, spc["seed"]),
              "B3": M.evaluar_reales(False, 3, rep, spc["seed"])}
    return dict(clf_A=clf_A, clf_B=clf_B, ruido=ruido, reg=reg, clu=clu, reales=reales,
                barrido_rf=None)


def paso_1():
    with st.spinner("Calculando todos los casos con los parámetros activos…"):
        res = _resultados()
    e05._estado["res"] = res
    try:
        receptor.ejecutar(e05.paso_1_resultados, {"CV 3×10": f"CV 3×{_repeticiones()}"})
    finally:
        e05._estado.pop("res", None)


RENDER = {"1": paso_1}

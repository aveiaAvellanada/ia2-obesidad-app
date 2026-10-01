"""Etapa 04 — Clustering K-Means."""
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.evaluation import metricas_por_clase
from src.preprocessing import CLASS_ORDER, NUMERIC_COLS
from webapp import clustering as CL
from webapp import controles as C
from webapp import ui

N_REFS = 10   # muestras de referencia del estadístico gap


def _b(variante, p):
    return CL.barrido(variante, p["k_min"], p["k_max"], p["n_init"], p["init"], p["seed"])


def _g(variante, p, n_refs=N_REFS):
    return CL.gap(variante, p["k_min"], p["k_max"], n_refs, p["seed"])


def _km(variante, p):
    return CL.ajustar(variante, p["k"], p["n_init"], p["init"], p["seed"])


def _con_gap(variante, p, n_refs=N_REFS):
    with st.spinner("Calculando el estadístico gap (tarda unos segundos)…"):
        return _g(variante, p, n_refs)


# ----------------------------------------------------------------------------
# 1–2. Datos y elección de K
# ----------------------------------------------------------------------------
def paso_1():
    m = CL.matriz("base")
    c1, c2 = st.columns(2)
    c1.metric("Filas", m["X_scaled"].shape[0])
    c2.metric("Features (sin la clase)", m["X_scaled"].shape[1])
    ui.tabla(pd.DataFrame(m["X_scaled"], columns=m["X"].columns).head(),
             "Primeras filas escaladas (StandardScaler)", 3)


def paso_2():
    p = C.form_clu(rango=True)
    ui.tabla(_b("base", p)["metrics"], "Métricas internas por K", 3)


def paso_2_1():
    p = C.form_clu(rango=True)
    b = _b("base", p)
    codo = CL.knee_point(b["ks"], b["metrics"]["inercia"].values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(b["metrics"].index, b["metrics"]["inercia"], marker="o")
    ax.axvline(codo, c="r", ls="--", label=f"codo detectado: K={codo}")
    ax.set_xlabel("K")
    ax.set_ylabel("Inercia (WCSS)")
    ax.set_title("Método del codo")
    ax.legend()
    ui.figura(fig)


def paso_2_2():
    p = C.form_clu(rango=True)
    m = _b("base", p)["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for ax, (col, mejor, sentido) in zip(axes, [
            ("silhouette", int(m["silhouette"].idxmax()), "max"),
            ("calinski_harabasz", int(m["calinski_harabasz"].idxmax()), "max"),
            ("davies_bouldin", int(m["davies_bouldin"].idxmin()), "min")]):
        ax.plot(m.index, m[col], marker="o")
        ax.axvline(mejor, c="r", ls="--", label=f"{sentido} en K={mejor}")
        ax.set_xlabel("K")
        ax.set_title(col)
        ax.legend()
    ui.figura(fig)


def paso_2_3():
    p = C.form_clu(rango=True)
    n_refs = st.slider("Muestras de referencia (gap)", 3, 20, N_REFS, key="e04_refs")
    gap_df, k_gap = _con_gap("base", p, n_refs)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(gap_df.index, gap_df["gap"], yerr=gap_df["s_k"], marker="o", capsize=3)
    if k_gap is not None:
        ax.axvline(k_gap, c="r", ls="--", label=f"gap: K={k_gap}")
        ax.legend()
    ax.set_xlabel("K")
    ax.set_ylabel("gap(K)")
    ax.set_title("Estadístico gap")
    ui.figura(fig)
    ui.tabla(gap_df, "gap(K) y s_k", 3)


def paso_2_4():
    p = C.form_clu(rango=True)
    g = _con_gap("base", p)
    ui.tabla(CL.resumen_k(_b("base", p), g), "K óptimo según cada método")


# ----------------------------------------------------------------------------
# 3. K vs clases reales
# ----------------------------------------------------------------------------
def paso_3():
    p = C.form_clu(k=True)
    a = CL.analizar("base", _km("base", p).labels_)
    ui.tabla(pd.Series(a["metricas"], name=f"K = {p['k']}"), "Silueta, ARI, NMI, accuracy con mapeo y pureza")


def paso_3_1():
    p = C.form_clu(k=True)
    ct = CL.analizar("base", _km("base", p).labels_)["ct"]
    ui.tabla(ct, f"Clases reales (filas) vs clusters K={p['k']} (columnas)", 0)
    fig, ax = plt.subplots(figsize=(9, 6))
    CL.heatmap_ct(ct, ax, f"Clases reales (filas) vs clusters K-Means K={p['k']} (columnas)")
    ui.figura(fig)


def paso_3_2():
    p = C.form_clu(k=True)
    a = CL.analizar("base", _km("base", p).labels_)
    fig, ax = plt.subplots(figsize=(9, 7))
    CL.heatmap_cm(a["cm"], ax, f"Matriz de confusión — clusters K={p['k']} mapeados a clases")
    ui.figura(fig)
    ui.tabla(pd.Series(a["mapping"], name="clase asignada").rename_axis("Cluster"),
             "Mapeo cluster → clase (algoritmo húngaro)")
    ui.tabla(metricas_por_clase(a["cm"], CLASS_ORDER), f"Métricas por clase — K={p['k']}", 3)


def _centroides(variante, p, columnas, ordenar=False):
    m, km = CL.matriz(variante), _km(variante, p)
    a = CL.analizar(variante, km.labels_)
    c = pd.DataFrame(m["scaler"].inverse_transform(km.cluster_centers_), columns=m["X"].columns)
    c.index.name = "Cluster"
    c.insert(0, "clase_mapeada", [a["mapping"].get(i, "sin_clase") for i in c.index])
    c.insert(0, "n", pd.Series(km.labels_).value_counts().sort_index().values)
    c["BMI_centroide"] = c["Weight"] / c["Height"] ** 2
    extra = [] if "BMI_centroide" in columnas else ["BMI_centroide"]
    c = c[["n", "clase_mapeada", *columnas, *extra]]
    return c.sort_values("BMI_centroide") if ordenar else c


def paso_3_3():
    p = C.form_clu(k=True)
    cols = ["Gender", "Age", "Height", "Weight", "BMI_centroide", "family_history_with_overweight",
            "FAVC", "FCVC", "CAEC", "FAF", "MTRANS_Public_Transportation"]
    ui.tabla(_centroides("base", p, cols), f"Centroides (K={p['k']}) en unidades originales", 2)


def paso_3_4():
    p = C.form_clu(k=True)
    ui.figura(CL.fig_pca("base", _km("base", p).labels_, f"Clusters K-Means (K={p['k']})"))


def paso_3_5():
    p = C.form_lloyd()
    with st.spinner("Calculando las iteraciones…"):
        pasos, convergio, Z, Zc = CL.lloyd_cacheado("base", p["k"], p["init"], p["seed"],
                                                     p["max_iter"])
    if convergio:
        st.success(f"Convergió en {len(pasos) - 1} iteraciones.")
    else:
        st.info(f"No convergió en {p['max_iter']} iteraciones (se muestran las {len(pasos)} calculadas).")
    # Con un único paso (max_iter = 1) no hay rango que recorrer: st.slider(0, 0) falla
    i = st.slider("Iteración", 0, len(pasos) - 1, 0, key="e04_iter") if len(pasos) > 1 else 0
    espacio = st.empty()
    if st.button("▶ Reproducir", key="e04_play"):
        for j in range(len(pasos)):
            with espacio.container():
                ui.figura(CL.fig_iteracion(Z, Zc, pasos, j, p["k"]))
            time.sleep(0.6)
    else:
        with espacio.container():
            ui.figura(CL.fig_iteracion(Z, Zc, pasos, i, p["k"]))
    cambian = [0] + [int((pasos[j]["etiquetas"] != pasos[j - 1]["etiquetas"]).sum())
                     for j in range(1, len(pasos))]
    ui.tabla(pd.DataFrame({"inercia": [s["inercia"] for s in pasos],
                           "movimiento_max_centroide": [s["movimiento"] for s in pasos],
                           "puntos_que_cambian": cambian}).rename_axis("iteración"),
             "Evolución por iteración", 3)


# ----------------------------------------------------------------------------
# 4. Con la columna de clase
# ----------------------------------------------------------------------------
def paso_4():
    filas = {"sin clase": "base", "con clase (ordinal)": "ord", "con clase (one-hot)": "oh",
             "sólo numéricas": "num"}
    ui.tabla(pd.DataFrame({n: CL.matriz(v)["X_scaled"].shape for n, v in filas.items()},
                          index=["filas", "columnas"]).T, "Dimensiones de cada variante", 0)


def paso_4_1():
    p = C.form_clu(rango=True)
    g = _con_gap("ord", p)
    b = _b("ord", p)
    ui.figura(CL.fig_cinco_metodos(b, g))
    ui.tabla(CL.resumen_k(b, g).rename("K óptimo (con clase ordinal)"),
             "K óptimo por método (con clase ordinal)")


def paso_4_2():
    p = C.form_clu(k=True)
    an = {nombre: CL.analizar(v, _km(v, p).labels_)
          for nombre, v in (("ordinal", "ord"), ("one-hot", "oh"))}
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, (nombre, a) in zip(axes, an.items()):
        CL.heatmap_ct(a["ct"], ax, f"Con clase ({nombre}) — K={p['k']}, contingencia cruda")
    ui.figura(fig)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, (nombre, a) in zip(axes, an.items()):
        CL.heatmap_cm(a["cm"], ax, f"Con clase ({nombre}) — K={p['k']} mapeado\n"
                      f"ARI={a['metricas']['ARI']:.3f}  Pureza={a['metricas']['pureza']:.3f}")
    ui.figura(fig)
    for nombre, a in an.items():
        ui.tabla(metricas_por_clase(a["cm"], CLASS_ORDER),
                 f"Métricas por clase — K={p['k']} con clase {nombre}", 3)


def paso_4_3():
    p = C.form_clu(k=True)
    filas = {"sin clase": "base", "con clase (ordinal)": "ord", "con clase (one-hot)": "oh"}
    t = pd.DataFrame({n: CL.analizar(v, _km(v, p).labels_)["metricas"]
                      for n, v in filas.items()}).T[["silueta", "ARI", "NMI", "acc_mapeo", "pureza"]]
    ui.tabla(t, f"Comparativa de clustering (K={p['k']})")


def paso_4_4():
    p = C.form_clu(k=True)
    ui.figura(CL.fig_pca("ord", _km("ord", p).labels_,
                         f"Clusters K-Means K={p['k']} (con clase ordinal)"))


# ----------------------------------------------------------------------------
# 5. Sólo numéricas
# ----------------------------------------------------------------------------
def paso_5():
    p = C.form_clu(rango=True, k=True)
    b = _b("num", p)
    resumen = pd.concat([CL.resumen_k(_b("base", p), _con_gap("base", p)).rename("K óptimo (todas)"),
                         CL.resumen_k(b, _con_gap("num", p)).rename("K óptimo (sólo numéricas)")],
                        axis=1)
    ui.tabla(resumen, "K óptimo por método: todas las features vs sólo numéricas")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, col in zip(axes, ["inercia", "silhouette", "calinski_harabasz", "davies_bouldin"]):
        ax.plot(b["metrics"].index, b["metrics"][col], marker="o")
        ax.set_xlabel("K")
        ax.set_title(f"{col} (sólo numéricas)")
    ui.figura(fig)
    a = CL.analizar("num", _km("num", p).labels_)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    CL.heatmap_ct(a["ct"], axes[0], f"Clases reales vs clusters (K={p['k']}, sólo numéricas)")
    CL.heatmap_cm(a["cm"], axes[1], "Clusters mapeados a clases (asignación óptima)")
    ui.figura(fig)
    ui.tabla(metricas_por_clase(a["cm"], CLASS_ORDER), f"Métricas por clase — K={p['k']} sólo numéricas", 3)
    ui.tabla(pd.Series(a["metricas"]), "Métricas del clustering")
    ui.tabla(_centroides("num", p, NUMERIC_COLS, ordenar=True),
             "Centroides (sólo numéricas) ordenados por IMC", 2)


RENDER = {
    "1": paso_1, "2": paso_2, "2.1": paso_2_1, "2.2": paso_2_2, "2.3": paso_2_3, "2.4": paso_2_4,
    "3": paso_3, "3.1": paso_3_1, "3.2": paso_3_2, "3.3": paso_3_3, "3.4": paso_3_4,
    "3.5": paso_3_5, "4": paso_4, "4.1": paso_4_1, "4.2": paso_4_2, "4.3": paso_4_3,
    "4.4": paso_4_4, "5": paso_5,
}

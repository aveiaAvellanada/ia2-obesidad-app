"""ETAPA 04 — Clustering K-Means
================================

Sobre la base de clasificación SIN la columna de clase (NObeyesdad).

Protocolo:
1. Features codificadas de la etapa 01, escaladas con StandardScaler (K-Means usa distancia
   euclídea, así que las escalas tienen que ser comparables).
2. Elegir K con CINCO métodos estadísticos, para K en [2, 12]:
   - Método del codo (inercia / WCSS) con detección automática ("kneedle").
   - Coeficiente de silueta (maximizar).
   - Índice de Calinski-Harabasz (maximizar).
   - Índice de Davies-Bouldin (minimizar).
   - Estadístico gap (Tibshirani et al., 2001).
3. Comparar si los métodos coinciden.
4. K-Means con K=7 (número de clases reales) y matriz de confusión clusters vs NObeyesdad.
5. Ejercicio académico: repetir INCLUYENDO la clase (ordinal y one-hot).
6. Variante complementaria: sólo features numéricas.

Pasos:
  1    Datos escalados                 3.4  PCA 2D: clusters vs clases
  2    Barrido de K (tabla)            4    Variantes con clase (ordinal / one-hot)
  2.1  Método del codo                 4.1  Elección de K con clase ordinal (5 métodos)
  2.2  Silueta, CH y DB                4.2  K=7 con clase: contingencia + matrices mapeadas
  2.3  Estadístico gap                 4.3  Tabla comparativa sin/con clase
  2.4  Comparación de los 5 métodos    4.4  PCA 2D con clase ordinal
  3    K=7 vs clases: métricas
  3.1  Tabla de contingencia           5    Variante sólo numéricas
  3.2  Matriz mapeada (húngaro)
  3.3  Centroides en unidades originales
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, confusion_matrix,
                             davies_bouldin_score, normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE
from src import cache
from src.data import TARGET_CLF
from src.evaluation import metricas_por_clase
from src.pasos import Etapa, cli
from src.preprocessing import CLASS_ORDER, NUMERIC_COLS, cargar_clf
from src.salida import subtitulo, tabla

etapa = Etapa("04", "Clustering K-Means", "clustering",
              "Elección de K con 5 métodos, K=7 vs clases reales, con/sin clase, sólo numéricas")

K_RANGE = list(range(2, 13))
_estado = {}


# ----------------------------------------------------------------------------
# Datos y utilidades
# ----------------------------------------------------------------------------
def datos():
    """X (20 features sin la clase), y_true, X_scaled y el scaler."""
    if "datos" not in _estado:
        df = cargar_clf()
        y_true = pd.Categorical(df[TARGET_CLF], categories=CLASS_ORDER)
        X = df.drop(columns=[TARGET_CLF])
        scaler = StandardScaler()
        _estado["datos"] = dict(X=X, y_true=y_true, scaler=scaler, X_scaled=scaler.fit_transform(X))
    return _estado["datos"]


def fit_kmeans(k, X_, seed=RANDOM_STATE):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X_)


def knee_point(ks, values):
    """Codo = punto con máxima distancia perpendicular a la recta entre el primer y último
    punto (curva normalizada a [0,1]). Es la idea del algoritmo 'kneedle'."""
    x = (np.asarray(ks) - ks[0]) / (ks[-1] - ks[0])
    v = np.asarray(values)
    yv = (v - v.min()) / (v.max() - v.min())
    p1, p2 = np.array([x[0], yv[0]]), np.array([x[-1], yv[-1]])
    v12 = p2 - p1
    pts = np.c_[x, yv] - p1
    d = np.abs(v12[0] * pts[:, 1] - v12[1] * pts[:, 0]) / np.linalg.norm(v12)
    return int(np.asarray(ks)[np.argmax(d)])


def gap_statistic(X_, ks, n_refs=10, seed=RANDOM_STATE):
    """Estadístico gap: compara log(inercia) observada con la esperada bajo una referencia
    uniforme. Se elige el menor K tal que gap(K) >= gap(K+1) - s(K+1)."""
    rng = np.random.default_rng(seed)
    lo, hi = X_.min(axis=0), X_.max(axis=0)
    gaps, sks = [], []
    for k in ks:
        ref_log_w = []
        for _ in range(n_refs):
            X_ref = rng.uniform(lo, hi, size=X_.shape)
            ref_log_w.append(np.log(KMeans(k, n_init=3, random_state=seed).fit(X_ref).inertia_))
        ref_log_w = np.asarray(ref_log_w)
        log_w = np.log(KMeans(k, n_init=10, random_state=seed).fit(X_).inertia_)
        gaps.append(ref_log_w.mean() - log_w)
        sks.append(ref_log_w.std(ddof=0) * np.sqrt(1 + 1 / n_refs))
    gaps, sks = np.asarray(gaps), np.asarray(sks)
    ks = list(ks)
    k_gap = None
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            k_gap = ks[i]
            break
    return pd.DataFrame({"gap": gaps, "s_k": sks}, index=pd.Index(ks, name="K")), k_gap


def _barrido(X_):
    """Inercia, silueta, CH y DB para cada K + gap. Devuelve dict con todo y los K óptimos."""
    rows = []
    for k in K_RANGE:
        km = fit_kmeans(k, X_)
        rows.append({"K": k, "inercia": km.inertia_,
                     "silhouette": silhouette_score(X_, km.labels_),
                     "calinski_harabasz": calinski_harabasz_score(X_, km.labels_),
                     "davies_bouldin": davies_bouldin_score(X_, km.labels_)})
    metrics = pd.DataFrame(rows).set_index("K")
    gap_df, k_gap = gap_statistic(X_, K_RANGE)
    resumen = pd.Series({
        "Codo (inercia)": knee_point(K_RANGE, metrics["inercia"].values),
        "Silueta (max)": int(metrics["silhouette"].idxmax()),
        "Calinski-Harabasz (max)": int(metrics["calinski_harabasz"].idxmax()),
        "Davies-Bouldin (min)": int(metrics["davies_bouldin"].idxmin()),
        "Gap statistic": k_gap,
    }, name="K óptimo")
    return dict(metrics=metrics, gap_df=gap_df, k_gap=k_gap, resumen=resumen)


def barrido(variante="base"):
    """'base' = 20 features, 'ord' = + clase ordinal, 'num' = sólo 8 numéricas. Cacheado."""
    X_ = {"base": lambda: datos()["X_scaled"],
          "ord": lambda: con_clase()["X_ord_scaled"],
          "num": lambda: solo_numericas()["X_num"]}[variante]()
    return cache.obtener(f"clu_barrido_{variante}", lambda: _barrido(X_))


def km7():
    if "km7" not in _estado:
        _estado["km7"] = fit_kmeans(7, datos()["X_scaled"])
    return _estado["km7"]


def analizar_clustering(X_scaled_matrix, labels):
    """Contingencia, mapeo húngaro cluster->clase, matriz de confusión y métricas."""
    y_true = datos()["y_true"]
    ct = pd.crosstab(pd.Series(y_true, name="Clase real"), pd.Series(labels, name="Cluster")).reindex(CLASS_ORDER)
    r_ind, c_ind = linear_sum_assignment(-ct.values)
    mapping = {int(ct.columns[c]): ct.index[r] for r, c in zip(r_ind, c_ind)}
    pred_class = pd.Series(labels).map(mapping).values
    cm = confusion_matrix(np.asarray(y_true.astype(str)), pred_class, labels=CLASS_ORDER)
    return {
        "ct": ct, "cm": cm, "mapping": mapping,
        "metricas": {
            "silueta": silhouette_score(X_scaled_matrix, labels),
            "ARI": adjusted_rand_score(y_true, labels),
            "NMI": normalized_mutual_info_score(y_true, labels),
            "acc_mapeo": np.trace(cm) / cm.sum(),
            "pureza": ct.max(axis=0).sum() / ct.values.sum(),
        },
    }


def analisis_base():
    if "analisis_base" not in _estado:
        _estado["analisis_base"] = analizar_clustering(datos()["X_scaled"], km7().labels_)
    return _estado["analisis_base"]


def con_clase():
    """Variantes con la clase añadida: ordinal (21 cols) y one-hot (27 cols), escaladas."""
    if "con_clase" not in _estado:
        d = datos()
        y_ord = pd.Categorical(d["y_true"], categories=CLASS_ORDER, ordered=True).codes
        X_ord = d["X"].assign(NObeyesdad_ord=y_ord)
        X_oh = pd.concat([d["X"], pd.get_dummies(d["y_true"], prefix="clase", dtype="int8")], axis=1)
        X_ord_scaled = StandardScaler().fit_transform(X_ord)
        X_oh_scaled = StandardScaler().fit_transform(X_oh)
        km7_ord = KMeans(n_clusters=7, n_init=10, random_state=RANDOM_STATE).fit(X_ord_scaled)
        km7_oh = KMeans(n_clusters=7, n_init=10, random_state=RANDOM_STATE).fit(X_oh_scaled)
        _estado["con_clase"] = dict(
            X_ord=X_ord, X_oh=X_oh, X_ord_scaled=X_ord_scaled, X_oh_scaled=X_oh_scaled,
            km7_ord=km7_ord, km7_oh=km7_oh,
            analisis_ord=analizar_clustering(X_ord_scaled, km7_ord.labels_),
            analisis_oh=analizar_clustering(X_oh_scaled, km7_oh.labels_),
        )
    return _estado["con_clase"]


def solo_numericas():
    if "num" not in _estado:
        d = datos()
        scaler_num = StandardScaler().fit(d["X"][NUMERIC_COLS])
        X_num = scaler_num.transform(d["X"][NUMERIC_COLS])
        km7n = fit_kmeans(7, X_num)
        _estado["num"] = dict(X_num=X_num, scaler_num=scaler_num, km7n=km7n,
                              analisis=analizar_clustering(X_num, km7n.labels_))
    return _estado["num"]


def _heatmap_ct(ct, ax, titulo):
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Cluster (número arbitrario)")
    ax.set_ylabel("Clase real")
    ax.set_title(titulo)


def _heatmap_cm(cm, ax, titulo):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax)
    ax.set_xlabel("Clase asignada al cluster")
    ax.set_ylabel("Clase real")
    ax.set_title(titulo)
    ax.tick_params(axis="x", rotation=45)


def _pca_2d(X_, labels, titulo_clusters, nombre_fig):
    y_true = datos()["y_true"]
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(X_)
    ratio = pca.explained_variance_ratio_
    print(f"Varianza explicada por PC1+PC2: {ratio.sum()*100:.2f}% "
          f"(PC1: {ratio[0]*100:.1f}%, PC2: {ratio[1]*100:.1f}%)")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=labels, palette="tab10", s=12, ax=axes[0], legend="full")
    axes[0].set_title(titulo_clusters)
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y_true.astype(str), hue_order=CLASS_ORDER,
                    palette="tab10", s=12, ax=axes[1])
    axes[1].set_title("Clases reales NObeyesdad")
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, markerscale=1.5)
    etapa.figura(fig, nombre_fig)


# ----------------------------------------------------------------------------
# 1–2. Datos y barrido de K
# ----------------------------------------------------------------------------
@etapa.paso("1", "Datos: features sin la clase, escaladas con StandardScaler")
def paso_1_datos():
    d = datos()
    print("X:", d["X_scaled"].shape, "| columnas:", d["X"].columns.tolist())


@etapa.paso("2", "Barrido de K ∈ [2, 12]: inercia, silueta, Calinski-Harabasz y Davies-Bouldin")
def paso_2_barrido():
    b = barrido("base")
    tabla(b["metrics"], "Métricas internas por K", 3)


@etapa.paso("2.1", "Método del codo (con detección automática del codo)", figuras=1)
def paso_2_1_codo():
    b = barrido("base")
    k_elbow = b["resumen"]["Codo (inercia)"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(b["metrics"].index, b["metrics"]["inercia"], marker="o")
    ax.axvline(k_elbow, c="r", ls="--", label=f"codo detectado: K={k_elbow}")
    ax.set_xlabel("K")
    ax.set_ylabel("Inercia (WCSS)")
    ax.set_title("Método del codo")
    ax.legend()
    etapa.figura(fig, "2.1_codo")
    print("K por método del codo:", k_elbow)


@etapa.paso("2.2", "Silueta, Calinski-Harabasz y Davies-Bouldin vs K", figuras=1)
def paso_2_2_sil_ch_db():
    b = barrido("base")
    m, r = b["metrics"], b["resumen"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for ax, (col, best, sense) in zip(axes, [
            ("silhouette", r["Silueta (max)"], "max"),
            ("calinski_harabasz", r["Calinski-Harabasz (max)"], "max"),
            ("davies_bouldin", r["Davies-Bouldin (min)"], "min")]):
        ax.plot(m.index, m[col], marker="o")
        ax.axvline(best, c="r", ls="--", label=f"{sense} en K={best}")
        ax.set_xlabel("K")
        ax.set_title(col)
        ax.legend()
    etapa.figura(fig, "2.2_silueta_ch_db")
    print(f"K por silueta: {r['Silueta (max)']} | Calinski-Harabasz: {r['Calinski-Harabasz (max)']} "
          f"| Davies-Bouldin: {r['Davies-Bouldin (min)']}")


@etapa.paso("2.3", "Estadístico gap", figuras=1)
def paso_2_3_gap():
    b = barrido("base")
    gap_df, k_gap = b["gap_df"], b["k_gap"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(gap_df.index, gap_df["gap"], yerr=gap_df["s_k"], marker="o", capsize=3)
    if k_gap is not None:
        ax.axvline(k_gap, c="r", ls="--", label=f"gap: K={k_gap}")
        ax.legend()
    ax.set_xlabel("K")
    ax.set_ylabel("gap(K)")
    ax.set_title("Estadístico gap")
    etapa.figura(fig, "2.3_gap")
    print("K por estadístico gap:", k_gap, "(None = ningún K en el rango cumple el criterio)")
    tabla(gap_df, "gap(K) y s_k", 3)


@etapa.paso("2.4", "Comparación de los 5 métodos para elegir K")
def paso_2_4_comparacion():
    tabla(barrido("base")["resumen"], "K óptimo según cada método")


# ----------------------------------------------------------------------------
# 3. K = 7 vs clases reales
# ----------------------------------------------------------------------------
@etapa.paso("3", "K-Means con K=7 vs clases reales: silueta, ARI, NMI")
def paso_3_k7():
    d = datos()
    labels7 = km7().labels_
    print(f"Silueta (K=7): {silhouette_score(d['X_scaled'], labels7):.3f}")
    print(f"ARI  (clusters vs NObeyesdad): {adjusted_rand_score(d['y_true'], labels7):.3f}")
    print(f"NMI  (clusters vs NObeyesdad): {normalized_mutual_info_score(d['y_true'], labels7):.3f}")


@etapa.paso("3.1", "Matriz de confusión clusters × clases (tabla de contingencia)", figuras=1)
def paso_3_1_contingencia():
    ct = analisis_base()["ct"]
    tabla(ct, "Clases reales (filas) vs clusters K=7 (columnas)", 0)
    fig, ax = plt.subplots(figsize=(9, 6))
    _heatmap_ct(ct, ax, "Clases reales (filas) vs clusters K-Means K=7 (columnas)")
    etapa.figura(fig, "3.1_contingencia_k7")


@etapa.paso("3.2", "Matriz con clusters reasignados a clases (algoritmo húngaro)", figuras=1)
def paso_3_2_hungaro():
    an = analisis_base()
    fig, ax = plt.subplots(figsize=(9, 7))
    _heatmap_cm(an["cm"], ax, "Matriz de confusión — clusters K=7 mapeados a clases (asignación óptima)")
    etapa.figura(fig, "3.2_matriz_mapeada_k7")
    print("Mapeo cluster -> clase:", an["mapping"])
    print(f"Accuracy con mapeo óptimo: {an['metricas']['acc_mapeo']:.3f}")
    print(f"Pureza de los clusters:    {an['metricas']['pureza']:.3f}")
    tabla(metricas_por_clase(an["cm"], CLASS_ORDER),
          "Métricas por clase — K=7 sin la clase", 3)


@etapa.paso("3.3", "¿Qué separa a cada cluster? Centroides en unidades originales")
def paso_3_3_centroides():
    d = datos()
    an = analisis_base()
    centroids = pd.DataFrame(d["scaler"].inverse_transform(km7().cluster_centers_), columns=d["X"].columns)
    centroids.index.name = "Cluster"
    centroids["clase_mapeada"] = [an["mapping"][i] for i in centroids.index]
    centroids["n"] = pd.Series(km7().labels_).value_counts().sort_index().values
    centroids["BMI_centroide"] = centroids["Weight"] / centroids["Height"] ** 2
    tabla(centroids[["n", "clase_mapeada", "Gender", "Age", "Height", "Weight", "BMI_centroide",
                     "family_history_with_overweight", "FAVC", "FCVC", "CAEC", "FAF",
                     "MTRANS_Public_Transportation"]], "Centroides (K=7) en unidades originales", 2)


@etapa.paso("3.4", "Visualización 2D (PCA): clusters K=7 vs clases reales", figuras=1)
def paso_3_4_pca():
    _pca_2d(datos()["X_scaled"], km7().labels_, "Clusters K-Means (K=7)", "3.4_pca_2d")


# ----------------------------------------------------------------------------
# 4. Con la columna de clase (ejercicio académico)
# ----------------------------------------------------------------------------
@etapa.paso("4", "Variantes con la columna de clase: ordinal (21 cols) y one-hot (27 cols)")
def paso_4_con_clase():
    c = con_clase()
    print(f"X_con_ord: {c['X_ord'].shape} | X_con_oh: {c['X_oh'].shape}")


@etapa.paso("4.1", "Elección de K para la variante con clase ordinal (5 métodos)", figuras=1)
def paso_4_1_k_con_clase():
    b = barrido("ord")
    m, r, gap_df, k_gap = b["metrics"], b["resumen"], b["gap_df"], b["k_gap"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    paneles = [
        (axes[0, 0], "inercia", r["Codo (inercia)"], "codo", "1. Método del codo", "Inercia (WCSS)"),
        (axes[0, 1], "silhouette", r["Silueta (max)"], "max", "2. Coeficiente de silueta", "Silueta"),
        (axes[0, 2], "calinski_harabasz", r["Calinski-Harabasz (max)"], "max", "3. Calinski-Harabasz", "Calinski-Harabasz"),
        (axes[1, 0], "davies_bouldin", r["Davies-Bouldin (min)"], "min", "4. Davies-Bouldin", "Davies-Bouldin"),
    ]
    for ax, col, best, sense, titulo_ax, ylabel in paneles:
        ax.plot(m.index, m[col], marker="o", color="steelblue")
        ax.axvline(best, c="r", ls="--", label=f"{sense}: K={best}")
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        ax.set_title(titulo_ax)
        ax.legend()
    ax = axes[1, 1]
    ax.errorbar(gap_df.index, gap_df["gap"], yerr=gap_df["s_k"], marker="o", capsize=3, color="steelblue")
    if k_gap is not None:
        ax.axvline(k_gap, c="r", ls="--", label=f"gap: K={k_gap}")
        ax.legend()
    ax.set_xlabel("K")
    ax.set_ylabel("gap(K)")
    ax.set_title("5. Estadístico gap")
    axes[1, 2].axis("off")
    etapa.figura(fig, "4.1_cinco_metodos_con_clase_ordinal")
    tabla(r.rename("K óptimo (con clase ordinal)"), "K óptimo por método (con clase ordinal)")


@etapa.paso("4.2", "K=7 con clase: tablas de contingencia crudas y matrices mapeadas", figuras=2)
def paso_4_2_k7_con_clase():
    c = con_clase()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, (nombre, an) in zip(axes, [("ordinal", c["analisis_ord"]), ("one-hot", c["analisis_oh"])]):
        _heatmap_ct(an["ct"], ax, f"Con clase ({nombre}) — K=7, tabla de contingencia cruda")
    etapa.figura(fig, "4.2_contingencia_cruda_con_clase")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, (nombre, an) in zip(axes, [("ordinal", c["analisis_ord"]), ("one-hot", c["analisis_oh"])]):
        _heatmap_cm(an["cm"], ax, f"Con clase ({nombre}) — K=7 mapeado\n"
                    f"ARI={an['metricas']['ARI']:.3f}  Pureza={an['metricas']['pureza']:.3f}")
    etapa.figura(fig, "4.2_matrices_mapeadas_con_clase")
    print("Mapeo Ordinal:", c["analisis_ord"]["mapping"])
    print("Mapeo One-Hot:", c["analisis_oh"]["mapping"])
    for nombre, an in [("ordinal", c["analisis_ord"]), ("one-hot", c["analisis_oh"])]:
        tabla(metricas_por_clase(an["cm"], CLASS_ORDER),
              f"Métricas por clase — K=7 con clase {nombre}", 3)


@etapa.paso("4.3", "Tabla comparativa final: sin clase vs con clase (ordinal / one-hot)")
def paso_4_3_tabla_comparativa():
    c = con_clase()
    df_comp = pd.DataFrame({
        "sin clase": analisis_base()["metricas"],
        "con clase (ordinal)": c["analisis_ord"]["metricas"],
        "con clase (one-hot)": c["analisis_oh"]["metricas"],
    }).T[["silueta", "ARI", "NMI", "acc_mapeo", "pureza"]]
    tabla(df_comp, "Comparativa de clustering (K=7)")


@etapa.paso("4.4", "Visualización 2D (PCA) para la variante con clase ordinal", figuras=1)
def paso_4_4_pca_con_clase():
    c = con_clase()
    _pca_2d(c["X_ord_scaled"], c["km7_ord"].labels_, "Clusters K-Means K=7 (con clase ordinal)",
            "4.4_pca_2d_con_clase_ordinal")


# ----------------------------------------------------------------------------
# 5. Sólo numéricas
# ----------------------------------------------------------------------------
@etapa.paso("5", "Variante complementaria: sólo las 8 features numéricas", figuras=2)
def paso_5_solo_numericas():
    n = solo_numericas()
    b = barrido("num")
    tabla(pd.concat([barrido("base")["resumen"].rename("K óptimo (todas)"),
                     b["resumen"].rename("K óptimo (sólo numéricas)")], axis=1),
          "K óptimo por método: todas las features vs sólo numéricas")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, col in zip(axes, ["inercia", "silhouette", "calinski_harabasz", "davies_bouldin"]):
        ax.plot(b["metrics"].index, b["metrics"][col], marker="o")
        ax.set_xlabel("K")
        ax.set_title(f"{col} (sólo numéricas)")
    etapa.figura(fig, "5_metricas_solo_numericas")

    an = n["analisis"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    _heatmap_ct(an["ct"], axes[0], "Clases reales vs clusters (K=7, sólo numéricas)")
    _heatmap_cm(an["cm"], axes[1], "Clusters mapeados a clases (asignación óptima)")
    etapa.figura(fig, "5_matrices_solo_numericas")
    tabla(metricas_por_clase(an["cm"], CLASS_ORDER),
          "Métricas por clase — K=7 sólo numéricas", 3)
    for k, v in an["metricas"].items():
        print(f"   {k:<10}: {v:.3f}")

    cent = pd.DataFrame(n["scaler_num"].inverse_transform(n["km7n"].cluster_centers_), columns=NUMERIC_COLS)
    cent.index.name = "Cluster"
    cent.insert(0, "clase_mapeada", [an["mapping"][i] for i in cent.index])
    cent.insert(0, "n", pd.Series(n["km7n"].labels_).value_counts().sort_index().values)
    cent["BMI_centroide"] = cent["Weight"] / cent["Height"] ** 2
    tabla(cent.sort_values("BMI_centroide"), "Centroides (sólo numéricas) ordenados por IMC", 2)


if __name__ == "__main__":
    sys.exit(cli(etapa))

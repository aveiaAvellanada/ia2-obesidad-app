"""Núcleo de clustering de la app (sin Streamlit): matrices, barridos, análisis, PCA y Lloyd."""
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, confusion_matrix,
                             davies_bouldin_score, normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE
from src.data import TARGET_CLF
from src.preprocessing import CLASS_ORDER, NUMERIC_COLS, cargar_clf
from etapas.e04_clustering import _heatmap_cm as heatmap_cm
from etapas.e04_clustering import _heatmap_ct as heatmap_ct
from etapas.e04_clustering import gap_statistic, knee_point

VARIANTES = ("base", "num", "ord", "oh")


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _base():
    df = cargar_clf()
    y = pd.Categorical(df[TARGET_CLF], categories=CLASS_ORDER)
    return df.drop(columns=[TARGET_CLF]), y


@lru_cache(maxsize=4)
def matriz(variante):
    """'base' = 20 features sin la clase; 'num' = 8 numéricas; 'ord' = base + clase ordinal;
    'oh' = base + clase one-hot. Siempre escaladas con StandardScaler."""
    X, y = _base()
    if variante == "base":
        Xv = X
    elif variante == "num":
        Xv = X[NUMERIC_COLS]
    elif variante == "ord":
        Xv = X.assign(NObeyesdad_ord=pd.Categorical(y, categories=CLASS_ORDER, ordered=True).codes)
    elif variante == "oh":
        Xv = pd.concat([X, pd.get_dummies(y, prefix="clase", dtype="int8")], axis=1)
    else:
        raise KeyError(variante)
    scaler = StandardScaler()
    return dict(X=Xv, X_scaled=scaler.fit_transform(Xv), scaler=scaler, y_true=y)


@lru_cache(maxsize=64)
def ajustar(variante, k, n_init, init, seed):
    return KMeans(n_clusters=int(k), n_init=int(n_init), init=init,
                  random_state=int(seed)).fit(matriz(variante)["X_scaled"])


def analizar(variante, labels):
    """Contingencia, mapeo húngaro cluster->clase, matriz de confusión y métricas.
    Funciona con cualquier K: los clusters que sobran quedan 'sin_clase' (cuentan como error)."""
    m = matriz(variante)
    y_true = m["y_true"]
    ct = pd.crosstab(pd.Series(y_true, name="Clase real"),
                     pd.Series(labels, name="Cluster")).reindex(CLASS_ORDER)
    filas, cols = linear_sum_assignment(-ct.values)
    mapping = {int(ct.columns[c]): ct.index[r] for r, c in zip(filas, cols)}
    pred = pd.Series(labels).map(mapping).fillna("sin_clase").to_numpy()
    cm = confusion_matrix(np.asarray(y_true.astype(str)), pred, labels=CLASS_ORDER)
    return {
        "ct": ct, "cm": cm, "mapping": mapping,
        "metricas": {
            "silueta": silhouette_score(m["X_scaled"], labels),
            "ARI": adjusted_rand_score(y_true, labels),
            "NMI": normalized_mutual_info_score(y_true, labels),
            "acc_mapeo": float(np.trace(cm) / len(y_true)),
            "pureza": float(ct.max(axis=0).sum() / ct.values.sum()),
        },
    }


# ----------------------------------------------------------------------------
# Elección de K
# ----------------------------------------------------------------------------
@lru_cache(maxsize=16)
def barrido(variante, k_min, k_max, n_init, init, seed):
    X = matriz(variante)["X_scaled"]
    ks = list(range(int(k_min), int(k_max) + 1))
    filas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=int(n_init), init=init, random_state=int(seed)).fit(X)
        filas.append({"K": k, "inercia": km.inertia_,
                      "silhouette": silhouette_score(X, km.labels_),
                      "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
                      "davies_bouldin": davies_bouldin_score(X, km.labels_)})
    return dict(metrics=pd.DataFrame(filas).set_index("K"), ks=ks)


@lru_cache(maxsize=16)
def gap(variante, k_min, k_max, n_refs, seed):
    ks = list(range(int(k_min), int(k_max) + 1))
    return gap_statistic(matriz(variante)["X_scaled"], ks, n_refs=int(n_refs), seed=int(seed))


def resumen_k(b, g):
    m = b["metrics"]
    return pd.Series({
        "Codo (inercia)": knee_point(b["ks"], m["inercia"].values),
        "Silueta (max)": int(m["silhouette"].idxmax()),
        "Calinski-Harabasz (max)": int(m["calinski_harabasz"].idxmax()),
        "Davies-Bouldin (min)": int(m["davies_bouldin"].idxmin()),
        "Gap statistic": g[1],
    }, name="K óptimo")


def fig_cinco_metodos(b, g):
    """Panel 2x3 con los cinco métodos de elección de K (como el paso 4.1 de la etapa 04)."""
    m, (gap_df, k_gap) = b["metrics"], g
    r = resumen_k(b, g)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    paneles = [
        (axes[0, 0], "inercia", r["Codo (inercia)"], "codo", "1. Método del codo", "Inercia (WCSS)"),
        (axes[0, 1], "silhouette", r["Silueta (max)"], "max", "2. Coeficiente de silueta", "Silueta"),
        (axes[0, 2], "calinski_harabasz", r["Calinski-Harabasz (max)"], "max",
         "3. Calinski-Harabasz", "Calinski-Harabasz"),
        (axes[1, 0], "davies_bouldin", r["Davies-Bouldin (min)"], "min",
         "4. Davies-Bouldin", "Davies-Bouldin"),
    ]
    for ax, col, best, sentido, titulo, ylabel in paneles:
        ax.plot(m.index, m[col], marker="o", color="steelblue")
        ax.axvline(best, c="r", ls="--", label=f"{sentido}: K={best}")
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        ax.set_title(titulo)
        ax.legend()
    ax = axes[1, 1]
    ax.errorbar(gap_df.index, gap_df["gap"], yerr=gap_df["s_k"], marker="o", capsize=3,
                color="steelblue")
    if k_gap is not None:
        ax.axvline(k_gap, c="r", ls="--", label=f"gap: K={k_gap}")
        ax.legend()
    ax.set_xlabel("K")
    ax.set_ylabel("gap(K)")
    ax.set_title("5. Estadístico gap")
    axes[1, 2].axis("off")
    return fig


# ----------------------------------------------------------------------------
# PCA 2D
# ----------------------------------------------------------------------------
@lru_cache(maxsize=4)
def pca2d(variante):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(matriz(variante)["X_scaled"]), pca


def fig_pca(variante, labels, titulo):
    Z, pca = pca2d(variante)
    y_true = matriz(variante)["y_true"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=labels, palette="tab10", s=12, ax=axes[0],
                    legend="full")
    axes[0].set_title(titulo)
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y_true.astype(str), hue_order=CLASS_ORDER,
                    palette="tab10", s=12, ax=axes[1])
    axes[1].set_title("Clases reales NObeyesdad")
    r = pca.explained_variance_ratio_
    for ax in axes:
        ax.set_xlabel(f"PC1 ({r[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({r[1] * 100:.1f}%)")
        ax.legend(fontsize=7, markerscale=1.5)
    return fig


# ----------------------------------------------------------------------------
# Algoritmo de Lloyd paso a paso
# ----------------------------------------------------------------------------
def lloyd(X, k, init="k-means++", seed=RANDOM_STATE, max_iter=50, centros_iniciales=None):
    """Paso i = (centroides c_i, asignación de cada punto a su c_i más cercano, inercia).
    El paso siguiente recalcula cada centroide como la media de su grupo (un cluster vacío
    conserva su centroide). Termina cuando las etiquetas no cambian (convergió) o al llegar a
    `max_iter` pasos. Devuelve (pasos, convergio); cada paso es un dict con centroides,
    etiquetas, inercia y movimiento (desplazamiento máximo de un centroide respecto al paso
    anterior)."""
    X = np.asarray(X, dtype=float)
    if centros_iniciales is not None:
        centros = np.array(centros_iniciales, dtype=float)
    elif init == "k-means++":
        centros = np.array(kmeans_plusplus(X, k, random_state=seed)[0], dtype=float)
    else:
        rng = np.random.default_rng(seed)
        centros = X[rng.choice(len(X), size=k, replace=False)].astype(float)
    k = len(centros)
    pasos, previas, convergio = [], None, False
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2)
        etiquetas = d2.argmin(axis=1)
        movimiento = (float(np.linalg.norm(centros - pasos[-1]["centroides"], axis=1).max())
                      if pasos else 0.0)
        pasos.append(dict(centroides=centros.copy(), etiquetas=etiquetas,
                          inercia=float(d2[np.arange(len(X)), etiquetas].sum()),
                          movimiento=movimiento))
        if previas is not None and np.array_equal(etiquetas, previas):
            convergio = True
            break
        previas = etiquetas
        nuevos = centros.copy()
        for j in range(k):
            miembros = X[etiquetas == j]
            if len(miembros):
                nuevos[j] = miembros.mean(axis=0)
        centros = nuevos
    return pasos, convergio


@lru_cache(maxsize=16)
def lloyd_cacheado(variante, k, init, seed, max_iter):
    """Lloyd sobre una variante + la proyección PCA 2D de los puntos y de los centroides."""
    pasos, convergio = lloyd(matriz(variante)["X_scaled"], int(k), init, int(seed), int(max_iter))
    Z, pca = pca2d(variante)
    return pasos, convergio, Z, [pca.transform(p["centroides"]) for p in pasos]


def fig_iteracion(Z, Zc_all, pasos, i, k):
    """Izquierda: puntos coloreados por cluster y centroides (X) con su trayectoria hasta el
    paso i. Derecha: inercia por iteración con el paso actual marcado."""
    paso = pasos[i]
    vmax = max(len(Zc_all[0]) - 1, 1)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax.scatter(Z[:, 0], Z[:, 1], c=paso["etiquetas"], cmap="tab20", vmin=0, vmax=vmax,
               s=10, alpha=0.6)
    for j in range(len(Zc_all[0])):
        tray = np.array([zc[j] for zc in Zc_all[:i + 1]])
        ax.plot(tray[:, 0], tray[:, 1], "-", color="black", lw=1, alpha=0.6)
    actual = Zc_all[i]
    ax.scatter(actual[:, 0], actual[:, 1], c=range(len(actual)), cmap="tab20", vmin=0, vmax=vmax,
               marker="X", s=240, edgecolors="black", linewidths=1.6, zorder=3)
    ax.set_xlim(Z[:, 0].min() - 0.5, Z[:, 0].max() + 0.5)
    ax.set_ylim(Z[:, 1].min() - 0.5, Z[:, 1].max() + 0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"K = {k} · iteración {i} de {len(pasos) - 1}")

    ine = [p["inercia"] for p in pasos]
    ax2.plot(range(len(ine)), ine, marker="o", color="steelblue")
    ax2.scatter([i], [ine[i]], s=120, color="red", zorder=3)
    ax2.set_xlabel("Iteración")
    ax2.set_ylabel("Inercia (WCSS)")
    ax2.set_title("Inercia por iteración")
    ax2.grid(alpha=0.3)
    return fig

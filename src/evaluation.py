"""Helpers de evaluación y gráficos compartidos por las etapas de modelado."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import NullFormatter, ScalarFormatter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config import RANDOM_STATE  # noqa: E402  (misma semilla en todo el proyecto)

# Colores fijos para las curvas train / test (par azul-naranja, seguro para daltonismo)
COLOR_TRAIN = "#1f77b4"
COLOR_TEST = "#ff7f0e"


def clf_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """R² ajustado. n = nº de observaciones, p = nº de predictores."""
    if n - p - 1 <= 0:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def reg_metrics(y_true, y_pred, n_features: int | None = None) -> dict:
    """MAE, RMSE, R2 y R2_adj. Mantener el orden de claves.
    Si n_features es None, R2_adj = nan (retrocompatibilidad)."""
    r2 = r2_score(y_true, y_pred)
    r2_adj = adjusted_r2(r2, len(y_true), n_features) if n_features is not None else float("nan")
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2,
        "R2_adj": r2_adj,
    }


def tree_summary(model, feature_names, nombre: str) -> dict:
    """Resumen de un árbol ya entrenado.
    Claves exactas: {"modelo", "profundidad", "n_hojas", "raiz_atributo", "raiz_umbral"}
    profundidad -> model.get_depth() ; n_hojas -> model.get_n_leaves()
    raiz_atributo -> feature_names[model.tree_.feature[0]]
    raiz_umbral   -> round(float(model.tree_.threshold[0]), 3)"""
    feature_names = list(feature_names)
    idx_raiz = int(model.tree_.feature[0])
    # Un árbol de una sola hoja no tiene split en la raíz (feature = -2)
    sin_split = idx_raiz < 0
    return {
        "modelo": nombre,
        "profundidad": int(model.get_depth()),
        "n_hojas": int(model.get_n_leaves()),
        "raiz_atributo": None if sin_split else feature_names[idx_raiz],
        "raiz_umbral": None if sin_split else round(float(model.tree_.threshold[0]), 3),
    }


def _annotate_optimum(ax, texto, xy, text_xy, va):
    """Cuadro de texto en coordenadas de ejes (text_xy) con flecha al punto xy (datos)."""
    ax.annotate(
        texto,
        xy=xy,
        xycoords="data",
        xytext=text_xy,
        textcoords="axes fraction",
        ha="left",
        va=va,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
    )


def plot_k_curve(cv_results, param="knn__n_neighbors", ylabel="macro-F1 (CV)", ax=None,
                 negate=False):
    """Curva score de CV vs K para KNN, a partir de gs.cv_results_.
    Para cada valor de K toma el MEJOR score entre las variantes de 'weights'.
    Marca el K ganador con línea vertical punteada y anotación con el valor.
    Devuelve (ax, mejor_k).

    negate=True dibuja -score (para scorings 'neg_*' de sklearn, p. ej.
    neg_root_mean_squared_error → RMSE). El mejor K se elige siempre sobre el
    score original de sklearn (mayor = mejor), así coincide con gs.best_params_."""
    res = pd.DataFrame({
        "k": np.asarray(cv_results[f"param_{param}"], dtype=int),
        "score": np.asarray(cv_results["mean_test_score"], dtype=float),
    })
    curva = res.groupby("k", sort=True)["score"].max()
    mejor_k = int(curva.idxmax())
    mejor_score = float(curva.max())

    if negate:
        curva = -curva
        mejor_score = -mejor_score

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curva.index, curva.values, marker="o", ms=5, lw=2, color=COLOR_TRAIN)
    ax.axvline(mejor_k, ls="--", lw=1.2, color="gray")
    # El texto va en la banda vacía: arriba si el óptimo es un máximo (todo lo demás
    # queda por debajo), abajo si es un mínimo; y al lado contrario del K ganador.
    frac_x = (mejor_k - curva.index.min()) / max(curva.index.max() - curva.index.min(), 1)
    _annotate_optimum(
        ax,
        f"mejor K = {mejor_k}\n{ylabel.split(' (')[0]} = {mejor_score:.3f}",
        xy=(mejor_k, mejor_score),
        text_xy=(0.6 if frac_x < 0.5 else 0.15, 0.08 if negate else 0.92),
        va="bottom" if negate else "top",
    )
    ax.set_xticks(curva.index)
    ax.set_xlabel("K (número de vecinos)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"KNN: {ylabel} según K")
    ax.grid(alpha=0.3)
    return ax, mejor_k


def plot_pruning_curve(alphas, score_train, score_test, ax=None, ylabel="accuracy"):
    """Curva de score en train y test vs ccp_alpha (eje x en escala log).
    Dos líneas rotuladas 'train' y 'test' + leyenda."""
    alphas = np.asarray(alphas, dtype=float)
    score_train = np.asarray(score_train, dtype=float)
    score_test = np.asarray(score_test, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(alphas, score_train, marker="o", ms=5, lw=2, color=COLOR_TRAIN, label="train")
    ax.plot(alphas, score_test, marker="s", ms=5, lw=2, color=COLOR_TEST, label="test")

    # Mejor alpha según test (primer máximo)
    i_best = int(np.argmax(score_test))
    ax.axvline(alphas[i_best], ls="--", lw=1.2, color="gray")
    # Esquina inferior izquierda: las curvas bajan al crecer alpha, así que queda vacía
    _annotate_optimum(
        ax,
        f"mejor test: α = {alphas[i_best]:.2e}\n{ylabel} = {score_test[i_best]:.3f}",
        xy=(alphas[i_best], score_test[i_best]),
        text_xy=(0.05, 0.08),
        va="bottom",
    )
    ax.set_xscale("log")
    ax.set_xlabel("ccp_alpha (escala log)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Postpoda: {ylabel} en train y test según ccp_alpha")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return ax


def plot_confusion(y_true, y_pred, labels, title, ax=None, normalize=False, vmin=None, vmax=None):
    """Matriz de confusión. vmin/vmax fijan la escala de color (útil para comparar
    varios paneles con la misma escala)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    return ax


def plot_pred_vs_real(y_true, y_pred, title, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=10, alpha=0.5)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicho")
    ax.set_title(title)
    return ax


def results_table(results: dict) -> pd.DataFrame:
    """results: {nombre_modelo: {metrica: valor}} -> DataFrame ordenado."""
    return pd.DataFrame(results).T.round(4)


def plot_n_estimators_curve(df, col_cv, col_test, ylabel, titulo, col_std=None,
                            col_tiempo="tiempo_fit_s", n_proyecto=None, mejor="max",
                            captura=0.90):
    """Dos paneles para el barrido del número de árboles del Random Forest.

    Izquierda: rendimiento (validación cruzada y test) según n_estimators.
    Derecha:   coste, es decir el tiempo de entrenamiento según n_estimators.

    Esa pareja es el argumento completo: la métrica se estanca a partir de cierto
    número de árboles mientras el coste sigue creciendo, así que añadir más árboles
    deja de compensar (nunca empeora el modelo, sólo se paga más).

    df    : DataFrame indexado por n_estimators.
    mejor : "max" si la métrica es mejor cuanto más alta (macro-F1, accuracy),
            "min" si es mejor cuanto más baja (RMSE).
    captura : fracción de la mejora total que define el punto "suficiente".

    El punto suficiente es el número de árboles MÁS PEQUEÑO que ya consigue esa
    fracción (90% por defecto) de toda la mejora del barrido, medida entre el peor
    valor (el del bosque más pequeño) y el mejor. Es un criterio transparente y
    fácil de defender, y no se deja engañar por el ruido entre folds: da igual que
    la curva siga subiendo unas milésimas, lo que se mide es cuánta de la mejora
    disponible ya está conseguida.

    Devuelve (fig, n_suficiente).
    """
    n = np.asarray(df.index, dtype=int)
    cv = np.asarray(df[col_cv], dtype=float)
    test = np.asarray(df[col_test], dtype=float)

    i_opt = int(np.nanargmax(cv)) if mejor == "max" else int(np.nanargmin(cv))
    optimo = float(cv[i_opt])
    partida = float(cv[0])                      # bosque más pequeño del barrido
    mejora_total = optimo - partida             # con signo: + si sube, - si baja

    if not np.isfinite(mejora_total) or mejora_total == 0:
        i_suf = i_opt
    else:
        umbral = partida + mejora_total * captura
        dentro = cv >= umbral if mejor == "max" else cv <= umbral
        i_suf = int(np.argmax(dentro))          # el primero (menor n) que lo alcanza
    n_suficiente = int(n[i_suf])
    pct = int(round(captura * 100))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))

    # --- Panel 1: rendimiento ---
    ax1.plot(n, cv, marker="o", ms=6, lw=2, color=COLOR_TRAIN, label="validación cruzada (train)")
    if col_std is not None and col_std in df.columns:
        s = np.asarray(df[col_std], dtype=float)
        ax1.fill_between(n, cv - s, cv + s, color=COLOR_TRAIN, alpha=0.15,
                         label="± 1 desv. típica entre folds")
    ax1.plot(n, test, marker="s", ms=6, lw=2, color=COLOR_TEST, label="test")
    ax1.axvline(n_suficiente, ls="--", lw=1.3, color="gray",
                label=f"{n_suficiente} árboles: {pct}% de la mejora")
    if n_proyecto is not None:
        ax1.axvline(n_proyecto, ls=":", lw=1.6, color="#2ca02c",
                    label=f"el proyecto usa {n_proyecto}")
    # La curva sube (mejor="max") o baja (mejor="min"), así que el hueco libre del
    # panel cambia de lado: leyenda y anotación se colocan donde no tapen los datos.
    if mejor == "max":
        loc_leyenda, xy_texto = "upper left", (0.42, 0.03)
    else:
        loc_leyenda, xy_texto = "upper right", (0.05, 0.03)
    _annotate_optimum(
        ax1,
        f"con {n_suficiente} árboles: {ylabel} = {cv[i_suf]:.4f}\n"
        f"(el {pct}% de toda la mejora del barrido)\n"
        f"mejor = {optimo:.4f} con {int(n[i_opt])} árboles",
        xy=(n_suficiente, cv[i_suf]),
        text_xy=xy_texto,
        va="bottom",
    )
    ax1.set_xlabel("n_estimators (número de árboles, escala log)")
    ax1.set_ylabel(ylabel)
    ax1.set_title(titulo)
    ax1.legend(fontsize=8.5, loc=loc_leyenda, framealpha=0.92)
    ax1.margins(y=0.16)

    # --- Panel 2: coste ---
    if col_tiempo in df.columns:
        t = np.asarray(df[col_tiempo], dtype=float)
        ax2.plot(n, t, marker="D", ms=6, lw=2, color="#9467bd")
        for xi, yi in zip(n, t):
            ax2.annotate(f"{yi:.2f}s", (xi, yi), textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=8)
        ax2.set_ylabel("segundos en entrenar")
        ax2.set_title("Coste: tiempo de entrenamiento")
        ax2.margins(y=0.18)
    ax2.set_xlabel("n_estimators (número de árboles, escala log)")

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks(n)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.3, which="major")
    return fig, n_suficiente

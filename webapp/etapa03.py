"""Etapa 03 — Regresión de Weight."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.tree import plot_tree

from src.evaluation import plot_k_curve, plot_n_estimators_curve, reg_metrics, tree_summary
from webapp import controles as C
from webapp import modelos as M
from webapp import ui


def _entrenar(hp):
    sp = C.params_split("reg")
    with st.spinner("Entrenando…"):
        return M.tabla_reg(hp, sp["features"], sp["test_size"], sp["seed"])


def _pred_vs_real(y_test, preds, titulos):
    """Paneles predicho vs real con la misma escala en todos los ejes."""
    lo = min(y_test.min(), min(p.min() for p in preds.values())) - 2
    hi = max(y_test.max(), max(p.max() for p in preds.values())) + 2
    fig, axes = plt.subplots(1, len(preds), figsize=(6 * len(preds), 5))
    for ax, (nombre, y_pred) in zip(axes, preds.items()):
        ax.scatter(y_test, y_pred, s=12, alpha=0.5, color="steelblue")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Real (kg)")
        ax.set_ylabel("Predicho (kg)")
        ax.set_title(titulos[nombre])
    ui.figura(fig)


def paso_1():
    sp = C.form_split("reg")
    d = M.split_reg(sp["features"], sp["test_size"], sp["seed"])
    ui.tabla(pd.DataFrame({"filas": [len(d["X_train"]), len(d["X_test"])],
                           "features": [d["n_features"]] * 2}, index=["train", "test"]),
             "Split (estratificado por deciles de Weight)")
    ui.tabla(d["y_train"].describe().round(2), "Weight en train")


def paso_2():
    hp = C.form_hp("reg")
    _, t, _ = _entrenar(hp)
    ui.tabla(t, "Resultados: RMSE de CV y métricas en test (MAE, RMSE, R², R²_adj)")


def paso_2_1():
    C.form_hp("reg")
    p = C.form_curva_k("reg")
    sp = C.params_split("reg")
    ks = tuple(range(p["k_min"], p["k_max"] + 1, p["paso"]))
    with st.spinner("Calculando la curva…"):
        cv = M.curva_k("reg", sp["features"], sp["test_size"], sp["seed"], ks, p["weights"])
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_k_curve(cv, ylabel="RMSE (CV)", negate=True, ax=ax)
    ui.figura(fig)
    ui.tabla(pd.Series([-s for s in cv["mean_test_score"]], index=cv["param_knn__n_neighbors"],
                       name="RMSE (CV)"), "RMSE por K", 3)


def paso_2_2():
    p = C.form_n_arboles("reg")
    sp = C.params_split("reg")
    hp_rf = C.params_hp("reg")["Random Forest"]
    with st.spinner("Entrenando los bosques…"):
        df = M.barrido_arboles("reg", sp["features"], sp["test_size"], sp["seed"],
                               sorted(p["ns"]), p["max_depth"], p["min_samples_leaf"])
    ui.tabla(df, "Random Forest (regresión): error y coste según el número de árboles", 4)
    fig, _ = plot_n_estimators_curve(
        df, "cv_RMSE", "test_RMSE", "RMSE (kg)", "Random Forest: RMSE según el número de árboles",
        col_std="cv_std", n_proyecto=int(hp_rf["n_estimators"]), mejor="min")
    ui.figura(fig)


def paso_3():
    sp = C.params_split("reg")
    p = C.form_arboles("reg")
    nivel = st.slider("Niveles a dibujar (Completo y Podado)", 1, 6, 3, key="e03_niv")
    feats = list(sp["features"])
    d = M.split_reg(feats, sp["test_size"], sp["seed"])
    arboles = {n: M.ajustar_reg("Árbol de regresión", {"criterion": "squared_error", **q}, feats,
                                sp["test_size"], sp["seed"], con_cv=False)["modelo"]
               for n, q in p.items()}
    resumen, preds, titulos = [], {}, {}
    for nombre, arbol in arboles.items():
        s = tree_summary(arbol, d["X"].columns, nombre)
        m = reg_metrics(d["y_test"], arbol.predict(d["X_test"]), n_features=d["n_features"])
        resumen.append({**s, **m})
        trunc = None if nombre == "Muy podado" else nivel
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(arbol, max_depth=trunc, feature_names=d["X"].columns, filled=True,
                  rounded=True, ax=ax, fontsize=9 if trunc is None else 8)
        ax.set_title(f"Árbol {nombre} — profundidad {s['profundidad']}, {s['n_hojas']} hojas, "
                     f"raíz: {s['raiz_atributo']} ≤ {s['raiz_umbral']}"
                     + (f" (truncado a {nivel} niveles)" if trunc else ""))
        ui.figura(fig)
        preds[nombre] = arbol.predict(d["X_test"])
        titulos[nombre] = (f"Árbol {nombre}\nRMSE={m['RMSE']:.2f} kg  R²={m['R2']:.3f}  "
                           f"R²adj={m['R2_adj']:.3f}")
    cols = ["modelo", "profundidad", "n_hojas", "raiz_atributo", "MAE", "RMSE", "R2", "R2_adj"]
    ui.tabla(pd.DataFrame(resumen)[cols].set_index("modelo"), "Resumen de los tres árboles")
    _pred_vs_real(d["y_test"], preds, titulos)


def paso_4():
    hp = C.form_hp("reg")
    res, _, d = _entrenar(hp)
    modelos = {n: r for n, r in res.items()}
    titulos = {n: (f"{n}\nRMSE={r['test_RMSE']:.2f} kg  R²={r['test_R2']:.3f}  "
                   f"R²adj={r['test_R2_adj']:.3f}") for n, r in modelos.items()}
    _pred_vs_real(d["y_test"], {n: r["y_pred"] for n, r in modelos.items()}, titulos)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (nombre, r) in zip(axes, modelos.items()):
        sns.histplot(d["y_test"] - r["y_pred"], bins=40, kde=True, ax=ax, color="steelblue")
        ax.axvline(0, c="r", ls="--", lw=1.2)
        ax.set_title(f"Residuos — {nombre}")
        ax.set_xlabel("Real − Predicho (kg)")
        ax.set_ylabel("Frecuencia")
    ui.figura(fig)


def paso_5():
    hp = C.form_hp("reg")
    res, _, d = _entrenar(hp)
    rf = res["Random Forest"]["modelo"]
    imp = pd.Series(rf.feature_importances_, index=d["X_train"].columns).sort_values(
        ascending=False)
    ui.tabla(imp.rename("importancia"), "feature_importances_")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color="steelblue")
    ax.set_title("Importancia de features — Random Forest (regresión de Weight)")
    ax.set_xlabel("Importancia (reducción de impureza)")
    ui.figura(fig)


def paso_6():
    hp = C.form_hp("reg")
    _, t, _ = _entrenar(hp)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    t["test_RMSE"].plot.bar(ax=axes[0], color="steelblue", title="RMSE en test (kg)")
    t["test_R2"].plot.bar(ax=axes[1], color="indianred", title="R² en test")
    t["test_R2_adj"].plot.bar(ax=axes[2], color="forestgreen", title="R² ajustado en test")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    ui.figura(fig)
    ui.tabla(t, "Tabla de resultados")


RENDER = {"1": paso_1, "2": paso_2, "2.1": paso_2_1, "2.2": paso_2_2, "3": paso_3,
          "4": paso_4, "5": paso_5, "6": paso_6}

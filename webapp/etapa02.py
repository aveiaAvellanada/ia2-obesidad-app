"""Etapa 02 — Clasificación de NObeyesdad (variantes A y B)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.tree import plot_tree

from src.evaluation import (metricas_por_clase, plot_confusion, plot_k_curve,
                            plot_n_estimators_curve, tree_summary)
from src.preprocessing import CLASS_ORDER
from webapp import controles as C
from webapp import modelos as M
from webapp import ui


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def _entrenar(variante, hp):
    sp = C.params_split("clf")
    feats = list(sp["features"]) if variante == "A" else C.features_b(sp)
    with st.spinner("Entrenando…"):
        return M.tabla_clf(hp, feats, sp["test_size"], sp["seed"])


def _features(variante):
    sp = C.params_split("clf")
    return (list(sp["features"]) if variante == "A" else C.features_b(sp)), sp


def _reportes_y_matrices(res, split, sufijo):
    y_test = split["y_test"]
    for nombre, r in res.items():
        rep = pd.DataFrame(classification_report(y_test, r["y_pred"], labels=CLASS_ORDER,
                                                 digits=3, output_dict=True)).T
        ui.tabla(rep.loc[CLASS_ORDER, ["precision", "recall", "f1-score"]],
                 f"classification_report por clase — {nombre} ({sufijo})", 3)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (nombre, r) in zip(axes, res.items()):
        plot_confusion(y_test, r["y_pred"], CLASS_ORDER, f"{nombre} ({sufijo})", ax=ax)
    ui.figura(fig)
    for nombre, r in res.items():
        cm = confusion_matrix(y_test, r["y_pred"], labels=CLASS_ORDER)
        ui.tabla(metricas_por_clase(cm, CLASS_ORDER), f"Métricas por clase — {nombre} ({sufijo})", 3)


def _importancias(res, split, color, titulo):
    rf = res["Random Forest"]["modelo"]
    imp = pd.Series(rf.feature_importances_, index=split["X_train"].columns).sort_values(
        ascending=False)
    ui.tabla(imp.rename("importancia"), "feature_importances_ (reducción media de impureza)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color=color)
    ax.set_title(titulo)
    ui.figura(fig)


def _curva_k(variante):
    p = C.form_curva_k("clf")
    feats, sp = _features(variante)
    ks = tuple(range(p["k_min"], p["k_max"] + 1, p["paso"]))
    with st.spinner("Calculando la curva…"):
        cv = M.curva_k("clf", feats, sp["test_size"], sp["seed"], ks, p["weights"])
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_k_curve(cv, ax=ax)
    ui.figura(fig)
    ui.tabla(pd.Series(cv["mean_test_score"], index=cv["param_knn__n_neighbors"],
                       name="macro-F1 (CV)"), "Score por K", 4)


# ----------------------------------------------------------------------------
# Pasos
# ----------------------------------------------------------------------------
def paso_1():
    sp = C.form_split("clf")
    d = M.split_clf(sp["features"], sp["test_size"], sp["seed"])
    ui.tabla(pd.DataFrame({"filas": [len(d["X_train"]), len(d["X_test"])],
                           "features": [d["X_train"].shape[1]] * 2}, index=["train", "test"]),
             "Split estratificado")
    ui.tabla((d["y_train"].value_counts(normalize=True).reindex(CLASS_ORDER) * 100)
             .rename("% en train"), "Distribución de clases en train (%)", 1)


def paso_2():
    hp = C.form_hp("clf")
    ui.tabla(pd.DataFrame(hp).T.astype(str), "Hiperparámetros vigentes")


def paso_3():
    hp = C.form_hp("clf")
    _, t, _ = _entrenar("A", hp)
    ui.tabla(t, "Variante A — macro-F1 en CV y métricas en test")


def paso_3_1():
    hp = C.form_hp("clf")
    res, _, split = _entrenar("A", hp)
    _reportes_y_matrices(res, split, "A")


def paso_3_2():
    hp = C.form_hp("clf")
    res, _, split = _entrenar("A", hp)
    _importancias(res, split, "steelblue", "Importancia de features — Random Forest (A)")


def paso_3_3():
    hp = C.form_hp("clf")
    res, _, split = _entrenar("A", hp)
    nivel = st.slider("Niveles a dibujar", 1, 6, 3, key="e02_niv33")
    arbol = res["Árbol de decisión"]["modelo"]
    ui.tabla(pd.DataFrame({"profundidad": [arbol.get_depth()], "hojas": [arbol.get_n_leaves()]}),
             "Árbol de decisión (A)", 0)
    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(arbol, feature_names=split["X_train"].columns, class_names=list(arbol.classes_),
              max_depth=nivel, filled=True, fontsize=8, ax=ax)
    ax.set_title(f"Árbol de decisión (variante A) — dibujo truncado a {nivel} niveles")
    ui.figura(fig)


def paso_3_4():
    sp = C.params_split("clf")
    p = C.form_arboles("clf")
    nivel = st.slider("Niveles a dibujar (Completo y Podado)", 1, 6, 3, key="e02_niv34")
    feats = list(sp["features"])
    d = M.split_clf(feats, sp["test_size"], sp["seed"])
    arboles = {n: M.ajustar_clf("Árbol de decisión", {"criterion": "gini", **q}, feats,
                                sp["test_size"], sp["seed"], con_cv=False)["modelo"]
               for n, q in p.items()}
    resumen = []
    for nombre, arbol in arboles.items():
        s = tree_summary(arbol, d["X"].columns, nombre)
        trunc = None if nombre == "Muy podado" else nivel
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(arbol, feature_names=d["X"].columns, class_names=list(arbol.classes_),
                  filled=True, max_depth=trunc, fontsize=8, ax=ax)
        ax.set_title(f"Árbol {nombre} — profundidad {s['profundidad']}, {s['n_hojas']} hojas, "
                     f"raíz: {s['raiz_atributo']} ≤ {s['raiz_umbral']}"
                     + (f"  (dibujo truncado a {nivel} niveles)" if trunc else ""))
        ui.figura(fig)
        resumen.append({**s,
                        "acc_train": accuracy_score(d["y_train"], arbol.predict(d["X_train"])),
                        "acc_test": accuracy_score(d["y_test"], arbol.predict(d["X_test"])),
                        "f1_macro_test": f1_score(d["y_test"], arbol.predict(d["X_test"]),
                                                  average="macro")})
    cols = ["modelo", "profundidad", "n_hojas", "raiz_atributo", "acc_train", "acc_test",
            "f1_macro_test"]
    ui.tabla(pd.DataFrame(resumen)[cols].set_index("modelo"), "Resumen de los tres árboles", 3)

    preds = {n: a.predict(d["X_test"]) for n, a in arboles.items()}
    vmax = max(pd.crosstab(d["y_test"], p_).values.max() for p_ in preds.values())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (nombre, y_pred) in zip(axes, preds.items()):
        plot_confusion(d["y_test"], y_pred, CLASS_ORDER,
                       f"Árbol {nombre} — acc test {accuracy_score(d['y_test'], y_pred):.3f}",
                       ax=ax, vmin=0, vmax=vmax)
    ui.figura(fig)
    for nombre, y_pred in preds.items():
        cm = confusion_matrix(d["y_test"], y_pred, labels=CLASS_ORDER)
        ui.tabla(metricas_por_clase(cm, CLASS_ORDER), f"Métricas por clase — árbol {nombre}", 3)


def paso_3_6():
    C.form_hp("clf")      # el K de KNN de los demás pasos se edita aquí también
    _curva_k("A")


def paso_3_7():
    p = C.form_n_arboles("clf")
    feats, sp = _features("A")
    hp_rf = C.params_hp("clf")["Random Forest"]
    with st.spinner("Entrenando los bosques…"):
        df = M.barrido_arboles("clf", feats, sp["test_size"], sp["seed"], sorted(p["ns"]),
                               p["max_depth"], p["min_samples_leaf"])
    ui.tabla(df, "Random Forest: rendimiento y coste según el número de árboles", 4)
    fig, _ = plot_n_estimators_curve(
        df, "cv_f1_macro", "test_f1_macro", "macro-F1",
        "Random Forest: macro-F1 según el número de árboles (variante A)",
        col_std="cv_std", n_proyecto=int(hp_rf["n_estimators"]), mejor="max")
    ui.figura(fig)


def paso_4():
    hp = C.form_hp("clf")
    res, t, split = _entrenar("B", hp)
    ui.tabla(t, "Variante B — macro-F1 en CV y métricas en test")
    _reportes_y_matrices(res, split, "B")
    _importancias(res, split, "indianred",
                  "Importancia de features — Random Forest (B, sin Weight/Height)")


def paso_4_1():
    C.form_hp("clf")
    _curva_k("B")


def paso_5():
    hp = C.params_hp("clf")
    _, tA, _ = _entrenar("A", hp)
    _, tB, _ = _entrenar("B", hp)
    comp = pd.concat({"A (todas)": tA, "B (sin Weight/Height)": tB}, axis=0)
    ui.tabla(comp, "Comparación A vs B")
    fig, ax = plt.subplots(figsize=(8, 4))
    datos = comp.reset_index().rename(columns={"level_0": "variante", "level_1": "modelo"})
    sns.barplot(data=datos, x="modelo", y="test_f1_macro", hue="variante", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Macro-F1 en test por modelo y variante")
    ui.figura(fig)


def paso_7():
    p = C.form_ruido()
    sp = C.params_split("clf")
    niveles = tuple(x / 100 for x in range(0, p["max"] + 1, p["paso"]))
    with st.spinner("Entrenando con ruido…"):
        df = M.barrido_ruido(sp["test_size"], sp["seed"], niveles)
    for metrica, titulo in (("accuracy", "Accuracy en test por nivel de ruido"),
                            ("f1_macro", "Macro-F1 en test por nivel de ruido")):
        ui.tabla(df.pivot(index="nivel", columns="modelo", values=metrica)
                 .rename_axis("Nivel de ruido (%)"), titulo)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ancho = df.pivot(index="nivel", columns="modelo", values="accuracy")
    for marcador, modelo in zip(["o", "s", "^", "D"], ancho.columns):
        ax.plot(ancho.index, ancho[modelo], marker=marcador, lw=2.2, label=modelo)
    ax.set_title("Degradación del accuracy vs nivel de ruido en hábitos (variante B)")
    ax.set_xlabel("Nivel de perturbación gaussiana (% de desviación estándar)")
    ax.set_ylabel("Accuracy en test")
    ax.set_xticks(list(ancho.index))
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(title="Modelo", loc="center left", bbox_to_anchor=(1.01, 0.5))
    ui.figura(fig)


def paso_8():
    p = C.form_reales()
    sp = C.params_split("clf")
    with st.spinner("Validación cruzada sobre las filas reales…"):
        t7 = M.evaluar_reales(p["incluir_wh"], 7, p["repeticiones"], sp["seed"])
        t3 = M.evaluar_reales(p["incluir_wh"], 3, p["repeticiones"], sp["seed"])
    ui.tabla(t7, "Sólo filas reales · 7 clases (CV de 3 folds repetida)")
    ui.tabla(t3, "Sólo filas reales · 3 clases agrupadas (CV de 3 folds repetida)")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, t, azar, titulo in [(axes[0], t7, 1 / 7, "7 clases"),
                                (axes[1], t3, 1 / 3, "3 clases agrupadas")]:
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["balanced_accuracy"], 0.4, label="balanced accuracy", color="#4c78a8")
        ax.bar(x + 0.2, t["f1_macro"], 0.4, yerr=t["f1_macro_std"], capsize=3,
               label="macro-F1 (± desv. entre folds)", color="#f58518")
        ax.axhline(azar, ls="--", color="gray", lw=1.2,
                   label=f"azar en balanced acc. (1/{round(1 / azar)})")
        ax.set_xticks(x, [n.replace(" (clase mayoritaria)", "") for n in t.index],
                      rotation=20, ha="right")
        ax.set_title(f"Filas reales · {titulo} · {'con' if p['incluir_wh'] else 'sin'} Weight/Height")
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("métrica")
    ui.figura(fig)


RENDER = {
    "1": paso_1, "2": paso_2, "3": paso_3, "3.1": paso_3_1, "3.2": paso_3_2, "3.3": paso_3_3,
    "3.4": paso_3_4, "3.6": paso_3_6, "3.7": paso_3_7, "4": paso_4, "4.1": paso_4_1,
    "5": paso_5, "7": paso_7, "8": paso_8,
}

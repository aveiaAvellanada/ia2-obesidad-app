"""ETAPA 02 — Clasificación de `NObeyesdad`
============================================

Modelos: árbol de decisión, random forest y KNN.

Protocolo:
- Dataset preparado en la etapa 01 (data/clf.csv): categóricas codificadas, sin duplicados.
- Split estratificado 80/20 (random_state=42). El test se usa una única vez, al final.
- Hiperparámetros con GridSearchCV (5-fold estratificado, criterio macro-F1) sobre el train.
- KNN va dentro de un Pipeline con StandardScaler (ajustado sólo con train). Los árboles no
  necesitan escalado.
- Métricas en test: accuracy, macro-F1, classification_report y matriz de confusión.

Dos variantes:
- A. Todas las features (incluye Weight y Height), como pide el enunciado.
- B. Sin Weight ni Height: como NObeyesdad se define por IMC = Weight/Height², la variante A
  tiene fuga de datos (ver etapa 01). B mide cuánto predicen por sí solos los hábitos.

Pasos (misma numeración que el notebook):
  1    Datos y split
  2    Modelos y grillas                 3.6  Curva del mejor K de KNN (A)
  3    Variante A: tabla de resultados   4    Variante B: resultados, reportes, matrices, importancias
  3.1  Reportes + matrices de confusión  4.1  Curva del mejor K de KNN (B)
  3.2  Importancia de features (RF)      5    Comparación final A vs B
  3.3  Árbol de GridSearch (dibujo)
  3.4  Tres árboles: completo/podado/muy podado   7  Robustez ante ruido (jittering)
                                         8    Sólo filas reales (sin SMOTE) y sin Weight/Height

Los GridSearch tardan ~40 s la primera vez; después quedan en cache/ y cargan al instante.
Si cambias un hiperparámetro en este archivo (make_models, N_ARBOLES...) y guardas, se
recalculan solos la siguiente vez (src/cache.py compara una huella del código).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from config import RANDOM_STATE
from src import cache
from src.data import TARGET_CLF
from src.evaluation import (clf_metrics, metricas_por_clase, plot_confusion, plot_k_curve,
                            plot_n_estimators_curve, results_table, tree_summary)
from src.pasos import Etapa, cli
from src.preprocessing import CLASS_ORDER, cargar_clf
from src.salida import subtitulo, tabla

etapa = Etapa("02", "Clasificación de NObeyesdad", "clasificacion",
              "Árbol, Random Forest y KNN en dos variantes (con y sin Weight/Height)",
              globales=("RANDOM_STATE", "datos", "cv_estratificado"))

DROP_WH = ["Weight", "Height"]   # columnas que se quitan en la variante B

# ----------------------------------------------------------------------------
# Estado compartido (se calcula bajo demanda; los GridSearch van a cache/)
# ----------------------------------------------------------------------------
_estado = {}


def datos():
    """Split estratificado 80/20. Devuelve dict con X, y, X_train, X_test, y_train, y_test."""
    if "datos" not in _estado:
        df = cargar_clf()
        X = df.drop(columns=[TARGET_CLF])
        y = df[TARGET_CLF].astype(str)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        _estado["datos"] = dict(X=X, y=y, X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test)
    return _estado["datos"]


def cv_estratificado():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def make_models():
    """Los 3 modelos con su grilla de hiperparámetros."""
    return {
        "Árbol de decisión": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [4, 6, 8, 10, None], "min_samples_leaf": [1, 3, 5]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [None, 12], "min_samples_leaf": [1, 2]},
        ),
        "KNN": (
            Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
            {"knn__n_neighbors": list(range(1, 32, 2)), "knn__weights": ["uniform", "distance"]},
        ),
    }


def fit_evaluate(X_tr, y_tr, X_te, y_te, tag):
    """Ajusta los 3 modelos con GridSearchCV y devuelve (fitted, tabla_resultados)."""
    fitted, results = {}, {}
    for name, (model, grid) in make_models().items():
        gs = GridSearchCV(model, grid, cv=cv_estratificado(), scoring="f1_macro", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        y_pred = gs.predict(X_te)
        fitted[name] = gs
        results[name] = {"cv_f1_macro": gs.best_score_,
                         **{f"test_{k}": v for k, v in clf_metrics(y_te, y_pred).items()}}
        print(f"   [{tag}] {name}: mejores params = {gs.best_params_}")
    return fitted, results_table(results)


def variante(tag: str):
    """'A' = todas las features, 'B' = sin Weight/Height. Devuelve (fitted, tabla, X_tr, X_te)."""
    clave = f"clf_variante_{tag}"
    d = datos()
    if tag == "A":
        X_tr, X_te = d["X_train"], d["X_test"]
    else:
        X_tr, X_te = d["X_train"].drop(columns=DROP_WH), d["X_test"].drop(columns=DROP_WH)
    fitted, tabla_res = cache.obtener(
        clave, lambda: fit_evaluate(X_tr, d["y_train"], X_te, d["y_test"], tag))
    return fitted, tabla_res, X_tr, X_te


def tabla_por_clase(y_true, y_pred, titulo):
    """Tabla TP/FN/FP/TN/Precision/Recall/F1 por clase de la matriz de confusión (test)."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    tabla(metricas_por_clase(cm, CLASS_ORDER), titulo, 3)


def arboles_prepoda():
    """Los tres árboles del paso 3.4 entrenados sobre el train de la variante A."""
    if "arboles" not in _estado:
        d = datos()
        arboles = {
            "Completo":   DecisionTreeClassifier(random_state=RANDOM_STATE),
            "Podado":     DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE),
            "Muy podado": DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=RANDOM_STATE),
        }
        for a in arboles.values():
            a.fit(d["X_train"], d["y_train"])
        _estado["arboles"] = arboles
    return _estado["arboles"]


# ----------------------------------------------------------------------------
# 1. Datos y split
# ----------------------------------------------------------------------------
@etapa.paso("1", "Datos y split estratificado 80/20")
def paso_1_datos():
    d = datos()
    print("train:", d["X_train"].shape, "| test:", d["X_test"].shape)
    print("features:", d["X"].columns.tolist())
    tabla((d["y_train"].value_counts(normalize=True).reindex(CLASS_ORDER) * 100).round(1)
          .rename("% en train"), "Distribución de clases en train (%)", 1)


# ----------------------------------------------------------------------------
# 2. Modelos y grillas
# ----------------------------------------------------------------------------
@etapa.paso("2", "Definición de modelos y grillas de hiperparámetros",
            parametros=("make_models",))
def paso_2_modelos():
    for name, (model, grid) in make_models().items():
        subtitulo(name)
        print("   modelo:", model)
        for k, v in grid.items():
            print(f"   {k}: {v}")


# ----------------------------------------------------------------------------
# 3. Variante A
# ----------------------------------------------------------------------------
@etapa.paso("3", "Variante A (todas las features): GridSearch y tabla de resultados",
            parametros=("make_models", "fit_evaluate"))
def paso_3_variante_A():
    _, tabla_A, _, _ = variante("A")
    tabla(tabla_A, "Variante A — macro-F1 en CV y métricas en test")


@etapa.paso("3.1", "Variante A: reportes por clase y matrices de confusión (test)", figuras=1,
            parametros=("make_models",))
def paso_3_1_reportes_A():
    fitted, _, _, X_te = variante("A")
    y_test = datos()["y_test"]
    for name, gs in fitted.items():
        subtitulo(name)
        y_pred = gs.predict(X_te)
        print(classification_report(y_test, y_pred, labels=CLASS_ORDER, digits=3))
        # además del texto, el reporte por clase queda como figura en figures/<etapa>/tablas/
        rep = pd.DataFrame(classification_report(y_test, y_pred, labels=CLASS_ORDER,
                                                 digits=3, output_dict=True)).T
        tabla(rep.loc[CLASS_ORDER, ["precision", "recall", "f1-score"]],
              f"classification_report por clase — {name} (A)", 3, texto=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (name, gs) in zip(axes, fitted.items()):
        plot_confusion(y_test, gs.predict(X_te), CLASS_ORDER, f"{name} (A)", ax=ax)
    etapa.figura(fig, "3.1_matrices_confusion_A")
    for name, gs in fitted.items():
        tabla_por_clase(y_test, gs.predict(X_te), f"Métricas por clase — {name} (A)")


@etapa.paso("3.2", "Variante A: importancia de features del Random Forest", figuras=1,
            parametros=("make_models",))
def paso_3_2_importancias_A():
    fitted, _, X_tr, _ = variante("A")
    rf_A = fitted["Random Forest"].best_estimator_
    imp = pd.Series(rf_A.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    tabla(imp.rename("importancia"), "feature_importances_ (reducción media de impureza Gini)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color="steelblue")
    ax.set_title("Importancia de features — Random Forest (A)")
    etapa.figura(fig, "3.2_importancias_rf_A")


@etapa.paso("3.3", "Variante A: árbol de decisión elegido por GridSearch (primeros 3 niveles)", figuras=1,
            parametros=("make_models",))
def paso_3_3_arbol_gridsearch():
    fitted, _, X_tr, _ = variante("A")
    tree_A = fitted["Árbol de decisión"].best_estimator_
    print("Profundidad del árbol:", tree_A.get_depth(), "| hojas:", tree_A.get_n_leaves())
    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(tree_A, feature_names=X_tr.columns, class_names=tree_A.classes_, max_depth=3,
              filled=True, fontsize=8, ax=ax)
    ax.set_title("Árbol de GridSearch (variante A) — dibujo truncado a 3 niveles")
    etapa.figura(fig, "3.3_arbol_gridsearch_A")


@etapa.paso("3.4", "Tres árboles: completo, podado y muy podado (prepoda)", figuras=4,
            parametros=("arboles_prepoda",))
def paso_3_4_tres_arboles():
    d = datos()
    X, y_train, y_test = d["X"], d["y_train"], d["y_test"]
    resumen = []
    for nombre, arbol in arboles_prepoda().items():
        s = tree_summary(arbol, X.columns, nombre)
        print(s)
        fig, ax = plt.subplots(figsize=(20, 8))
        # class_names debe ir en el orden de arbol.classes_ (alfabético), que es el que usa
        # plot_tree para rotular cada nodo; pasar CLASS_ORDER pondría nombres equivocados.
        plot_tree(arbol, feature_names=X.columns, class_names=list(arbol.classes_), filled=True,
                  max_depth=None if nombre == "Muy podado" else 3, fontsize=8, ax=ax)
        ax.set_title(f"Árbol {nombre} — profundidad {s['profundidad']}, {s['n_hojas']} hojas, "
                     f"raíz: {s['raiz_atributo']} ≤ {s['raiz_umbral']}"
                     + ("" if nombre == "Muy podado" else "  (dibujo truncado a 3 niveles)"))
        etapa.figura(fig, f"3.4_arbol_{nombre.lower().replace(' ', '_')}")
        resumen.append({
            **s,
            "acc_train": accuracy_score(y_train, arbol.predict(d["X_train"])),
            "acc_test": accuracy_score(y_test, arbol.predict(d["X_test"])),
            "f1_macro_test": f1_score(y_test, arbol.predict(d["X_test"]), average="macro"),
        })

    df_arboles = (pd.DataFrame(resumen)
                  [["modelo", "profundidad", "n_hojas", "raiz_atributo", "acc_train", "acc_test", "f1_macro_test"]]
                  .set_index("modelo"))
    tabla(df_arboles, "Resumen de los tres árboles", 3)

    # Tres matrices de confusión con la misma escala de color
    preds = {n: a.predict(d["X_test"]) for n, a in arboles_prepoda().items()}
    vmax = max(pd.crosstab(y_test, p).values.max() for p in preds.values())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (nombre, y_pred) in zip(axes, preds.items()):
        plot_confusion(y_test, y_pred, CLASS_ORDER,
                       f"Árbol {nombre} — acc test {accuracy_score(y_test, y_pred):.3f}",
                       ax=ax, vmin=0, vmax=vmax)
    etapa.figura(fig, "3.4_matrices_tres_arboles")
    for nombre, y_pred in preds.items():
        tabla_por_clase(y_test, y_pred, f"Métricas por clase — árbol {nombre}")


@etapa.paso("3.6", "Curva del mejor K de KNN — variante A", figuras=1,
            parametros=("make_models",))
def paso_3_6_curva_k_A():
    fitted, _, _, _ = variante("A")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax, mejor_k = plot_k_curve(fitted["KNN"].cv_results_, ax=ax)
    etapa.figura(fig, "3.6_curva_k_knn_A")
    print("Mejor K (variante A):", mejor_k, "| params:", fitted["KNN"].best_params_)


# ----------------------------------------------------------------------------
# 3.7 ¿Cuántos árboles necesita el Random Forest?
# ----------------------------------------------------------------------------
N_ARBOLES = [10, 25, 50, 100, 200, 400, 800]


def _barrido_n_arboles(n_arboles=N_ARBOLES):
    """Para cada nº de árboles: macro-F1 de CV (5 folds), métricas en test y tiempo de fit.

    Se usa la variante A y la MISMA validación cruzada estratificada del GridSearch,
    para que las cifras sean comparables con las del paso 3.
    """
    d = datos()
    filas = []
    for n in n_arboles:
        rf = RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1)
        scores = cross_val_score(rf, d["X_train"], d["y_train"], cv=cv_estratificado(),
                                 scoring="f1_macro", n_jobs=-1)
        t0 = time.perf_counter()
        rf.fit(d["X_train"], d["y_train"])
        t_fit = time.perf_counter() - t0
        m = clf_metrics(d["y_test"], rf.predict(d["X_test"]))
        filas.append({"n_estimators": n, "cv_f1_macro": scores.mean(), "cv_std": scores.std(),
                      "test_accuracy": m["accuracy"], "test_f1_macro": m["f1_macro"],
                      "tiempo_fit_s": t_fit})
    return pd.DataFrame(filas).set_index("n_estimators")


def barrido_n_arboles():
    return cache.obtener("clf_barrido_n_arboles", _barrido_n_arboles)


@etapa.paso("3.7", "¿Cuántos árboles? Random Forest con "
            + ", ".join(map(str, N_ARBOLES[:-1])) + f" y {N_ARBOLES[-1]}", figuras=1,
            parametros=("N_ARBOLES", "_barrido_n_arboles"))
def paso_3_7_n_arboles():
    df = barrido_n_arboles()
    tabla(df, "Random Forest: rendimiento y coste según el número de árboles", 4)

    fig, n_suf = plot_n_estimators_curve(
        df, "cv_f1_macro", "test_f1_macro", "macro-F1",
        "Random Forest: macro-F1 según el número de árboles (variante A)",
        col_std="cv_std", n_proyecto=200, mejor="max")
    etapa.figura(fig, "3.7_n_estimators_rf")

    cv, t = df["cv_f1_macro"], df["tiempo_fit_s"]
    n_min, n_max, n_mejor = int(df.index.min()), int(df.index.max()), int(cv.idxmax())
    print(f"\nmacro-F1 de CV: {cv.loc[n_min]:.4f} con {n_min} árboles  ->  {cv.loc[n_max]:.4f} con "
          f"{n_max} (ganancia total {cv.loc[n_max] - cv.loc[n_min]:+.4f}; el mejor valor de la "
          f"tabla se da con {n_mejor} árboles)")
    print(f"Con {n_suf} árboles ya se consigue el 90% de toda la mejora del barrido.")
    print(f"Pasar de {n_suf} a {n_max} árboles multiplica el tiempo de entrenamiento por "
          f"{t.loc[n_max] / max(float(t.loc[n_suf]), 1e-9):.1f} "
          f"({t.loc[n_suf]:.2f} s -> {t.loc[n_max]:.2f} s) a cambio de "
          f"{cv.loc[n_max] - cv.loc[n_suf]:+.4f} de macro-F1.")


# ----------------------------------------------------------------------------
# 4. Variante B
# ----------------------------------------------------------------------------
@etapa.paso("4", "Variante B (sin Weight/Height): resultados, reportes, matrices e importancias", figuras=2,
            parametros=("make_models", "DROP_WH"))
def paso_4_variante_B():
    fitted, tabla_B, X_tr, X_te = variante("B")
    y_test = datos()["y_test"]
    tabla(tabla_B, "Variante B — macro-F1 en CV y métricas en test")

    for name, gs in fitted.items():
        subtitulo(name)
        y_pred = gs.predict(X_te)
        print(classification_report(y_test, y_pred, labels=CLASS_ORDER, digits=3))
        # además del texto, el reporte por clase queda como figura en figures/<etapa>/tablas/
        rep = pd.DataFrame(classification_report(y_test, y_pred, labels=CLASS_ORDER,
                                                 digits=3, output_dict=True)).T
        tabla(rep.loc[CLASS_ORDER, ["precision", "recall", "f1-score"]],
              f"classification_report por clase — {name} (B)", 3, texto=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (name, gs) in zip(axes, fitted.items()):
        plot_confusion(y_test, gs.predict(X_te), CLASS_ORDER, f"{name} (B, sin Weight/Height)", ax=ax)
    etapa.figura(fig, "4_matrices_confusion_B")
    for name, gs in fitted.items():
        tabla_por_clase(y_test, gs.predict(X_te), f"Métricas por clase — {name} (B)")

    rf_B = fitted["Random Forest"].best_estimator_
    imp_B = pd.Series(rf_B.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    tabla(imp_B.rename("importancia"), "Importancias RF (B)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=imp_B.values, y=imp_B.index, ax=ax, color="indianred")
    ax.set_title("Importancia de features — Random Forest (B, sin Weight/Height)")
    etapa.figura(fig, "4_importancias_rf_B")


@etapa.paso("4.1", "Curva del mejor K de KNN — variante B", figuras=1,
            parametros=("make_models", "DROP_WH"))
def paso_4_1_curva_k_B():
    fitted, _, _, _ = variante("B")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax, mejor_k = plot_k_curve(fitted["KNN"].cv_results_, ax=ax)
    etapa.figura(fig, "4.1_curva_k_knn_B")
    print("Mejor K (variante B):", mejor_k, "| params:", fitted["KNN"].best_params_)


# ----------------------------------------------------------------------------
# 5. Comparación final
# ----------------------------------------------------------------------------
@etapa.paso("5", "Comparación final: variante A vs variante B", figuras=1,
            parametros=("make_models", "DROP_WH"))
def paso_5_comparacion():
    _, tabla_A, _, _ = variante("A")
    _, tabla_B, _, _ = variante("B")
    comp = pd.concat({"A (todas)": tabla_A, "B (sin Weight/Height)": tabla_B}, axis=0)
    tabla(comp, "Comparación A vs B")

    fig, ax = plt.subplots(figsize=(8, 4))
    datos_plot = comp.reset_index().rename(columns={"level_0": "variante", "level_1": "modelo"})
    sns.barplot(data=datos_plot, x="modelo", y="test_f1_macro", hue="variante", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Macro-F1 en test por modelo y variante")
    etapa.figura(fig, "5_comparacion_A_vs_B")


# ----------------------------------------------------------------------------
# 7. Robustez ante ruido sintético (jittering)
# ----------------------------------------------------------------------------
HABIT_NUM_COLS = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]   # atributos continuos a perturbar
RANGOS_HABITOS = {"Age": (14, 70), "FCVC": (1, 3), "NCP": (1, 4), "CH2O": (1, 3), "FAF": (0, 3), "TUE": (0, 2)}


def _modelos_robustez():
    return {
        "Árbol de decisión": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "KNN (k=1)": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=1))]),
        "KNN (k=3, distance)": Pipeline([("scaler", StandardScaler()),
                                         ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance"))]),
    }


def _con_ruido(X_tr, X_te, nivel):
    """Copia de X_tr/X_te con ruido gaussiano N(0, (nivel·σ_j)²) en los hábitos, recortado a
    rangos fisiológicos. La semilla se fija aquí para que el resultado sea reproducible."""
    np.random.seed(RANDOM_STATE)
    X_tr_n, X_te_n = X_tr.copy(), X_te.copy()
    for col in HABIT_NUM_COLS:
        std = X_tr[col].std()
        X_tr_n[col] += np.random.normal(0, nivel * std, size=len(X_tr))
        X_te_n[col] += np.random.normal(0, nivel * std, size=len(X_te))
    for col, (lo, hi) in RANGOS_HABITOS.items():
        X_tr_n[col] = X_tr_n[col].clip(lo, hi)
        X_te_n[col] = X_te_n[col].clip(lo, hi)
    return X_tr_n, X_te_n


@etapa.paso("7", "Robustez ante ruido sintético (jittering) en variante B: 5% y barrido 0–20%", figuras=1,
            parametros=("_modelos_robustez", "_con_ruido", "HABIT_NUM_COLS", "RANGOS_HABITOS"))
def paso_7_ruido():
    d = datos()
    y_train, y_test = d["y_train"], d["y_test"]
    X_tr_B, X_te_B = d["X_train"].drop(columns=DROP_WH), d["X_test"].drop(columns=DROP_WH)

    # --- 5% de ruido: comparación limpio vs ruidoso ---
    X_tr_n, X_te_n = _con_ruido(d["X_train"], d["X_test"], 0.05)
    X_tr_n, X_te_n = X_tr_n.drop(columns=DROP_WH), X_te_n.drop(columns=DROP_WH)
    modelos = _modelos_robustez()
    filas = []
    for name, model in modelos.items():
        model.fit(X_tr_B, y_train)
        acc_c = accuracy_score(y_test, model.predict(X_te_B))
        f1_c = f1_score(y_test, model.predict(X_te_B), average="macro")
        model.fit(X_tr_n, y_train)
        acc_n = accuracy_score(y_test, model.predict(X_te_n))
        f1_n = f1_score(y_test, model.predict(X_te_n), average="macro")
        filas.append({"Modelo": name, "Test Acc (Limpio)": acc_c, "Test Acc (Ruido 5%)": acc_n,
                      "Dif Acc": acc_n - acc_c, "Test F1 (Limpio)": f1_c, "Test F1 (Ruido 5%)": f1_n,
                      "Dif F1": f1_n - f1_c})
    tabla(pd.DataFrame(filas).set_index("Modelo"), "Robustez ante 5% de ruido en hábitos (variante B)")

    # --- Barrido 0%, 5%, 10%, 15%, 20% ---
    niveles = [0.0, 0.05, 0.10, 0.15, 0.20]
    registros = []
    for nl in niveles:
        if nl > 0:
            X_tr_s, X_te_s = _con_ruido(X_tr_B, X_te_B, nl)
        else:
            X_tr_s, X_te_s = X_tr_B.copy(), X_te_B.copy()
        fila = {"Nivel de ruido (%)": int(nl * 100)}
        for name, model in modelos.items():
            model.fit(X_tr_s, y_train)
            fila[name] = round(accuracy_score(y_test, model.predict(X_te_s)), 4)
        registros.append(fila)
    df_sweep = pd.DataFrame(registros)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    markers = ["o", "s", "^", "D"]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    for i, name in enumerate(modelos):
        ax.plot(df_sweep["Nivel de ruido (%)"], df_sweep[name], marker=markers[i], linewidth=2.2,
                label=name, color=colors[i])
    ax.set_title("Degradación del Accuracy vs Nivel de Ruido en Hábitos (Variante B)")
    ax.set_xlabel("Nivel de perturbación gaussiana (% de desviación estándar)")
    ax.set_ylabel("Accuracy en Test")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(0.58, 0.90)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(title="Modelo", loc="center left", bbox_to_anchor=(1.01, 0.5))
    etapa.figura(fig, "7_barrido_ruido")
    tabla(df_sweep.set_index("Nivel de ruido (%)"), "Accuracy en test por nivel de ruido")


# ----------------------------------------------------------------------------
# 8. Sólo filas reales (sin SMOTE) y sin Weight/Height
# ----------------------------------------------------------------------------
# Respuestas de encuesta que SMOTE deja con decimales. Una fila con TODAS ellas enteras no fue
# interpolada: es una respuesta original. Da 491 de 2087 filas (23.5 %, el ~23 % real que
# declara UCI). Es una heurística, no una etiqueta del dataset.
COLS_ENCUESTA = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
GRUPOS_3 = {"Insufficient_Weight": "Bajo/Normal", "Normal_Weight": "Bajo/Normal",
            "Overweight_Level_I": "Sobrepeso", "Overweight_Level_II": "Sobrepeso",
            "Obesity_Type_I": "Obesidad", "Obesity_Type_II": "Obesidad", "Obesity_Type_III": "Obesidad"}
METRICAS_REALES = ["accuracy", "balanced_accuracy", "f1_macro"]
BASELINE_REALES = "Línea base (clase mayoritaria)"


def filas_reales(X):
    """Máscara booleana: True en las filas cuyas respuestas de encuesta son todas enteras."""
    return (X[COLS_ENCUESTA] % 1 == 0).all(axis=1)


def _modelos_reales():
    """Configuraciones fijas y razonables (sin GridSearch: con 491 filas haría falta CV anidada)."""
    return {
        BASELINE_REALES: DummyClassifier(strategy="most_frequent"),
        "Árbol (depth 4, leaf 5)": DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                                                          random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "RF con pesos por clase": RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                                         class_weight="balanced",
                                                         random_state=RANDOM_STATE, n_jobs=-1),
        "KNN (k=15, distance)": Pipeline([("scaler", StandardScaler()),
                                          ("knn", KNeighborsClassifier(n_neighbors=15, weights="distance"))]),
        "KNN (k=1)": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=1))]),
    }


def _cv_reales(X, y):
    """Media y desviación de cada métrica con 3 folds estratificados repetidos 10 veces."""
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=RANDOM_STATE)
    filas = {}
    for nombre, modelo in _modelos_reales().items():
        s = cross_validate(modelo, X, y, cv=cv, scoring=METRICAS_REALES, n_jobs=-1)
        filas[nombre] = {**{m: s[f"test_{m}"].mean() for m in METRICAS_REALES},
                         "f1_macro_std": s["test_f1_macro"].std()}
    return pd.DataFrame(filas).T


def _evaluacion_reales():
    d = datos()
    X, y = d["X"], d["y"]
    r = filas_reales(X)
    X_r, y_r = X[r], y[r]
    X_rB = X_r.drop(columns=DROP_WH)

    # (a) Los modelos del proyecto (entrenados con todo) evaluados por separado en el test
    m_te = filas_reales(d["X_test"])
    real_vs_sint = {}
    for tag in ("A", "B"):
        fitted, _, _, X_te = variante(tag)
        for nombre in ("Árbol de decisión", "Random Forest", "KNN"):
            p = fitted[nombre].predict(X_te)
            real_vs_sint[f"{tag} · {nombre}"] = {
                "acc filas reales": accuracy_score(d["y_test"][m_te], p[m_te]),
                "acc filas sintéticas": accuracy_score(d["y_test"][~m_te], p[~m_te])}

    # (b) Coherencia de la etiqueta con el IMC en las filas reales (¿están bien etiquetadas?)
    imc = X_r["Weight"] / X_r["Height"] ** 2
    cortes = [-np.inf, 18.5, 25, 27.5, 30, 35, 40, np.inf]
    clase_oms = pd.cut(imc, cortes, right=False, labels=CLASS_ORDER).astype(str)

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1).fit(X_rB, y_r.map(GRUPOS_3))
    return dict(
        n_reales=int(r.sum()), n_total=len(X), n_test_reales=int(m_te.sum()),
        conteo=y_r.value_counts().reindex(CLASS_ORDER),
        real_vs_sint=pd.DataFrame(real_vs_sint).T,
        coincide_imc=float((clase_oms == y_r).mean()),
        B7=_cv_reales(X_rB, y_r),
        B3=_cv_reales(X_rB, y_r.map(GRUPOS_3)),
        A7=_cv_reales(X_r, y_r),
        importancias=pd.Series(rf.feature_importances_, index=X_rB.columns).sort_values(ascending=False),
    )


def evaluacion_reales():
    return cache.obtener("clf_filas_reales", _evaluacion_reales)


@etapa.paso("8", "Sólo filas reales (sin SMOTE) y sin Weight/Height: el caso sin datos sintéticos", figuras=1,
            parametros=("_modelos_reales", "_cv_reales", "_evaluacion_reales", "COLS_ENCUESTA", "GRUPOS_3"))
def paso_8_filas_reales():
    ev = evaluacion_reales()
    B7, B3, A7 = ev["B7"], ev["B3"], ev["A7"]

    tabla(ev["real_vs_sint"], "Modelos del proyecto: accuracy en las filas reales vs sintéticas del test")
    tabla(ev["conteo"].rename("filas"), "Filas reales por clase (distribución de la encuesta original)")

    tabla(B7, "Sólo filas reales · variante B · 7 clases (CV 3×10)")

    tabla(B3, "Sólo filas reales · variante B · 3 clases agrupadas (CV 3×10)")
    tabla(ev["importancias"].head(8).rename("importancia"),
          "Importancia de atributos (RF con pesos por clase, 3 clases, filas reales)")

    tabla(A7, "Referencia: sólo filas reales · variante A (con Weight/Height) · 7 clases")

    # --- Figura: macro-F1 y balanced accuracy vs línea base, 7 y 3 clases ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, t, azar, titulo in [(axes[0], B7, 1 / 7, "7 clases"), (axes[1], B3, 1 / 3, "3 clases agrupadas")]:
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["balanced_accuracy"], 0.4, label="balanced accuracy", color="#4c78a8")
        ax.bar(x + 0.2, t["f1_macro"], 0.4, yerr=t["f1_macro_std"], capsize=3,
               label="macro-F1 (± desv. entre folds)", color="#f58518")
        ax.axhline(azar, ls="--", color="gray", lw=1.2, label=f"azar en balanced acc. (1/{round(1 / azar)})")
        ax.set_xticks(x, [n.replace(" (clase mayoritaria)", "") for n in t.index], rotation=20, ha="right")
        ax.set_title(f"Filas reales, sin Weight/Height · {titulo}")
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("métrica (CV 3 folds × 10 repeticiones)")
    axes[0].set_ylim(0, 0.6)
    for ax in axes:
        ax.legend(loc="upper right", fontsize=9)
    etapa.figura(fig, "8_filas_reales_sin_WH")


if __name__ == "__main__":
    sys.exit(cli(etapa))

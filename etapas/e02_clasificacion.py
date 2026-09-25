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

Pasos (misma numeración que el notebook y la guía de estudio):
  1    Datos y split                     3.5  Postpoda (ccp_alpha)
  2    Modelos y grillas                 3.6  Curva del mejor K de KNN (A)
  3    Variante A: tabla de resultados   4    Variante B: resultados, reportes, matrices, importancias
  3.1  Reportes + matrices de confusión  4.1  Curva del mejor K de KNN (B)
  3.2  Importancia de features (RF)      5    Comparación final A vs B
  3.3  Árbol de GridSearch (dibujo)      6    Conclusiones
  3.4  Tres árboles: completo/podado/muy podado   7  Robustez ante ruido (jittering)

Los GridSearch tardan ~40 s la primera vez; después quedan en cache/ y cargan al instante.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from config import RANDOM_STATE
from src import cache
from src.data import TARGET_CLF
from src.evaluation import (clf_metrics, plot_confusion, plot_k_curve,
                            plot_n_estimators_curve, plot_pruning_curve,
                            results_table, tree_summary)
from src.pasos import Etapa, cli
from src.preprocessing import CLASS_ORDER, cargar_clf
from src.salida import nota, subtitulo, tabla

etapa = Etapa("02", "Clasificación de NObeyesdad", "clasificacion",
              "Árbol, Random Forest y KNN en dos variantes (con y sin Weight/Height)")

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
    nota("""
    `stratify=y` garantiza que las 7 clases tengan la misma proporción en train y test.
    El test (418 filas) se usa UNA sola vez por modelo, al final; todas las decisiones de
    hiperparámetros se toman con validación cruzada dentro del train (1669 filas).
    """)


# ----------------------------------------------------------------------------
# 2. Modelos y grillas
# ----------------------------------------------------------------------------
@etapa.paso("2", "Definición de modelos y grillas de hiperparámetros")
def paso_2_modelos():
    for name, (model, grid) in make_models().items():
        subtitulo(name)
        print("   modelo:", model)
        for k, v in grid.items():
            print(f"   {k}: {v}")
    nota("""
    GridSearchCV prueba todas las combinaciones de la grilla con 5-fold estratificado y se queda
    con la de mayor macro-F1 promedio. Para KNN se prueban K impares de 1 a 31 y dos formas de
    votar (uniform / distance): ésa es "la rutinita" para elegir el mejor K (pasos 3.6 y 4.1).
    """)


# ----------------------------------------------------------------------------
# 3. Variante A
# ----------------------------------------------------------------------------
@etapa.paso("3", "Variante A (todas las features): GridSearch y tabla de resultados")
def paso_3_variante_A():
    _, tabla_A, _, _ = variante("A")
    tabla(tabla_A, "Variante A — macro-F1 en CV y métricas en test")
    nota("""
    Random Forest gana (acc 0.957 / macro-F1 0.956), seguido del árbol (0.938) y KNN (0.821).
    El árbol y RF eligen max_depth=None: con Weight y Height disponibles, cuanto más crecen,
    mejor reconstruyen los cortes de IMC. KNN elige K=1 (ver paso 3.6).
    """)


@etapa.paso("3.1", "Variante A: reportes por clase y matrices de confusión (test)", figuras=1)
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
    nota("""
    Las confusiones restantes están entre clases CONTIGUAS (Overweight_I <-> Overweight_II,
    Normal <-> Overweight_I): los niveles de obesidad son intervalos consecutivos de IMC y los
    errores caen justo en las fronteras. Obesity_Type_III es casi perfecta en los tres modelos.
    """)


@etapa.paso("3.2", "Variante A: importancia de features del Random Forest", figuras=1)
def paso_3_2_importancias_A():
    fitted, _, X_tr, _ = variante("A")
    rf_A = fitted["Random Forest"].best_estimator_
    imp = pd.Series(rf_A.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    tabla(imp.rename("importancia"), "feature_importances_ (reducción media de impureza Gini)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color="steelblue")
    ax.set_title("Importancia de features — Random Forest (A)")
    etapa.figura(fig, "3.2_importancias_rf_A")
    nota("""
    Weight es de lejos la más importante (0.31, tres veces la siguiente), luego Age, Height y
    FCVC: el modelo está reconstruyendo el IMC más algunas correcciones. Es la fuga de datos
    vista desde las importancias.
    """)


@etapa.paso("3.3", "Variante A: árbol de decisión elegido por GridSearch (primeros 3 niveles)", figuras=1)
def paso_3_3_arbol_gridsearch():
    fitted, _, X_tr, _ = variante("A")
    tree_A = fitted["Árbol de decisión"].best_estimator_
    print("Profundidad del árbol:", tree_A.get_depth(), "| hojas:", tree_A.get_n_leaves())
    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(tree_A, feature_names=X_tr.columns, class_names=tree_A.classes_, max_depth=3,
              filled=True, fontsize=8, ax=ax)
    ax.set_title("Árbol de GridSearch (variante A) — dibujo truncado a 3 niveles")
    etapa.figura(fig, "3.3_arbol_gridsearch_A")


@etapa.paso("3.4", "Tres árboles: completo, podado y muy podado (prepoda)", figuras=4)
def paso_3_4_tres_arboles():
    nota("""
    El profesor pidió ver TRES árboles: uno completo, uno podado "un poquito" y uno con poda
    muy agresiva. Se entrenan a mano sobre el train de la variante A (sin GridSearch), con
    PREPODA: se limita el crecimiento antes de entrenar con max_depth y min_samples_leaf.

    | Árbol      | max_depth | min_samples_leaf |
    |------------|-----------|------------------|
    | Completo   | sin límite| 1                |
    | Podado     | 6         | 5                |
    | Muy podado | 3         | 10               |

    Para cada uno se muestra el dibujo (los dos primeros truncados a 3 niveles sólo para el
    dibujo; el muy podado completo), su profundidad, número de hojas y el atributo de la raíz
    con su umbral. criterion="gini" (por defecto).
    """)
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

    nota("""
    QUÉ PASA AL PODAR.

    - acc_train baja de 1.000 (completo: 16 niveles, 115 hojas, memoriza el train) a 0.913
      (podado) y 0.649 (muy podado): a menos capacidad, menos ajuste al train. Eso es la poda
      quitando sobreajuste.
    - acc_test también baja (0.938 -> 0.888 -> 0.653), pero mucho menos que el train en el podado:
      la brecha train–test pasa de 0.062 a 0.025. Con 6 niveles el árbol ya captura casi todo lo
      que generaliza.
    - El muy podado (3 niveles, 8 hojas) sigue lejos del azar (1/7 ≈ 0.14) porque la raíz corta por
      Weight ≤ 99.5 kg y los siguientes niveles siguen usando Weight, Height, Age y Gender: con 8
      hojas para 7 clases está aproximando el IMC a trozos. Con tan pocos cortes se queda sin hoja
      para Overweight_Level_II (esa columna de la matriz queda en cero) y mezcla Normal_Weight con
      Overweight_Level_I.
    - Los tres árboles tienen la MISMA raíz (Weight ≤ 99.536): la poda cambia la profundidad, no
      el primer corte.
    """)


@etapa.paso("3.5", "Postpoda por coste-complejidad (ccp_alpha): accuracy train/test vs alpha", figuras=1)
def paso_3_5_postpoda():
    nota("""
    Alternativa a la prepoda: se deja crecer el árbol completo y luego se CORTAN ramas con el
    parámetro ccp_alpha (poda por coste-complejidad). A mayor alpha, más ramas se eliminan. Se
    entrena un árbol por cada alpha y se mide accuracy en train y test.
    """)
    d = datos()
    path = arboles_prepoda()["Completo"].cost_complexity_pruning_path(d["X_train"], d["y_train"])
    alphas = np.unique(path.ccp_alphas)
    alphas = alphas[alphas > 0][::max(1, len(alphas) // 25)]   # ~25 valores

    score_train, score_test = [], []
    for a in alphas:
        t = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=a).fit(d["X_train"], d["y_train"])
        score_train.append(accuracy_score(d["y_train"], t.predict(d["X_train"])))
        score_test.append(accuracy_score(d["y_test"], t.predict(d["X_test"])))

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_pruning_curve(alphas, score_train, score_test, ax=ax, ylabel="accuracy")
    etapa.figura(fig, "3.5_postpoda_ccp_alpha")
    i_best = int(np.argmax(score_test))
    print(f"{len(alphas)} valores de alpha | mejor accuracy test = {score_test[i_best]:.3f} "
          f"con ccp_alpha = {alphas[i_best]:.2e}")
    nota("""
    PREPODA vs POSTPODA. La prepoda limita el crecimiento antes de entrenar (max_depth,
    min_samples_leaf); la postpoda corta ramas de un árbol ya crecido (ccp_alpha). Con alpha
    pequeño el árbol memoriza el train (accuracy 1.0) y el test se mantiene ≈0.94; a partir de
    alpha ≈ 1e-2 ambas curvas caen juntas porque el árbol queda demasiado simple.
    """)


@etapa.paso("3.6", "Curva del mejor K de KNN — variante A", figuras=1)
def paso_3_6_curva_k_A():
    fitted, _, _, _ = variante("A")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax, mejor_k = plot_k_curve(fitted["KNN"].cv_results_, ax=ax)
    etapa.figura(fig, "3.6_curva_k_knn_A")
    print("Mejor K (variante A):", mejor_k, "| params:", fitted["KNN"].best_params_)
    nota("""
    Con esta rutina determinamos que el mejor K era 1 (weights=uniform, macro-F1 de CV 0.815), y
    el rendimiento baja de forma monótona al aumentar K.

    Que gane K=1 no es lo habitual y tiene explicación: ~77% del dataset es sintético (SMOTE
    interpola nuevos puntos entre vecinos de la misma clase). Cada fila sintética tiene "hermanas"
    casi idénticas de su misma clase a distancia mínima, así que el vecino más cercano casi
    siempre acierta; al ampliar el vecindario entran puntos de clases contiguas y el macro-F1
    cae. Con datos 100% reales esperaríamos un K óptimo mayor (ver paso 7).
    """)


# ----------------------------------------------------------------------------
# 3.7 ¿Cuántos árboles necesita el Random Forest?
# ----------------------------------------------------------------------------
N_ARBOLES = [10, 25, 50, 100, 200, 400, 800]


def _barrido_n_arboles():
    """Para cada nº de árboles: macro-F1 de CV (5 folds), métricas en test y tiempo de fit.

    Se usa la variante A y la MISMA validación cruzada estratificada del GridSearch,
    para que las cifras sean comparables con las del paso 3.
    """
    d = datos()
    filas = []
    for n in N_ARBOLES:
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


@etapa.paso("3.7", "¿Cuántos árboles? Random Forest con 10, 25, 50, 100, 200, 400 y 800", figuras=1)
def paso_3_7_n_arboles():
    nota("""
    El número de árboles (n_estimators) es el hiperparámetro propio del Random Forest, y se
    comporta distinto a los demás: NO produce sobreajuste. Cada árbol se entrena sobre una
    muestra bootstrap distinta y el bosque promedia sus votos; añadir árboles sólo reduce la
    varianza de ese promedio, nunca aumenta el sesgo. Por eso la curva sube y se aplana, en
    vez de subir y volver a bajar como pasa con la profundidad del árbol o con K en KNN.

    La pregunta práctica entonces no es "cuál es el mejor número" (siempre el más alto, por un
    margen despreciable) sino "a partir de cuántos árboles deja de compensar", y eso sólo se
    responde mirando a la vez el rendimiento y el coste.
    """)
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
    nota("""
    CÓMO LEER LA GRÁFICA. La curva sube rápido al principio y después se aplana: es la forma
    típica de este hiperparámetro. Fíjate además en la banda sombreada, que es la desviación
    típica entre los 5 folds: ancha a la izquierda (con pocos árboles el promedio del bosque es
    inestable y depende de qué muestras bootstrap tocaron) y estrecha a la derecha. Buena parte
    del beneficio de añadir árboles no es subir la métrica, sino hacerla más fiable.

    El criterio para marcar el punto "suficiente" es sencillo y verificable: el menor número de
    árboles que ya consigue el 90% de toda la mejora del barrido (medida entre el bosque más
    pequeño y el mejor resultado). Se prefiere ese criterio al de "quién saca el score más alto"
    porque el más alto, en un hiperparámetro que no sobreajusta, siempre será el mayor de la
    grilla por un margen irrelevante.

    LECTURA PARA EL INFORME. El coste crece de forma aproximadamente lineal con el número de
    árboles (cada uno se entrena por separado), mientras la ganancia se agota. Por eso la
    pregunta correcta no es "¿cuál es el mejor número?" — un GridSearch que sólo mire el score
    siempre elegirá el valor más alto de la grilla — sino "¿a partir de cuántos deja de
    compensar?". Los números concretos están impresos justo arriba.

    Un matiz importante para defenderlo: a diferencia de la profundidad del árbol o de K en KNN,
    aquí NO hay sobreajuste por pasarse. Cada árbol se entrena sobre una muestra bootstrap
    distinta y el bosque promedia; añadir árboles sólo reduce la varianza de ese promedio. Por eso
    la curva se aplana en vez de bajar. Las oscilaciones pequeñas que se ven al final son ruido de
    muestreo, no degradación del modelo.
    """)


# ----------------------------------------------------------------------------
# 4. Variante B
# ----------------------------------------------------------------------------
@etapa.paso("4", "Variante B (sin Weight/Height): resultados, reportes, matrices e importancias", figuras=2)
def paso_4_variante_B():
    nota("Mismo protocolo que en A, eliminando las dos columnas que determinan el IMC.")
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

    rf_B = fitted["Random Forest"].best_estimator_
    imp_B = pd.Series(rf_B.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    tabla(imp_B.rename("importancia"), "Importancias RF (B)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=imp_B.values, y=imp_B.index, ax=ax, color="indianred")
    ax.set_title("Importancia de features — Random Forest (B, sin Weight/Height)")
    etapa.figura(fig, "4_importancias_rf_B")
    nota("""
    Sin Weight/Height el rendimiento cae ~10 puntos (RF: 0.957 -> 0.857) pero sigue muy por encima
    del azar (1/7 ≈ 0.14). Suben los hábitos y la condición física: Age, FCVC, NCP, FAF y TUE.
    """)


@etapa.paso("4.1", "Curva del mejor K de KNN — variante B", figuras=1)
def paso_4_1_curva_k_B():
    fitted, _, _, _ = variante("B")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax, mejor_k = plot_k_curve(fitted["KNN"].cv_results_, ax=ax)
    etapa.figura(fig, "4.1_curva_k_knn_B")
    print("Mejor K (variante B):", mejor_k, "| params:", fitted["KNN"].best_params_)
    nota("""
    Con esta rutina determinamos que el mejor K era 3 (weights=distance, macro-F1 de CV 0.764).
    K=1 y K=3 quedan prácticamente empatados (≈0.764) y gana K=3 por el ponderado por distancia;
    sin las dos columnas que definen el IMC el vecino único ya no basta por sí solo. De ahí en
    adelante la curva vuelve a bajar por la misma razón que en A.
    """)


# ----------------------------------------------------------------------------
# 5. Comparación final
# ----------------------------------------------------------------------------
@etapa.paso("5", "Comparación final: variante A vs variante B", figuras=1)
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
# 6. Conclusiones
# ----------------------------------------------------------------------------
@etapa.paso("6", "Conclusiones (texto) + nodo raíz e importancia de atributos")
def paso_6_conclusiones():
    nota("""
    | Variante    | Modelo              | CV macro-F1 | Test acc | Test macro-F1 |
    |-------------|---------------------|-------------|----------|---------------|
    | A (todas)   | Árbol               | 0.923       | 0.938    | 0.936         |
    | A (todas)   | Random Forest       | 0.944       | 0.957    | 0.956         |
    | A (todas)   | KNN (k=1)           | 0.815       | 0.821    | 0.808         |
    | B (sin W/H) | Árbol               | 0.744       | 0.746    | 0.740         |
    | B (sin W/H) | Random Forest       | 0.851       | 0.857    | 0.853         |
    | B (sin W/H) | KNN (k=3, distance) | 0.764       | 0.768    | 0.757         |

    - VARIANTE A: Random Forest es el mejor (acc 0.957 / macro-F1 0.956), seguido del árbol
      (0.938) y KNN (0.821). Weight es de lejos la feature más importante (0.31): el modelo está
      aprendiendo esencialmente el IMC. Las confusiones restantes son entre clases contiguas.
    - KNN queda claramente por debajo de los árboles en ambas variantes: la distancia euclídea
      mezcla variables de escalas y naturalezas muy distintas (binarias, ordinales, one-hot,
      continuas), y el escalado estándar no resuelve que Weight/Height sean las que realmente
      separan las clases. Que el mejor k sea 1 en A es coherente con que ~77% del dataset sea
      sintético.
    - VARIANTE B: al quitar Weight/Height el rendimiento cae ~10 puntos (RF: 0.957 -> 0.857), pero
      sigue muy por encima del azar. Los hábitos y la condición física sí llevan información
      sobre el nivel de obesidad. Igual conviene cautela: SMOTE infla el rendimiento de RF/KNN
      respecto a datos 100% reales.
    - PARA EL INFORME: reportar la variante A como modelo principal (es lo que pide el enunciado)
      y presentar la fuga de datos y la variante B como análisis complementario.

    6.1 NODO RAÍZ E IMPORTANCIA DE ATRIBUTOS

    Atributo raíz de cada árbol (variante A): los tres árboles del paso 3.4 y el árbol de
    GridSearch cortan la raíz por Weight ≤ 99.536. Es el primer corte porque es el que más
    reduce la impureza Gini: separa de un golpe las tres clases de obesidad de las demás.

    Top-5 de feature_importances_ del Random Forest:

    | # | Variante A (todas) | Variante B (sin Weight/Height) |
    |---|--------------------|--------------------------------|
    | 1 | Weight 0.311       | Age 0.160                      |
    | 2 | Age 0.096          | FCVC 0.142                     |
    | 3 | Height 0.095       | NCP 0.097                      |
    | 4 | FCVC 0.088         | FAF 0.096                      |
    | 5 | Gender 0.054       | TUE 0.094                      |

    En A manda Weight: el modelo reconstruye el IMC (fuga de datos). En B suben los hábitos y la
    condición física: edad, consumo de vegetales (FCVC), número de comidas (NCP), actividad
    física (FAF) y tiempo frente a pantallas (TUE). Ésos son los atributos que realmente
    describen el fenómeno, aunque con ellos solos el acierto baje de 0.957 a 0.857.
    """)


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


@etapa.paso("7", "Robustez ante ruido sintético (jittering) en variante B: 5% y barrido 0–20%", figuras=1)
def paso_7_ruido():
    nota("""
    7.1 MOTIVACIÓN: romper la artificialidad de SMOTE. ~77% de las filas fueron generadas por
    SMOTE (interpolación lineal entre vecinos). Esto crea (1) puntos cuasi-duplicados que
    favorecen a KNN con k=1, y (2) límites de decisión irreales para los árboles.

    Para evaluar la robustez empírica inyectamos ruido gaussiano controlado sobre los atributos
    continuos de hábitos:  x_ruidoso = x + ε,  ε ~ N(0, (α·σ_j)²), con α de 5% a 20% de la
    desviación estándar del atributo en train. Se evalúa en la variante B (donde los hábitos son
    las únicas variables predictivas). Los valores se recortan a rangos fisiológicos realistas.
    """)
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

    nota("""
    7.2 CONCLUSIONES DE LA PRUEBA DE ROBUSTEZ

    1. Random Forest es el modelo más robusto: pasa de 85.7% (sin ruido) a 79.0% con 20% de
       perturbación (pierde 6.7 puntos). El promedio de muchos árboles (bagging) cancela el ruido
       de media cero.
    2. El árbol individual es el más frágil: cae de 74.6% a 64.8% (casi 10 puntos; ya al 5% pierde
       5). Los cortes rígidos x <= umbral son inestables ante perturbaciones de frontera.
    3. SMOTE y KNN (k=1 vs k=3): en datos limpios k=1 superaba ligeramente a k=3 (0.770 vs 0.768)
       gracias a los cuasi-duplicados de SMOTE. Con ruido (5%–20%) k=1 baja sostenidamente hasta
       0.737, mientras k=3 se mantiene entre 0.768 y 0.778, superando a k=1 en todos los niveles.
       Lectura para la sustentación: el desempeño de k=1 era un artefacto de SMOTE; con ruido
       real, promediar la vecindad (k=3) generaliza mejor.
    """)


if __name__ == "__main__":
    sys.exit(cli(etapa))

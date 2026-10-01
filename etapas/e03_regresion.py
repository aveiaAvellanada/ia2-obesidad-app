"""ETAPA 03 — Regresión de `Weight`
===================================

Modelos: árbol de regresión, random forest y KNN.

Protocolo:
- Dataset preparado en la etapa 01 (data/reg.csv): mismas features codificadas, SIN
  NObeyesdad (la clase codifica directamente el rango de peso -> fuga de datos). Height sí se
  mantiene: es una medida física independiente, no se deriva del target.
- Split 80/20 (random_state=42), estratificado por deciles de Weight para que sea comparable
  al de clasificación.
- Hiperparámetros con GridSearchCV (5-fold, criterio RMSE negativo) sobre el train.
- KNN dentro de Pipeline con StandardScaler.
- Métricas en test: MAE, RMSE, R² y R² ajustado. Baseline que predice la media.
- Tres árboles con prepoda, postpoda por ccp_alpha y curva del mejor K de KNN.

Pasos:
  1    Datos y split                       3.1  Postpoda (ccp_alpha) con R²
  2    Modelos, grillas y ajuste (tabla)   4    Predicho vs real y residuos (modelos principales)
  2.1  Curva del mejor K de KNN            5    Importancia de features (RF)
  3    Tres árboles de regresión           6    Comparación y R² ajustado
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

from config import RANDOM_STATE
from src import cache
from src.data import TARGET_REG
from src.evaluation import (plot_k_curve, plot_n_estimators_curve, plot_pruning_curve,
                            reg_metrics, results_table, tree_summary)
from src.pasos import Etapa, cli
from src.preprocessing import cargar_reg
from src.salida import subtitulo, tabla

etapa = Etapa("03", "Regresión de Weight", "regresion",
              "Árbol, Random Forest y KNN para predecir el peso sin usar NObeyesdad")

_estado = {}


def datos():
    if "datos" not in _estado:
        df = cargar_reg()
        X = df.drop(columns=[TARGET_REG])
        y = df[TARGET_REG]
        strata = pd.qcut(y, q=10, labels=False)   # deciles de peso para estratificar
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=strata, random_state=RANDOM_STATE)
        _estado["datos"] = dict(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train,
                                y_test=y_test, n_features=X.shape[1])
    return _estado["datos"]


def make_models():
    return {
        "Árbol de regresión": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {"max_depth": [4, 6, 8, 12, None], "min_samples_leaf": [1, 3, 5, 10]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [None, 12], "min_samples_leaf": [1, 2]},
        ),
        "KNN": (
            Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor())]),
            {"knn__n_neighbors": list(range(1, 32, 2)), "knn__weights": ["uniform", "distance"]},
        ),
    }


def _ajustar():
    """GridSearch de los 3 modelos + baseline. Devuelve (fitted, results, tabla)."""
    d = datos()
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted, results = {}, {}
    dummy = DummyRegressor(strategy="mean").fit(d["X_train"], d["y_train"])
    results["Baseline (media)"] = {
        "cv_RMSE": np.nan,
        **{f"test_{k}": v for k, v in
           reg_metrics(d["y_test"], dummy.predict(d["X_test"]), n_features=d["n_features"]).items()},
    }
    for name, (model, grid) in make_models().items():
        gs = GridSearchCV(model, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(d["X_train"], d["y_train"])
        fitted[name] = gs
        y_pred = gs.predict(d["X_test"])
        results[name] = {
            "cv_RMSE": -gs.best_score_,
            **{f"test_{k}": v for k, v in
               reg_metrics(d["y_test"], y_pred, n_features=d["n_features"]).items()},
        }
        print(f"   {name}: mejores params = {gs.best_params_}")
    return fitted, results, results_table(results)


def modelos():
    """(fitted, results, tabla) — cacheado en disco."""
    return cache.obtener("reg_modelos", _ajustar)


def arboles_prepoda():
    if "arboles" not in _estado:
        d = datos()
        arboles = {
            "Completo": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "Podado": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE),
            "Muy podado": DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=RANDOM_STATE),
        }
        for a in arboles.values():
            a.fit(d["X_train"], d["y_train"])
        _estado["arboles"] = arboles
    return _estado["arboles"]


def _pred_vs_real(items, titulos, nombre_fig):
    """Panel 1x3 predicho vs real con la MISMA escala en los tres ejes."""
    d = datos()
    y_test, X_test = d["y_test"], d["X_test"]
    preds = {n: m.predict(X_test) for n, m in items.items()}
    lo = min(y_test.min(), min(p.min() for p in preds.values())) - 2
    hi = max(y_test.max(), max(p.max() for p in preds.values())) + 2
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (nombre, y_pred) in zip(axes, preds.items()):
        ax.scatter(y_test, y_pred, s=12, alpha=0.5, color="steelblue")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Real (kg)")
        ax.set_ylabel("Predicho (kg)")
        ax.set_title(titulos[nombre])
    etapa.figura(fig, nombre_fig)


# ----------------------------------------------------------------------------
@etapa.paso("1", "Datos y split (estratificado por deciles de Weight)")
def paso_1_datos():
    d = datos()
    print("train:", d["X_train"].shape, "| test:", d["X_test"].shape, "| n_features:", d["n_features"])
    print("features:", d["X"].columns.tolist())
    print("¿NObeyesdad entre las features?:", "NObeyesdad" in d["X"].columns, "(debe ser False)")
    tabla(d["y_train"].describe().round(2), "Weight en train")


@etapa.paso("2", "Modelos, grillas y ajuste con GridSearchCV (tabla con R² ajustado)")
def paso_2_modelos():
    for name, (model, grid) in make_models().items():
        subtitulo(name)
        for k, v in grid.items():
            print(f"   {k}: {v}")
    _, _, t = modelos()
    tabla(t, "Resultados: RMSE de CV y métricas en test (MAE, RMSE, R², R²_adj)")


@etapa.paso("2.1", "Curva del mejor K de KNN (RMSE de CV vs K)", figuras=1)
def paso_2_1_curva_k():
    fitted, _, _ = modelos()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax, mejor_k = plot_k_curve(fitted["KNN"].cv_results_, ylabel="RMSE (CV)", negate=True, ax=ax)
    etapa.figura(fig, "2.1_curva_k_knn")
    print("Mejor K (regresión):", mejor_k, "| params:", fitted["KNN"].best_params_)


# ----------------------------------------------------------------------------
# 2.2 ¿Cuántos árboles necesita el Random Forest?
# ----------------------------------------------------------------------------
N_ARBOLES = [10, 25, 50, 100, 200, 400, 800]


def _barrido_n_arboles(n_arboles=N_ARBOLES):
    """Para cada nº de árboles: RMSE de CV (5 folds), métricas en test y tiempo de fit."""
    d = datos()
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    filas = []
    for n in n_arboles:
        rf = RandomForestRegressor(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1)
        scores = -cross_val_score(rf, d["X_train"], d["y_train"], cv=cv,
                                  scoring="neg_root_mean_squared_error", n_jobs=-1)
        t0 = time.perf_counter()
        rf.fit(d["X_train"], d["y_train"])
        t_fit = time.perf_counter() - t0
        m = reg_metrics(d["y_test"], rf.predict(d["X_test"]), n_features=d["n_features"])
        filas.append({"n_estimators": n, "cv_RMSE": scores.mean(), "cv_std": scores.std(),
                      "test_RMSE": m["RMSE"], "test_R2": m["R2"], "tiempo_fit_s": t_fit})
    return pd.DataFrame(filas).set_index("n_estimators")


def barrido_n_arboles():
    return cache.obtener("reg_barrido_n_arboles", _barrido_n_arboles)


@etapa.paso("2.2", "¿Cuántos árboles? Random Forest con 10, 25, 50, 100, 200, 400 y 800", figuras=1)
def paso_2_2_n_arboles():
    df = barrido_n_arboles()
    tabla(df, "Random Forest (regresión): error y coste según el número de árboles", 4)

    fig, n_suf = plot_n_estimators_curve(
        df, "cv_RMSE", "test_RMSE", "RMSE (kg)",
        "Random Forest: RMSE según el número de árboles",
        col_std="cv_std", n_proyecto=200, mejor="min")
    etapa.figura(fig, "2.2_n_estimators_rf")

    cv_r, t = df["cv_RMSE"], df["tiempo_fit_s"]
    n_min, n_max, n_mejor = int(df.index.min()), int(df.index.max()), int(cv_r.idxmin())
    print(f"\nRMSE de CV: {cv_r.loc[n_min]:.4f} kg con {n_min} árboles  ->  {cv_r.loc[n_max]:.4f} kg "
          f"con {n_max} (mejora total {cv_r.loc[n_min] - cv_r.loc[n_max]:.4f} kg; el mejor valor "
          f"de la tabla se da con {n_mejor} árboles)")
    print(f"Con {n_suf} árboles ya se consigue el 90% de toda la mejora del barrido.")
    print(f"Pasar de {n_suf} a {n_max} árboles multiplica el tiempo de entrenamiento por "
          f"{t.loc[n_max] / max(float(t.loc[n_suf]), 1e-9):.1f} "
          f"({t.loc[n_suf]:.2f} s -> {t.loc[n_max]:.2f} s) a cambio de "
          f"{cv_r.loc[n_suf] - cv_r.loc[n_max]:.4f} kg de RMSE.")


@etapa.paso("3", "Tres árboles de regresión: completo, podado y muy podado", figuras=4)
def paso_3_tres_arboles():
    d = datos()
    X = d["X"]
    resumen = []
    for nombre, modelo in arboles_prepoda().items():
        s = tree_summary(modelo, X.columns, nombre)
        m = reg_metrics(d["y_test"], modelo.predict(d["X_test"]), n_features=d["n_features"])
        resumen.append({**s, **m})
        print(s)
        max_d = None if nombre == "Muy podado" else 3
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(modelo, max_depth=max_d, feature_names=X.columns, filled=True, rounded=True,
                  ax=ax, fontsize=9 if nombre == "Muy podado" else 8)
        ax.set_title(f"Árbol {nombre} — profundidad {s['profundidad']}, {s['n_hojas']} hojas, "
                     f"raíz: {s['raiz_atributo']} ≤ {s['raiz_umbral']}"
                     + (" (truncado a profundidad 3 para visualización)" if max_d else ""))
        etapa.figura(fig, f"3_arbol_{nombre.lower().replace(' ', '_')}")

    cols = ["modelo", "profundidad", "n_hojas", "raiz_atributo", "MAE", "RMSE", "R2", "R2_adj"]
    tabla(pd.DataFrame(resumen)[cols].set_index("modelo"), "Resumen de los tres árboles")

    titulos = {}
    for nombre, modelo in arboles_prepoda().items():
        m = reg_metrics(d["y_test"], modelo.predict(d["X_test"]), n_features=d["n_features"])
        titulos[nombre] = (f"Árbol {nombre}\nRMSE={m['RMSE']:.2f} kg  R²={m['R2']:.3f}  "
                           f"R²adj={m['R2_adj']:.3f}")
    _pred_vs_real(arboles_prepoda(), titulos, "3_pred_vs_real_tres_arboles")


@etapa.paso("3.1", "Postpoda por coste-complejidad (ccp_alpha): R² train/test vs alpha", figuras=1)
def paso_3_1_postpoda():
    d = datos()
    path = arboles_prepoda()["Completo"].cost_complexity_pruning_path(d["X_train"], d["y_train"])
    alphas = np.unique(path.ccp_alphas)
    alphas = alphas[alphas > 0][::max(1, len(alphas) // 25)]
    score_train, score_test = [], []
    for a in alphas:
        t = DecisionTreeRegressor(random_state=RANDOM_STATE, ccp_alpha=a).fit(d["X_train"], d["y_train"])
        score_train.append(r2_score(d["y_train"], t.predict(d["X_train"])))
        score_test.append(r2_score(d["y_test"], t.predict(d["X_test"])))
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_pruning_curve(alphas, score_train, score_test, ax=ax, ylabel="R²")
    etapa.figura(fig, "3.1_postpoda_ccp_alpha")
    i_best = int(np.argmax(score_test))
    print(f"{len(alphas)} valores de alpha | mejor R² test = {score_test[i_best]:.3f} "
          f"con ccp_alpha = {alphas[i_best]:.2e}")


@etapa.paso("4", "Predicho vs real y residuos — modelos principales (test)", figuras=2)
def paso_4_pred_vs_real():
    fitted, results, _ = modelos()
    titulos = {name: (f"{name}\nRMSE={results[name]['test_RMSE']:.2f} kg  "
                      f"R²={results[name]['test_R2']:.3f}  R²adj={results[name]['test_R2_adj']:.3f}")
               for name in fitted}
    _pred_vs_real(fitted, titulos, "4_pred_vs_real_modelos")

    d = datos()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (name, gs) in zip(axes, fitted.items()):
        resid = d["y_test"] - gs.predict(d["X_test"])
        sns.histplot(resid, bins=40, kde=True, ax=ax, color="steelblue")
        ax.axvline(0, c="r", ls="--", lw=1.2)
        ax.set_title(f"Residuos — {name}")
        ax.set_xlabel("Real − Predicho (kg)")
        ax.set_ylabel("Frecuencia")
    etapa.figura(fig, "4_residuos_modelos")


@etapa.paso("5", "Importancia de features (Random Forest)", figuras=1)
def paso_5_importancias():
    fitted, _, _ = modelos()
    d = datos()
    rf = fitted["Random Forest"].best_estimator_
    imp = pd.Series(rf.feature_importances_, index=d["X_train"].columns).sort_values(ascending=False)
    tabla(imp.rename("importancia"), "feature_importances_")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color="steelblue")
    ax.set_title("Importancia de features — Random Forest (regresión de Weight)")
    ax.set_xlabel("Importancia (reducción de impureza)")
    etapa.figura(fig, "5_importancias_rf")


@etapa.paso("6", "Comparación de modelos y explicación del R² ajustado", figuras=1)
def paso_6_comparacion():
    _, _, t = modelos()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    t["test_RMSE"].plot.bar(ax=axes[0], color="steelblue", title="RMSE en test (kg)")
    t["test_R2"].plot.bar(ax=axes[1], color="indianred", title="R² en test")
    t["test_R2_adj"].plot.bar(ax=axes[2], color="forestgreen", title="R² ajustado en test")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    etapa.figura(fig, "6_comparacion_metricas")
    tabla(t, "Tabla de resultados")


if __name__ == "__main__":
    sys.exit(cli(etapa))

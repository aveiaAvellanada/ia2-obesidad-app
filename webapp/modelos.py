"""Núcleo de clasificación y regresión de la app (sin Streamlit, para poder probarlo).

Todo se cachea por parámetros con lru_cache: dos combinaciones distintas de hiperparámetros
nunca comparten resultado. Los objetos devueltos son compartidos: no se deben modificar.
"""
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (KFold, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.data import TARGET_CLF, TARGET_REG
from src.evaluation import clf_metrics, reg_metrics, results_table
from src.preprocessing import cargar_clf, cargar_reg
from etapas import e02_clasificacion as e02

MODELOS_CLF = ("Árbol de decisión", "Random Forest", "KNN")
MODELOS_REG = ("Árbol de regresión", "Random Forest", "KNN")
DROP_WH = ["Weight", "Height"]

HP_CLF = {
    "Árbol de decisión": {"criterion": "gini", "max_depth": 10, "min_samples_leaf": 1},
    "Random Forest": {"n_estimators": 200, "max_depth": 0, "min_samples_leaf": 1,
                      "max_features": "sqrt"},
    "KNN": {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"},
}
HP_REG = {
    "Árbol de regresión": {"criterion": "squared_error", "max_depth": 8, "min_samples_leaf": 5},
    "Random Forest": {"n_estimators": 200, "max_depth": 0, "min_samples_leaf": 1,
                      "max_features": "todas"},
    "KNN": {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"},
}


def _prof(v):
    """0 (o None) = sin límite de profundidad."""
    return None if not v else int(v)


def _mf(v):
    return None if v == "todas" else v


def features_clf():
    return [c for c in cargar_clf().columns if c != TARGET_CLF]


def features_reg():
    return [c for c in cargar_reg().columns if c != TARGET_REG]


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _split_clf(features, test_size, seed):
    df = cargar_clf()
    X, y = df[list(features)], df[TARGET_CLF].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    return dict(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def split_clf(features, test_size, seed):
    return _split_clf(tuple(features), float(test_size), int(seed))


@lru_cache(maxsize=32)
def _split_reg(features, test_size, seed):
    df = cargar_reg()
    X, y = df[list(features)], df[TARGET_REG]
    strata = pd.qcut(y, q=10, labels=False)   # deciles de peso, como en la etapa 03
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=strata, random_state=seed)
    return dict(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                n_features=X.shape[1])


def split_reg(features, test_size, seed):
    return _split_reg(tuple(features), float(test_size), int(seed))


# ----------------------------------------------------------------------------
# Construcción de modelos
# ----------------------------------------------------------------------------
def _knn(cls, p):
    return Pipeline([("scaler", StandardScaler()),
                     ("knn", cls(n_neighbors=int(p["n_neighbors"]), weights=p["weights"],
                                 metric=p["metric"]))])


def construir_clf(nombre, p, seed):
    if nombre == "Árbol de decisión":
        return DecisionTreeClassifier(criterion=p["criterion"], max_depth=_prof(p["max_depth"]),
                                      min_samples_leaf=int(p["min_samples_leaf"]),
                                      random_state=seed)
    if nombre == "Random Forest":
        return RandomForestClassifier(n_estimators=int(p["n_estimators"]),
                                      max_depth=_prof(p["max_depth"]),
                                      min_samples_leaf=int(p["min_samples_leaf"]),
                                      max_features=_mf(p["max_features"]),
                                      random_state=seed, n_jobs=-1)
    if nombre == "KNN":
        return _knn(KNeighborsClassifier, p)
    raise KeyError(nombre)


def construir_reg(nombre, p, seed):
    if nombre == "Árbol de regresión":
        return DecisionTreeRegressor(criterion=p["criterion"], max_depth=_prof(p["max_depth"]),
                                     min_samples_leaf=int(p["min_samples_leaf"]),
                                     random_state=seed)
    if nombre == "Random Forest":
        return RandomForestRegressor(n_estimators=int(p["n_estimators"]),
                                     max_depth=_prof(p["max_depth"]),
                                     min_samples_leaf=int(p["min_samples_leaf"]),
                                     max_features=_mf(p["max_features"]),
                                     random_state=seed, n_jobs=-1)
    if nombre == "KNN":
        return _knn(KNeighborsRegressor, p)
    raise KeyError(nombre)


# ----------------------------------------------------------------------------
# Entrenamiento (cacheado por parámetros)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=128)
def _ajustar_clf(nombre, p_json, features, test_size, seed, con_cv):
    d = _split_clf(features, test_size, seed)
    modelo = construir_clf(nombre, json.loads(p_json), seed)
    cv = float("nan")
    if con_cv:
        cv = float(cross_val_score(
            modelo, d["X_train"], d["y_train"], scoring="f1_macro",
            cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean())
    modelo.fit(d["X_train"], d["y_train"])
    y_pred = modelo.predict(d["X_test"])
    return dict(modelo=modelo, cv_f1_macro=cv, y_pred=y_pred,
                **{f"test_{k}": v for k, v in clf_metrics(d["y_test"], y_pred).items()})


def ajustar_clf(nombre, p, features, test_size, seed, con_cv=True):
    return _ajustar_clf(nombre, json.dumps(p, sort_keys=True), tuple(features),
                        float(test_size), int(seed), bool(con_cv))


@lru_cache(maxsize=128)
def _ajustar_reg(nombre, p_json, features, test_size, seed, con_cv):
    d = _split_reg(features, test_size, seed)
    modelo = construir_reg(nombre, json.loads(p_json), seed)
    cv = float("nan")
    if con_cv:
        cv = float(-cross_val_score(
            modelo, d["X_train"], d["y_train"], scoring="neg_root_mean_squared_error",
            cv=KFold(5, shuffle=True, random_state=seed)).mean())
    modelo.fit(d["X_train"], d["y_train"])
    y_pred = modelo.predict(d["X_test"])
    return dict(modelo=modelo, cv_RMSE=cv, y_pred=y_pred,
                **{f"test_{k}": v for k, v in
                   reg_metrics(d["y_test"], y_pred, n_features=d["n_features"]).items()})


def ajustar_reg(nombre, p, features, test_size, seed, con_cv=True):
    return _ajustar_reg(nombre, json.dumps(p, sort_keys=True), tuple(features),
                        float(test_size), int(seed), bool(con_cv))


def tabla_clf(hp, features, test_size, seed, con_cv=True):
    """(res, tabla, split): los tres modelos entrenados con los hiperparámetros `hp`."""
    res = {n: ajustar_clf(n, hp[n], features, test_size, seed, con_cv) for n in MODELOS_CLF}
    tabla = results_table({n: {"cv_f1_macro": r["cv_f1_macro"],
                               "test_accuracy": r["test_accuracy"],
                               "test_f1_macro": r["test_f1_macro"]} for n, r in res.items()})
    return res, tabla, split_clf(features, test_size, seed)


def tabla_reg(hp, features, test_size, seed, con_cv=True):
    """(res, tabla, split) con la fila 'Baseline (media)' al principio."""
    d = split_reg(features, test_size, seed)
    dummy = DummyRegressor(strategy="mean").fit(d["X_train"], d["y_train"])
    filas = {"Baseline (media)": {
        "cv_RMSE": float("nan"),
        **{f"test_{k}": v for k, v in reg_metrics(
            d["y_test"], dummy.predict(d["X_test"]), n_features=d["n_features"]).items()}}}
    res = {n: ajustar_reg(n, hp[n], features, test_size, seed, con_cv) for n in MODELOS_REG}
    cols = ["cv_RMSE", "test_MAE", "test_RMSE", "test_R2", "test_R2_adj"]
    for n, r in res.items():
        filas[n] = {c: r[c] for c in cols}
    return res, results_table(filas), d


# ----------------------------------------------------------------------------
# Barridos
# ----------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _curva_k(tarea, features, test_size, seed, ks, weights):
    if tarea == "clf":
        d = _split_clf(features, test_size, seed)
        cls, scoring = KNeighborsClassifier, "f1_macro"
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    else:
        d = _split_reg(features, test_size, seed)
        cls, scoring = KNeighborsRegressor, "neg_root_mean_squared_error"
        cv = KFold(5, shuffle=True, random_state=seed)
    scores = []
    for k in ks:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("knn", cls(n_neighbors=int(k), weights=weights))])
        scores.append(float(cross_val_score(pipe, d["X_train"], d["y_train"],
                                            scoring=scoring, cv=cv).mean()))
    return {"param_knn__n_neighbors": list(ks), "mean_test_score": scores}


def curva_k(tarea, features, test_size, seed, ks, weights):
    return _curva_k(tarea, tuple(features), float(test_size), int(seed), tuple(ks), weights)


@lru_cache(maxsize=16)
def _barrido_arboles(tarea, features, test_size, seed, ns, max_depth, min_samples_leaf):
    clf = tarea == "clf"
    d = (_split_clf if clf else _split_reg)(features, test_size, seed)
    cv = (StratifiedKFold if clf else KFold)(5, shuffle=True, random_state=seed)
    filas = []
    for n in ns:
        kw = dict(n_estimators=int(n), max_depth=_prof(max_depth),
                  min_samples_leaf=int(min_samples_leaf), random_state=seed, n_jobs=-1)
        rf = RandomForestClassifier(**kw) if clf else RandomForestRegressor(**kw)
        scoring = "f1_macro" if clf else "neg_root_mean_squared_error"
        s = cross_val_score(rf, d["X_train"], d["y_train"], cv=cv, scoring=scoring)
        t0 = time.perf_counter()
        rf.fit(d["X_train"], d["y_train"])
        t_fit = time.perf_counter() - t0
        if clf:
            m = clf_metrics(d["y_test"], rf.predict(d["X_test"]))
            filas.append({"n_estimators": n, "cv_f1_macro": s.mean(), "cv_std": s.std(),
                          "test_accuracy": m["accuracy"], "test_f1_macro": m["f1_macro"],
                          "tiempo_fit_s": t_fit})
        else:
            m = reg_metrics(d["y_test"], rf.predict(d["X_test"]), n_features=d["n_features"])
            filas.append({"n_estimators": n, "cv_RMSE": -s.mean(), "cv_std": s.std(),
                          "test_RMSE": m["RMSE"], "test_R2": m["R2"], "tiempo_fit_s": t_fit})
    return pd.DataFrame(filas).set_index("n_estimators")


def barrido_arboles(tarea, features, test_size, seed, ns, max_depth, min_samples_leaf):
    return _barrido_arboles(tarea, tuple(features), float(test_size), int(seed),
                            tuple(ns), int(max_depth), int(min_samples_leaf))


@lru_cache(maxsize=8)
def _barrido_ruido(test_size, seed, niveles):
    d = _split_clf(tuple(features_clf()), test_size, seed)
    X_tr, X_te = d["X_train"].drop(columns=DROP_WH), d["X_test"].drop(columns=DROP_WH)
    filas = []
    for nl in niveles:
        Xa, Xb = e02._con_ruido(X_tr, X_te, nl) if nl > 0 else (X_tr, X_te)
        for nombre, modelo in e02._modelos_robustez().items():
            modelo.fit(Xa, d["y_train"])
            p = modelo.predict(Xb)
            filas.append({"nivel": int(round(nl * 100)), "modelo": nombre,
                          "accuracy": accuracy_score(d["y_test"], p),
                          "f1_macro": f1_score(d["y_test"], p, average="macro")})
    return pd.DataFrame(filas)


def barrido_ruido(test_size, seed, niveles):
    return _barrido_ruido(float(test_size), int(seed), tuple(float(n) for n in niveles))


@lru_cache(maxsize=16)
def _evaluar_reales(incluir_wh, n_clases, repeticiones, seed):
    df = cargar_clf()
    X, y = df[features_clf()], df[TARGET_CLF].astype(str)
    real = e02.filas_reales(X)
    X, y = X[real], y[real]
    if not incluir_wh:
        X = X.drop(columns=DROP_WH)
    if n_clases == 3:
        y = y.map(e02.GRUPOS_3)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=repeticiones, random_state=seed)
    filas = {}
    for nombre, modelo in e02._modelos_reales().items():
        s = cross_validate(modelo, X, y, cv=cv, scoring=e02.METRICAS_REALES, n_jobs=-1)
        filas[nombre] = {**{m: s[f"test_{m}"].mean() for m in e02.METRICAS_REALES},
                         "f1_macro_std": s["test_f1_macro"].std()}
    return pd.DataFrame(filas).T


def evaluar_reales(incluir_wh, n_clases, repeticiones, seed):
    return _evaluar_reales(bool(incluir_wh), int(n_clases), int(repeticiones), int(seed))

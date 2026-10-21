import pytest

from webapp import modelos as M

F = tuple(M.features_clf())
FR = tuple(M.features_reg())


def test_split_estratificado_reproducible_y_cacheado():
    a, b = M.split_clf(F, 0.2, 42), M.split_clf(F, 0.2, 42)
    assert a is b
    assert len(a["X_test"]) / len(a["X"]) == pytest.approx(0.2, abs=0.01)
    assert set(a["y_train"]) == set(a["y_test"])


def test_split_reg_no_incluye_weight_ni_nobeyesdad():
    d = M.split_reg(FR, 0.2, 42)
    assert "Weight" not in d["X"].columns and "NObeyesdad" not in d["X"].columns
    assert d["n_features"] == len(FR)


def test_max_depth_cero_es_sin_limite_y_max_features_todas_es_none():
    arbol = M.construir_clf("Árbol de decisión",
                            {"criterion": "gini", "max_depth": 0, "min_samples_leaf": 1}, 42)
    assert arbol.max_depth is None
    rf = M.construir_clf("Random Forest", {"n_estimators": 10, "max_depth": 7,
                                           "min_samples_leaf": 2, "max_features": "todas"}, 42)
    assert rf.max_depth == 7 and rf.max_features is None and rf.min_samples_leaf == 2
    rf2 = M.construir_reg("Random Forest", {"n_estimators": 10, "max_depth": 0,
                                            "min_samples_leaf": 1, "max_features": "sqrt"}, 42)
    assert rf2.max_depth is None and rf2.max_features == "sqrt"


def test_parametros_distintos_dan_resultados_distintos():
    poco = M.ajustar_clf("Árbol de decisión", {"criterion": "gini", "max_depth": 2,
                                               "min_samples_leaf": 1}, F, 0.2, 42, con_cv=False)
    mucho = M.ajustar_clf("Árbol de decisión", {"criterion": "gini", "max_depth": 0,
                                                "min_samples_leaf": 1}, F, 0.2, 42, con_cv=False)
    assert poco["modelo"].get_depth() == 2
    assert mucho["modelo"].get_depth() > 2
    assert mucho["test_accuracy"] > poco["test_accuracy"]


def test_tabla_clf_tiene_los_tres_modelos_y_las_columnas_esperadas():
    res, tabla, split = M.tabla_clf(M.HP_CLF, F, 0.2, 42, con_cv=False)
    assert list(tabla.index) == list(M.MODELOS_CLF)
    assert {"cv_f1_macro", "test_accuracy", "test_f1_macro"} <= set(tabla.columns)
    assert tabla["test_accuracy"].between(0, 1).all()
    assert len(res["KNN"]["y_pred"]) == len(split["y_test"])


def test_tabla_reg_incluye_baseline_y_el_rf_lo_supera():
    _, tabla, _ = M.tabla_reg(M.HP_REG, FR, 0.2, 42, con_cv=False)
    assert tabla.index[0] == "Baseline (media)"
    assert {"cv_RMSE", "test_MAE", "test_RMSE", "test_R2", "test_R2_adj"} <= set(tabla.columns)
    assert tabla.loc["Random Forest", "test_R2"] > tabla.loc["Baseline (media)", "test_R2"]


def test_curva_k_tiene_el_formato_de_plot_k_curve():
    r = M.curva_k("clf", F, 0.2, 42, (1, 3, 5), "uniform")
    assert r["param_knn__n_neighbors"] == [1, 3, 5]
    assert len(r["mean_test_score"]) == 3
    rr = M.curva_k("reg", FR, 0.2, 42, (1, 3), "distance")
    assert all(s < 0 for s in rr["mean_test_score"])   # -RMSE


def test_barrido_arboles_clf_y_reg():
    df = M.barrido_arboles("clf", F, 0.2, 42, (10, 25), 0, 1)
    assert list(df.index) == [10, 25]
    assert {"cv_f1_macro", "cv_std", "test_f1_macro", "tiempo_fit_s"} <= set(df.columns)
    dr = M.barrido_arboles("reg", FR, 0.2, 42, (10, 25), 0, 1)
    assert {"cv_RMSE", "cv_std", "test_RMSE", "test_R2", "tiempo_fit_s"} <= set(dr.columns)


def test_barrido_ruido_incluye_el_nivel_cero_y_cuatro_modelos():
    df = M.barrido_ruido(0.2, 42, (0.0, 0.05))
    assert sorted(df["nivel"].unique()) == [0, 5]
    assert df["modelo"].nunique() == 4


def test_evaluar_reales_devuelve_la_linea_base_y_metricas():
    t = M.evaluar_reales(False, 3, 1, 42)
    assert "Línea base (clase mayoritaria)" in t.index
    assert {"accuracy", "balanced_accuracy", "f1_macro", "f1_macro_std"} <= set(t.columns)

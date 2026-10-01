"""Cada paso de la app se renderiza con sus parámetros por defecto sin excepciones ni errores."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from webapp.navegacion import PASOS

RAIZ = Path(__file__).resolve().parents[1]
CASOS = [(e, i) for e, lista in PASOS.items() for i, _ in lista]


def abrir(etapa, paso, **estado):
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=600)
    at.session_state["etapa"] = etapa
    at.session_state[f"paso_{etapa}"] = paso
    for clave, valor in estado.items():
        at.session_state[clave] = valor
    return at.run()


@pytest.mark.parametrize("etapa,paso", CASOS, ids=[f"{e}-{i}" for e, i in CASOS])
def test_paso_se_renderiza(etapa, paso):
    at = abrir(etapa, paso)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.subheader[0].value.startswith(paso)


def test_variante_b_sin_features_muestra_un_aviso_y_no_falla():
    solo_wh = {"features": ["Weight", "Height"], "test_size": 0.2, "seed": 42}
    for paso in ("4", "4.1", "7"):
        at = abrir("02", paso, **{"params::split_clf": solo_wh})
        assert not at.exception, [e.value for e in at.exception]
        if paso != "7":    # el paso 7 siempre usa todas las features menos Weight/Height
            assert at.warning, f"el paso {paso} debería avisar"


def test_sin_features_se_avisa_con_un_error_claro():
    vacio = {"features": [], "test_size": 0.2, "seed": 42}
    at = abrir("02", "3", **{"params::split_clf": vacio})
    assert not at.exception
    assert at.error and "al menos una feature" in at.error[0].value


@pytest.mark.parametrize("k,init,max_iter", [(3, "random", 1), (7, "k-means++", 30),
                                             (12, "random", 100)])
def test_iteraciones_kmeans_con_parametros_extremos(k, init, max_iter):
    at = abrir("04", "3.5", **{"params::lloyd": {"k": k, "init": init, "seed": 42,
                                                  "max_iter": max_iter}})
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    filas = len(at.dataframe[-1].value)
    if max_iter == 1:
        assert filas == 1 and at.info and "No convergió" in at.info[0].value
    else:
        assert 1 <= filas <= max_iter


def test_iteraciones_kmeans_slider_y_reproduccion():
    at = abrir("04", "3.5")
    n = len(at.dataframe[-1].value)
    at.slider(key="e04_iter").set_value(n - 1).run()
    assert not at.exception, [e.value for e in at.exception]
    at.button(key="e04_play").click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_comparacion_final_usa_los_parametros_activos():
    def tabla_a(at):
        return next(d.value for d in at.dataframe
                    if "Árbol de decisión" in d.value.index and "test_accuracy" in d.value.columns)

    def en_cache():
        return set((RAIZ / "cache").glob("*")) if (RAIZ / "cache").exists() else set()

    antes = en_cache()
    at = abrir("05", "1")
    base = tabla_a(at).loc["Árbol de decisión", "test_accuracy"]
    titulos = [m.value for m in at.markdown]
    assert any("CV 3×3" in t for t in titulos) and not any("3×10" in t for t in titulos)
    at.session_state["params::hp_clf"] = {
        "Árbol de decisión": {"criterion": "gini", "max_depth": 2, "min_samples_leaf": 1},
        "Random Forest": {"n_estimators": 20, "max_depth": 0, "min_samples_leaf": 1,
                          "max_features": "sqrt"},
        "KNN": {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"}}
    at.session_state["params::reales"] = {"incluir_wh": False, "repeticiones": 2}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert tabla_a(at).loc["Árbol de decisión", "test_accuracy"] < base
    assert any("CV 3×2" in m.value for m in at.markdown)
    assert en_cache() == antes      # la app no escribe en cache/


# El multiselect de números de árboles acepta valores escritos a mano: llegan como texto.
@pytest.mark.parametrize("etapa,paso,tarea", [("02", "3.7", "clf"), ("03", "2.2", "reg")])
def test_numeros_de_arboles_escritos_a_mano(etapa, paso, tarea):
    p = {"ns": [10, "30", " 75 ", "30"], "max_depth": 0, "min_samples_leaf": 1}
    at = abrir(etapa, paso, **{f"params::narb_{tarea}": p})
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    tabla = next(d.value for d in at.dataframe if "tiempo_fit_s" in d.value.columns)
    assert list(tabla.index) == [10, 30, 75]
    assert set(at.multiselect(key=f"narb_{tarea}_ns").options) >= {"10", "30", "75", "800"}


@pytest.mark.parametrize("malos", [["abc"], ["0"], ["2.5"], ["-4"], ["5000"]])
def test_numeros_de_arboles_no_validos_dan_un_error_claro(malos):
    p = {"ns": [10, 25, *malos], "max_depth": 0, "min_samples_leaf": 1}
    at = abrir("02", "3.7", **{"params::narb_clf": p})
    assert not at.exception, [e.value for e in at.exception]
    assert at.error and malos[0] in at.error[0].value


def test_el_multiselect_de_arboles_admite_opciones_nuevas():
    at = abrir("02", "3.7")
    ms = at.multiselect(key="narb_clf_ns")
    ms.set_value([10, 25, 60]).run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    tabla = next(d.value for d in at.dataframe if "tiempo_fit_s" in d.value.columns)
    assert list(tabla.index) == [10, 25, 60]

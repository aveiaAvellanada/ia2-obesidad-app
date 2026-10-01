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

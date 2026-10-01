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

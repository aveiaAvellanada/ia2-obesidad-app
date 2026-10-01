"""El menú de cada etapa indica dónde empieza y termina cada paso, y el CLI no escribe archivos
salvo que se le pida con --guardar."""
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import salida

RAIZ = Path(__file__).resolve().parents[1]


def test_cada_paso_indica_su_linea_de_inicio_y_de_fin():
    from etapas import e02_clasificacion as e02
    for paso in e02.etapa.pasos:
        lineas, inicio = inspect.getsourcelines(paso.fn)
        fin = inicio + len(lineas) - 1
        assert paso.ubicacion == f"etapas/e02_clasificacion.py:{inicio} (hasta {fin})"
        fuente = (RAIZ / "etapas/e02_clasificacion.py").read_text(encoding="utf-8").splitlines()
        assert fuente[inicio - 1].startswith(f'@etapa.paso("{paso.id}"')
        assert fuente[fin - 1].strip()                       # última línea del cuerpo
    assert "(hasta " in e02.etapa.pasos[0].etiqueta


def _entradas(carpeta):
    return set(carpeta.rglob("*")) if carpeta.exists() else set()


def test_ejecutar_un_paso_solo_muestra_y_no_escribe_archivos():
    carpetas = [RAIZ / "figures", RAIZ / "resultados_texto"]
    antes = [_entradas(c) for c in carpetas]
    r = subprocess.run([sys.executable, "etapas/e01_eda.py", "--paso", "2.1", "--sin-ventanas"],
                       cwd=RAIZ, capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
                       timeout=300)
    assert r.returncode == 0, r.stderr
    assert "Distribución de las 7 clases" in r.stdout and "Filas por clase" in r.stdout
    assert "[figura guardada]" not in r.stdout
    assert [_entradas(c) for c in carpetas] == antes


def test_sin_pantalla_las_figuras_se_guardan_en_vez_de_perderse(monkeypatch):
    opciones = dict(salida.OPCIONES)
    monkeypatch.setattr(salida, "_backend_grafico_funciona", lambda: False)
    try:
        salida.configurar(mostrar=True, guardar=False)
        assert salida.OPCIONES["mostrar"] is False and salida.OPCIONES["guardar"] is True
    finally:
        salida.OPCIONES.update(opciones)
        salida.configurar(mostrar=False)


def test_guardar_tablas_tiene_sus_dependencias():
    """--guardar y run_all.py exportan las tablas con DataFrame.to_markdown() (tabulate)."""
    import pandas as pd
    md = pd.DataFrame({"a": [1]}).to_markdown()      # sin tabulate lanza ImportError
    assert md.splitlines()[0].split("|")[2].strip() == "a" and "|---" in md


# ----------------------------------------------------------------------------
# Dónde se modifican los parámetros de cada paso
# ----------------------------------------------------------------------------
def _etapas():
    from etapas import (e01_eda, e02_clasificacion, e03_regresion, e04_clustering,
                        e05_comparacion)
    return [m.etapa for m in (e01_eda, e02_clasificacion, e03_regresion, e04_clustering,
                              e05_comparacion)]


def _linea(ubicacion):
    archivo, linea = ubicacion.rsplit(":", 1)
    return (RAIZ / archivo).read_text(encoding="utf-8").splitlines()[int(linea) - 1]


def test_todos_los_parametros_declarados_apuntan_a_su_definicion():
    from src.pasos import ubicar
    for etapa in _etapas():
        modulo = etapa.pasos[0].fn.__globals__
        nombres = list(etapa.globales) + [n for p in etapa.pasos for n in p.parametros]
        for nombre in nombres:
            ubicacion = ubicar(nombre, modulo)
            assert ubicacion, f"etapa {etapa.numero}: no se encuentra {nombre!r}"
            corto = nombre.rsplit(".", 1)[-1]
            linea = _linea(ubicacion)
            assert f"def {corto}(" in linea or linea.startswith(f"{corto} ="), (nombre, linea)


def test_el_paso_de_numero_de_arboles_apunta_a_n_arboles():
    from etapas import e02_clasificacion as e02
    paso = e02.etapa.buscar("3.7")
    assert "N_ARBOLES" in paso.parametros
    assert paso.linea_parametros.startswith("N_ARBOLES etapas/e02_clasificacion.py:")
    assert _linea(paso.linea_parametros.split()[1]).startswith("N_ARBOLES = [")


def test_la_semilla_se_resuelve_en_config_y_va_en_la_cabecera(capsys):
    from etapas import e02_clasificacion as e02
    from src.pasos import ubicar
    assert _linea(ubicar("RANDOM_STATE", e02.etapa.pasos[0].fn.__globals__)).startswith(
        "RANDOM_STATE = ")
    e02.etapa.listar()
    salida_menu = capsys.readouterr().out.splitlines()
    cabecera = next(l for l in salida_menu if "Comunes a todos los pasos:" in l)
    assert "RANDOM_STATE config.py:" in cabecera and "datos etapas/e02_clasificacion.py:" in cabecera
    i = next(i for i, l in enumerate(salida_menu) if l.lstrip().startswith("[ 3.7]"))
    assert salida_menu[i + 1].strip().startswith("N_ARBOLES etapas/e02_clasificacion.py:")


def test_un_paso_sin_parametros_externos_no_lleva_segunda_linea(capsys):
    from etapas import e01_eda as e01
    assert e01.etapa.buscar("2.2").linea_parametros == ""
    e01.etapa.listar()
    lineas = capsys.readouterr().out.splitlines()
    i = next(i for i, l in enumerate(lineas) if l.lstrip().startswith("[ 2.2]"))
    assert lineas[i + 1].lstrip().startswith("[ 2.3]")


def test_los_nombres_con_punto_se_resuelven_en_otra_etapa():
    from etapas import e05_comparacion as e05
    from src.pasos import ubicar
    u = ubicar("e02.make_models", e05.etapa.pasos[0].fn.__globals__)
    assert u.startswith("etapas/e02_clasificacion.py:") and "def make_models(" in _linea(u)

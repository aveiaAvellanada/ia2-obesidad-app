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

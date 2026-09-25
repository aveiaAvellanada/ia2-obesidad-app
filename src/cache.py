"""Caché en disco de resultados costosos (GridSearchCV, barridos de K-Means).

La primera vez que se pide algo se calcula y se guarda en cache/<nombre>.joblib.
Las siguientes veces se carga en < 1 s. Así, en el menú, pedir "ver la matriz de
confusión" no obliga a repetir 40 s de GridSearch. `limpiar()` borra todo.
"""
from pathlib import Path

import joblib

from config import CACHE_DIR

_memoria = {}  # caché en RAM para la sesión actual


def obtener(nombre: str, calcular):
    """Devuelve el resultado de `calcular()` usando RAM -> disco -> cálculo."""
    if nombre in _memoria:
        return _memoria[nombre]
    ruta = Path(CACHE_DIR) / f"{nombre}.joblib"
    if ruta.exists():
        print(f"   [caché] cargando {ruta.name} (borra la carpeta cache/ para recalcular)")
        valor = joblib.load(ruta)
    else:
        print(f"   [calculando] {nombre} ... (la primera vez tarda; luego queda en cache/)")
        valor = calcular()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(valor, ruta)
    _memoria[nombre] = valor
    return valor


def limpiar():
    _memoria.clear()
    n = 0
    for f in Path(CACHE_DIR).glob("*.joblib"):
        f.unlink()
        n += 1
    print(f"Caché limpiada ({n} archivos).")

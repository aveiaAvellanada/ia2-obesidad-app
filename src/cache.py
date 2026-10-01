"""Caché en disco de resultados costosos (GridSearchCV, barridos de K-Means).

La primera vez que se pide algo se calcula y se guarda en cache/<nombre>.joblib.
Las siguientes veces se carga en < 1 s. Así, en el menú, pedir "ver la matriz de
confusión" no obliga a repetir 40 s de GridSearch. `limpiar()` borra todo.

Cada resultado se guarda con una HUELLA del código del que depende: el archivo de la etapa
que lo calcula y los módulos del proyecto que importa (src/, config.py, otras etapas). Si
cambias un hiperparámetro en el código (p. ej. N_ARBOLES o la grilla de make_models()) y
guardas, la huella cambia y el resultado se recalcula solo; si no tocas nada, se carga de
disco. No hace falta borrar cache/ a mano.
"""
import ast
import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

import joblib

from config import CACHE_DIR, ROOT

_memoria = {}  # caché en RAM para la sesión actual: nombre -> (huella, valor)

# Carpetas y archivos del proyecto que cuentan para la huella (no .venv ni librerías)
_CARPETAS_PROYECTO = ("etapas", "src", "webapp")
_ARCHIVOS_PROYECTO = ("config.py",)


def _ruta_del_proyecto(archivo):
    """Ruta relativa a la raíz si `archivo` es código del proyecto; None si no lo es."""
    if not archivo:
        return None
    try:
        rel = Path(archivo).resolve().relative_to(Path(ROOT).resolve())
    except ValueError:
        return None
    if rel.parts[0] in _CARPETAS_PROYECTO or rel.as_posix() in _ARCHIVOS_PROYECTO:
        return rel
    return None


def _archivo_de_modulo(nombre):
    """Archivo .py del módulo `nombre` (sin ejecutarlo); None si no es un módulo."""
    modulo = sys.modules.get(nombre)
    if modulo is not None:
        return getattr(modulo, "__file__", None)
    try:
        spec = importlib.util.find_spec(nombre)
    except (ImportError, ValueError):
        return None
    return spec.origin if spec is not None and spec.has_location else None


def _importados(ruta):
    """Nombres de los módulos que importa un archivo, leídos de su código con ast.
    `from a import b` da 'a' y 'a.b' (b puede ser un submódulo o sólo un nombre)."""
    nombres = set()
    for nodo in ast.walk(ast.parse(ruta.read_text(encoding="utf-8"))):
        if isinstance(nodo, ast.Import):
            nombres.update(a.name for a in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.level == 0:
            nombres.add(nodo.module)
            nombres.update(f"{nodo.module}.{a.name}" for a in nodo.names)
    return nombres


def _dependencias(calcular) -> dict:
    """{ruta relativa: ruta absoluta} de los archivos del proyecto de los que depende
    `calcular`: el archivo donde está definido y, recursivamente, los archivos del proyecto
    que ese archivo importa (también los `from config import RANDOM_STATE`)."""
    pendientes, rutas = [inspect.getsourcefile(calcular)], {}
    while pendientes:
        archivo = pendientes.pop()
        rel = _ruta_del_proyecto(archivo)
        if rel is None or rel in rutas:
            continue
        rutas[rel] = Path(archivo).resolve()
        pendientes.extend(_archivo_de_modulo(n) for n in _importados(rutas[rel]))
    return rutas


def archivos_de(calcular) -> list[Path]:
    """Archivos del proyecto de los que depende `calcular`, relativos a la raíz y ordenados."""
    return sorted(_dependencias(calcular))


def huella(calcular) -> str:
    """SHA-256 del contenido de los archivos de los que depende `calcular`."""
    h = hashlib.sha256()
    for rel, ruta in sorted(_dependencias(calcular).items()):
        h.update(rel.as_posix().encode())
        h.update(ruta.read_bytes())
    return h.hexdigest()


def obtener(nombre: str, calcular):
    """Devuelve el resultado de `calcular()` usando RAM -> disco -> cálculo.

    Lo guardado sólo se reutiliza si el código del que depende no ha cambiado."""
    h = huella(calcular)
    if nombre in _memoria and _memoria[nombre][0] == h:
        return _memoria[nombre][1]
    ruta = Path(CACHE_DIR) / f"{nombre}.joblib"
    guardado = None
    if ruta.exists():
        try:
            guardado = joblib.load(ruta)
        except Exception:  # archivo corrupto o de otra versión de las librerías
            guardado = None
    if isinstance(guardado, dict) and set(guardado) == {"huella", "valor"} and guardado["huella"] == h:
        print(f"   [caché] cargando {ruta.name} (el código del que depende no ha cambiado)")
        valor = guardado["valor"]
    else:
        if ruta.exists():
            print(f"   [recalculando] {nombre}: cambió el código del que depende ... (tarda; luego queda en cache/)")
        else:
            print(f"   [calculando] {nombre} ... (la primera vez tarda; luego queda en cache/)")
        valor = calcular()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"huella": h, "valor": valor}, ruta)
    _memoria[nombre] = (h, valor)
    return valor


def limpiar():
    _memoria.clear()
    n = 0
    for f in Path(CACHE_DIR).glob("*.joblib"):
        f.unlink()
        n += 1
    print(f"Caché limpiada ({n} archivos).")

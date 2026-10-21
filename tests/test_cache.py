"""La caché en disco se invalida sola cuando cambia el código del que depende el cálculo."""
import importlib
import sys

import joblib
import pytest

from src import cache

CODIGO = '''
from src import cache

LLAMADAS = []
N_ARBOLES = {valores}


def calcular():
    LLAMADAS.append(1)
    return sum(N_ARBOLES)


def obtener():
    return cache.obtener("prueba_huella", calcular)
'''


@pytest.fixture
def modulo(tmp_path, monkeypatch):
    """Un módulo 'etapas_prueba/e99_prueba.py' temporal dentro de un proyecto falso en
    tmp_path (otro nombre de carpeta para no chocar con el paquete real `etapas`)."""
    (tmp_path / "etapas_prueba").mkdir()
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache, "ROOT", tmp_path)
    monkeypatch.setattr(cache, "_CARPETAS_PROYECTO", ("etapas_prueba",))
    monkeypatch.syspath_prepend(str(tmp_path))
    cache._memoria.clear()
    archivo = tmp_path / "etapas_prueba" / "e99_prueba.py"

    def cargar(valores):
        archivo.write_text(CODIGO.format(valores=valores), encoding="utf-8")
        for nombre in ("etapas_prueba.e99_prueba", "etapas_prueba"):
            sys.modules.pop(nombre, None)
        importlib.invalidate_caches()
        cache._memoria.clear()          # como si fuera una ejecución nueva del programa
        return importlib.import_module("etapas_prueba.e99_prueba")

    yield cargar
    for nombre in ("etapas_prueba.e99_prueba", "etapas_prueba"):
        sys.modules.pop(nombre, None)
    cache._memoria.clear()


def test_sin_cambios_carga_de_disco_sin_recalcular(modulo):
    m = modulo("[10, 20]")
    assert m.obtener() == 30 and len(m.LLAMADAS) == 1
    m2 = modulo("[10, 20]")
    assert m2.obtener() == 30 and len(m2.LLAMADAS) == 0


def test_si_cambia_el_codigo_recalcula(modulo):
    assert modulo("[10, 20]").obtener() == 30
    m = modulo("[10, 20, 400]")         # se editó N_ARBOLES y se guardó el archivo
    assert m.obtener() == 430 and len(m.LLAMADAS) == 1
    m = modulo("[10, 20, 400]")
    assert m.obtener() == 430 and len(m.LLAMADAS) == 0


def test_una_cache_del_formato_antiguo_se_recalcula(modulo, tmp_path):
    (tmp_path / "cache").mkdir()
    joblib.dump(999, tmp_path / "cache" / "prueba_huella.joblib")   # valor suelto, sin huella
    m = modulo("[1, 2]")
    assert m.obtener() == 3 and len(m.LLAMADAS) == 1


def test_un_archivo_de_cache_corrupto_se_recalcula(modulo, tmp_path):
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "prueba_huella.joblib").write_bytes(b"no es un joblib")
    m = modulo("[1, 2]")
    assert m.obtener() == 3 and len(m.LLAMADAS) == 1


def test_la_huella_de_una_etapa_incluye_src_y_config():
    from etapas import e02_clasificacion as e02
    archivos = cache.archivos_de(e02._barrido_n_arboles)
    nombres = {p.as_posix() for p in archivos}
    assert "etapas/e02_clasificacion.py" in nombres
    assert {"src/evaluation.py", "src/preprocessing.py", "config.py"} <= nombres
    assert not any(n.startswith(("etapas/e03", "etapas/e04", "webapp/")) for n in nombres)
    assert not any(".venv" in n for n in nombres)

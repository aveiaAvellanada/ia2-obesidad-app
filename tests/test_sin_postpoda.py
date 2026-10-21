"""La postpoda (ccp_alpha) no debe quedar en ningún .py del proyecto."""
import os
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PATRON = re.compile(r"ccp_alpha|postpoda|cost_complexity|pruning_path|plot_pruning_curve", re.I)
IGNORAR = {".venv", ".venv-1", ".git", "tests", "docs", "__pycache__", "cache", "figures"}


def _archivos_py():
    for dirpath, dirnames, filenames in os.walk(RAIZ):
        dirnames[:] = [d for d in dirnames if d not in IGNORAR]
        for f in filenames:
            if f.endswith(".py"):
                yield Path(dirpath) / f


def test_no_queda_postpoda_en_el_codigo():
    hallazgos = [
        f"{p.relative_to(RAIZ).as_posix()}:{i}: {linea.strip()}"
        for p in _archivos_py()
        for i, linea in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if PATRON.search(linea)
    ]
    assert hallazgos == [], "\n".join(hallazgos)


def test_las_etapas_no_registran_pasos_de_postpoda():
    from etapas import e02_clasificacion as e02, e03_regresion as e03
    assert "3.5" not in [p.id for p in e02.etapa.pasos]
    assert "3.1" not in [p.id for p in e03.etapa.pasos]

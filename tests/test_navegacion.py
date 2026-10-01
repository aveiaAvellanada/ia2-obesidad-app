"""El registro de pasos debe ser el del HTML (mismos ids y títulos), más los añadidos."""
import html
import re
from pathlib import Path

from webapp.navegacion import ETAPAS, PASOS

RAIZ = Path(__file__).resolve().parents[1]
ANADIDOS = {("01", "2.2"), ("01", "2.4"), ("04", "3.5")}
TEXTO = {("02", "6"), ("03", "7"), ("04", "4.5"), ("04", "6"), ("05", "2"), ("05", "3")}


def _pasos_html():
    s = (RAIZ / "estudio_etapas.html").read_text(encoding="utf-8")
    patron = (r'<article class="step" id="etapa(\d\d)-p[^"]+"><header class="step-head">'
              r'<span class="step-id">([^<]*)</span><h3>(.*?)</h3>')
    return {(e, i): html.unescape(re.sub(r"<.*?>", "", t)).strip()
            for e, i, t in re.findall(patron, s)}


def _registro():
    return {(e, i): t for e, lista in PASOS.items() for i, t in lista}


def test_las_etapas_son_cinco():
    assert list(ETAPAS) == ["01", "02", "03", "04", "05"]
    assert set(PASOS) == set(ETAPAS)


def test_titulos_iguales_al_html():
    h, r = _pasos_html(), _registro()
    for clave, titulo in r.items():
        if clave in ANADIDOS:
            continue
        assert clave in h, f"{clave} no existe en el HTML"
        assert h[clave].startswith(titulo), f"{clave}: {titulo!r} vs {h[clave]!r}"


def test_solo_faltan_los_pasos_de_texto_y_solo_sobran_los_anadidos():
    h, r = _pasos_html(), _registro()
    assert set(h) - set(r) == TEXTO
    assert set(r) - set(h) == ANADIDOS


def test_orden_de_los_pasos_es_el_del_html():
    h = list(_pasos_html())
    r = [(e, i) for e, lista in PASOS.items() for i, _ in lista if (e, i) not in ANADIDOS]
    assert r == [c for c in h if c not in TEXTO]

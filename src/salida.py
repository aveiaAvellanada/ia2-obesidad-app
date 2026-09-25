"""Cómo se muestran las cosas: texto en consola y figuras (ventana + PNG).

Hay dos modos, controlados por `configurar()`:
- mostrar=True  -> cada figura abre una VENTANA (hay que cerrarla para seguir).
- guardar=True  -> cada figura se guarda como PNG en figures/<etapa>/<nombre>.png

`run_all.py` usa mostrar=False (sólo guarda). El menú de `main.py` usa ambos.

TABLAS COMO FIGURA
------------------
Además de imprimirse en la consola, cada tabla (`tabla()`) se guarda también como
imagen en `figures/<etapa>/tablas/<paso>_tablaN_<titulo>.png`:

- tabla numérica de UNA columna      -> gráfico de barras horizontales
- tabla numérica de 2 a 6 columnas   -> un panel de barras por columna (escalas independientes)
- cualquier otra (texto, muy ancha)  -> la tabla dibujada como imagen, con sus valores

Estas figuras NO abren ventana (sólo se guardan), para no tener que cerrar decenas
de ventanas al recorrer los pasos. Se pueden apagar con `configurar(graficar_tablas=False)`
o pasando `grafica=False` a una llamada concreta de `tabla()`.
"""
import numbers
import os
import re
import textwrap
import unicodedata
from pathlib import Path

import matplotlib
import pandas as pd

from config import FIG_DIR, ESTILO_MPL

OPCIONES = {"mostrar": True, "guardar": True, "graficar_tablas": True}
_backend_listo = False

# Etapa/paso en curso: lo fija src/pasos.py para poder nombrar las figuras de las tablas
_CONTEXTO = {"carpeta": "tablas_sueltas", "paso": "suelta", "n_tabla": 0}


def configurar(mostrar: bool | None = None, guardar: bool | None = None,
               graficar_tablas: bool | None = None):
    """Fija las opciones de salida y elige el backend de matplotlib.

    QtAgg (PyQt6) abre ventanas; Agg no abre nada y sólo sirve para guardar.
    """
    global _backend_listo
    if mostrar is not None:
        OPCIONES["mostrar"] = mostrar
    if guardar is not None:
        OPCIONES["guardar"] = guardar
    if graficar_tablas is not None:
        OPCIONES["graficar_tablas"] = graficar_tablas

    if OPCIONES["mostrar"]:
        if _backend_grafico_funciona():
            matplotlib.use("QtAgg")
        else:
            print("[aviso] No se pudo iniciar el backend gráfico (QtAgg). Las gráficas NO se abrirán "
                  "en ventana; sólo se guardarán como PNG en figures/.\n"
                  "        Revisa que PyQt5 esté instalado en el .venv (pip install -r requirements.txt)\n"
                  "        y que estés en una sesión con pantalla (no por SSH sin -X).")
            OPCIONES["mostrar"] = False
            matplotlib.use("Agg")
    else:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(ESTILO_MPL)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 40)
    _backend_listo = True


def _backend_grafico_funciona() -> bool:
    """Prueba QtAgg en un subproceso: si Qt falla, falla allí y no tumba este programa."""
    import subprocess
    import sys

    prueba = ("import matplotlib; matplotlib.use('QtAgg'); import matplotlib.pyplot as plt; "
              "f = plt.figure(); f.canvas.manager; plt.close(f)")
    try:
        r = subprocess.run([sys.executable, "-c", prueba], capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def _asegurar():
    if not _backend_listo:
        configurar()


def fijar_contexto(carpeta: str, paso: str):
    """Registra en qué etapa/paso estamos, para nombrar las figuras de las tablas.

    Lo llama `Etapa.ejecutar()` (src/pasos.py) antes de cada paso; reinicia el
    contador de tablas para que numeren 1, 2, 3... dentro del paso.
    """
    _CONTEXTO["carpeta"] = carpeta
    _CONTEXTO["paso"] = paso
    _CONTEXTO["n_tabla"] = 0


def mostrar(fig, nombre: str, etapa: str, ventana: bool | None = None):
    """Guarda la figura en figures/<etapa>/<nombre>.png y/o la abre en ventana.

    `ventana=False` fuerza que no se abra ventana aunque el modo interactivo esté
    activo (lo usan las tablas convertidas en figura).
    """
    _asegurar()
    import matplotlib.pyplot as plt

    abrir = OPCIONES["mostrar"] if ventana is None else (ventana and OPCIONES["mostrar"])

    if OPCIONES["guardar"]:
        carpeta = Path(FIG_DIR) / etapa
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / f"{nombre}.png"
        fig.savefig(ruta, bbox_inches="tight")
        print(f"   [figura guardada] {ruta.relative_to(FIG_DIR.parent)}")
    if abrir:
        print("   [ventana abierta] cierra la ventana de la gráfica para continuar...")
        # Variable de entorno para pruebas automáticas: cierra la ventana sola tras N ms
        auto_ms = os.environ.get("IA2_AUTOCERRAR_MS")
        if auto_ms:
            t = fig.canvas.new_timer(interval=int(auto_ms))
            t.add_callback(lambda: plt.close("all"))
            t.single_shot = True
            t.start()
        plt.show()
    plt.close(fig)


# ---------- Texto ----------

def titulo(texto: str):
    print("\n" + "=" * 100)
    print(texto)
    print("=" * 100)


def subtitulo(texto: str):
    print("\n--- " + texto + " ---")


def nota(texto: str):
    """Imprime una explicación (equivalente a las celdas markdown del notebook)."""
    print()
    for parrafo in textwrap.dedent(texto).strip().split("\n\n"):
        lineas = parrafo.splitlines()
        # Las tablas / listas se imprimen tal cual; los párrafos se reajustan a 100 columnas
        if any(l.lstrip().startswith(("|", "-", "*", "1", "2", "3", "4", "5")) for l in lineas):
            print("\n".join(lineas))
        else:
            print(textwrap.fill(" ".join(l.strip() for l in lineas), width=100))
        print()


def tabla(df, titulo_tabla: str | None = None, decimales: int = 4, grafica: bool = True,
          texto: bool = True):
    """Imprime un DataFrame/Series completo, redondeado, y lo guarda además como figura.

    `grafica=False` imprime sólo el texto (sin generar PNG) para esa tabla concreta.
    `texto=False` hace lo contrario: sólo genera la figura (útil cuando el texto ya
    se imprimió con otro formato, como el classification_report de sklearn).
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    d = df.round(decimales)
    if texto:
        if titulo_tabla:
            subtitulo(titulo_tabla)
        with pd.option_context("display.max_rows", 200, "display.max_columns", 40,
                               "display.width", 140):
            print(d.to_string())
        print()

    if grafica and OPCIONES["graficar_tablas"] and OPCIONES["guardar"]:
        _guardar_figura_de_tabla(d, titulo_tabla, decimales)


# ---------- Tablas convertidas en figura ----------

_MAX_FILAS_IMAGEN = 40      # filas que caben legibles en la imagen de una tabla
_MAX_COLS_IMAGEN = 14       # columnas idem
_MAX_FILAS_BARRAS = 40      # más filas que esto: mejor imagen que barras
_MAX_FILAS_PANELES = 20
_MAX_COLS_PANELES = 6

_AZUL = "#4c78a8"
_AZUL_OSCURO = "#40567a"


def _slug(texto: str, limite: int = 40) -> str:
    """'Resultados: RMSE de CV' -> 'resultados_rmse_de_cv' (para el nombre del PNG)."""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    t = re.sub(r"[^\w]+", "_", t).strip("_").lower()
    return t[:limite].strip("_") or "tabla"


def _etiqueta(v, ancho: int = 22) -> str:
    """Texto de un índice o encabezado (aplana los MultiIndex y recorta lo muy largo)."""
    s = " · ".join(str(x) for x in v) if isinstance(v, tuple) else str(v)
    return s if len(s) <= ancho else s[:ancho - 1] + "…"


def _es_nulo(v) -> bool:
    try:
        return v is None or bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _decimales_necesarios(serie, tope: int) -> int:
    """Decimales mínimos que hacen falta en la columna (para no escribir 64.0000)."""
    valores = [float(v) for v in serie if not _es_nulo(v)]
    if not valores:
        return 0
    for d in range(0, max(int(tope), 0) + 1):
        if all(round(v, d) == round(v, int(tope)) for v in valores):
            return d
    return max(int(tope), 0)


def _formatear_columna(serie, decimales: int, ancho: int = 24) -> list:
    """Formatea una columna entera con el mismo criterio (misma cantidad de decimales)."""
    if pd.api.types.is_bool_dtype(serie):
        return ["—" if _es_nulo(v) else ("sí" if bool(v) else "no") for v in serie]
    if pd.api.types.is_integer_dtype(serie):
        return ["—" if _es_nulo(v) else f"{int(v):,}" for v in serie]
    if pd.api.types.is_float_dtype(serie):
        d = _decimales_necesarios(serie, decimales)
        return ["—" if _es_nulo(v) else f"{float(v):,.{d}f}" for v in serie]

    salida_col = []
    for v in serie:
        if _es_nulo(v):
            salida_col.append("—")
        elif isinstance(v, numbers.Integral) and not isinstance(v, bool):
            salida_col.append(f"{int(v):,}")
        elif isinstance(v, numbers.Real) and not isinstance(v, bool):
            f = float(v)
            salida_col.append(f"{int(f):,}" if f.is_integer() else f"{f:,.{max(int(decimales), 1)}f}")
        else:
            s = str(v)
            salida_col.append(s if len(s) <= ancho else s[:ancho - 1] + "…")
    return salida_col


def _es_numerica(df) -> bool:
    return df.shape[1] > 0 and all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)


def _fig_barras(df, titulo_tabla):
    """Tabla de una sola columna numérica -> barras horizontales con su valor."""
    import matplotlib.pyplot as plt

    col = df.columns[0]
    serie = df[col].astype(float)
    etiquetas = [_etiqueta(i, 28) for i in serie.index]
    alto = min(max(2.6, 0.42 * len(serie) + 1.6), 12)

    fig, ax = plt.subplots(figsize=(9, alto))
    barras = ax.barh(range(len(serie)), serie.values, color=_AZUL)
    ax.set_yticks(range(len(serie)))
    ax.set_yticklabels(etiquetas)
    ax.invert_yaxis()
    ax.set_xlabel(str(col))
    ax.bar_label(barras, fmt="%.4g", padding=3, fontsize=9)
    ax.margins(x=0.15)
    if titulo_tabla:
        ax.set_title(titulo_tabla)
    return fig


def _fig_paneles(df, titulo_tabla):
    """Tabla numérica de 2-6 columnas -> un panel de barras por columna.

    Se usa un panel por columna (y no barras agrupadas) porque las columnas suelen
    tener escalas muy distintas (RMSE ~8 junto a R² ~0.89).
    """
    import matplotlib.pyplot as plt

    columnas = list(df.columns)
    etiquetas = [_etiqueta(i, 20) for i in df.index]
    n_filas = len(df)
    ancho = min(max(3.4, 0.45 * n_filas + 2.4) * len(columnas), 22)
    alto = max(3.4, 0.16 * max(len(e) for e in etiquetas) + 2.8)

    fig, axes = plt.subplots(1, len(columnas), figsize=(ancho, alto), squeeze=False)
    for ax, c in zip(axes[0], columnas):
        valores = df[c].astype(float).values
        barras = ax.bar(range(n_filas), valores, color=_AZUL)
        ax.set_xticks(range(n_filas))
        ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=8)
        ax.set_title(_etiqueta(c, 30), fontsize=10)
        ax.bar_label(barras, fmt="%.4g", fontsize=7, padding=2)
        ax.margins(y=0.18)
    if titulo_tabla:
        fig.suptitle(titulo_tabla)
    return fig


def _fig_tabla_imagen(df, titulo_tabla, decimales):
    """Cualquier otra tabla -> se dibuja tal cual como imagen, con sus valores."""
    import matplotlib.pyplot as plt

    d = df
    aviso = ""
    if len(d) > _MAX_FILAS_IMAGEN:
        aviso += f"  (primeras {_MAX_FILAS_IMAGEN} de {len(d)} filas)"
        d = d.head(_MAX_FILAS_IMAGEN)
    if d.shape[1] > _MAX_COLS_IMAGEN:
        aviso += f"  (primeras {_MAX_COLS_IMAGEN} de {df.shape[1]} columnas)"
        d = d.iloc[:, :_MAX_COLS_IMAGEN]

    encabezados = [""] + [_etiqueta(c, 18) for c in d.columns]
    # Se formatea columna por columna para que todas usen los mismos decimales
    columnas_txt = [_formatear_columna(d[c], decimales) for c in d.columns]
    celdas = [[_etiqueta(ix, 26)] + [col[i] for col in columnas_txt]
              for i, ix in enumerate(d.index)]

    n_f, n_c = len(celdas) or 1, len(encabezados)
    ancho_texto = [max(len(str(encabezados[j])), *(len(f[j]) for f in celdas)) if celdas
                   else len(str(encabezados[j])) for j in range(n_c)]
    ancho = min(max(5.5, 0.13 * sum(ancho_texto) + 1.2 * n_c), 22)
    alto = min(max(1.6, 0.34 * (n_f + 1) + (0.55 if titulo_tabla else 0.15)), 14)

    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.axis("off")
    t = ax.table(cellText=celdas or [[""] * n_c], colLabels=encabezados,
                 cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(max(6.0, min(10.0, 130.0 / n_c)))
    # ancho de cada columna proporcional al texto más largo que contiene
    total = sum(ancho_texto) or 1
    for (fila, col), celda in t.get_celld().items():
        celda.set_width(max(ancho_texto[col] / total, 0.04))
    for (fila, col), celda in t.get_celld().items():
        celda.set_edgecolor("#cfd6e0")
        if fila == 0:
            celda.set_text_props(weight="bold", color="white")
            celda.set_facecolor(_AZUL_OSCURO)
        elif col == 0:
            celda.set_text_props(weight="bold")
            celda.set_facecolor("#eef1f6")
        elif fila % 2 == 0:
            celda.set_facecolor("#f7f9fc")
    if titulo_tabla:
        ax.set_title(str(titulo_tabla) + aviso, pad=14)
    return fig


def _figura_de_tabla(df, titulo_tabla, decimales):
    """Elige la mejor representación gráfica según la forma de la tabla."""
    if _es_numerica(df):
        if df.shape[1] == 1 and 1 <= len(df) <= _MAX_FILAS_BARRAS:
            return _fig_barras(df, titulo_tabla)
        if 2 <= df.shape[1] <= _MAX_COLS_PANELES and 1 <= len(df) <= _MAX_FILAS_PANELES:
            return _fig_paneles(df, titulo_tabla)
    return _fig_tabla_imagen(df, titulo_tabla, decimales)


def _guardar_figura_de_tabla(df, titulo_tabla, decimales):
    """Genera el PNG de la tabla en figures/<etapa>/tablas/. Nunca interrumpe el paso."""
    _asegurar()
    _CONTEXTO["n_tabla"] += 1
    nombre = f"{_CONTEXTO['paso']}_tabla{_CONTEXTO['n_tabla']}"
    if titulo_tabla:
        nombre += "_" + _slug(titulo_tabla)
    try:
        fig = _figura_de_tabla(df, titulo_tabla, decimales)
    except Exception as e:  # una tabla rara no debe tumbar el paso entero
        print(f"   [aviso] no se pudo dibujar esta tabla como figura ({type(e).__name__}: {e});"
              " queda sólo el texto de arriba.")
        return
    mostrar(fig, nombre, f"{_CONTEXTO['carpeta']}/tablas", ventana=False)

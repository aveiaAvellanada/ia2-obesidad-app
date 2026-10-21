"""Cómo se muestran las cosas: texto en consola y figuras (ventana + PNG).

Hay dos modos, controlados por `configurar()`:
- mostrar=True  -> cada figura abre una VENTANA (hay que cerrarla para seguir).
- guardar=True  -> cada figura se guarda como PNG en figures/<etapa>/<nombre>.png

El menú de cada etapa y el de `main.py` sólo MUESTRAN (ventanas + consola, sin escribir
archivos); `run_all.py` y la opción --guardar son los que guardan.

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

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import FIG_DIR, ESTILO_MPL, ROOT
import numpy as np

RESULTADOS_TEXTO_DIR = ROOT / "resultados_texto"
_PASOS_INICIADOS = set()

OPCIONES = {"mostrar": True, "guardar": True, "graficar_tablas": True}
_backend_listo = False

# Receptor opcional de la salida (p. ej. una interfaz gráfica). Si es None todo va a
# la consola como siempre; si no, títulos, notas, tablas y figuras se le entregan a él con
# sus métodos titulo(texto), subtitulo(texto), nota(texto), tabla(df, titulo) y figura(fig, nombre).
RECEPTOR = None

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
                  "en ventana; se guardarán como PNG en figures/ para no perderlas.\n"
                  "        Revisa que PyQt5 esté instalado en el .venv (pip install -r requirements.txt)\n"
                  "        y que estés en una sesión con pantalla (no por SSH sin -X).")
            OPCIONES["mostrar"] = False
            OPCIONES["guardar"] = True   # sin ventana ni PNG la figura se perdería
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


def mostrar(fig, nombre: str, etapa: str, ventana: bool | None = None, datos=None,
            es_tabla: bool = False):
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
        if not es_tabla:
            _guardar_figura_texto(fig, nombre, etapa, datos=datos)
    if RECEPTOR is not None and ventana is not False:
        RECEPTOR.figura(fig, nombre)
    elif abrir:
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
    if RECEPTOR is not None:
        return RECEPTOR.titulo(texto)
    print("\n" + "=" * 100)
    print(texto)
    print("=" * 100)


def subtitulo(texto: str):
    if RECEPTOR is not None:
        return RECEPTOR.subtitulo(texto)
    print("\n--- " + texto + " ---")


def nota(texto: str):
    """Imprime una explicación (equivalente a las celdas markdown del notebook)."""
    if RECEPTOR is not None:
        return RECEPTOR.nota(textwrap.dedent(texto).strip())
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
    if RECEPTOR is not None and texto:
        RECEPTOR.tabla(d, titulo_tabla)
    elif texto:
        if titulo_tabla:
            subtitulo(titulo_tabla)
        with pd.option_context("display.max_rows", 200, "display.max_columns", 40,
                               "display.width", 140):
            print(d.to_string())
        print()

    _CONTEXTO["n_tabla"] += 1
    nombre = f"{_CONTEXTO['paso']}_tabla{_CONTEXTO['n_tabla']}"
    if titulo_tabla:
        nombre += "_" + _slug(titulo_tabla)

    if OPCIONES["guardar"]:
        _guardar_tabla_texto(d, titulo_tabla, decimales, nombre)

    if grafica and OPCIONES["graficar_tablas"] and OPCIONES["guardar"]:
        _guardar_figura_de_tabla(d, titulo_tabla, decimales, nombre)


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


def _formatear_df(df: pd.DataFrame, decimales: int) -> pd.DataFrame:
    """Aplica el mismo formateo numérico que la imagen de la tabla."""
    df_f = pd.DataFrame(index=df.index)
    for col in df.columns:
        df_f[col] = _formatear_columna(df[col], decimales)
    return df_f


def _guardar_tabla_texto(df: pd.DataFrame, titulo_tabla: str | None, decimales: int, nombre: str):
    """Guarda la tabla en Markdown y CSV en resultados_texto/ y figures/."""
    etapa = _CONTEXTO["carpeta"]
    paso = str(_CONTEXTO["paso"])
    df_formateado = _formatear_df(df, decimales)

    md_titulo = f"### {titulo_tabla}\n\n" if titulo_tabla else f"### Tabla {nombre}\n\n"
    md_contenido = md_titulo + df_formateado.to_markdown() + "\n\n"

    # 1. Archivo del paso: resultados_texto/<etapa>/<paso>.md
    dir_etapa_res = RESULTADOS_TEXTO_DIR / etapa
    dir_etapa_res.mkdir(parents=True, exist_ok=True)
    ruta_paso_md = dir_etapa_res / f"{paso}.md"

    modo = "a" if (etapa, paso) in _PASOS_INICIADOS else "w"
    _PASOS_INICIADOS.add((etapa, paso))
    with open(ruta_paso_md, modo, encoding="utf-8") as f:
        if modo == "w":
            f.write(f"# Etapa {etapa} · Paso {paso}\n\n")
        f.write(md_contenido)

    # 2. Archivos individuales en resultados_texto/<etapa>/tablas/
    dir_tablas_res = dir_etapa_res / "tablas"
    dir_tablas_res.mkdir(parents=True, exist_ok=True)
    with open(dir_tablas_res / f"{nombre}.md", "w", encoding="utf-8") as f:
        f.write(md_contenido)
    df_formateado.to_csv(dir_tablas_res / f"{nombre}.csv", encoding="utf-8")

    # 3. Junto al PNG en figures/<etapa>/tablas/
    dir_tablas_fig = Path(FIG_DIR) / etapa / "tablas"
    dir_tablas_fig.mkdir(parents=True, exist_ok=True)
    with open(dir_tablas_fig / f"{nombre}.md", "w", encoding="utf-8") as f:
        f.write(md_contenido)
    df_formateado.to_csv(dir_tablas_fig / f"{nombre}.csv", encoding="utf-8")


def extraer_datos_figura(fig, nombre: str, datos=None):
    """Extrae datos numéricos y estructurados de una figura de matplotlib."""
    if datos is not None:
        if isinstance(datos, pd.DataFrame):
            return f"### {nombre}\n\n" + datos.to_markdown() + "\n\n", [(nombre, datos)]
        elif isinstance(datos, pd.Series):
            df_s = datos.to_frame()
            return f"### {nombre}\n\n" + df_s.to_markdown() + "\n\n", [(nombre, df_s)]
        elif isinstance(datos, dict):
            blocks, dfs = [], []
            for k, v in datos.items():
                if isinstance(v, (pd.DataFrame, pd.Series)):
                    v_df = v.to_frame() if isinstance(v, pd.Series) else v
                    blocks.append(f"#### {k}\n\n" + v_df.to_markdown())
                    dfs.append((f"{nombre}_{_slug(k)}", v_df))
            if blocks:
                return f"### {nombre}\n\n" + "\n\n".join(blocks) + "\n\n", dfs

    md_parts = []
    dfs = []
    suptitle = fig._suptitle.get_text() if hasattr(fig, "_suptitle") and fig._suptitle else ""

    for idx, ax in enumerate(fig.axes):
        ax_title = ax.get_title() or (suptitle if len(fig.axes) == 1 else f"Panel_{idx+1}")

        # 1. Decision Tree nodes
        tree_texts = [t.get_text().strip() for t in ax.texts if any(k in t.get_text() for k in ("samples =", "gini =", "squared_error =", "value ="))]
        if tree_texts:
            node_records = []
            for t in ax.texts:
                txt = t.get_text().strip()
                if not txt or txt in ("True", "False"):
                    continue
                lines = [l.strip() for l in txt.split("\n") if l.strip()]
                rec = {"nodo": len(node_records) + 1}
                for l in lines:
                    if "<=" in l:
                        rec["condicion"] = l
                    elif "=" in l:
                        k, v = l.split("=", 1)
                        rec[k.strip()] = v.strip()
                    else:
                        rec["info"] = l
                node_records.append(rec)
            df_tree = pd.DataFrame(node_records)
            md_parts.append(f"#### {ax_title or 'Estructura del árbol'}\n\n" + df_tree.to_markdown() + "\n")
            dfs.append((f"{nombre}_{_slug(ax_title or 'arbol')}", df_tree))
            continue

        # 2. QuadMesh / imshow (heatmaps / matrices de confusión / contingencia)
        meshes = [c for c in ax.collections if hasattr(c, "get_array") and "QuadMesh" in type(c).__name__]
        if meshes and meshes[0].get_array() is not None:
            arr = meshes[0].get_array()
            xt = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
            yt = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
            if xt and yt and len(xt) * len(yt) == arr.size:
                df_cm = pd.DataFrame(np.asarray(arr).reshape(len(yt), len(xt)), index=yt, columns=xt)
            else:
                df_cm = pd.DataFrame(np.asarray(arr))
            md_parts.append(f"#### {ax_title or 'Matriz'}\n\n" + df_cm.to_markdown() + "\n")
            dfs.append((f"{nombre}_{_slug(ax_title or 'matriz')}", df_cm))
            continue
        elif ax.images and ax.images[0].get_array() is not None:
            arr = ax.images[0].get_array()
            xt = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
            yt = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
            if xt and yt and len(xt) * len(yt) == arr.size:
                df_cm = pd.DataFrame(np.asarray(arr).reshape(len(yt), len(xt)), index=yt, columns=xt)
            else:
                df_cm = pd.DataFrame(np.asarray(arr))
            md_parts.append(f"#### {ax_title or 'Matriz'}\n\n" + df_cm.to_markdown() + "\n")
            dfs.append((f"{nombre}_{_slug(ax_title or 'matriz')}", df_cm))
            continue

        # 3. Bar Containers (gráficos de barras)
        bar_containers = [c for c in ax.containers if "BarContainer" in type(c).__name__]
        if bar_containers:
            xt = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
            yt = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
            for c_idx, c in enumerate(bar_containers):
                lbl = c.get_label() if c.get_label() and not c.get_label().startswith("_") else ("valor" if len(bar_containers) == 1 else f"serie_{c_idx+1}")
                heights = [p.get_height() for p in c]
                widths = [p.get_width() for p in c]
                if xt and len(xt) == len(heights):
                    df_bar = pd.DataFrame({lbl: heights}, index=xt)
                    df_bar.index.name = ax.get_xlabel() or "categoria"
                elif yt and len(yt) == len(widths):
                    df_bar = pd.DataFrame({lbl: widths}, index=yt)
                    df_bar.index.name = ax.get_ylabel() or "categoria"
                else:
                    df_bar = pd.DataFrame({lbl: heights})
                md_parts.append(f"#### {ax_title or 'Valores de barras'}\n\n" + df_bar.to_markdown() + "\n")
                dfs.append((f"{nombre}_{_slug(ax_title or 'barras')}", df_bar))
            continue

        # 4. Errorbar Containers (p. ej. estadístico gap)
        err_containers = [c for c in ax.containers if "ErrorbarContainer" in type(c).__name__]
        if err_containers:
            for c in err_containers:
                data_line = c[0]
                x_vals = data_line.get_xdata()
                y_vals = data_line.get_ydata()
                err_dict = {"K": x_vals, ax.get_ylabel() or "valor": y_vals}
                if hasattr(c, "has_yerr") and c.has_yerr and len(c) > 2 and len(c[2]) > 0:
                    try:
                        segs = c[2][0].get_segments()
                        err_dict["s_k"] = [(seg[1][1] - seg[0][1]) / 2.0 for seg in segs]
                    except Exception:
                        pass
                df_err = pd.DataFrame(err_dict).set_index("K")
                md_parts.append(f"#### {ax_title or 'Estadístico Gap'}\n\n" + df_err.to_markdown() + "\n")
                dfs.append((f"{nombre}_{_slug(ax_title or 'gap')}", df_err))
            continue

        # 5. Lines (curvas de hiperparámetros, codo, silueta, etc.)
        lines = [l for l in ax.get_lines() if len(l.get_xdata()) >= 2 and not (len(l.get_xdata()) == 2 and l.get_xdata()[0] == l.get_xdata()[1])]
        if lines:
            ldict = {}
            x_name = ax.get_xlabel() or "x"
            y_name = ax.get_ylabel() or "y"
            for l_idx, l in enumerate(lines):
                lbl = l.get_label()
                if not lbl or lbl.startswith("_child") or lbl.startswith("_"):
                    lbl = f"{y_name}_{l_idx+1}" if len(lines) > 1 else y_name
                s = pd.Series(l.get_ydata(), index=l.get_xdata(), name=lbl)
                s = s[~s.index.duplicated(keep="first")]
                ldict[lbl] = s
            df_lines = pd.DataFrame(ldict)
            df_lines.index.name = x_name
            md_parts.append(f"#### {ax_title or 'Curva'}\n\n" + df_lines.to_markdown() + "\n")
            dfs.append((f"{nombre}_{_slug(ax_title or 'curva')}", df_lines))
            continue

        # 6. Scatter plots (PCA, pred vs real)
        scatters = [c for c in ax.collections if "PathCollection" in type(c).__name__]
        if scatters:
            offsets = scatters[0].get_offsets()
            if len(offsets) > 0:
                x_vals = offsets[:, 0]
                y_vals = offsets[:, 1]
                xlab = ax.get_xlabel() or "x"
                ylab = ax.get_ylabel() or "y"
                df_pts = pd.DataFrame({xlab: x_vals, ylab: y_vals})
                summary_data = {
                    "N_puntos": len(df_pts),
                    f"{xlab}_min": round(float(x_vals.min()), 4),
                    f"{xlab}_max": round(float(x_vals.max()), 4),
                    f"{xlab}_media": round(float(x_vals.mean()), 4),
                    f"{ylab}_min": round(float(y_vals.min()), 4),
                    f"{ylab}_max": round(float(y_vals.max()), 4),
                    f"{ylab}_media": round(float(y_vals.mean()), 4),
                }
                if len(df_pts) > 1 and np.std(x_vals) > 0 and np.std(y_vals) > 0:
                    summary_data["correlacion_r"] = round(float(np.corrcoef(x_vals, y_vals)[0, 1]), 4)
                df_sum = pd.DataFrame([summary_data]).T
                df_sum.columns = ["Valor"]
                md_parts.append(f"#### {ax_title or 'Distribución de puntos'}\n\n" + df_sum.to_markdown() + "\n")
                dfs.append((f"{nombre}_{_slug(ax_title or 'scatter')}_resumen", df_sum))
                dfs.append((f"{nombre}_{_slug(ax_title or 'scatter')}_puntos", df_pts))
                continue

    if not md_parts:
        md_parts.append(f"#### {nombre}\n\nNo se detectaron datos numéricos directamente en la figura.\n")

    md_output = f"### Datos de la figura: {nombre}\n\n" + "\n\n".join(md_parts) + "\n\n"
    return md_output, dfs


def _guardar_figura_texto(fig, nombre: str, etapa: str, datos=None):
    """Extrae y guarda los datos de la figura en Markdown y CSV."""
    try:
        md_str, dfs = extraer_datos_figura(fig, nombre, datos=datos)
    except Exception as e:
        md_str = f"### Datos de la figura: {nombre}\n\n[aviso: no se pudieron extraer datos de la figura: {e}]\n\n"
        dfs = []

    # 1. figures/<etapa>/<nombre>.md y CSV
    carpeta_fig = Path(FIG_DIR) / etapa
    carpeta_fig.mkdir(parents=True, exist_ok=True)
    with open(carpeta_fig / f"{nombre}.md", "w", encoding="utf-8") as f:
        f.write(md_str)
    if dfs:
        if len(dfs) == 1:
            dfs[0][1].to_csv(carpeta_fig / f"{nombre}.csv", encoding="utf-8")
        else:
            for sub_name, df_sub in dfs:
                df_sub.to_csv(carpeta_fig / f"{sub_name}.csv", encoding="utf-8")

    # 2. resultados_texto/<etapa>/
    etapa_limpia = etapa.split("/")[0]
    dir_etapa_res = RESULTADOS_TEXTO_DIR / etapa_limpia
    dir_etapa_res.mkdir(parents=True, exist_ok=True)
    with open(dir_etapa_res / f"{nombre}.md", "w", encoding="utf-8") as f:
        f.write(md_str)
    if dfs:
        if len(dfs) == 1:
            dfs[0][1].to_csv(dir_etapa_res / f"{nombre}.csv", encoding="utf-8")
        else:
            for sub_name, df_sub in dfs:
                df_sub.to_csv(dir_etapa_res / f"{sub_name}.csv", encoding="utf-8")

    # 3. resultados_texto/<etapa>/<paso>.md
    paso = str(_CONTEXTO["paso"])
    ruta_paso_md = dir_etapa_res / f"{paso}.md"
    modo = "a" if (etapa_limpia, paso) in _PASOS_INICIADOS else "w"
    _PASOS_INICIADOS.add((etapa_limpia, paso))
    with open(ruta_paso_md, modo, encoding="utf-8") as f:
        if modo == "w":
            f.write(f"# Etapa {etapa_limpia} · Paso {paso}\n\n")
        f.write(md_str)


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


def _guardar_figura_de_tabla(df, titulo_tabla, decimales, nombre: str):
    """Genera el PNG de la tabla en figures/<etapa>/tablas/. Nunca interrumpe el paso."""
    _asegurar()
    try:
        fig = _figura_de_tabla(df, titulo_tabla, decimales)
    except Exception as e:  # una tabla rara no debe tumbar el paso entero
        print(f"   [aviso] no se pudo dibujar esta tabla como figura ({type(e).__name__}: {e});"
              " queda sólo el texto de arriba.")
        return
    mostrar(fig, nombre, f"{_CONTEXTO['carpeta']}/tablas", ventana=False, es_tabla=True)

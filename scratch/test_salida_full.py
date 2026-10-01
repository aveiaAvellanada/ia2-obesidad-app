import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Test extraction logic
def _slug(texto: str, limite: int = 40) -> str:
    import unicodedata, re
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    t = re.sub(r"[^\w]+", "_", t).strip("_").lower()
    return t[:limite].strip("_") or "tabla"

def extraer_datos_figura(fig, nombre: str, datos=None):
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

        # 2. QuadMesh (heatmaps / confusion matrices)
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

        # 3. Bar Containers (bar plots)
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

        # 4. Lines (line plots)
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

        # 5. Scatter plots
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

print("Extractor defined successfully")

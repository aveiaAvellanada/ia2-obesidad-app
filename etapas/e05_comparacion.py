"""ETAPA 05 — Comparación final de modelos, discusión y conclusiones
=====================================================================

Cierre global del proyecto: reúne en un solo sitio los resultados de TODOS los casos
estudiados en las etapas 02-04 y los compara entre sí.

Casos:
- Clasificación A     (todas las features)             árbol, RF y KNN con GridSearch
- Clasificación B     (sin Weight/Height)              árbol, RF y KNN con GridSearch
- Clasificación B + ruido del 20 % en los hábitos      árbol, RF, KNN k=1 y KNN k=3
- Clasificación B sólo con filas reales (sin SMOTE)    7 y 3 clases, frente a una línea base
- Regresión de Weight (sin NObeyesdad)                 baseline, árbol, RF y KNN
- Clustering K-Means K=7: sin clase, sólo numéricas, con clase ordinal y con clase one-hot

Todas las cifras de esta etapa se CALCULAN a partir de los modelos de las etapas anteriores
(la mayoría ya en cache/), no están escritas a mano: si algo cambia allí, cambia aquí.

Pasos:
  1    Resultados: tablas por caso, mejor modelo de cada caso y figura comparativa
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src import cache
from src.pasos import Etapa, cli
from src.salida import tabla

from etapas import e02_clasificacion as e02
from etapas import e03_regresion as e03
from etapas import e04_clustering as e04

etapa = Etapa("05", "Comparación final y conclusiones", "comparacion",
              "Todos los casos juntos: resultados y mejor modelo por caso")

NIVEL_RUIDO = 0.20   # el nivel más alto del barrido de la etapa 02, paso 7

_estado = {}


# ----------------------------------------------------------------------------
# Recolección de resultados (se reutilizan los modelos cacheados de 02-04)
# ----------------------------------------------------------------------------
def _ruido():
    """Accuracy y macro-F1 en test, limpio y con ruido, de los modelos de robustez (variante B)."""
    d = e02.datos()
    X_tr = d["X_train"].drop(columns=e02.DROP_WH)
    X_te = d["X_test"].drop(columns=e02.DROP_WH)
    X_tr_n, X_te_n = e02._con_ruido(X_tr, X_te, NIVEL_RUIDO)
    filas = {}
    for nombre, modelo in e02._modelos_robustez().items():
        modelo.fit(X_tr, d["y_train"])
        acc_c = accuracy_score(d["y_test"], modelo.predict(X_te))
        modelo.fit(X_tr_n, d["y_train"])
        p = modelo.predict(X_te_n)
        filas[nombre] = {"test_accuracy_limpio": acc_c,
                         "test_accuracy": accuracy_score(d["y_test"], p),
                         "test_f1_macro": f1_score(d["y_test"], p, average="macro")}
    return pd.DataFrame(filas).T


def resultados():
    """Dict con una tabla por caso más los datos auxiliares de la discusión."""
    if "res" not in _estado:
        _, clf_A, _, _ = e02.variante("A")
        _, clf_B, _, _ = e02.variante("B")
        ruido = cache.obtener(f"cmp_ruido_{int(NIVEL_RUIDO * 100)}", _ruido)
        _, _, reg = e03.modelos()
        c = e04.con_clase()
        clu = pd.DataFrame({
            "Sin clase (20 features)": e04.analisis_base()["metricas"],
            "Sólo numéricas (8)": e04.solo_numericas()["analisis"]["metricas"],
            "Con clase ordinal": c["analisis_ord"]["metricas"],
            "Con clase one-hot": c["analisis_oh"]["metricas"],
        }).T[["silueta", "ARI", "NMI", "pureza"]].astype(float)
        _estado["res"] = dict(clf_A=clf_A, clf_B=clf_B, ruido=ruido, reg=reg, clu=clu,
                              reales=e02.evaluacion_reales(), barrido_rf=e02.barrido_n_arboles())
    return _estado["res"]


def mejores():
    """Una fila por caso con su mejor modelo y la métrica principal."""
    r = resultados()
    filas = []
    for caso, t in [("Clasificación A (todas)", r["clf_A"]), ("Clasificación B (sin W/H)", r["clf_B"])]:
        m = t["test_f1_macro"].idxmax()
        filas.append({"Caso": caso, "Mejor modelo": m, "Métrica principal": "macro-F1 test",
                      "Valor": t.loc[m, "test_f1_macro"],
                      "Otra métrica": f"accuracy {t.loc[m, 'test_accuracy']:.4f}"})
    m = r["ruido"]["test_accuracy"].idxmax()
    filas.append({"Caso": f"Clasificación B + ruido {int(NIVEL_RUIDO * 100)} %", "Mejor modelo": m,
                  "Métrica principal": "macro-F1 test", "Valor": r["ruido"].loc[m, "test_f1_macro"],
                  "Otra métrica": f"accuracy {r['ruido'].loc[m, 'test_accuracy']:.4f}"})
    for n_cl in (7, 3):
        t = r["reales"][f"B{n_cl}"]
        base = t.loc[e02.BASELINE_REALES, "f1_macro"]
        m = t.drop(index=e02.BASELINE_REALES)["f1_macro"].idxmax()
        filas.append({"Caso": f"Clasificación B · filas reales · {n_cl} clases", "Mejor modelo": m,
                      "Métrica principal": "macro-F1 CV 3×10", "Valor": t.loc[m, "f1_macro"],
                      "Otra métrica": f"línea base {base:.4f}"})
    reg = r["reg"].drop(index="Baseline (media)")
    m = reg["test_RMSE"].idxmin()
    filas.append({"Caso": "Regresión de Weight", "Mejor modelo": m, "Métrica principal": "RMSE test (kg)",
                  "Valor": reg.loc[m, "test_RMSE"], "Otra métrica": f"R² {reg.loc[m, 'test_R2']:.4f}"})
    sin = r["clu"].drop(index=["Con clase ordinal", "Con clase one-hot"])
    m = sin["ARI"].idxmax()
    filas.append({"Caso": "Clustering K=7 (sin la clase)", "Mejor modelo": f"K-Means · {m}",
                  "Métrica principal": "ARI vs clases", "Valor": sin.loc[m, "ARI"],
                  "Otra métrica": f"pureza {sin.loc[m, 'pureza']:.4f}"})
    return pd.DataFrame(filas).set_index("Caso")


# ----------------------------------------------------------------------------
# 1. Resultados
# ----------------------------------------------------------------------------
@etapa.paso("1", "Resultados: tablas por caso, mejor modelo de cada caso y figura comparativa", figuras=1,
            parametros=("NIVEL_RUIDO", "e02.make_models", "e03.make_models", "e04.fit_kmeans"))
def paso_1_resultados():
    r = resultados()
    tabla(r["clf_A"], "Clasificación A (todas las features)")
    tabla(r["clf_B"], "Clasificación B (sin Weight ni Height)")
    tabla(r["ruido"], f"Clasificación B con {int(NIVEL_RUIDO * 100)} % de ruido en los hábitos "
                      "(test_accuracy_limpio = mismo modelo sin ruido)")
    tabla(r["reales"]["B7"], "Clasificación B sólo con filas reales (sin SMOTE), 7 clases (CV 3×10)")
    tabla(r["reales"]["B3"], "Clasificación B sólo con filas reales (sin SMOTE), 3 clases (CV 3×10)")
    tabla(r["reg"], "Regresión de Weight (RMSE y MAE en kg)")
    tabla(r["clu"], "Clustering K-Means con K=7 comparado con las clases reales")
    tabla(mejores(), "Mejor modelo de cada caso", grafica=False)

    # --- Figura: una fila de paneles, un panel por tipo de tarea ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.4), gridspec_kw={"width_ratios": [1.5, 1, 1, 1.1]})
    colores = {"Árbol": "#4c78a8", "Random Forest": "#54a24b", "KNN": "#f58518", "Otro": "#9d9d9d"}

    def color(nombre):
        for k, v in colores.items():
            if nombre.startswith(k) or k in nombre:
                return v
        return colores["Otro"]

    # Panel 1: accuracy en test de los tres casos de clasificación
    ax = axes[0]
    casos = [("A (todas)", r["clf_A"]["test_accuracy"]), ("B (sin W/H)", r["clf_B"]["test_accuracy"]),
             (f"B + ruido {int(NIVEL_RUIDO * 100)} %", r["ruido"]["test_accuracy"])]
    x0 = 0
    ticks, etiquetas = [], []
    for caso, serie in casos:
        for i, (modelo, v) in enumerate(serie.items()):
            ax.bar(x0 + i, v, color=color(modelo), width=0.85)
            ax.text(x0 + i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
            if "k=" in modelo:   # distinguir las dos variantes de KNN del caso con ruido
                ax.text(x0 + i, 0.515, modelo.split("(")[1].split(",")[0].rstrip(")"),
                        ha="center", fontsize=8, color="white", fontweight="bold")
        ticks.append(x0 + (len(serie) - 1) / 2)
        etiquetas.append(caso)
        x0 += len(serie) + 1
    ax.set_xticks(ticks, etiquetas)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Accuracy en test")
    ax.set_title("Clasificación de NObeyesdad")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in list(colores.values())[:3]],
              labels=list(colores)[:3], loc="upper right", fontsize=9)

    # Panel 2: R² en test de la regresión
    ax = axes[1]
    reg = r["reg"]["test_R2"]
    ax.bar(range(len(reg)), reg.values, color=[color(m) for m in reg.index])
    for i, v in enumerate(reg.values):
        ax.text(i, max(v, 0) + 0.02, f"{v + 0.0:.3f}".replace("-0.000", "0.000"), ha="center",
                fontsize=8)
    ax.set_xticks(range(len(reg)), [m.replace(" de regresión", "") for m in reg.index], rotation=15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("R² en test")
    ax.set_title("Regresión de Weight")

    # Panel 3: ARI del clustering (qué tanto recupera las clases)
    ax = axes[2]
    ari = r["clu"]["ARI"]
    cols = ["#9d9d9d", "#9d9d9d", "#e45756", "#e45756"]
    ax.bar(range(len(ari)), ari.values, color=cols)
    for i, v in enumerate(ari.values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(ari)), [s.replace(" (20 features)", "").replace("Con clase ", "clase ")
                                    for s in ari.index], rotation=15)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("ARI (clusters vs clases reales)")
    ax.set_title("Clustering K-Means, K=7\n(rojo: con la clase como feature)")
    # Panel 4: macro-F1 de la variante B con y sin SMOTE, frente a la línea base
    ax = axes[3]
    B7, B3 = r["reales"]["B7"], r["reales"]["B3"]
    base = e02.BASELINE_REALES
    mejor7 = B7.drop(index=base)["f1_macro"].idxmax()
    mejor3 = B3.drop(index=base)["f1_macro"].idxmax()
    barras = [("Dataset completo\n7 clases (RF)", r["clf_B"].loc["Random Forest", "test_f1_macro"], "#54a24b"),
              ("Filas reales\n7 clases", B7.loc[mejor7, "f1_macro"], "#b279a2"),
              ("Filas reales\n3 clases", B3.loc[mejor3, "f1_macro"], "#b279a2")]
    for i, (et, v, c) in enumerate(barras):
        ax.bar(i, v, color=c, width=0.7)
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    for i, t in [(1, B7), (2, B3)]:
        ax.hlines(t.loc[base, "f1_macro"], i - 0.35, i + 0.35, colors="black", linestyles="--", lw=1.4)
    ax.plot([], [], ls="--", color="black", label="línea base (clase mayoritaria)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xticks(range(3), [b[0] for b in barras], fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("macro-F1")
    ax.set_title("Variante B (sin Weight/Height):\ncon SMOTE vs sólo filas reales")
    for ax in axes:
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Comparación final: cuánto de la variable objetivo recupera cada enfoque", y=1.02)
    etapa.figura(fig, "1_comparacion_final")


if __name__ == "__main__":
    sys.exit(cli(etapa))

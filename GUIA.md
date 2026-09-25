# GUÍA: dónde hacer clic para ver cada gráfica

Este proyecto es la versión "Python normal" (sin Jupyter) de la entrega. Todo el análisis está en
4 scripts, uno por etapa, y cada script está dividido en **pasos numerados** (misma numeración que
las secciones de los notebooks originales y que la guía de estudio). Cada paso imprime en la
terminal las tablas y explicaciones, y las gráficas se abren en una **ventana** y además se guardan
como **PNG** en `figures/`.

---

## 0. Preparar el proyecto (sólo la primera vez)

1. Abre VS Code → `File > Open Folder...` → elige la carpeta `ia2-obesidad-py`.
2. Abre una terminal dentro de VS Code: menú `Terminal > New Terminal` (o `Ctrl+ñ` / `` Ctrl+` ``).
3. Crea el entorno e instala dependencias (copia y pega en esa terminal):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Selecciona el intérprete: `Ctrl+Shift+P` → escribe **Python: Select Interpreter** → elige el que
   dice `.venv` (`./.venv/bin/python`). Si VS Code te propone instalar la extensión **Python**
   (de Microsoft), acepta: es la que pone el botón ▶ de ejecutar.

> El dataset ya está en `data/obesity_raw.csv`. Si lo borras, la etapa 01 lo vuelve a descargar de
> UCI con `ucimlrepo`.

---

## 1. Las tres formas de ejecutar (elige la que prefieras)

### Forma A — Menú interactivo (la más cómoda)

1. En el explorador de archivos de VS Code (panel izquierdo) haz clic en **`main.py`**.
2. Pulsa el botón **▶ (Run Python File)** que aparece **arriba a la derecha** del editor.
3. En la terminal aparece el menú principal:

   ```
   [1] 01 — EDA y preprocesamiento
   [2] 02 — Clasificación de NObeyesdad
   [3] 03 — Regresión de Weight
   [4] 04 — Clustering K-Means
   [t] Ejecutar TODO ...
   ```

4. Escribe el número de la etapa (p. ej. `2`) y `Enter`. Verás la lista de pasos de esa etapa.
5. Escribe el id del paso (p. ej. `3.4`) y `Enter`. Se imprimen las tablas y se abre la ventana
   con la gráfica. **Cierra la ventana** (la X) para que el programa continúe con la siguiente
   gráfica o vuelva al menú.
6. `t` ejecuta todos los pasos de la etapa; `v` vuelve al menú principal; `q` sale.

### Forma B — Panel "Ejecutar y depurar" de VS Code (botón verde)

1. Abre el panel con `Ctrl+Shift+D` (icono ▷ con un bicho, en la barra izquierda).
2. Arriba hay un **desplegable** con configuraciones ya preparadas:
   - `Menú principal (main.py)`
   - `Etapa 01 — EDA (menú de pasos)` … `Etapa 04 — Clustering (menú de pasos)`
   - `Un paso concreto (pregunta etapa y paso)` ← te pregunta la etapa y el paso y lo ejecuta directo
   - `TODO el proyecto (run_all.py, sólo guarda PNG)`
3. Elige una y pulsa el **▶ verde** (o `F5`).

### Forma C — Terminal (útil en la sustentación para ir directo a una gráfica)

```bash
source .venv/bin/activate            # una vez por terminal

python main.py 02 3.4                # etapa 02, paso 3.4 (abre ventanas)
python main.py 04 2.1 2.2 2.3        # varios pasos seguidos
python etapas/e03_regresion.py       # menú de pasos de la etapa 03
python etapas/e03_regresion.py --paso 2.1 --sin-ventanas   # sólo guarda el PNG
python etapas/e02_clasificacion.py --lista                  # lista de pasos
python run_all.py                    # todo el proyecto, sin ventanas, ~1 min
python run_all.py --limpiar-cache    # recalcula todo desde cero
```

---

## 2. Cosas que conviene saber

- **Caché**: los GridSearch (etapas 02 y 03) y los barridos de K-Means (04) tardan ~40 s la primera
  vez y se guardan en `cache/`. Después, cualquier paso carga en 1–2 s. Para recalcular: opción
  `[c]` del menú principal o `--limpiar-cache`.
- **PNG**: cada gráfica se guarda en `figures/<etapa>/<paso>_<nombre>.png` (para el informe).
- **Tablas como gráfica**: además de imprimirse en la terminal, **cada tabla se guarda también como
  imagen** en `figures/<etapa>/tablas/<paso>_tablaN_<titulo>.png`. Si la tabla es numérica de una
  sola columna sale un **gráfico de barras**; si tiene entre 2 y 6 columnas numéricas, **un panel de
  barras por columna** (así no se mezclan escalas como RMSE 8.2 con R² 0.89); y si tiene texto o es
  muy ancha, se dibuja **la tabla misma como imagen** lista para pegar en el informe. Estas figuras
  no abren ventana, sólo se guardan; con `[b]` en el menú principal se apagan o encienden.
- **Ventanas**: las abre PyQt5 (en `requirements.txt`). Si no se abrieran, el programa lo avisa y
  sigue guardando los PNG. Con `[v]` en el menú apagas/enciendes las ventanas.
- Las **cifras** (accuracy 0.957, RMSE 8.25 kg, ARI 0.088, etc.) son idénticas a las de los
  notebooks porque se usa la misma semilla (`RANDOM_STATE = 42` en `config.py`) y los mismos splits.
- Ejecuta los pasos en el orden que quieras: cada uno calcula lo que necesita. El único requisito
  real es que exista `data/clf.csv` / `data/reg.csv` (los genera el paso 3 de la etapa 01, y si no
  existen las etapas 02–04 los regeneran solas en memoria).

---

## 3. Mapa completo: qué paso ejecutar para ver cada cosa

### Etapa 01 — EDA y preprocesamiento (`etapas/e01_eda.py`)

| Paso | Qué muestra | Figura (`figures/01_eda/`) |
|---|---|---|
| `1` | Carga, `describe`, `info`, faltantes (0), duplicados (24), valores de categóricas | — |
| `2.1` | Distribución de las 7 clases (balanceadas 13–17 %) | `2.1_distribucion_clases.png` |
| `2.2` | Histograma de `Weight` + boxplot de `Weight` por clase | `2.2_distribucion_weight.png` |
| `2.3` | **Fuga de datos**: IMC por clase con umbrales OMS; árbol con sólo Weight+Height → 0.954 | `2.3_imc_por_clase.png` |
| `2.4` | Heatmap de correlaciones | `2.4_correlaciones.png` |
| `3` | Tabla de encoding; genera `data/clf.csv` y `data/reg.csv` | — |

### Etapa 02 — Clasificación (`etapas/e02_clasificacion.py`)

| Paso | Qué muestra | Figura (`figures/02_clasificacion/`) |
|---|---|---|
| `1` | Split estratificado 80/20 (1669 / 418) | — |
| `2` | Modelos y grillas de GridSearch | — |
| `3` | **Variante A** (todas): tabla CV/test → RF acc 0.957 | — |
| `3.1` | `classification_report` por clase + 3 matrices de confusión (A) | `3.1_matrices_confusion_A.png` |
| `3.2` | Importancia de features RF (A): Weight 0.31 | `3.2_importancias_rf_A.png` |
| `3.3` | Dibujo del árbol de GridSearch (3 niveles) | `3.3_arbol_gridsearch_A.png` |
| `3.4` | **Tres árboles** completo / podado / muy podado + tabla + 3 matrices | `3.4_arbol_completo.png`, `3.4_arbol_podado.png`, `3.4_arbol_muy_podado.png`, `3.4_matrices_tres_arboles.png` |
| `3.5` | **Postpoda** `ccp_alpha`: accuracy train/test | `3.5_postpoda_ccp_alpha.png` |
| `3.6` | **Curva del mejor K** de KNN (A) → K=1 | `3.6_curva_k_knn_A.png` |
| `3.7` | **¿Cuántos árboles?** Random Forest con 10, 25, 50, 100, 200, 400 y 800: macro-F1 de CV y test + tiempo de entrenamiento | `3.7_n_estimators_rf.png` |
| `4` | **Variante B** (sin Weight/Height): tabla, reportes, matrices, importancias | `4_matrices_confusion_B.png`, `4_importancias_rf_B.png` |
| `4.1` | Curva del mejor K de KNN (B) → K=3 distance | `4.1_curva_k_knn_B.png` |
| `5` | Comparación A vs B (barras macro-F1) | `5_comparacion_A_vs_B.png` |
| `6` | Conclusiones + nodo raíz (`Weight ≤ 99.536`) + top-5 importancias | — |
| `7` | **Robustez ante ruido** (jittering 0–20 %): tabla 5 % + curva | `7_barrido_ruido.png` |

### Etapa 03 — Regresión (`etapas/e03_regresion.py`)

| Paso | Qué muestra | Figura (`figures/03_regresion/`) |
|---|---|---|
| `1` | Split estratificado por deciles de Weight; sin NObeyesdad | — |
| `2` | GridSearch + tabla con MAE, RMSE, R², **R² ajustado** → RF RMSE 8.25 | — |
| `2.1` | Curva del mejor K de KNN (RMSE de CV) → K=3 | `2.1_curva_k_knn.png` |
| `2.2` | **¿Cuántos árboles?** Random Forest con 10, 25, 50, 100, 200, 400 y 800: RMSE de CV y test + tiempo de entrenamiento | `2.2_n_estimators_rf.png` |
| `3` | Tres árboles de regresión + tabla + predicho vs real | `3_arbol_completo.png`, `3_arbol_podado.png`, `3_arbol_muy_podado.png`, `3_pred_vs_real_tres_arboles.png` |
| `3.1` | Postpoda `ccp_alpha` (R²) → pico 0.824 | `3.1_postpoda_ccp_alpha.png` |
| `4` | Predicho vs real + residuos de los 3 modelos | `4_pred_vs_real_modelos.png`, `4_residuos_modelos.png` |
| `5` | Importancia de features RF → family_history 0.25, Height 0.17 | `5_importancias_rf.png` |
| `6` | Barras RMSE / R² / R²adj + explicación del R² ajustado | `6_comparacion_metricas.png` |
| `7` | Conclusiones | — |

### Etapa 04 — Clustering (`etapas/e04_clustering.py`)

| Paso | Qué muestra | Figura (`figures/04_clustering/`) |
|---|---|---|
| `1` | 20 features sin la clase, escaladas | — |
| `2` | Tabla de inercia / silueta / CH / DB para K = 2..12 | — |
| `2.1` | **Método del codo** (kneedle) → K=7 | `2.1_codo.png` |
| `2.2` | Silueta (K=2), Calinski-Harabasz (K=2), Davies-Bouldin (K=9) | `2.2_silueta_ch_db.png` |
| `2.3` | **Estadístico gap** → K=2 | `2.3_gap.png` |
| `2.4` | Tabla comparando los 5 métodos (no coinciden) | — |
| `3` | K=7: silueta 0.132, ARI 0.088, NMI 0.158 | — |
| `3.1` | **Matriz clusters × clases** (contingencia) | `3.1_contingencia_k7.png` |
| `3.2` | Matriz mapeada con **algoritmo húngaro**: acc 0.302, pureza 0.306 | `3.2_matriz_mapeada_k7.png` |
| `3.3` | Centroides en unidades originales (qué separa cada cluster) | — |
| `3.4` | PCA 2D: clusters vs clases | `3.4_pca_2d.png` |
| `4` | Variantes **con la clase** (ordinal / one-hot) | — |
| `4.1` | 5 métodos para K con clase ordinal (panel 2×3) | `4.1_cinco_metodos_con_clase_ordinal.png` |
| `4.2` | K=7 con clase: contingencia cruda + matrices mapeadas | `4.2_contingencia_cruda_con_clase.png`, `4.2_matrices_mapeadas_con_clase.png` |
| `4.3` | Tabla comparativa sin / ordinal / one-hot (ARI 0.088 / 0.252 / 0.999) | — |
| `4.4` | PCA 2D con clase ordinal | `4.4_pca_2d_con_clase_ordinal.png` |
| `4.5` | Interpretación: por qué la clase cambia el clustering | — |
| `5` | Variante sólo numéricas: métricas, matrices, centroides por IMC | `5_metricas_solo_numericas.png`, `5_matrices_solo_numericas.png` |
| `6` | Conclusiones | — |

---

## 4. Requisitos del profesor → dónde están

| Requisito | Paso(s) |
|---|---|
| Clasificación con árbol, RF y KNN + preprocesamiento + métricas | 01·3, 02·3, 02·3.1, 02·4 |
| Regresión de Weight sin NObeyesdad (sin fuga) | 03·1, 03·2, 03·4 |
| Tres árboles (completo / podado / muy podado), nodo raíz | 02·3.4, 03·3, 02·6 |
| Prepoda vs postpoda | 02·3.4 + 02·3.5, 03·3 + 03·3.1 |
| "Rutinita" del mejor K de KNN | 02·3.6, 02·4.1, 03·2.1 |
| Número de árboles del Random Forest (10 vs 50 vs 500...) con gráfica comparativa | 02·3.7, 03·2.2 |
| R² ajustado | 03·2, 03·6 |
| K-Means sin la clase; K con ≥ 3 métodos; ¿coinciden? | 04·2.1, 04·2.2, 04·2.3, 04·2.4 |
| Matriz de confusión clusters (K=7) vs clases | 04·3.1, 04·3.2 |
| K-Means con la columna de clase (mejor con ella) | 04·4 … 04·4.5 |
| Fuga de datos NObeyesdad ≈ f(Weight, Height) | 01·2.3, 02·4, 02·6 |
| Efecto de SMOTE / robustez | 02·3.6, 02·7 |

---

## 5. Estructura del proyecto

```
ia2-obesidad-py/
├── main.py              ← MENÚ PRINCIPAL (ejecuta este con ▶)
├── run_all.py           ← todo el proyecto de una vez, sólo PNG
├── config.py            ← rutas, RANDOM_STATE = 42, estilo de gráficos
├── requirements.txt
├── GUIA.md              ← este archivo
├── README.md
├── etapas/
│   ├── e01_eda.py             etapa 01
│   ├── e02_clasificacion.py   etapa 02
│   ├── e03_regresion.py       etapa 03
│   └── e04_clustering.py      etapa 04
├── src/
│   ├── data.py          carga del dataset (UCI 544)
│   ├── preprocessing.py encoding, duplicados, clf.csv / reg.csv
│   ├── evaluation.py    métricas, R² ajustado, curvas de K, postpoda, matrices
│   ├── pasos.py         mini-framework de pasos + menú + línea de comandos
│   ├── salida.py        ventanas / PNG / impresión de tablas y notas
│   └── cache.py         caché en disco de los GridSearch
├── data/                obesity_raw.csv (+ clf.csv, reg.csv generados)
├── figures/             PNG generados, por etapa
├── cache/               modelos entrenados (se puede borrar)
└── .vscode/             launch.json (botón verde) y settings.json (intérprete)
```

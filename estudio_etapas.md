> **Inteligencia Artificial 2 · proyecto UCI 544**

# Niveles de obesidad, de los datos a la sustentación

Cada paso de las cinco etapas con sus figuras y tablas. Las cifras salen de ejecutar `python run_all.py` sobre el repositorio.

### Resumen de Métricas Principales (KPIs)

| Indicador | Valor | Descripción |
| :--- | :--- | :--- |
| **Clasificación A · Random Forest** | `0.957` | accuracy en test (con Weight y Height) |
| **Clasificación B · Random Forest** | `0.857` | accuracy sin Weight ni Height |
| **Regresión de Weight · Random Forest** | `8.25 kg` | RMSE · R² 0.899 |
| **Clustering K-Means, K=7** | `0.088` | ARI contra las clases reales |

---

> **ETAPAS**

## Etapa 01 — EDA y preprocesamiento, paso a paso

**Archivo:** `etapas/e01_eda.py`. **Figuras:** `figures/01_eda/`. **EDA** = *Exploratory Data
Analysis* (análisis exploratorio): conocer los datos antes de modelar.

Estado compartido: `df_raw()` carga el CSV una sola vez; `df_bmi()` añade la columna
`BMI = Weight / Height**2`.

---

### Paso 1: Carga y EDA mínimo

> *Ver código e01_eda.py:64* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e01_eda.py#L64](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e01_eda.py#L64)

- **Qué hace:** imprime dimensiones (2111, 17), `head()`, `describe().T` (conteo, media, desviación, mínimo, cuartiles, máximo de las 8 numéricas), `info()` (tipos y no-nulos), faltantes (**0**), duplicados exactos (**24**) y `value_counts()` de cada categórica.
- **Qué debes observar:** variables que eran respuestas discretas tienen decimales (FCVC = 2.386, mediana de Age = 22.778) → huella de SMOTE. SMOKE (44 "yes") y SCC (96) están muy desbalanceadas; MTRANS tiene categorías rarísimas (Bike 7, Motorbike 11).
- **Sin gráfica** (las tablas se guardan como PNG en `figures/01_eda/tablas/`).

#### Tablas de este paso como imagen (2)

![1 · primeras 5 filas](imagenes/01_eda_tablas_1_tabla1_primeras_5_filas.png)

### Primeras 5 filas

|    | Gender   |   Age |   Height |   Weight | family_history_with_overweight   | FAVC   |   FCVC |   NCP | CAEC      | SMOKE   |   CH2O | SCC   |   FAF |   TUE | CALC       | MTRANS                | NObeyesdad          |
|---:|:---------|------:|---------:|---------:|:---------------------------------|:-------|-------:|------:|:----------|:--------|-------:|:------|------:|------:|:-----------|:----------------------|:--------------------|
|  0 | Female   |    21 |     1.62 |     64   | yes                              | no     |      2 |     3 | Sometimes | no      |      2 | no    |     0 |     1 | no         | Public_Transportation | Normal_Weight       |
|  1 | Female   |    21 |     1.52 |     56   | yes                              | no     |      3 |     3 | Sometimes | yes     |      3 | yes   |     3 |     0 | Sometimes  | Public_Transportation | Normal_Weight       |
|  2 | Male     |    23 |     1.8  |     77   | yes                              | no     |      2 |     3 | Sometimes | no      |      2 | no    |     2 |     1 | Frequently | Public_Transportation | Normal_Weight       |
|  3 | Male     |    27 |     1.8  |     87   | no                               | no     |      3 |     3 | Sometimes | no      |      2 | no    |     2 |     0 | Frequently | Walking               | Overweight_Level_I  |
|  4 | Male     |    22 |     1.78 |     89.8 | no                               | no     |      2 |     1 | Sometimes | no      |      2 | no    |     0 |     0 | Sometimes  | Public_Transportation | Overweight_Level_II |

![1 · estadisticos de las columnas numericas](imagenes/01_eda_tablas_1_tabla2_estadisticos_de_las_columnas_numericas.png)

### Estadísticos de las columnas numéricas

|        |   count |   mean |    std |   min |    25% |    50% |     75% |    max |
|:-------|--------:|-------:|-------:|------:|-------:|-------:|--------:|-------:|
| Age    |   2,111 | 24.313 |  6.346 | 14    | 19.947 | 22.778 |  26     |  61    |
| Height |   2,111 |  1.702 |  0.093 |  1.45 |  1.63  |  1.7   |   1.768 |   1.98 |
| Weight |   2,111 | 86.586 | 26.191 | 39    | 65.473 | 83     | 107.431 | 173    |
| FCVC   |   2,111 |  2.419 |  0.534 |  1    |  2     |  2.386 |   3     |   3    |
| NCP    |   2,111 |  2.686 |  0.778 |  1    |  2.659 |  3     |   3     |   4    |
| CH2O   |   2,111 |  2.008 |  0.613 |  1    |  1.585 |  2     |   2.477 |   3    |
| FAF    |   2,111 |  1.01  |  0.851 |  0    |  0.125 |  1     |   1.667 |   3    |
| TUE    |   2,111 |  0.658 |  0.609 |  0    |  0     |  0.625 |   1     |   2    |

### Paso 2.1: Distribución de las 7 clases

> *Ver código e01_eda.py:86* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e01_eda.py#L86](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e01_eda.py#L86)

![2.1 · distribucion clases](imagenes/01_eda_2.1_distribucion_clases.png)

### Datos de la figura: 2.1_distribucion_clases

#### Distribución de NObeyesdad

| NObeyesdad          |   valor |
|:--------------------|--------:|
| Insufficient_Weight |     272 |
| Normal_Weight       |     287 |
| Overweight_Level_I  |     290 |
| Overweight_Level_II |     290 |
| Obesity_Type_I      |     351 |
| Obesity_Type_II     |     297 |
| Obesity_Type_III    |     324 |

- **Gráfica `2.1_distribucion_clases.png`:** barras con el número de filas por clase, en `CLASS_ORDER`.
- **Cómo se lee:** todas las barras tienen alturas parecidas (272–351) → clases **balanceadas** (12.9 %–16.6 %).
- **Código:** `counts = df[TARGET_CLF].value_counts().reindex(CLASS_ORDER)` y `sns.barplot(x=counts.index, y=counts.values)`. `reindex` ordena por IMC en vez de por frecuencia.
- **Conclusión:** no hace falta re-muestrear; basta con `stratify` en el split y macro-F1.

#### Tablas de este paso como imagen (2)

![2.1 · filas por clase](imagenes/01_eda_tablas_2.1_tabla1_filas_por_clase.png)

### Filas por clase

| NObeyesdad          |   n |
|:--------------------|----:|
| Insufficient_Weight | 272 |
| Normal_Weight       | 287 |
| Overweight_Level_I  | 290 |
| Overweight_Level_II | 290 |
| Obesity_Type_I      | 351 |
| Obesity_Type_II     | 297 |
| Obesity_Type_III    | 324 |

![2.1 · proporcion por clase](imagenes/01_eda_tablas_2.1_tabla2_proporcion_por_clase.png)

### Proporción por clase

| NObeyesdad          |   proporción |
|:--------------------|-------------:|
| Insufficient_Weight |        0.129 |
| Normal_Weight       |        0.136 |
| Overweight_Level_I  |        0.137 |
| Overweight_Level_II |        0.137 |
| Obesity_Type_I      |        0.166 |
| Obesity_Type_II     |        0.141 |
| Obesity_Type_III    |        0.153 |

### Paso 2.3: Chequeo de fuga de datos (el paso más importante de la etapa)

> *Ver código e01_eda.py:121* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e01_eda.py#L121](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e01_eda.py#L121)

- **Prueba cuantitativa:** accuracy en 5-fold CV de un árbol de decisión usando sólo: Columnas Accuracy CV BMI 0.937 **Weight + Height** **0.954** Weight 0.644 Height 0.328 Con sólo dos columnas se acierta 95 % → **fuga de datos**. (Weight + Height supera a BMI solo porque el árbol puede aprovechar también Height por separado, p. ej. para el sexo.)
- **Tabla de correlaciones:** Weight–BMI 0.935, Weight–Height 0.463, Height–BMI 0.132.
- **Decisiones que salen de aquí:** Clasificación: variante **A** con todas las features (lo que pide el enunciado) y variante **B** sin Weight ni Height (para medir lo que aportan los hábitos). Regresión de Weight: se **excluye** `NObeyesdad` (la clase contiene el rango de peso).

#### Tablas de este paso como imagen (1)

![2.3 · correlacion weight height bmi age](imagenes/01_eda_tablas_2.3_tabla2_correlacion_weight_height_bmi_age.png)

### Correlación Weight / Height / BMI / Age

|        |   Weight |   Height |   BMI |    Age |
|:-------|---------:|---------:|------:|-------:|
| Weight |    1     |    0.463 | 0.935 |  0.203 |
| Height |    0.463 |    1     | 0.132 | -0.026 |
| BMI    |    0.935 |    0.132 | 1     |  0.244 |
| Age    |    0.203 |   -0.026 | 0.244 |  1     |

### Paso 3: Preprocesamiento →clf.csvyreg.csv

> *Ver código e01_eda.py:161* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e01_eda.py#L161](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e01_eda.py#L161)

- Llama a `build_datasets(df_raw())`: quita los 24 duplicados (2111 → **2087**), codifica (tabla de la sección 2.3) y genera: `data/clf.csv`: **2087 × 21** (20 features + NObeyesdad). `data/reg.csv`: **2087 × 20** (19 features + Weight; **sin** NObeyesdad).
- Las 20 features finales: Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS_Automobile, MTRANS_Bike, MTRANS_Motorbike, MTRANS_Public_Transportation, MTRANS_Walking.
- El **escalado no se hace aquí** (va dentro del Pipeline de KNN y, en clustering, sobre la matriz completa). En los árboles `criterion="gini"`.

#### Tablas de este paso como imagen (2)

![3 · clf csv primeras filas](imagenes/01_eda_tablas_3_tabla1_clf_csv_primeras_filas.png)

### clf.csv — primeras filas

|    |   Gender |   Age |   Height |   Weight |   family_history_with_overweight |   FAVC |   FCVC |   NCP |   CAEC |   SMOKE |   CH2O |   SCC |   FAF |   TUE |   CALC | NObeyesdad          |   MTRANS_Automobile |   MTRANS_Bike |   MTRANS_Motorbike |   MTRANS_Public_Transportation |   MTRANS_Walking |
|---:|---------:|------:|---------:|---------:|---------------------------------:|-------:|-------:|------:|-------:|--------:|-------:|------:|------:|------:|-------:|:--------------------|--------------------:|--------------:|-------------------:|-------------------------------:|-----------------:|
|  0 |        0 |    21 |     1.62 |     64   |                                1 |      0 |      2 |     3 |      1 |       0 |      2 |     0 |     0 |     1 |      0 | Normal_Weight       |                   0 |             0 |                  0 |                              1 |                0 |
|  1 |        0 |    21 |     1.52 |     56   |                                1 |      0 |      3 |     3 |      1 |       1 |      3 |     1 |     3 |     0 |      1 | Normal_Weight       |                   0 |             0 |                  0 |                              1 |                0 |
|  2 |        1 |    23 |     1.8  |     77   |                                1 |      0 |      2 |     3 |      1 |       0 |      2 |     0 |     2 |     1 |      2 | Normal_Weight       |                   0 |             0 |                  0 |                              1 |                0 |
|  3 |        1 |    27 |     1.8  |     87   |                                0 |      0 |      3 |     3 |      1 |       0 |      2 |     0 |     2 |     0 |      2 | Overweight_Level_I  |                   0 |             0 |                  0 |                              0 |                1 |
|  4 |        1 |    22 |     1.78 |     89.8 |                                0 |      0 |      2 |     1 |      1 |       0 |      2 |     0 |     0 |     0 |      1 | Overweight_Level_II |                   0 |             0 |                  0 |                              1 |                0 |

![3 · reg csv primeras filas](imagenes/01_eda_tablas_3_tabla2_reg_csv_primeras_filas.png)

### reg.csv — primeras filas

|    |   Gender |   Age |   Height |   family_history_with_overweight |   FAVC |   FCVC |   NCP |   CAEC |   SMOKE |   CH2O |   SCC |   FAF |   TUE |   CALC |   MTRANS_Automobile |   MTRANS_Bike |   MTRANS_Motorbike |   MTRANS_Public_Transportation |   MTRANS_Walking |   Weight |
|---:|---------:|------:|---------:|---------------------------------:|-------:|-------:|------:|-------:|--------:|-------:|------:|------:|------:|-------:|--------------------:|--------------:|-------------------:|-------------------------------:|-----------------:|---------:|
|  0 |        0 |    21 |     1.62 |                                1 |      0 |      2 |     3 |      1 |       0 |      2 |     0 |     0 |     1 |      0 |                   0 |             0 |                  0 |                              1 |                0 |     64   |
|  1 |        0 |    21 |     1.52 |                                1 |      0 |      3 |     3 |      1 |       1 |      3 |     1 |     3 |     0 |      1 |                   0 |             0 |                  0 |                              1 |                0 |     56   |
|  2 |        1 |    23 |     1.8  |                                1 |      0 |      2 |     3 |      1 |       0 |      2 |     0 |     2 |     1 |      2 |                   0 |             0 |                  0 |                              1 |                0 |     77   |
|  3 |        1 |    27 |     1.8  |                                0 |      0 |      3 |     3 |      1 |       0 |      2 |     0 |     2 |     0 |      2 |                   0 |             0 |                  0 |                              0 |                1 |     87   |
|  4 |        1 |    22 |     1.78 |                                0 |      0 |      2 |     1 |      1 |       0 |      2 |     0 |     0 |     0 |      1 |                   0 |             0 |                  0 |                              1 |                0 |     89.8 |

> **ETAPAS**

## Etapa 02 — Clasificación de NObeyesdad, paso a paso

**Archivo:** `etapas/e02_clasificacion.py`. **Figuras:** `figures/02_clasificacion/`.

**Protocolo:** split estratificado 80/20 → GridSearchCV (5-fold estratificado, `f1_macro`) en el
train → evaluar **una vez** en test. KNN dentro de `Pipeline([StandardScaler, KNN])`.

**Funciones clave del archivo:**

| Función | Qué hace |
| :--- | :--- |
| `datos()` | lee `clf.csv`, separa `X`/`y`, hace `train_test_split(test_size=0.2, stratify=y, random_state=42)` |
| `cv_estratificado()` | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| `make_models()` | los 3 modelos con su grilla |
| `fit_evaluate(...)` | para cada modelo: GridSearchCV → `fit` → `predict` en test → métricas |
| `variante("A"/"B")` | entrena (o carga de caché) la variante; en B hace `drop(columns=DROP_WH)` |
| `arboles_prepoda()` | los 3 árboles del paso 3.4 |
| `_barrido_n_arboles()` | RF con 10…800 árboles (paso 3.7) |
| `_con_ruido(X_tr, X_te, nivel)` | añade ruido gaussiano a los hábitos (paso 7) |
| `_modelos_robustez()` | árbol, RF(200), KNN k=1, KNN k=3 distance (paso 7) |
| `filas_reales(X)` | máscara de filas con todas las respuestas de encuesta enteras (paso 8) |
| `_modelos_reales()` | línea base, árbol, RF, RF con pesos por clase, KNN k=15 y k=1 (paso 8) |
| `_cv_reales(X, y)` | accuracy, balanced accuracy y macro-F1 con CV 3×10 (paso 8) |
| `evaluacion_reales()` | todo el paso 8, cacheado en `cache/clf_filas_reales.joblib` |

---

### Paso 1: Datos y split

> *Ver código e02_clasificacion.py:158* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L158](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L158)

- train **(1669, 20)**, test **(418, 20)**. La tabla muestra el % de cada clase en train (12.8 % a 16.8 %) — idéntico al del dataset gracias a `stratify=y`.

#### Tablas de este paso como imagen (1)

![1 · distribucion de clases en train](imagenes/02_clasificacion_tablas_1_tabla1_distribucion_de_clases_en_train.png)

### Distribución de clases en train (%)

| NObeyesdad          |   % en train |
|:--------------------|-------------:|
| Insufficient_Weight |         12.8 |
| Normal_Weight       |         13.5 |
| Overweight_Level_I  |         13.2 |
| Overweight_Level_II |         13.9 |
| Obesity_Type_I      |         16.8 |
| Obesity_Type_II     |         14.2 |
| Obesity_Type_III    |         15.5 |

### Paso 2: Modelos y grillas

> *Ver código e02_clasificacion.py:170* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L170](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L170)

| Modelo | Grilla | Combinaciones |
| :--- | :--- | :--- |
| Árbol de decisión | `max_depth` ∈ {4, 6, 8, 10, None}, `min_samples_leaf` ∈ {1, 3, 5} | 15 |
| Random Forest | `n_estimators` ∈ {200, 400}, `max_depth` ∈ {None, 12}, `min_samples_leaf` ∈ {1, 2} | 8 |
| KNN (Pipeline) | `knn__n_neighbors` ∈ {1, 3, …, 31} (16 impares), `knn__weights` ∈ {uniform, distance} | 32 |

Cada combinación se evalúa con 5 folds → p. ej. KNN = 32 × 5 = 160 entrenamientos.

### Paso 3: Variante A (todas las features): tabla de resultados

> *Ver código e02_clasificacion.py:182* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L182](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L182)

| Modelo | Mejores hiperparámetros | macro-F1 CV | accuracy test | macro-F1 test |
| :--- | :--- | :--- | :--- | :--- |
| Árbol de decisión | max_depth=None, min_samples_leaf=1 | 0.9230 | 0.9378 | 0.9362 |
| **Random Forest** | n_estimators=400, max_depth=None, min_samples_leaf=1 | **0.9437** | **0.9569** | **0.9558** |
| KNN | K=1, weights=uniform | 0.8152 | 0.8206 | 0.8075 |

- **Lectura:** RF gana. Árbol y RF eligen `max_depth=None` (crecer sin límite): con Weight y Height disponibles, cuanto más cortes, mejor reconstruyen el IMC. CV y test son parecidos → no hay sobreajuste "oculto" en la elección de hiperparámetros.

#### Tablas de este paso como imagen (1)

![3 · variante a macro f1 en cv y metricas en](imagenes/02_clasificacion_tablas_3_tabla1_variante_a_macro_f1_en_cv_y_metricas_en.png)

### Variante A — macro-F1 en CV y métricas en test

|                   |   cv_f1_macro |   test_accuracy |   test_f1_macro |
|:------------------|--------------:|----------------:|----------------:|
| Árbol de decisión |        0.923  |          0.9378 |          0.9362 |
| Random Forest     |        0.9437 |          0.9569 |          0.9558 |
| KNN               |        0.8152 |          0.8206 |          0.8075 |

### Paso 3.1: Reportes por clase y matrices de confusión (A)

> *Ver código e02_clasificacion.py:188* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L188](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L188)

![3.1 · matrices confusion A](imagenes/02_clasificacion_3.1_matrices_confusion_A.png)

### Datos de la figura: 3.1_matrices_confusion_A

#### Árbol de decisión (A)

|                     |   Insufficient_Weight |   Normal_Weight |   Overweight_Level_I |   Overweight_Level_II |   Obesity_Type_I |   Obesity_Type_II |   Obesity_Type_III |
|:--------------------|----------------------:|----------------:|---------------------:|----------------------:|-----------------:|------------------:|-------------------:|
| Insufficient_Weight |                    49 |               4 |                    0 |                     0 |                0 |                 0 |                  0 |
| Normal_Weight       |                     1 |              53 |                    3 |                     0 |                0 |                 0 |                  0 |
| Overweight_Level_I  |                     0 |               7 |                   45 |                     2 |                1 |                 0 |                  0 |
| Overweight_Level_II |                     0 |               0 |                    0 |                    56 |                2 |                 0 |                  0 |
| Obesity_Type_I      |                     0 |               1 |                    1 |                     2 |               66 |                 0 |                  0 |
| Obesity_Type_II     |                     0 |               0 |                    0 |                     0 |                1 |                59 |                  0 |
| Obesity_Type_III    |                     0 |               0 |                    0 |                     0 |                0 |                 1 |                 64 |

#### Random Forest (A)

|                     |   Insufficient_Weight |   Normal_Weight |   Overweight_Level_I |   Overweight_Level_II |   Obesity_Type_I |   Obesity_Type_II |   Obesity_Type_III |
|:--------------------|----------------------:|----------------:|---------------------:|----------------------:|-----------------:|------------------:|-------------------:|
| Insufficient_Weight |                    50 |               3 |                    0 |                     0 |                0 |                 0 |                  0 |
| Normal_Weight       |                     0 |              55 |                    2 |                     0 |                0 |                 0 |                  0 |
| Overweight_Level_I  |                     0 |               5 |                   50 |                     0 |                0 |                 0 |                  0 |
| Overweight_Level_II |                     0 |               2 |                    3 |                    53 |                0 |                 0 |                  0 |
| Obesity_Type_I      |                     0 |               0 |                    0 |                     2 |               68 |                 0 |                  0 |
| Obesity_Type_II     |                     0 |               0 |                    0 |                     0 |                0 |                60 |                  0 |
| Obesity_Type_III    |                     0 |               0 |                    0 |                     0 |                1 |                 0 |                 64 |

#### KNN (A)

|                     |   Insufficient_Weight |   Normal_Weight |   Overweight_Level_I |   Overweight_Level_II |   Obesity_Type_I |   Obesity_Type_II |   Obesity_Type_III |
|:--------------------|----------------------:|----------------:|---------------------:|----------------------:|-----------------:|------------------:|-------------------:|
| Insufficient_Weight |                    49 |               3 |                    1 |                     0 |                0 |                 0 |                  0 |
| Normal_Weight       |                    14 |              25 |                    4 |                    12 |                2 |                 0 |                  0 |
| Overweight_Level_I  |                     1 |               6 |                   39 |                     4 |                5 |                 0 |                  0 |
| Overweight_Level_II |                     2 |               1 |                    6 |                    46 |                2 |                 0 |                  1 |
| Obesity_Type_I      |                     0 |               3 |                    1 |                     2 |               62 |                 2 |                  0 |
| Obesity_Type_II     |                     0 |               2 |                    0 |                     1 |                0 |                57 |                  0 |
| Obesity_Type_III    |                     0 |               0 |                    0 |                     0 |                0 |                 0 |                 65 |

- **`classification_report`** de cada modelo (precision, recall, f1, support por clase). Datos clave del RF: Obesity_Type_II F1 = 1.000, Obesity_Type_III 0.992; la clase más difícil es Normal_Weight (F1 0.902). En KNN, Normal_Weight cae a F1 0.515 (recall 0.439).
- **Gráfica `3.1_matrices_confusion_A.png`:** tres matrices (árbol, RF, KNN), filas = clase real, columnas = predicha, en `CLASS_ORDER`. **Cómo se lee:** la diagonal casi llena; los pocos errores están **justo al lado de la diagonal** (Normal ↔ Overweight_I, Overweight_I ↔ Overweight_II): son clases **contiguas** de IMC y los errores caen en las fronteras. Nunca se confunde bajo peso con obesidad III. **Código:** `plot_confusion(y_test, gs.predict(X_te), CLASS_ORDER, ...)` para cada modelo en `plt.subplots(1, 3)`.
- **Tablas por clase** (`3.1_tabla4…6_metricas_por_clase_*`): debajo de las matrices, una tabla por modelo con TP, FN, FP, TN, Precision, Recall y F1 de cada clase (ver 6.6). Es la forma más directa de leer la matriz: p. ej. en RF, Normal_Weight tiene 10 FP (Precision 0.846) porque recibe casos de las clases vecinas.

#### Tablas de este paso como imagen (6)

![3.1 · classification report por clase arbol de](imagenes/02_clasificacion_tablas_3.1_tabla1_classification_report_por_clase_arbol_de.png)

### classification_report por clase — Árbol de decisión (A)

|                     |   precision |   recall |   f1-score |
|:--------------------|------------:|---------:|-----------:|
| Insufficient_Weight |       0.98  |    0.925 |      0.951 |
| Normal_Weight       |       0.815 |    0.93  |      0.869 |
| Overweight_Level_I  |       0.918 |    0.818 |      0.865 |
| Overweight_Level_II |       0.933 |    0.966 |      0.949 |
| Obesity_Type_I      |       0.943 |    0.943 |      0.943 |
| Obesity_Type_II     |       0.983 |    0.983 |      0.983 |
| Obesity_Type_III    |       1     |    0.985 |      0.992 |

![3.1 · classification report por clase random f](imagenes/02_clasificacion_tablas_3.1_tabla2_classification_report_por_clase_random_f.png)

### classification_report por clase — Random Forest (A)

|                     |   precision |   recall |   f1-score |
|:--------------------|------------:|---------:|-----------:|
| Insufficient_Weight |       1     |    0.943 |      0.971 |
| Normal_Weight       |       0.846 |    0.965 |      0.902 |
| Overweight_Level_I  |       0.909 |    0.909 |      0.909 |
| Overweight_Level_II |       0.964 |    0.914 |      0.938 |
| Obesity_Type_I      |       0.986 |    0.971 |      0.978 |
| Obesity_Type_II     |       1     |    1     |      1     |
| Obesity_Type_III    |       1     |    0.985 |      0.992 |

![3.1 · classification report por clase knn a](imagenes/02_clasificacion_tablas_3.1_tabla3_classification_report_por_clase_knn_a.png)

### classification_report por clase — KNN (A)

|                     |   precision |   recall |   f1-score |
|:--------------------|------------:|---------:|-----------:|
| Insufficient_Weight |       0.742 |    0.925 |      0.824 |
| Normal_Weight       |       0.625 |    0.439 |      0.515 |
| Overweight_Level_I  |       0.765 |    0.709 |      0.736 |
| Overweight_Level_II |       0.708 |    0.793 |      0.748 |
| Obesity_Type_I      |       0.873 |    0.886 |      0.879 |
| Obesity_Type_II     |       0.966 |    0.95  |      0.958 |
| Obesity_Type_III    |       0.985 |    1     |      0.992 |

![3.1 · metricas por clase arbol de decision a](imagenes/02_clasificacion_tablas_3.1_tabla4_metricas_por_clase_arbol_de_decision_a.png)

### Métricas por clase — Árbol de decisión (A)

| Clase               |   TP |   FN |   FP |   TN |   Precision |   Recall |    F1 |
|:--------------------|-----:|-----:|-----:|-----:|------------:|---------:|------:|
| Insufficient_Weight |   49 |    4 |    1 |  364 |       0.98  |    0.925 | 0.951 |
| Normal_Weight       |   53 |    4 |   12 |  349 |       0.815 |    0.93  | 0.869 |
| Overweight_Level_I  |   45 |   10 |    4 |  359 |       0.918 |    0.818 | 0.865 |
| Overweight_Level_II |   56 |    2 |    4 |  356 |       0.933 |    0.966 | 0.949 |
| Obesity_Type_I      |   66 |    4 |    4 |  344 |       0.943 |    0.943 | 0.943 |
| Obesity_Type_II     |   59 |    1 |    1 |  357 |       0.983 |    0.983 | 0.983 |
| Obesity_Type_III    |   64 |    1 |    0 |  353 |       1     |    0.985 | 0.992 |

![3.1 · metricas por clase random forest a](imagenes/02_clasificacion_tablas_3.1_tabla5_metricas_por_clase_random_forest_a.png)

### Métricas por clase — Random Forest (A)

| Clase               |   TP |   FN |   FP |   TN |   Precision |   Recall |    F1 |
|:--------------------|-----:|-----:|-----:|-----:|------------:|---------:|------:|
| Insufficient_Weight |   50 |    3 |    0 |  365 |       1     |    0.943 | 0.971 |
| Normal_Weight       |   55 |    2 |   10 |  351 |       0.846 |    0.965 | 0.902 |
| Overweight_Level_I  |   50 |    5 |    5 |  358 |       0.909 |    0.909 | 0.909 |
| Overweight_Level_II |   53 |    5 |    2 |  358 |       0.964 |    0.914 | 0.938 |
| Obesity_Type_I      |   68 |    2 |    1 |  347 |       0.986 |    0.971 | 0.978 |
| Obesity_Type_II     |   60 |    0 |    0 |  358 |       1     |    1     | 1     |
| Obesity_Type_III    |   64 |    1 |    0 |  353 |       1     |    0.985 | 0.992 |

![3.1 · metricas por clase knn a](imagenes/02_clasificacion_tablas_3.1_tabla6_metricas_por_clase_knn_a.png)

### Métricas por clase — KNN (A)

| Clase               |   TP |   FN |   FP |   TN |   Precision |   Recall |    F1 |
|:--------------------|-----:|-----:|-----:|-----:|------------:|---------:|------:|
| Insufficient_Weight |   49 |    4 |   17 |  348 |       0.742 |    0.925 | 0.824 |
| Normal_Weight       |   25 |   32 |   15 |  346 |       0.625 |    0.439 | 0.515 |
| Overweight_Level_I  |   39 |   16 |   12 |  351 |       0.765 |    0.709 | 0.736 |
| Overweight_Level_II |   46 |   12 |   19 |  341 |       0.708 |    0.793 | 0.748 |
| Obesity_Type_I      |   62 |    8 |    9 |  339 |       0.873 |    0.886 | 0.879 |
| Obesity_Type_II     |   57 |    3 |    2 |  356 |       0.966 |    0.95  | 0.958 |
| Obesity_Type_III    |   65 |    0 |    1 |  352 |       0.985 |    1     | 0.992 |

### Paso 3.2: Importancia de features del RF (A)

> *Ver código e02_clasificacion.py:210* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L210](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L210)

![3.2 · importancias rf A](imagenes/02_clasificacion_3.2_importancias_rf_A.png)

### Datos de la figura: 3.2_importancias_rf_A

#### Importancia de features — Random Forest (A)

| None                           |       valor |
|:-------------------------------|------------:|
| Weight                         | 0.311482    |
| Age                            | 0.0961661   |
| Height                         | 0.0952862   |
| FCVC                           | 0.0881645   |
| Gender                         | 0.0542296   |
| NCP                            | 0.0537821   |
| FAF                            | 0.0502255   |
| TUE                            | 0.047912    |
| CH2O                           | 0.0476404   |
| family_history_with_overweight | 0.0328215   |
| CAEC                           | 0.0322337   |
| CALC                           | 0.0316903   |
| FAVC                           | 0.0175593   |
| MTRANS_Public_Transportation   | 0.0144603   |
| MTRANS_Automobile              | 0.0131365   |
| SCC                            | 0.00642348  |
| MTRANS_Walking                 | 0.00314282  |
| SMOKE                          | 0.00272976  |
| MTRANS_Motorbike               | 0.000490114 |
| MTRANS_Bike                    | 0.000424088 |

- **Gráfica `3.2_importancias_rf_A.png`:** barras horizontales ordenadas.
- **Top-5:** **Weight 0.311**, Age 0.096, Height 0.095, FCVC 0.088, Gender 0.054. Las últimas: MTRANS_Bike 0.0004, MTRANS_Motorbike 0.0005.
- **Lectura:** Weight vale tres veces la siguiente → el modelo reconstruye el IMC. Es la fuga de datos vista desde las importancias.
- **Código:** `pd.Series(rf_A.feature_importances_, index=X_tr.columns).sort_values(ascending=False)` y `sns.barplot(x=imp.values, y=imp.index)`.

#### Tablas de este paso como imagen (1)

![3.2 · feature importances  reduccion media de](imagenes/02_clasificacion_tablas_3.2_tabla1_feature_importances__reduccion_media_de.png)

### feature_importances_ (reducción media de impureza Gini)

|                                |   importancia |
|:-------------------------------|--------------:|
| Weight                         |        0.3115 |
| Age                            |        0.0962 |
| Height                         |        0.0953 |
| FCVC                           |        0.0882 |
| Gender                         |        0.0542 |
| NCP                            |        0.0538 |
| FAF                            |        0.0502 |
| TUE                            |        0.0479 |
| CH2O                           |        0.0476 |
| family_history_with_overweight |        0.0328 |
| CAEC                           |        0.0322 |
| CALC                           |        0.0317 |
| FAVC                           |        0.0176 |
| MTRANS_Public_Transportation   |        0.0145 |
| MTRANS_Automobile              |        0.0131 |
| SCC                            |        0.0064 |
| MTRANS_Walking                 |        0.0031 |
| SMOKE                          |        0.0027 |
| MTRANS_Motorbike               |        0.0005 |
| MTRANS_Bike                    |        0.0004 |

### Paso 3.3: Árbol de GridSearch dibujado (A)

> *Ver código e02_clasificacion.py:222* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L222](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L222)

![3.3 · arbol gridsearch A](imagenes/02_clasificacion_3.3_arbol_gridsearch_A.png)

### Datos de la figura: 3.3_arbol_gridsearch_A

#### Árbol de GridSearch (variante A) — dibujo truncado a 3 niveles

|    |   nodo | condicion         |    gini |   samples | value                                             | class               | info   |
|---:|-------:|:------------------|--------:|----------:|:--------------------------------------------------|:--------------------|:-------|
|  0 |      1 | Weight <= 99.536  |   0.856 |      1669 | [214.0, 225.0, 281.0, 237.0, 259.0, 221.0, 232.0] | Obesity_Type_I      | nan    |
|  1 |      2 | Weight <= 60.059  |   0.801 |      1086 | [214, 225, 192, 5, 0, 221, 229]                   | Overweight_Level_II | nan    |
|  2 |      3 | Height <= 1.66    |   0.488 |       335 | [209, 117, 0, 0, 0, 8, 1]                         | Insufficient_Weight | nan    |
|  3 |      4 | Weight <= 46.828  |   0.538 |       184 | [76.0, 99.0, 0.0, 0.0, 0.0, 8.0, 1.0]             | Normal_Weight       | nan    |
|  4 |      5 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
|  5 |      6 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
|  6 |      7 | Weight <= 59.995  |   0.21  |       151 | [133, 18, 0, 0, 0, 0, 0]                          | Insufficient_Weight | nan    |
|  7 |      8 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
|  8 |      9 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
|  9 |     10 | Weight <= 76.041  |   0.741 |       751 | [5, 108, 192, 5, 0, 213, 228]                     | Overweight_Level_II | nan    |
| 10 |     11 | Height <= 1.719   |   0.625 |       271 | [5, 94, 2, 0, 0, 131, 39]                         | Overweight_Level_I  | nan    |
| 11 |     12 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 12 |     13 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 13 |     14 | Height <= 1.673   |   0.658 |       480 | [0, 14, 190, 5, 0, 82, 189]                       | Obesity_Type_I      | nan    |
| 14 |     15 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 15 |     16 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 16 |     17 | Gender <= 0.5     |   0.621 |       583 | [0, 0, 89, 232, 259, 0, 3]                        | Obesity_Type_III    | nan    |
| 17 |     18 | Weight <= 101.25  |   0.008 |       260 | [0, 0, 0, 1, 259, 0, 0]                           | Obesity_Type_III    | nan    |
| 18 |     19 | nan               |   0     |         1 | [0, 0, 0, 1, 0, 0, 0]                             | Obesity_Type_II     | nan    |
| 19 |     20 | nan               |   0     |       259 | [0, 0, 0, 0, 259, 0, 0]                           | Obesity_Type_III    | nan    |
| 20 |     21 | Age <= 22.34      |   0.413 |       323 | [0, 0, 89, 231, 0, 0, 3]                          | Obesity_Type_II     | nan    |
| 21 |     22 | FCVC <= 2.705     |   0.313 |        67 | [0, 0, 54, 13, 0, 0, 0]                           | Obesity_Type_I      | nan    |
| 22 |     23 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 23 |     24 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 24 |     25 | Weight <= 109.744 |   0.256 |       256 | [0, 0, 35, 218, 0, 0, 3]                          | Obesity_Type_II     | nan    |
| 25 |     26 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 26 |     27 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |

- **Gráfica `3.3_arbol_gridsearch_A.png`:** el árbol elegido por GridSearch (profundidad 16, 115 hojas) dibujado **sólo hasta el nivel 3** (`max_depth=3` en `plot_tree`; el árbol real es más profundo, es sólo para que se pueda leer).
- **Cómo se lee la raíz:** `Weight <= 99.536`, `gini = 0.856`, `samples = 1669`, `value = [...]` (conteo de cada clase **en orden alfabético** de `classes_`), `class =` la mayoritaria. Izquierda = se cumple (≤ 99.5 kg), derecha = no. Nivel 2: izquierda vuelve a cortar por Weight (≤ 60.06), derecha corta por **Gender** (separar Obesity II —hombres— de Obesity III —mujeres—).

### Paso 3.4: Tres árboles: completo, podado y muy podado (prepoda)

> *Ver código e02_clasificacion.py:234* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L234](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L234)

![3.4 · arbol completo](imagenes/02_clasificacion_3.4_arbol_completo.png)

### Datos de la figura: 3.4_arbol_completo

#### Árbol Completo — profundidad 16, 115 hojas, raíz: Weight ≤ 99.536  (dibujo truncado a 3 niveles)

|    |   nodo | condicion         |    gini |   samples | value                                             | class               | info   |
|---:|-------:|:------------------|--------:|----------:|:--------------------------------------------------|:--------------------|:-------|
|  0 |      1 | Weight <= 99.536  |   0.856 |      1669 | [214.0, 225.0, 281.0, 237.0, 259.0, 221.0, 232.0] | Obesity_Type_I      | nan    |
|  1 |      2 | Weight <= 60.059  |   0.801 |      1086 | [214, 225, 192, 5, 0, 221, 229]                   | Overweight_Level_II | nan    |
|  2 |      3 | Height <= 1.66    |   0.488 |       335 | [209, 117, 0, 0, 0, 8, 1]                         | Insufficient_Weight | nan    |
|  3 |      4 | Weight <= 46.828  |   0.538 |       184 | [76.0, 99.0, 0.0, 0.0, 0.0, 8.0, 1.0]             | Normal_Weight       | nan    |
|  4 |      5 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
|  5 |      6 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
|  6 |      7 | Weight <= 59.995  |   0.21  |       151 | [133, 18, 0, 0, 0, 0, 0]                          | Insufficient_Weight | nan    |
|  7 |      8 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
|  8 |      9 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
|  9 |     10 | Weight <= 76.041  |   0.741 |       751 | [5, 108, 192, 5, 0, 213, 228]                     | Overweight_Level_II | nan    |
| 10 |     11 | Height <= 1.719   |   0.625 |       271 | [5, 94, 2, 0, 0, 131, 39]                         | Overweight_Level_I  | nan    |
| 11 |     12 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 12 |     13 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 13 |     14 | Height <= 1.673   |   0.658 |       480 | [0, 14, 190, 5, 0, 82, 189]                       | Obesity_Type_I      | nan    |
| 14 |     15 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 15 |     16 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 16 |     17 | Gender <= 0.5     |   0.621 |       583 | [0, 0, 89, 232, 259, 0, 3]                        | Obesity_Type_III    | nan    |
| 17 |     18 | Weight <= 101.25  |   0.008 |       260 | [0, 0, 0, 1, 259, 0, 0]                           | Obesity_Type_III    | nan    |
| 18 |     19 | nan               |   0     |         1 | [0, 0, 0, 1, 0, 0, 0]                             | Obesity_Type_II     | nan    |
| 19 |     20 | nan               |   0     |       259 | [0, 0, 0, 0, 259, 0, 0]                           | Obesity_Type_III    | nan    |
| 20 |     21 | Age <= 22.34      |   0.413 |       323 | [0, 0, 89, 231, 0, 0, 3]                          | Obesity_Type_II     | nan    |
| 21 |     22 | FCVC <= 2.705     |   0.313 |        67 | [0, 0, 54, 13, 0, 0, 0]                           | Obesity_Type_I      | nan    |
| 22 |     23 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 23 |     24 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 24 |     25 | Weight <= 109.744 |   0.256 |       256 | [0, 0, 35, 218, 0, 0, 3]                          | Obesity_Type_II     | nan    |
| 25 |     26 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 26 |     27 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |

![3.4 · arbol podado](imagenes/02_clasificacion_3.4_arbol_podado.png)

### Datos de la figura: 3.4_arbol_podado

#### Árbol Podado — profundidad 6, 38 hojas, raíz: Weight ≤ 99.536  (dibujo truncado a 3 niveles)

|    |   nodo | condicion         |    gini |   samples | value                                             | class               | info   |
|---:|-------:|:------------------|--------:|----------:|:--------------------------------------------------|:--------------------|:-------|
|  0 |      1 | Weight <= 99.536  |   0.856 |      1669 | [214.0, 225.0, 281.0, 237.0, 259.0, 221.0, 232.0] | Obesity_Type_I      | nan    |
|  1 |      2 | Weight <= 60.059  |   0.801 |      1086 | [214, 225, 192, 5, 0, 221, 229]                   | Overweight_Level_II | nan    |
|  2 |      3 | Height <= 1.66    |   0.488 |       335 | [209, 117, 0, 0, 0, 8, 1]                         | Insufficient_Weight | nan    |
|  3 |      4 | Weight <= 46.828  |   0.538 |       184 | [76.0, 99.0, 0.0, 0.0, 0.0, 8.0, 1.0]             | Normal_Weight       | nan    |
|  4 |      5 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
|  5 |      6 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
|  6 |      7 | Weight <= 59.995  |   0.21  |       151 | [133, 18, 0, 0, 0, 0, 0]                          | Insufficient_Weight | nan    |
|  7 |      8 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
|  8 |      9 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
|  9 |     10 | Weight <= 76.041  |   0.741 |       751 | [5, 108, 192, 5, 0, 213, 228]                     | Overweight_Level_II | nan    |
| 10 |     11 | Height <= 1.719   |   0.625 |       271 | [5, 94, 2, 0, 0, 131, 39]                         | Overweight_Level_I  | nan    |
| 11 |     12 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 12 |     13 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 13 |     14 | Height <= 1.673   |   0.658 |       480 | [0, 14, 190, 5, 0, 82, 189]                       | Obesity_Type_I      | nan    |
| 14 |     15 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 15 |     16 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 16 |     17 | Gender <= 0.5     |   0.621 |       583 | [0, 0, 89, 232, 259, 0, 3]                        | Obesity_Type_III    | nan    |
| 17 |     18 | CH2O <= 1.005     |   0.008 |       260 | [0, 0, 0, 1, 259, 0, 0]                           | Obesity_Type_III    | nan    |
| 18 |     19 | nan               |   0.32  |         5 | [0, 0, 0, 1, 4, 0, 0]                             | Obesity_Type_III    | nan    |
| 19 |     20 | nan               |   0     |       255 | [0, 0, 0, 0, 255, 0, 0]                           | Obesity_Type_III    | nan    |
| 20 |     21 | Age <= 22.34      |   0.413 |       323 | [0, 0, 89, 231, 0, 0, 3]                          | Obesity_Type_II     | nan    |
| 21 |     22 | FCVC <= 2.705     |   0.313 |        67 | [0, 0, 54, 13, 0, 0, 0]                           | Obesity_Type_I      | nan    |
| 22 |     23 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 23 |     24 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 24 |     25 | Weight <= 109.744 |   0.256 |       256 | [0, 0, 35, 218, 0, 0, 3]                          | Obesity_Type_II     | nan    |
| 25 |     26 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |
| 26 |     27 | nan               | nan     |       nan | nan                                               | nan                 | (...)  |

![3.4 · arbol muy podado](imagenes/02_clasificacion_3.4_arbol_muy_podado.png)

### Datos de la figura: 3.4_arbol_muy_podado

#### Árbol Muy podado — profundidad 3, 8 hojas, raíz: Weight ≤ 99.536

|    |   nodo | condicion         |   gini |   samples | value                                             | class               |
|---:|-------:|:------------------|-------:|----------:|:--------------------------------------------------|:--------------------|
|  0 |      1 | Weight <= 99.536  |  0.856 |      1669 | [214.0, 225.0, 281.0, 237.0, 259.0, 221.0, 232.0] | Obesity_Type_I      |
|  1 |      2 | Weight <= 60.059  |  0.801 |      1086 | [214, 225, 192, 5, 0, 221, 229]                   | Overweight_Level_II |
|  2 |      3 | Height <= 1.66    |  0.488 |       335 | [209, 117, 0, 0, 0, 8, 1]                         | Insufficient_Weight |
|  3 |      4 | nan               |  0.538 |       184 | [76.0, 99.0, 0.0, 0.0, 0.0, 8.0, 1.0]             | Normal_Weight       |
|  4 |      5 | nan               |  0.21  |       151 | [133, 18, 0, 0, 0, 0, 0]                          | Insufficient_Weight |
|  5 |      6 | Weight <= 76.041  |  0.741 |       751 | [5, 108, 192, 5, 0, 213, 228]                     | Overweight_Level_II |
|  6 |      7 | nan               |  0.625 |       271 | [5, 94, 2, 0, 0, 131, 39]                         | Overweight_Level_I  |
|  7 |      8 | nan               |  0.658 |       480 | [0, 14, 190, 5, 0, 82, 189]                       | Obesity_Type_I      |
|  8 |      9 | Gender <= 0.5     |  0.621 |       583 | [0, 0, 89, 232, 259, 0, 3]                        | Obesity_Type_III    |
|  9 |     10 | Weight <= 102.313 |  0.008 |       260 | [0, 0, 0, 1, 259, 0, 0]                           | Obesity_Type_III    |
| 10 |     11 | nan               |  0.18  |        10 | [0, 0, 0, 1, 9, 0, 0]                             | Obesity_Type_III    |
| 11 |     12 | nan               |  0     |       250 | [0, 0, 0, 0, 250, 0, 0]                           | Obesity_Type_III    |
| 12 |     13 | Age <= 22.34      |  0.413 |       323 | [0, 0, 89, 231, 0, 0, 3]                          | Obesity_Type_II     |
| 13 |     14 | nan               |  0.313 |        67 | [0, 0, 54, 13, 0, 0, 0]                           | Obesity_Type_I      |
| 14 |     15 | nan               |  0.256 |       256 | [0, 0, 35, 218, 0, 0, 3]                          | Obesity_Type_II     |

![3.4 · matrices tres arboles](imagenes/02_clasificacion_3.4_matrices_tres_arboles.png)

### Datos de la figura: 3.4_matrices_tres_arboles

#### Árbol Completo — acc test 0.938

|                     |   Insufficient_Weight |   Normal_Weight |   Overweight_Level_I |   Overweight_Level_II |   Obesity_Type_I |   Obesity_Type_II |   Obesity_Type_III |
|:--------------------|----------------------:|----------------:|---------------------:|----------------------:|-----------------:|------------------:|-------------------:|
| Insufficient_Weight |                    49 |               4 |                    0 |                     0 |                0 |                 0 |                  0 |
| Normal_Weight       |                     1 |              53 |                    3 |                     0 |                0 |                 0 |                  0 |
| Overweight_Level_I  |                     0 |               7 |                   45 |                     2 |                1 |                 0 |                  0 |
| Overweight_Level_II |                     0 |               0 |                    0 |                    56 |                2 |                 0 |                  0 |
| Obesity_Type_I      |                     0 |               1 |                    1 |                     2 |               66 |                 0 |                  0 |
| Obesity_Type_II     |                     0 |               0 |                    0 |                     0 |                1 |                59 |                  0 |
| Obesity_Type_III    |                     0 |               0 |                    0 |                     0 |                0 |                 1 |                 64 |

#### Árbol Podado — acc test 0.888

|                     |   Insufficient_Weight |   Normal_Weight |   Overweight_Level_I |   Overweight_Level_II |   Obesity_Type_I |   Obesity_Type_II |   Obesity_Type_III |
|:--------------------|----------------------:|----------------:|---------------------:|----------------------:|-----------------:|------------------:|-------------------:|
| Insufficient_Weight |                    49 |               4 |                    0 |                     0 |                0 |                 0 |                  0 |
| Normal_Weight       |                     2 |              41 |                   14 |                     0 |                0 |                 0 |                  0 |
| Overweight_Level_I  |                     0 |               5 |                   48 |                     2 |                0 |                 0 |                  0 |
| Overweight_Level_II |                     0 |               0 |                   12 |                    43 |                3 |                 0 |                  0 |
| Obesity_Type_I      |                     0 |               1 |                    1 |                     1 |               67 |                 0 |                  0 |
| Obesity_Type_II     |                     0 |               0 |                    0 |                     0 |                1 |                59 |                  0 |
| Obesity_Type_III    |                     0 |               0 |                    0 |                     0 |                0 |                 1 |                 64 |

#### Árbol Muy podado — acc test 0.653

|                     |   Insufficient_Weight |   Normal_Weight |   Overweight_Level_I |   Overweight_Level_II |   Obesity_Type_I |   Obesity_Type_II |   Obesity_Type_III |
|:--------------------|----------------------:|----------------:|---------------------:|----------------------:|-----------------:|------------------:|-------------------:|
| Insufficient_Weight |                    41 |              11 |                    1 |                     0 |                0 |                 0 |                  0 |
| Normal_Weight       |                     4 |              19 |                   30 |                     0 |                4 |                 0 |                  0 |
| Overweight_Level_I  |                     0 |               2 |                   31 |                     0 |               22 |                 0 |                  0 |
| Overweight_Level_II |                     0 |               0 |                    5 |                     0 |               51 |                 2 |                  0 |
| Obesity_Type_I      |                     0 |               0 |                    2 |                     0 |               60 |                 8 |                  0 |
| Obesity_Type_II     |                     0 |               0 |                    0 |                     0 |                2 |                58 |                  0 |
| Obesity_Type_III    |                     0 |               0 |                    0 |                     0 |                1 |                 0 |                 64 |

Pedido del profesor: ver cómo cambia el árbol al podarlo. Se entrenan **a mano** (sin
GridSearch) sobre el train de la variante A:

| Árbol | max_depth | min_samples_leaf | Profundidad real | Hojas | Raíz | acc train | acc test | macro-F1 test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Completo | None | 1 | 16 | 115 | Weight ≤ 99.536 | **1.000** | 0.938 | 0.936 |
| Podado | 6 | 5 | 6 | 38 | Weight ≤ 99.536 | 0.913 | 0.888 | 0.885 |
| Muy podado | 3 | 10 | 3 | 8 | Weight ≤ 99.536 | 0.649 | 0.653 | 0.605 |

- **Gráficas `3.4_arbol_completo.png`, `3.4_arbol_podado.png`, `3.4_arbol_muy_podado.png`:** el dibujo de cada árbol (completo y podado truncados a 3 niveles para el dibujo; el muy podado completo). El título dice profundidad, hojas y raíz.
- **El árbol muy podado entero** (se puede explicar en voz alta):`Weight ≤ 99.54 ├── Weight ≤ 60.06 │ ├── Height ≤ 1.66 → Normal_Weight │ └── Height > 1.66 → Insufficient_Weight (misma masa, más alto = IMC menor) └── Weight > 60.06 ├── Weight ≤ 76.04 → Overweight_Level_I └── Weight > 76.04 → Obesity_Type_I Weight > 99.54 ├── Gender = Female (≤ 0.5) → Obesity_Type_III (dos hojas, ambas Obesity III) └── Gender = Male ├── Age ≤ 22.34 → Obesity_Type_I └── Age > 22.34 → Obesity_Type_II` Está aproximando el IMC "a trozos" con Weight y Height, y usa Gender para separar Obesity II/III. **Ninguna hoja predice Overweight_Level_II** → esa columna de su matriz de confusión queda en ceros.
- **Gráfica `3.4_matrices_tres_arboles.png`:** tres matrices de confusión **con la misma escala de color** (`vmin=0, vmax=...`) para compararlas. Completo: diagonal casi perfecta. Podado: aparecen errores entre clases contiguas (14 Normal→Overweight_I, 12 Overweight_II→Overweight_I). Muy podado: la columna Overweight_Level_II vacía, 51 Overweight_II → Obesity_I, 30 Normal → Overweight_I.
- **Qué hay que explicar:** El completo **memoriza** (train 1.000) → sobreajuste; brecha train–test 0.062. Al podar, train baja mucho y test baja menos: la brecha del podado es sólo **0.025**. El muy podado **subajusta** (train y test ≈ 0.65), aunque sigue muy lejos del azar (0.14). **Los tres tienen la misma raíz**: la poda cambia la profundidad, no el primer corte (el primer corte es el que más reduce el Gini en todo el train; no depende de los límites).
- **Tablas por clase** de los tres árboles (`3.4_tabla2…4_metricas_por_clase_arbol_*`): muestran en qué clases pierde el árbol muy podado (Recall bajo en las clases que no llega a separar).

#### Tablas de este paso como imagen (4)

![3.4 · resumen de los tres arboles](imagenes/02_clasificacion_tablas_3.4_tabla1_resumen_de_los_tres_arboles.png)

### Resumen de los tres árboles

| modelo     |   profundidad |   n_hojas | raiz_atributo   |   acc_train |   acc_test |   f1_macro_test |
|:-----------|--------------:|----------:|:----------------|------------:|-----------:|----------------:|
| Completo   |            16 |       115 | Weight          |       1     |      0.938 |           0.936 |
| Podado     |             6 |        38 | Weight          |       0.913 |      0.888 |           0.885 |
| Muy podado |             3 |         8 | Weight          |       0.649 |      0.653 |           0.605 |

![3.4 · metricas por clase arbol completo](imagenes/02_clasificacion_tablas_3.4_tabla2_metricas_por_clase_arbol_completo.png)

### Métricas por clase — árbol Completo

| Clase               |   TP |   FN |   FP |   TN |   Precision |   Recall |    F1 |
|:--------------------|-----:|-----:|-----:|-----:|------------:|---------:|------:|
| Insufficient_Weight |   49 |    4 |    1 |  364 |       0.98  |    0.925 | 0.951 |
| Normal_Weight       |   53 |    4 |   12 |  349 |       0.815 |    0.93  | 0.869 |
| Overweight_Level_I  |   45 |   10 |    4 |  359 |       0.918 |    0.818 | 0.865 |
| Overweight_Level_II |   56 |    2 |    4 |  356 |       0.933 |    0.966 | 0.949 |
| Obesity_Type_I      |   66 |    4 |    4 |  344 |       0.943 |    0.943 | 0.943 |
| Obesity_Type_II     |   59 |    1 |    1 |  357 |       0.983 |    0.983 | 0.983 |
| Obesity_Type_III    |   64 |    1 |    0 |  353 |       1     |    0.985 | 0.992 |

![3.4 · metricas por clase arbol podado](imagenes/02_clasificacion_tablas_3.4_tabla3_metricas_por_clase_arbol_podado.png)

### Métricas por clase — árbol Podado

| Clase               |   TP |   FN |   FP |   TN |   Precision |   Recall |    F1 |
|:--------------------|-----:|-----:|-----:|-----:|------------:|---------:|------:|
| Insufficient_Weight |   49 |    4 |    2 |  363 |       0.961 |    0.925 | 0.942 |
| Normal_Weight       |   41 |   16 |   10 |  351 |       0.804 |    0.719 | 0.759 |
| Overweight_Level_I  |   48 |    7 |   27 |  336 |       0.64  |    0.873 | 0.738 |
| Overweight_Level_II |   43 |   15 |    3 |  357 |       0.935 |    0.741 | 0.827 |
| Obesity_Type_I      |   67 |    3 |    4 |  344 |       0.944 |    0.957 | 0.95  |
| Obesity_Type_II     |   59 |    1 |    1 |  357 |       0.983 |    0.983 | 0.983 |
| Obesity_Type_III    |   64 |    1 |    0 |  353 |       1     |    0.985 | 0.992 |

![3.4 · metricas por clase arbol muy podado](imagenes/02_clasificacion_tablas_3.4_tabla4_metricas_por_clase_arbol_muy_podado.png)

### Métricas por clase — árbol Muy podado

| Clase               |   TP |   FN |   FP |   TN | Precision   |   Recall | F1    |
|:--------------------|-----:|-----:|-----:|-----:|:------------|---------:|:------|
| Insufficient_Weight |   41 |   12 |    4 |  361 | 0.911       |    0.774 | 0.837 |
| Normal_Weight       |   19 |   38 |   13 |  348 | 0.594       |    0.333 | 0.427 |
| Overweight_Level_I  |   31 |   24 |   38 |  325 | 0.449       |    0.564 | 0.500 |
| Overweight_Level_II |    0 |   58 |    0 |  360 | —           |    0     | —     |
| Obesity_Type_I      |   60 |   10 |   80 |  268 | 0.429       |    0.857 | 0.571 |
| Obesity_Type_II     |   58 |    2 |   10 |  348 | 0.853       |    0.967 | 0.906 |
| Obesity_Type_III    |   64 |    1 |    0 |  353 | 1.000       |    0.985 | 0.992 |

### Paso 3.6: Curva del mejor K de KNN (A)

> *Ver código e02_clasificacion.py:297* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L297](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L297)

![3.6 · curva k knn A](imagenes/02_clasificacion_3.6_curva_k_knn_A.png)

### Datos de la figura: 3.6_curva_k_knn_A

#### KNN: macro-F1 (CV) según K

|   K (número de vecinos) |   macro-F1 (CV) |
|------------------------:|----------------:|
|                       1 |        0.815197 |
|                       3 |        0.802068 |
|                       5 |        0.805642 |
|                       7 |        0.791698 |
|                       9 |        0.782556 |
|                      11 |        0.775982 |
|                      13 |        0.770677 |
|                      15 |        0.768447 |
|                      17 |        0.759779 |
|                      19 |        0.750065 |
|                      21 |        0.738902 |
|                      23 |        0.735932 |
|                      25 |        0.733872 |
|                      27 |        0.724715 |
|                      29 |        0.723573 |
|                      31 |        0.722121 |

- **Gráfica `3.6_curva_k_knn_A.png`:** macro-F1 de CV (eje y) para cada K impar 1…31 (eje x), tomando para cada K el mejor entre `uniform` y `distance`. Línea punteada en el mejor K.
- **Resultado:** **mejor K = 1** (macro-F1 CV 0.815). K=3: 0.802 (distance), K=5: 0.806, K=11: 0.776… La curva baja al crecer K (con una pequeña subida en K=5 con `distance`).
- **Por qué gana K=1 (explicación clave):** SMOTE crea puntos interpolando entre vecinos de la misma clase; cada fila tiene "hermanas" casi idénticas muy cerca. El vecino más cercano casi siempre es de la misma clase; al ampliar el vecindario entran vecinos de clases contiguas y el F1 baja. Con datos reales se esperaría un K mayor (lo confirma el paso 7).
- **Código:** `plot_k_curve(fitted["KNN"].cv_results_)` — reutiliza los resultados del GridSearch (no se vuelve a entrenar).

### Paso 3.7: ¿Cuántos árboles necesita el Random Forest?

> *Ver código e02_clasificacion.py:338* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L338](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L338)

![3.7 · n estimators rf](imagenes/02_clasificacion_3.7_n_estimators_rf.png)

### Datos de la figura: 3.7_n_estimators_rf

#### Random Forest: macro-F1 según el número de árboles (variante A)

|   n_estimators (número de árboles, escala log) |   validación cruzada (train) |     test |
|-----------------------------------------------:|-----------------------------:|---------:|
|                                             10 |                     0.910108 | 0.928422 |
|                                             25 |                     0.925296 | 0.944011 |
|                                             50 |                     0.933698 | 0.94378  |
|                                            100 |                     0.940923 | 0.955803 |
|                                            200 |                     0.937542 | 0.95576  |
|                                            400 |                     0.943731 | 0.95576  |
|                                            800 |                     0.946221 | 0.950813 |

#### Coste: tiempo de entrenamiento

|   n_estimators (número de árboles, escala log) |   segundos en entrenar |
|-----------------------------------------------:|-----------------------:|
|                                             10 |              0.042153  |
|                                             25 |              0.0605601 |
|                                             50 |              0.0874642 |
|                                            100 |              0.15358   |
|                                            200 |              0.277548  |
|                                            400 |              0.557933  |
|                                            800 |              1.1236    |

- **Qué hace:** para n ∈ {10, 25, 50, 100, 200, 400, 800} entrena un RF (variante A): macro-F1 en 5-fold CV (media y desviación), métricas en test y tiempo de `fit`. n_estimators macro-F1 CV desv. CV accuracy test macro-F1 test tiempo (s) 10 0.9101 0.0155 0.9306 0.9284 0.04 25 0.9253 0.0120 0.9450 0.9440 0.06 50 0.9337 0.0088 0.9450 0.9438 0.13 100 0.9409 0.0117 0.9569 0.9558 0.22 200 0.9375 0.0087 0.9569 0.9558 0.43 400 0.9437 0.0075 0.9569 0.9558 0.89 800 0.9462 0.0056 0.9522 0.9508 1.75 (Los tiempos dependen del computador.)
- **Gráfica `3.7_n_estimators_rf.png`** (dos paneles, eje x logarítmico): **Izquierda (rendimiento):** línea azul = macro-F1 de CV con **banda sombreada ± 1 desviación** entre folds; línea naranja = test; línea gris discontinua = **punto suficiente**; línea verde punteada = lo que usa el proyecto (200). **Derecha (coste):** tiempo de entrenamiento con su valor escrito sobre cada punto; crece ~linealmente (duplica al duplicar los árboles). **Criterio del "punto suficiente":** el **menor** número de árboles que ya consigue el 90 % de toda la mejora del barrido (entre el bosque más pequeño y el mejor valor de CV). Aquí: umbral = 0.9101 + 0.9 × (0.9462 − 0.9101) = 0.9426 → **400 árboles** (0.9437). Pasar de 400 a 800 duplica el tiempo a cambio de +0.0025 de macro-F1. **Cómo se lee:** la curva sube rápido y se aplana (no vuelve a bajar); la banda se **estrecha** hacia la derecha: más árboles = promedio más **estable**. La leve caída del test en 800 (0.957 → 0.952 = 2 filas de 418) es ruido de muestreo, no sobreajuste. Con 100 árboles el test ya es el mismo que con 400.
- **Idea para defender:** `n_estimators` es un **parámetro de coste**, no de complejidad. A diferencia de la profundidad o de K, pasarse no sobreajusta; la pregunta no es "¿cuál es el mejor?" (siempre el mayor, por un margen irrelevante) sino "¿a partir de cuántos deja de compensar?".

#### Tablas de este paso como imagen (1)

![3.7 · random forest rendimiento y coste segun](imagenes/02_clasificacion_tablas_3.7_tabla1_random_forest_rendimiento_y_coste_segun.png)

### Random Forest: rendimiento y coste según el número de árboles

|   n_estimators |   cv_f1_macro |   cv_std |   test_accuracy |   test_f1_macro |   tiempo_fit_s |
|---------------:|--------------:|---------:|----------------:|----------------:|---------------:|
|             10 |        0.9101 |   0.0155 |          0.9306 |          0.9284 |         0.0422 |
|             25 |        0.9253 |   0.012  |          0.945  |          0.944  |         0.0606 |
|             50 |        0.9337 |   0.0088 |          0.945  |          0.9438 |         0.0875 |
|            100 |        0.9409 |   0.0117 |          0.9569 |          0.9558 |         0.1536 |
|            200 |        0.9375 |   0.0087 |          0.9569 |          0.9558 |         0.2775 |
|            400 |        0.9437 |   0.0075 |          0.9569 |          0.9558 |         0.5579 |
|            800 |        0.9462 |   0.0056 |          0.9522 |          0.9508 |         1.1236 |

### Paso 4: Variante B (sin Weight ni Height)

> *Ver código e02_clasificacion.py:364* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L364](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L364)

![4 · matrices confusion B](imagenes/02_clasificacion_4_matrices_confusion_B.png)

### Datos de la figura: 4_matrices_confusion_B

#### Árbol de decisión (B, sin Weight/Height)

|                     |   Insufficient_Weight |   Normal_Weight |   Overweight_Level_I |   Overweight_Level_II |   Obesity_Type_I |   Obesity_Type_II |   Obesity_Type_III |
|:--------------------|----------------------:|----------------:|---------------------:|----------------------:|-----------------:|------------------:|-------------------:|
| Insufficient_Weight |                    46 |               3 |                    1 |                     1 |                2 |                 0 |                  0 |
| Normal_Weight       |                     4 |              29 |                    6 |                     5 |                9 |                 4 |                  0 |
| Overweight_Level_I  |                     4 |               7 |                   37 |                     1 |                5 |                 1 |                  0 |
| Overweight_Level_II |                     3 |               4 |                    5 |                    35 |                6 |                 5 |                  0 |
| Obesity_Type_I      |                     2 |               3 |                    7 |                     8 |               48 |                 2 |                  0 |
| Obesity_Type_II     |                     0 |               1 |                    2 |                     3 |                1 |                53 |                  0 |
| Obesity_Type_III    |                     0 |               1 |                    0 |                     0 |                0 |                 0 |                 64 |

#### Random Forest (B, sin Weight/Height)

|                     |   Insufficient_Weight |   Normal_Weight |   Overweight_Level_I |   Overweight_Level_II |   Obesity_Type_I |   Obesity_Type_II |   Obesity_Type_III |
|:--------------------|----------------------:|----------------:|---------------------:|----------------------:|-----------------:|------------------:|-------------------:|
| Insufficient_Weight |                    50 |               3 |                    0 |                     0 |                0 |                 0 |                  0 |
| Normal_Weight       |                     3 |              44 |                    4 |                     3 |                2 |                 1 |                  0 |
| Overweight_Level_I  |                     2 |               7 |                   41 |                     1 |                4 |                 0 |                  0 |
| Overweight_Level_II |                     1 |               6 |                    4 |                    40 |                5 |                 2 |                  0 |
| Obesity_Type_I      |                     0 |               3 |                    1 |                     1 |               62 |                 3 |                  0 |
| Obesity_Type_II     |                     0 |               2 |                    0 |                     1 |                0 |                57 |                  0 |
| Obesity_Type_III    |                     0 |               1 |                    0 |                     0 |                0 |                 0 |                 64 |

#### KNN (B, sin Weight/Height)

|                     |   Insufficient_Weight |   Normal_Weight |   Overweight_Level_I |   Overweight_Level_II |   Obesity_Type_I |   Obesity_Type_II |   Obesity_Type_III |
|:--------------------|----------------------:|----------------:|---------------------:|----------------------:|-----------------:|------------------:|-------------------:|
| Insufficient_Weight |                    48 |               3 |                    1 |                     0 |                1 |                 0 |                  0 |
| Normal_Weight       |                    12 |              29 |                    5 |                     8 |                2 |                 1 |                  0 |
| Overweight_Level_I  |                     2 |               8 |                   34 |                     3 |                6 |                 2 |                  0 |
| Overweight_Level_II |                     4 |               3 |                    6 |                    37 |                3 |                 4 |                  1 |
| Obesity_Type_I      |                     1 |               1 |                    3 |                     2 |               56 |                 6 |                  1 |
| Obesity_Type_II     |                     0 |               3 |                    2 |                     1 |                1 |                53 |                  0 |
| Obesity_Type_III    |                     0 |               1 |                    0 |                     0 |                0 |                 0 |                 64 |

![4 · importancias rf B](imagenes/02_clasificacion_4_importancias_rf_B.png)

### Datos de la figura: 4_importancias_rf_B

#### Importancia de features — Random Forest (B, sin Weight/Height)

| None                           |       valor |
|:-------------------------------|------------:|
| Age                            | 0.160071    |
| FCVC                           | 0.141832    |
| NCP                            | 0.096726    |
| FAF                            | 0.0963831   |
| TUE                            | 0.0935398   |
| CH2O                           | 0.0891509   |
| Gender                         | 0.0663871   |
| CALC                           | 0.0546993   |
| family_history_with_overweight | 0.0516438   |
| CAEC                           | 0.0511572   |
| FAVC                           | 0.028957    |
| MTRANS_Public_Transportation   | 0.0249686   |
| MTRANS_Automobile              | 0.0218927   |
| SCC                            | 0.0112505   |
| MTRANS_Walking                 | 0.0049649   |
| SMOKE                          | 0.00488416  |
| MTRANS_Motorbike               | 0.0010994   |
| MTRANS_Bike                    | 0.000392634 |

| Modelo | Mejores hiperparámetros | macro-F1 CV | accuracy test | macro-F1 test |
| :--- | :--- | :--- | :--- | :--- |
| Árbol de decisión | max_depth=None, min_samples_leaf=1 | 0.7440 | 0.7464 | 0.7401 |
| **Random Forest** | n_estimators=200, max_depth=None, min_samples_leaf=1 | **0.8506** | **0.8565** | **0.8528** |
| KNN | K=3, weights=distance | 0.7640 | 0.7679 | 0.7567 |

- **Gráfica `4_matrices_confusion_B.png`:** tres matrices; más dispersión fuera de la diagonal que en A, sobre todo en Normal_Weight, Overweight I/II. Obesity_Type_III sigue casi perfecta (F1 0.99): es la clase con perfil de hábitos más distintivo (y casi sólo mujeres).
- **Gráfica `4_importancias_rf_B.png`** (barras rojas): **Age 0.160, FCVC 0.142, NCP 0.097, FAF 0.096, TUE 0.094**, CH2O 0.089, Gender 0.066…
- **Lectura:** sin las dos columnas del IMC, RF cae ~10 puntos (0.957 → 0.857) pero sigue muy por encima del azar (0.14). Los hábitos y la condición física **sí** tienen información sobre el nivel de obesidad. Es el resultado "más honesto".
- **Tablas por clase** de los tres modelos (`4_tabla5…7_metricas_por_clase_*_b`), debajo de las matrices. La tabla de importancias pasa a ser `4_tabla8_importancias_rf_b`.

#### Tablas de este paso como imagen (8)

![4 · variante b macro f1 en cv y metricas en](imagenes/02_clasificacion_tablas_4_tabla1_variante_b_macro_f1_en_cv_y_metricas_en.png)

### Variante B — macro-F1 en CV y métricas en test

|                   |   cv_f1_macro |   test_accuracy |   test_f1_macro |
|:------------------|--------------:|----------------:|----------------:|
| Árbol de decisión |        0.744  |          0.7464 |          0.7401 |
| Random Forest     |        0.8506 |          0.8565 |          0.8528 |
| KNN               |        0.764  |          0.7679 |          0.7567 |

![4 · classification report por clase arbol de](imagenes/02_clasificacion_tablas_4_tabla2_classification_report_por_clase_arbol_de.png)

### classification_report por clase — Árbol de decisión (B)

|                     |   precision |   recall |   f1-score |
|:--------------------|------------:|---------:|-----------:|
| Insufficient_Weight |       0.78  |    0.868 |      0.821 |
| Normal_Weight       |       0.604 |    0.509 |      0.552 |
| Overweight_Level_I  |       0.638 |    0.673 |      0.655 |
| Overweight_Level_II |       0.66  |    0.603 |      0.631 |
| Obesity_Type_I      |       0.676 |    0.686 |      0.681 |
| Obesity_Type_II     |       0.815 |    0.883 |      0.848 |
| Obesity_Type_III    |       1     |    0.985 |      0.992 |

![4 · classification report por clase random f](imagenes/02_clasificacion_tablas_4_tabla3_classification_report_por_clase_random_f.png)

### classification_report por clase — Random Forest (B)

|                     |   precision |   recall |   f1-score |
|:--------------------|------------:|---------:|-----------:|
| Insufficient_Weight |       0.893 |    0.943 |      0.917 |
| Normal_Weight       |       0.667 |    0.772 |      0.715 |
| Overweight_Level_I  |       0.82  |    0.745 |      0.781 |
| Overweight_Level_II |       0.87  |    0.69  |      0.769 |
| Obesity_Type_I      |       0.849 |    0.886 |      0.867 |
| Obesity_Type_II     |       0.905 |    0.95  |      0.927 |
| Obesity_Type_III    |       1     |    0.985 |      0.992 |

![4 · classification report por clase knn b](imagenes/02_clasificacion_tablas_4_tabla4_classification_report_por_clase_knn_b.png)

### classification_report por clase — KNN (B)

|                     |   precision |   recall |   f1-score |
|:--------------------|------------:|---------:|-----------:|
| Insufficient_Weight |       0.716 |    0.906 |      0.8   |
| Normal_Weight       |       0.604 |    0.509 |      0.552 |
| Overweight_Level_I  |       0.667 |    0.618 |      0.642 |
| Overweight_Level_II |       0.725 |    0.638 |      0.679 |
| Obesity_Type_I      |       0.812 |    0.8   |      0.806 |
| Obesity_Type_II     |       0.803 |    0.883 |      0.841 |
| Obesity_Type_III    |       0.97  |    0.985 |      0.977 |

![4 · metricas por clase arbol de decision b](imagenes/02_clasificacion_tablas_4_tabla5_metricas_por_clase_arbol_de_decision_b.png)

### Métricas por clase — Árbol de decisión (B)

| Clase               |   TP |   FN |   FP |   TN |   Precision |   Recall |    F1 |
|:--------------------|-----:|-----:|-----:|-----:|------------:|---------:|------:|
| Insufficient_Weight |   46 |    7 |   13 |  352 |       0.78  |    0.868 | 0.821 |
| Normal_Weight       |   29 |   28 |   19 |  342 |       0.604 |    0.509 | 0.552 |
| Overweight_Level_I  |   37 |   18 |   21 |  342 |       0.638 |    0.673 | 0.655 |
| Overweight_Level_II |   35 |   23 |   18 |  342 |       0.66  |    0.603 | 0.631 |
| Obesity_Type_I      |   48 |   22 |   23 |  325 |       0.676 |    0.686 | 0.681 |
| Obesity_Type_II     |   53 |    7 |   12 |  346 |       0.815 |    0.883 | 0.848 |
| Obesity_Type_III    |   64 |    1 |    0 |  353 |       1     |    0.985 | 0.992 |

![4 · metricas por clase random forest b](imagenes/02_clasificacion_tablas_4_tabla6_metricas_por_clase_random_forest_b.png)

### Métricas por clase — Random Forest (B)

| Clase               |   TP |   FN |   FP |   TN |   Precision |   Recall |    F1 |
|:--------------------|-----:|-----:|-----:|-----:|------------:|---------:|------:|
| Insufficient_Weight |   50 |    3 |    6 |  359 |       0.893 |    0.943 | 0.917 |
| Normal_Weight       |   44 |   13 |   22 |  339 |       0.667 |    0.772 | 0.715 |
| Overweight_Level_I  |   41 |   14 |    9 |  354 |       0.82  |    0.745 | 0.781 |
| Overweight_Level_II |   40 |   18 |    6 |  354 |       0.87  |    0.69  | 0.769 |
| Obesity_Type_I      |   62 |    8 |   11 |  337 |       0.849 |    0.886 | 0.867 |
| Obesity_Type_II     |   57 |    3 |    6 |  352 |       0.905 |    0.95  | 0.927 |
| Obesity_Type_III    |   64 |    1 |    0 |  353 |       1     |    0.985 | 0.992 |

![4 · metricas por clase knn b](imagenes/02_clasificacion_tablas_4_tabla7_metricas_por_clase_knn_b.png)

### Métricas por clase — KNN (B)

| Clase               |   TP |   FN |   FP |   TN |   Precision |   Recall |    F1 |
|:--------------------|-----:|-----:|-----:|-----:|------------:|---------:|------:|
| Insufficient_Weight |   48 |    5 |   19 |  346 |       0.716 |    0.906 | 0.8   |
| Normal_Weight       |   29 |   28 |   19 |  342 |       0.604 |    0.509 | 0.552 |
| Overweight_Level_I  |   34 |   21 |   17 |  346 |       0.667 |    0.618 | 0.642 |
| Overweight_Level_II |   37 |   21 |   14 |  346 |       0.725 |    0.638 | 0.679 |
| Obesity_Type_I      |   56 |   14 |   13 |  335 |       0.812 |    0.8   | 0.806 |
| Obesity_Type_II     |   53 |    7 |   13 |  345 |       0.803 |    0.883 | 0.841 |
| Obesity_Type_III    |   64 |    1 |    2 |  351 |       0.97  |    0.985 | 0.977 |

![4 · importancias rf b](imagenes/02_clasificacion_tablas_4_tabla8_importancias_rf_b.png)

### Importancias RF (B)

|                                |   importancia |
|:-------------------------------|--------------:|
| Age                            |        0.1601 |
| FCVC                           |        0.1418 |
| NCP                            |        0.0967 |
| FAF                            |        0.0964 |
| TUE                            |        0.0935 |
| CH2O                           |        0.0892 |
| Gender                         |        0.0664 |
| CALC                           |        0.0547 |
| family_history_with_overweight |        0.0516 |
| CAEC                           |        0.0512 |
| FAVC                           |        0.029  |
| MTRANS_Public_Transportation   |        0.025  |
| MTRANS_Automobile              |        0.0219 |
| SCC                            |        0.0113 |
| MTRANS_Walking                 |        0.005  |
| SMOKE                          |        0.0049 |
| MTRANS_Motorbike               |        0.0011 |
| MTRANS_Bike                    |        0.0004 |

### Paso 4.1: Curva del mejor K de KNN (B)

> *Ver código e02_clasificacion.py:396* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L396](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L396)

![4.1 · curva k knn B](imagenes/02_clasificacion_4.1_curva_k_knn_B.png)

### Datos de la figura: 4.1_curva_k_knn_B

#### KNN: macro-F1 (CV) según K

|   K (número de vecinos) |   macro-F1 (CV) |
|------------------------:|----------------:|
|                       1 |        0.763719 |
|                       3 |        0.763958 |
|                       5 |        0.749448 |
|                       7 |        0.739906 |
|                       9 |        0.730159 |
|                      11 |        0.729408 |
|                      13 |        0.71718  |
|                      15 |        0.71002  |
|                      17 |        0.69879  |
|                      19 |        0.692875 |
|                      21 |        0.688015 |
|                      23 |        0.684344 |
|                      25 |        0.679342 |
|                      27 |        0.673851 |
|                      29 |        0.670767 |
|                      31 |        0.664894 |

- **Gráfica `4.1_curva_k_knn_B.png`.** Mejor **K = 3 con `distance`** (macro-F1 CV 0.7640), prácticamente empatado con K = 1 (0.7637). Después la curva baja.
- **Lectura:** sin las dos columnas que definen el IMC, el vecino único ya no basta por sí solo; ponderar 3 vecinos por distancia ayuda un poco.

### Paso 5: Comparación A vs B

> *Ver código e02_clasificacion.py:408* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L408](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L408)

![5 · comparacion A vs B](imagenes/02_clasificacion_5_comparacion_A_vs_B.png)

### Datos de la figura: 5_comparacion_A_vs_B

#### Macro-F1 en test por modelo y variante

| modelo            |   serie_1 |
|:------------------|----------:|
| Árbol de decisión |    0.9362 |
| Random Forest     |    0.9558 |
| KNN               |    0.8075 |

#### Macro-F1 en test por modelo y variante

| modelo            |   serie_2 |
|:------------------|----------:|
| Árbol de decisión |    0.7401 |
| Random Forest     |    0.8528 |
| KNN               |    0.7567 |

- **Gráfica `5_comparacion_A_vs_B.png`:** barras agrupadas: macro-F1 en test por modelo (eje x), un color por variante (hue). Eje y de 0 a 1.
- **Lectura:** en los tres modelos A > B; el orden RF > árbol en ambas; KNN queda último en A y segundo en B.
- **Código:** `pd.concat({"A (todas)": tabla_A, "B (...)": tabla_B})` y `sns.barplot(data=..., x="modelo", y="test_f1_macro", hue="variante")`.

#### Tablas de este paso como imagen (1)

![5 · comparacion a vs b](imagenes/02_clasificacion_tablas_5_tabla1_comparacion_a_vs_b.png)

### Comparación A vs B

|                                                |   cv_f1_macro |   test_accuracy |   test_f1_macro |
|:-----------------------------------------------|--------------:|----------------:|----------------:|
| ('A (todas)', 'Árbol de decisión')             |        0.923  |          0.9378 |          0.9362 |
| ('A (todas)', 'Random Forest')                 |        0.9437 |          0.9569 |          0.9558 |
| ('A (todas)', 'KNN')                           |        0.8152 |          0.8206 |          0.8075 |
| ('B (sin Weight/Height)', 'Árbol de decisión') |        0.744  |          0.7464 |          0.7401 |
| ('B (sin Weight/Height)', 'Random Forest')     |        0.8506 |          0.8565 |          0.8528 |
| ('B (sin Weight/Height)', 'KNN')               |        0.764  |          0.7679 |          0.7567 |

### Paso 6: Conclusiones de clasificación (texto)

- Tabla resumen de A y B (la de los pasos 3 y 4).
- **Nodo raíz** de todos los árboles de A: `Weight ≤ 99.536` — es el corte que más reduce el Gini: separa de un golpe las clases de obesidad alta del resto.
- **Top-5 de importancias:** A (Weight, Age, Height, FCVC, Gender) vs B (Age, FCVC, NCP, FAF, TUE).
- Recomendación: presentar A como modelo principal (lo pide el enunciado) y la fuga + B como análisis complementario.

### Paso 7: Robustez ante ruido (jittering) en la variante B

> *Ver código e02_clasificacion.py:455* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L455](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L455)

![7 · barrido ruido](imagenes/02_clasificacion_7_barrido_ruido.png)

### Datos de la figura: 7_barrido_ruido

#### Degradación del Accuracy vs Nivel de Ruido en Hábitos (Variante B)

|   Nivel de perturbación gaussiana (% de desviación estándar) |   Árbol de decisión |   Random Forest |   KNN (k=1) |   KNN (k=3, distance) |
|-------------------------------------------------------------:|--------------------:|----------------:|------------:|----------------------:|
|                                                            0 |              0.7464 |          0.8565 |      0.7703 |                0.7679 |
|                                                            5 |              0.6938 |          0.8445 |      0.7656 |                0.7727 |
|                                                           10 |              0.6962 |          0.8254 |      0.7632 |                0.7775 |
|                                                           15 |              0.6579 |          0.823  |      0.744  |                0.7751 |
|                                                           20 |              0.6483 |          0.7895 |      0.7368 |                0.7679 |

- **Motivación:** romper la artificialidad de SMOTE. Se suma ruido `N(0, (α·σ)²)` a Age, FCVC, NCP, CH2O, FAF, TUE (train y test), se recorta a rangos válidos (`RANGOS_HABITOS`: Age 14–70, FCVC 1–3, NCP 1–4, CH2O 1–3, FAF 0–3, TUE 0–2) y se reentrenan 4 modelos fijos (sin GridSearch).
- **Tabla con 5 % de ruido:** Modelo Acc limpio Acc ruido 5 % Diferencia Árbol de decisión 0.7464 0.6938 −0.053 Random Forest 0.8565 0.8445 −0.012 KNN (k=1) 0.7703 0.7656 −0.005 KNN (k=3, distance) 0.7679 0.7727 +0.005
- **Barrido 0–20 % (accuracy en test):** Ruido Árbol Random Forest KNN k=1 KNN k=3 distance 0 % 0.746 0.857 0.770 0.768 5 % 0.694 0.845 0.766 0.773 10 % 0.696 0.825 0.763 0.778 15 % 0.658 0.823 0.744 0.775 20 % 0.648 0.790 0.737 0.768
- **Gráfica `7_barrido_ruido.png`:** una línea por modelo (rojo árbol, verde RF, azul KNN k=1, morado KNN k=3); eje x = nivel de ruido (% de la desviación), eje y = accuracy en test (0.58–0.90). Leyenda fuera, a la derecha.
- **Cómo se lee y qué concluir:** **RF** es el más robusto: siempre arriba y con pendiente suave (−6.7 puntos al 20 %): el promedio de muchos árboles cancela un ruido de media cero. **El árbol** es el más frágil (−9.8 puntos; ya con 5 % pierde 5): los cortes rígidos `x ≤ umbral` cambian de lado con pequeñas perturbaciones. **KNN k=1 vs k=3:** sin ruido k=1 gana por poco (0.770 vs 0.768) gracias a los cuasi-duplicados de SMOTE; **con ruido k=3 supera a k=1 en todos los niveles**. La ventaja de k=1 era un **artefacto de SMOTE**; con datos más realistas promediar la vecindad generaliza mejor.
- **Lo que este paso NO resuelve:** el ruido perturba los datos pero no quita SMOTE: el 77 % sintético sigue en train y en test, con sus cuasi-duplicados y las clases balanceadas a mano. Eso lo hace el paso 8.

#### Tablas de este paso como imagen (2)

![7 · robustez ante 5 de ruido en habitos vari](imagenes/02_clasificacion_tablas_7_tabla1_robustez_ante_5_de_ruido_en_habitos_vari.png)

### Robustez ante 5% de ruido en hábitos (variante B)

| Modelo              |   Test Acc (Limpio) |   Test Acc (Ruido 5%) |   Dif Acc |   Test F1 (Limpio) |   Test F1 (Ruido 5%) |   Dif F1 |
|:--------------------|--------------------:|----------------------:|----------:|-------------------:|---------------------:|---------:|
| Árbol de decisión   |              0.7464 |                0.6938 |   -0.0526 |             0.7401 |               0.6804 |  -0.0596 |
| Random Forest       |              0.8565 |                0.8445 |   -0.012  |             0.8528 |               0.8382 |  -0.0146 |
| KNN (k=1)           |              0.7703 |                0.7656 |   -0.0048 |             0.7586 |               0.7527 |  -0.006  |
| KNN (k=3, distance) |              0.7679 |                0.7727 |    0.0048 |             0.7567 |               0.7613 |   0.0046 |

![7 · accuracy en test por nivel de ruido](imagenes/02_clasificacion_tablas_7_tabla2_accuracy_en_test_por_nivel_de_ruido.png)

### Accuracy en test por nivel de ruido

|   Nivel de ruido (%) |   Árbol de decisión |   Random Forest |   KNN (k=1) |   KNN (k=3, distance) |
|---------------------:|--------------------:|----------------:|------------:|----------------------:|
|                    0 |              0.7464 |          0.8565 |      0.7703 |                0.7679 |
|                    5 |              0.6938 |          0.8445 |      0.7656 |                0.7727 |
|                   10 |              0.6962 |          0.8254 |      0.7632 |                0.7775 |
|                   15 |              0.6579 |          0.823  |      0.744  |                0.7751 |
|                   20 |              0.6483 |          0.7895 |      0.7368 |                0.7679 |

### Paso 8: Sólo filas reales (sin SMOTE) y sin Weight/Height

> *Ver código e02_clasificacion.py:597* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L597](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e02_clasificacion.py#L597)

![8 · filas reales sin WH](imagenes/02_clasificacion_8_filas_reales_sin_WH.png)

### Datos de la figura: 8_filas_reales_sin_WH

#### Filas reales, sin Weight/Height · 7 clases

| categoria               |   balanced accuracy |
|:------------------------|--------------------:|
| Línea base              |            0.142857 |
| Árbol (depth 4, leaf 5) |            0.171317 |
| Random Forest           |            0.181934 |
| RF con pesos por clase  |            0.199357 |
| KNN (k=15, distance)    |            0.169264 |
| KNN (k=1)               |            0.194429 |

#### Filas reales, sin Weight/Height · 7 clases

| categoria               |   macro-F1 (± desv. entre folds) |
|:------------------------|---------------------------------:|
| Línea base              |                         0.104232 |
| Árbol (depth 4, leaf 5) |                         0.156424 |
| Random Forest           |                         0.173666 |
| RF con pesos por clase  |                         0.195007 |
| KNN (k=15, distance)    |                         0.152017 |
| KNN (k=1)               |                         0.194203 |

#### Filas reales, sin Weight/Height · 3 clases agrupadas

| categoria               |   balanced accuracy |
|:------------------------|--------------------:|
| Línea base              |            0.333333 |
| Árbol (depth 4, leaf 5) |            0.383993 |
| Random Forest           |            0.393853 |
| RF con pesos por clase  |            0.420001 |
| KNN (k=15, distance)    |            0.36868  |
| KNN (k=1)               |            0.396316 |

#### Filas reales, sin Weight/Height · 3 clases agrupadas

| categoria               |   macro-F1 (± desv. entre folds) |
|:------------------------|---------------------------------:|
| Línea base              |                         0.26155  |
| Árbol (depth 4, leaf 5) |                         0.371853 |
| Random Forest           |                         0.382403 |
| RF con pesos por clase  |                         0.421004 |
| KNN (k=15, distance)    |                         0.334994 |
| KNN (k=1)               |                         0.395685 |

- **Motivación (pedido del profesor):** un ejemplo que no pareciera tan artificial ni tan fácil de entrenar. Se elimina SMOTE por completo: sólo las respuestas originales de la encuesta y, además, sin Weight ni Height (sin la fuga del IMC).
- **Cómo se identifican las filas reales:** `filas_reales(X)` = todas las respuestas de encuesta (Age, FCVC, NCP, CH2O, FAF, TUE) son enteras. Resultan **491 de 2087 (23.5 %)**. Distribución: Normal_Weight 282, Overweight_II 58, Overweight_I 55, Obesity_I 47, Insufficient 35, Obesity_II 11, **Obesity_III 3**.
- **Primera evidencia (modelos del proyecto, test separado en 103 filas reales y 315 sintéticas):** Modelo Acc filas reales Acc filas sintéticas A · Árbol 0.864 0.962 A · Random Forest 0.835 0.997 A · KNN 0.456 0.940 B · Árbol 0.417 0.854 B · Random Forest **0.583** **0.946** B · KNN 0.447 0.873
- **Protocolo:** variante B sobre las 491 filas reales, CV estratificada 3 folds × 10 repeticiones, con **línea base** (siempre Normal_Weight) y tres métricas: accuracy, balanced accuracy y macro-F1 (ver 6.10). Hiperparámetros fijos (sin GridSearch: con tan pocas filas haría falta CV anidada).
- **7 clases:** Modelo Accuracy Balanced acc. Macro-F1 **Línea base** **0.574** 0.143 0.104 Árbol (depth 4, leaf 5) 0.555 0.171 0.156 Random Forest 0.576 0.182 0.174 RF con pesos por clase 0.297 0.196 0.171 KNN (k=15, distance) 0.590 0.169 0.152 KNN (k=1) 0.463 0.194 **0.194** **Lectura:** ningún modelo supera de forma útil a la línea base. Los que maximizan accuracy lo hacen prediciendo casi siempre Normal_Weight; al ponderar por clase la accuracy se desploma pero el acierto se reparte mejor. Con datos reales, los hábitos solos casi no distinguen 7 niveles.
- **3 clases agrupadas** (Bajo/Normal 317, Sobrepeso 113, Obesidad 61): Modelo Accuracy Balanced acc. Macro-F1 Línea base 0.646 0.333 0.262 Random Forest 0.648 0.394 0.382 **RF con pesos por clase** 0.549 **0.435** **0.424** KNN (k=1) 0.575 0.396 0.396 Aquí sí hay una **señal modesta pero real**. Importancias (RF con pesos, 3 clases): **Age 0.221**, family_history 0.093, FAF 0.084, CALC 0.074, CH2O 0.073.
- **Referencia con Weight y Height (A, 7 clases, filas reales):** el mejor llega a accuracy **0.705** (árbol), no a 0.957. No es que las etiquetas estén mal: en las filas reales la clase coincide con el IMC (umbrales OMS) en el **96.7 %**. Con 491 filas un árbol no aproxima bien la frontera Weight/Height² con cortes paralelos a los ejes; en el dataset completo SMOTE le daba miles de puntos para hacerlo.
- **Gráfica `8_filas_reales_sin_WH.png`:** dos paneles (7 clases | 3 clases). Por modelo, barra azul = balanced accuracy y naranja = macro-F1 (con barra de error = desviación entre folds); línea gris discontinua = azar en balanced accuracy (1/7 y 1/3). Se lee comparando cada modelo con la "Línea base": en 7 clases todas las barras quedan apenas por encima; en 3 clases el RF con pesos se separa claramente.
- **Conclusión del paso:** el rendimiento alto del proyecto se apoya en dos artificios, la fuga del IMC (A) y SMOTE (B). Sin ninguno de los dos, predecir 7 niveles no supera a la línea base; en 3 niveles los hábitos aportan algo, con la edad como factor principal. **Limitaciones:** identificación heurística, pocas filas (3 de Obesity_III), hiperparámetros sin ajustar.

#### Tablas de este paso como imagen (6)

![8 · modelos del proyecto accuracy en las fil](imagenes/02_clasificacion_tablas_8_tabla1_modelos_del_proyecto_accuracy_en_las_fil.png)

### Modelos del proyecto: accuracy en las filas reales vs sintéticas del test

|                       |   acc filas reales |   acc filas sintéticas |
|:----------------------|-------------------:|-----------------------:|
| A · Árbol de decisión |             0.8641 |                 0.9619 |
| A · Random Forest     |             0.835  |                 0.9968 |
| A · KNN               |             0.4563 |                 0.9397 |
| B · Árbol de decisión |             0.4175 |                 0.854  |
| B · Random Forest     |             0.5825 |                 0.946  |
| B · KNN               |             0.4466 |                 0.873  |

![8 · filas reales por clase distribucion de l](imagenes/02_clasificacion_tablas_8_tabla2_filas_reales_por_clase_distribucion_de_l.png)

### Filas reales por clase (distribución de la encuesta original)

| NObeyesdad          |   filas |
|:--------------------|--------:|
| Insufficient_Weight |      35 |
| Normal_Weight       |     282 |
| Overweight_Level_I  |      55 |
| Overweight_Level_II |      58 |
| Obesity_Type_I      |      47 |
| Obesity_Type_II     |      11 |
| Obesity_Type_III    |       3 |

![8 · solo filas reales variante b 7 clases cv](imagenes/02_clasificacion_tablas_8_tabla3_solo_filas_reales_variante_b_7_clases_cv.png)

### Sólo filas reales · variante B · 7 clases (CV 3×10)

|                                |   accuracy |   balanced_accuracy |   f1_macro |   f1_macro_std |
|:-------------------------------|-----------:|--------------------:|-----------:|---------------:|
| Línea base (clase mayoritaria) |     0.5743 |              0.1429 |     0.1042 |         0.0002 |
| Árbol (depth 4, leaf 5)        |     0.5546 |              0.1713 |     0.1564 |         0.0257 |
| Random Forest                  |     0.5758 |              0.1819 |     0.1737 |         0.0206 |
| RF con pesos por clase         |     0.4735 |              0.1994 |     0.195  |         0.0211 |
| KNN (k=15, distance)           |     0.5896 |              0.1693 |     0.152  |         0.0244 |
| KNN (k=1)                      |     0.4633 |              0.1944 |     0.1942 |         0.0253 |

![8 · solo filas reales variante b 3 clases ag](imagenes/02_clasificacion_tablas_8_tabla4_solo_filas_reales_variante_b_3_clases_ag.png)

### Sólo filas reales · variante B · 3 clases agrupadas (CV 3×10)

|                                |   accuracy |   balanced_accuracy |   f1_macro |   f1_macro_std |
|:-------------------------------|-----------:|--------------------:|-----------:|---------------:|
| Línea base (clase mayoritaria) |     0.6456 |              0.3333 |     0.2616 |         0.0003 |
| Árbol (depth 4, leaf 5)        |     0.631  |              0.384  |     0.3719 |         0.0374 |
| Random Forest                  |     0.6483 |              0.3939 |     0.3824 |         0.0345 |
| RF con pesos por clase         |     0.602  |              0.42   |     0.421  |         0.0318 |
| KNN (k=15, distance)           |     0.6578 |              0.3687 |     0.335  |         0.0305 |
| KNN (k=1)                      |     0.575  |              0.3963 |     0.3957 |         0.028  |

![8 · importancia de atributos rf con pesos po](imagenes/02_clasificacion_tablas_8_tabla5_importancia_de_atributos_rf_con_pesos_po.png)

### Importancia de atributos (RF con pesos por clase, 3 clases, filas reales)

|                                |   importancia |
|:-------------------------------|--------------:|
| Age                            |        0.2215 |
| family_history_with_overweight |        0.0888 |
| FAF                            |        0.0879 |
| CALC                           |        0.0727 |
| CH2O                           |        0.0715 |
| CAEC                           |        0.0705 |
| TUE                            |        0.0665 |
| FCVC                           |        0.0537 |

![8 · referencia solo filas reales variante a](imagenes/02_clasificacion_tablas_8_tabla6_referencia_solo_filas_reales_variante_a.png)

### Referencia: sólo filas reales · variante A (con Weight/Height) · 7 clases

|                                |   accuracy |   balanced_accuracy |   f1_macro |   f1_macro_std |
|:-------------------------------|-----------:|--------------------:|-----------:|---------------:|
| Línea base (clase mayoritaria) |     0.5743 |              0.1429 |     0.1042 |         0.0002 |
| Árbol (depth 4, leaf 5)        |     0.7051 |              0.4281 |     0.4241 |         0.045  |
| Random Forest                  |     0.7041 |              0.3386 |     0.3409 |         0.0211 |
| RF con pesos por clase         |     0.6914 |              0.4023 |     0.4119 |         0.0359 |
| KNN (k=15, distance)           |     0.599  |              0.1803 |     0.1696 |         0.0276 |
| KNN (k=1)                      |     0.5173 |              0.2417 |     0.2478 |         0.0366 |

> **ETAPAS**

## Etapa 03 — Regresión de Weight, paso a paso

**Archivo:** `etapas/e03_regresion.py`. **Figuras:** `figures/03_regresion/`.

**Protocolo:** `reg.csv` (19 features, **sin NObeyesdad**; Height sí se mantiene porque es una
medida física independiente, no se deriva del peso) → split 80/20 **estratificado por deciles
de Weight** → GridSearchCV con `KFold(5)` y `scoring="neg_root_mean_squared_error"` → test.
Además se compara con un **baseline** que predice la media.

---

### Paso 1: Datos y split

> *Ver código e03_regresion.py:151* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L151](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L151)

- train (1669, 19), test (418, 19), `n_features = 19`. Se comprueba que NObeyesdad **no** está.
- `strata = pd.qcut(y, q=10, labels=False)` divide el peso en 10 grupos de igual tamaño (deciles) y se pasa como `stratify=strata`: train y test tienen la misma distribución de pesos (medianas 83.1 y 83.2 kg). En regresión no se puede estratificar por el valor continuo directamente, por eso se discretiza.
- Los folds de la CV usan `KFold` (sin estratificar).

#### Tablas de este paso como imagen (1)

![1 · weight en train](imagenes/03_regresion_tablas_1_tabla1_weight_en_train.png)

### Weight en train

|       |   Weight |
|:------|---------:|
| count |  1669    |
| mean  |    86.92 |
| std   |    26.26 |
| min   |    39.37 |
| 25%   |    66    |
| 50%   |    83.1  |
| 75%   |   108    |
| max   |   173    |

### Paso 2: Modelos, grillas y tabla de resultados

> *Ver código e03_regresion.py:160* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L160](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L160)

| Modelo | Grilla | Mejores hiperparámetros |
| :--- | :--- | :--- |
| Árbol de regresión | `max_depth` ∈ {4, 6, 8, 12, None}, `min_samples_leaf` ∈ {1, 3, 5, 10} | depth 12, leaf 10 |
| Random Forest | `n_estimators` ∈ {200, 400}, `max_depth` ∈ {None, 12}, `min_samples_leaf` ∈ {1, 2} | 400 árboles, depth None, leaf 1 |
| KNN (Pipeline con escalado) | K impares 1…31 × {uniform, distance} | K = 3, distance |

**Resultados (test, 418 filas):**

| Modelo | RMSE CV | MAE | RMSE | R² | R² ajustado |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (media) | — | 21.71 | 25.91 | −0.0001 | −0.048 |
| Árbol de regresión | 12.47 | 6.58 | 10.47 | 0.837 | 0.829 |
| **Random Forest** | **9.12** | **4.84** | **8.25** | **0.899** | **0.894** |
| KNN (K=3, distance) | 10.37 | 5.05 | 9.91 | 0.854 | 0.847 |

- **Cómo interpretarlo:** RF gana en todo. Con un rango de pesos de 39 a 173 kg, un RMSE de 8.25 kg y un MAE de 4.84 kg (en promedio se equivoca unos 5 kg) es un buen resultado **sin usar la clase**. R² 0.899 = explica ~90 % de la variación del peso.
- **El baseline** tiene R² ≈ 0 por construcción (predecir la media); sale −0.0001 porque la media del train no es exactamente la del test. Su R² ajustado es negativo (−0.048): un modelo sin capacidad real queda penalizado.
- **RMSE > MAE siempre**, y la diferencia indica que hay algunos errores grandes (el RMSE los penaliza al cuadrado).

#### Tablas de este paso como imagen (1)

![2 · resultados rmse de cv y metricas en test](imagenes/03_regresion_tablas_2_tabla1_resultados_rmse_de_cv_y_metricas_en_test.png)

### Resultados: RMSE de CV y métricas en test (MAE, RMSE, R², R²_adj)

|                    | cv_RMSE   |   test_MAE |   test_RMSE |   test_R2 |   test_R2_adj |
|:-------------------|:----------|-----------:|------------:|----------:|--------------:|
| Baseline (media)   | —         |    21.7064 |     25.9126 |   -0.0001 |       -0.0479 |
| Árbol de regresión | 12.4728   |     6.581  |     10.4655 |    0.8369 |        0.8291 |
| Random Forest      | 9.1178    |     4.8292 |      8.2356 |    0.899  |        0.8942 |
| KNN                | 10.3709   |     5.0484 |      9.9085 |    0.8538 |        0.8468 |

### Paso 2.1: Curva del mejor K de KNN (regresión)

> *Ver código e03_regresion.py:170* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L170](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L170)

![2.1 · curva k knn](imagenes/03_regresion_2.1_curva_k_knn.png)

### Datos de la figura: 2.1_curva_k_knn

#### KNN: RMSE (CV) según K

|   K (número de vecinos) |   RMSE (CV) |
|------------------------:|------------:|
|                       1 |     12.206  |
|                       3 |     10.3709 |
|                       5 |     10.4136 |
|                       7 |     10.7047 |
|                       9 |     10.9677 |
|                      11 |     11.1643 |
|                      13 |     11.2625 |
|                      15 |     11.4152 |
|                      17 |     11.606  |
|                      19 |     11.8039 |
|                      21 |     11.9485 |
|                      23 |     12.111  |
|                      25 |     12.2518 |
|                      27 |     12.4444 |
|                      29 |     12.5867 |
|                      31 |     12.7611 |

- **Gráfica `2.1_curva_k_knn.png`:** RMSE de CV (kg) vs K; aquí **más bajo es mejor** (`negate=True` convierte el score negativo de sklearn en RMSE positivo).
- **Resultado:** K=1 → 12.21 kg; **K=3 distance → 10.37 kg (mejor)**; K=5 → 10.41; luego sube de forma monótona (K=11 → 11.16).
- **Lectura:** a diferencia de clasificación A (K=1), aquí K=3 gana: en una variable continua promediar 3 vecinos **suaviza** la predicción y reduce la varianza del error. Con K grande entran personas de complexión distinta y el error crece.

### Paso 2.2: ¿Cuántos árboles? (regresión)

> *Ver código e03_regresion.py:207* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L207](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L207)

![2.2 · n estimators rf](imagenes/03_regresion_2.2_n_estimators_rf.png)

### Datos de la figura: 2.2_n_estimators_rf

#### Random Forest: RMSE según el número de árboles

|   n_estimators (número de árboles, escala log) |   validación cruzada (train) |    test |
|-----------------------------------------------:|-----------------------------:|--------:|
|                                             10 |                      9.72323 | 9.17961 |
|                                             25 |                      9.32383 | 8.84025 |
|                                             50 |                      9.16052 | 8.61809 |
|                                            100 |                      9.15368 | 8.36156 |
|                                            200 |                      9.14645 | 8.32997 |
|                                            400 |                      9.1178  | 8.23562 |
|                                            800 |                      9.11951 | 8.25839 |

#### Coste: tiempo de entrenamiento

|   n_estimators (número de árboles, escala log) |   segundos en entrenar |
|-----------------------------------------------:|-----------------------:|
|                                             10 |              0.0428718 |
|                                             25 |              0.0695132 |
|                                             50 |              0.105366  |
|                                            100 |              0.190356  |
|                                            200 |              0.322608  |
|                                            400 |              0.636285  |
|                                            800 |              1.40258   |

| n_estimators | RMSE CV (kg) | desv. CV | RMSE test | R² test | tiempo (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 9.772 | 0.672 | 9.074 | 0.877 | 0.05 |
| 25 | 9.303 | 0.626 | 8.800 | 0.885 | 0.07 |
| 50 | 9.154 | 0.622 | 8.563 | 0.891 | 0.13 |
| 100 | 9.143 | 0.629 | 8.356 | 0.896 | 0.24 |
| 200 | 9.150 | 0.672 | 8.349 | 0.896 | 0.42 |
| 400 | 9.116 | 0.643 | 8.247 | 0.899 | 0.77 |
| 800 | 9.117 | 0.654 | 8.276 | 0.898 | 1.93 |

- **Gráfica `2.2_n_estimators_rf.png`:** igual que la 3.7 de clasificación pero con RMSE (la curva **baja** y se aplana; más bajo es mejor). **Punto suficiente: 50 árboles** (ya consigue el 90 % de la mejora). De 50 a 800 el tiempo se multiplica por ~15 a cambio de 0.04 kg de RMSE.
- **Lectura:** toda la mejora de recorrer la escala son **fracciones de kilo** frente a un error de ~8 kg: no cambia ninguna conclusión. Contraste con los otros hiperparámetros: la profundidad del árbol y K sí tienen un óptimo (pasarse sobreajusta); `n_estimators` no.

#### Tablas de este paso como imagen (1)

![2.2 · random forest regresion error y coste se](imagenes/03_regresion_tablas_2.2_tabla1_random_forest_regresion_error_y_coste_se.png)

### Random Forest (regresión): error y coste según el número de árboles

|   n_estimators |   cv_RMSE |   cv_std |   test_RMSE |   test_R2 |   tiempo_fit_s |
|---------------:|----------:|---------:|------------:|----------:|---------------:|
|             10 |    9.7232 |   0.6694 |      9.1796 |    0.8745 |         0.0429 |
|             25 |    9.3238 |   0.6403 |      8.8403 |    0.8836 |         0.0695 |
|             50 |    9.1605 |   0.6809 |      8.6181 |    0.8894 |         0.1054 |
|            100 |    9.1537 |   0.6675 |      8.3616 |    0.8959 |         0.1904 |
|            200 |    9.1465 |   0.6839 |      8.33   |    0.8966 |         0.3226 |
|            400 |    9.1178 |   0.6617 |      8.2356 |    0.899  |         0.6363 |
|            800 |    9.1195 |   0.661  |      8.2584 |    0.8984 |         1.4026 |

### Paso 3: Tres árboles de regresión (prepoda)

> *Ver código e03_regresion.py:230* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L230](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L230)

![3 · arbol completo](imagenes/03_regresion_3_arbol_completo.png)

### Datos de la figura: 3_arbol_completo

#### Árbol Completo — profundidad 27, 1574 hojas, raíz: family_history_with_overweight ≤ 0.5 (truncado a profundidad 3 para visualización)

|    |   nodo | condicion                             |   squared_error |   samples |   value | info   |
|---:|-------:|:--------------------------------------|----------------:|----------:|--------:|:-------|
|  0 |      1 | family_history_with_overweight <= 0.5 |         689.185 |      1669 |  86.919 | nan    |
|  1 |      2 | Gender <= 0.5                         |         195.079 |       289 |  58.482 | nan    |
|  2 |      3 | Age <= 23.607                         |          92.008 |       179 |  51.789 | nan    |
|  3 |      4 | NCP <= 2.661                          |          54.539 |       157 |  49.824 | nan    |
|  4 |      5 | nan                                   |         nan     |       nan | nan     | (...)  |
|  5 |      6 | nan                                   |         nan     |       nan | nan     | (...)  |
|  6 |      7 | Height <= 1.625                       |         135.321 |        22 |  65.808 | nan    |
|  7 |      8 | nan                                   |         nan     |       nan | nan     | (...)  |
|  8 |      9 | nan                                   |         nan     |       nan | nan     | (...)  |
|  9 |     10 | Age <= 23.5                           |         171.263 |       110 |  69.374 | nan    |
| 10 |     11 | Age <= 18.5                           |         138.253 |        80 |  65.135 | nan    |
| 11 |     12 | nan                                   |         nan     |       nan | nan     | (...)  |
| 12 |     13 | nan                                   |         nan     |       nan | nan     | (...)  |
| 13 |     14 | Height <= 1.72                        |          83.61  |        30 |  80.678 | nan    |
| 14 |     15 | nan                                   |         nan     |       nan | nan     | (...)  |
| 15 |     16 | nan                                   |         nan     |       nan | nan     | (...)  |
| 16 |     17 | Height <= 1.723                       |         587.85  |      1380 |  92.874 | nan    |
| 17 |     18 | FCVC <= 2.99                          |         395.353 |       764 |  84.007 | nan    |
| 18 |     19 | CAEC <= 1.5                           |         205.018 |       507 |  77.628 | nan    |
| 19 |     20 | nan                                   |         nan     |       nan | nan     | (...)  |
| 20 |     21 | nan                                   |         nan     |       nan | nan     | (...)  |
| 21 |     22 | CAEC <= 1.5                           |         532.152 |       257 |  96.593 | nan    |
| 22 |     23 | nan                                   |         nan     |       nan | nan     | (...)  |
| 23 |     24 | nan                                   |         nan     |       nan | nan     | (...)  |
| 24 |     25 | FAF <= 2.0                            |         608.149 |       616 | 103.871 | nan    |
| 25 |     26 | Gender <= 0.5                         |         485.241 |       486 | 110.195 | nan    |
| 26 |     27 | nan                                   |         nan     |       nan | nan     | (...)  |
| 27 |     28 | nan                                   |         nan     |       nan | nan     | (...)  |
| 28 |     29 | NCP <= 3.024                          |         359.186 |       130 |  80.229 | nan    |
| 29 |     30 | nan                                   |         nan     |       nan | nan     | (...)  |
| 30 |     31 | nan                                   |         nan     |       nan | nan     | (...)  |

![3 · arbol podado](imagenes/03_regresion_3_arbol_podado.png)

### Datos de la figura: 3_arbol_podado

#### Árbol Podado — profundidad 8, 116 hojas, raíz: family_history_with_overweight ≤ 0.5 (truncado a profundidad 3 para visualización)

|    |   nodo | condicion                             |   squared_error |   samples |   value | info   |
|---:|-------:|:--------------------------------------|----------------:|----------:|--------:|:-------|
|  0 |      1 | family_history_with_overweight <= 0.5 |         689.185 |      1669 |  86.919 | nan    |
|  1 |      2 | Gender <= 0.5                         |         195.079 |       289 |  58.482 | nan    |
|  2 |      3 | Age <= 23.607                         |          92.008 |       179 |  51.789 | nan    |
|  3 |      4 | NCP <= 2.661                          |          54.539 |       157 |  49.824 | nan    |
|  4 |      5 | nan                                   |         nan     |       nan | nan     | (...)  |
|  5 |      6 | nan                                   |         nan     |       nan | nan     | (...)  |
|  6 |      7 | Height <= 1.625                       |         135.321 |        22 |  65.808 | nan    |
|  7 |      8 | nan                                   |         nan     |       nan | nan     | (...)  |
|  8 |      9 | nan                                   |         nan     |       nan | nan     | (...)  |
|  9 |     10 | Age <= 23.5                           |         171.263 |       110 |  69.374 | nan    |
| 10 |     11 | Age <= 18.5                           |         138.253 |        80 |  65.135 | nan    |
| 11 |     12 | nan                                   |         nan     |       nan | nan     | (...)  |
| 12 |     13 | nan                                   |         nan     |       nan | nan     | (...)  |
| 13 |     14 | Height <= 1.72                        |          83.61  |        30 |  80.678 | nan    |
| 14 |     15 | nan                                   |         nan     |       nan | nan     | (...)  |
| 15 |     16 | nan                                   |         nan     |       nan | nan     | (...)  |
| 16 |     17 | Height <= 1.723                       |         587.85  |      1380 |  92.874 | nan    |
| 17 |     18 | FCVC <= 2.99                          |         395.353 |       764 |  84.007 | nan    |
| 18 |     19 | CAEC <= 1.5                           |         205.018 |       507 |  77.628 | nan    |
| 19 |     20 | nan                                   |         nan     |       nan | nan     | (...)  |
| 20 |     21 | nan                                   |         nan     |       nan | nan     | (...)  |
| 21 |     22 | CAEC <= 1.5                           |         532.152 |       257 |  96.593 | nan    |
| 22 |     23 | nan                                   |         nan     |       nan | nan     | (...)  |
| 23 |     24 | nan                                   |         nan     |       nan | nan     | (...)  |
| 24 |     25 | FAF <= 2.0                            |         608.149 |       616 | 103.871 | nan    |
| 25 |     26 | Gender <= 0.5                         |         485.241 |       486 | 110.195 | nan    |
| 26 |     27 | nan                                   |         nan     |       nan | nan     | (...)  |
| 27 |     28 | nan                                   |         nan     |       nan | nan     | (...)  |
| 28 |     29 | NCP <= 3.024                          |         359.186 |       130 |  80.229 | nan    |
| 29 |     30 | nan                                   |         nan     |       nan | nan     | (...)  |
| 30 |     31 | nan                                   |         nan     |       nan | nan     | (...)  |

![3 · arbol muy podado](imagenes/03_regresion_3_arbol_muy_podado.png)

### Datos de la figura: 3_arbol_muy_podado

#### Árbol Muy podado — profundidad 3, 8 hojas, raíz: family_history_with_overweight ≤ 0.5

|    |   nodo | condicion                             |   squared_error |   samples |   value |
|---:|-------:|:--------------------------------------|----------------:|----------:|--------:|
|  0 |      1 | family_history_with_overweight <= 0.5 |         689.185 |      1669 |  86.919 |
|  1 |      2 | Gender <= 0.5                         |         195.079 |       289 |  58.482 |
|  2 |      3 | Age <= 23.607                         |          92.008 |       179 |  51.789 |
|  3 |      4 | nan                                   |          54.539 |       157 |  49.824 |
|  4 |      5 | nan                                   |         135.321 |        22 |  65.808 |
|  5 |      6 | Age <= 23.5                           |         171.263 |       110 |  69.374 |
|  6 |      7 | nan                                   |         138.253 |        80 |  65.135 |
|  7 |      8 | nan                                   |          83.61  |        30 |  80.678 |
|  8 |      9 | Height <= 1.723                       |         587.85  |      1380 |  92.874 |
|  9 |     10 | FCVC <= 2.99                          |         395.353 |       764 |  84.007 |
| 10 |     11 | nan                                   |         205.018 |       507 |  77.628 |
| 11 |     12 | nan                                   |         532.152 |       257 |  96.593 |
| 12 |     13 | FAF <= 2.0                            |         608.149 |       616 | 103.871 |
| 13 |     14 | nan                                   |         485.241 |       486 | 110.195 |
| 14 |     15 | nan                                   |         359.186 |       130 |  80.229 |

![3 · pred vs real tres arboles](imagenes/03_regresion_3_pred_vs_real_tres_arboles.png)

### Datos de la figura: 3_pred_vs_real_tres_arboles

#### Árbol Completo
RMSE=12.47 kg  R²=0.768  R²adj=0.757

|   Real (kg) |   Predicho (kg) |
|------------:|----------------:|
|      37     |          37     |
|     162.935 |         162.935 |

#### Árbol Podado
RMSE=11.19 kg  R²=0.814  R²adj=0.805

|   Real (kg) |   Predicho (kg) |
|------------:|----------------:|
|      37     |          37     |
|     162.935 |         162.935 |

#### Árbol Muy podado
RMSE=18.02 kg  R²=0.516  R²adj=0.493

|   Real (kg) |   Predicho (kg) |
|------------:|----------------:|
|      37     |          37     |
|     162.935 |         162.935 |

| Árbol | max_depth | min_samples_leaf | Profundidad | Hojas | Raíz | MAE | RMSE | R² | R² adj |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Completo | None | 1 | 27 | 1576 | family_history ≤ 0.5 | 6.41 | 12.20 | 0.778 | 0.768 |
| **Podado** | 8 | 5 | 8 | 116 | family_history ≤ 0.5 | 7.09 | **11.19** | **0.814** | 0.805 |
| Muy podado | 3 | 20 | 3 | 8 | family_history ≤ 0.5 | 13.88 | 18.02 | 0.516 | 0.493 |

- **Gráficas `3_arbol_completo.png`, `3_arbol_podado.png`, `3_arbol_muy_podado.png`:** en cada nodo: condición, `squared_error` (varianza del peso en el nodo), `samples` y `value` (**peso medio** del nodo, que es la predicción si fuera hoja). Raíz: `squared_error = 689.2`, `samples = 1669`, `value = 86.9` kg.
- **Árbol muy podado completo** (para explicarlo):`family_history ≤ 0.5 (sin antecedente) ├── Gender = Female → Age ≤ 23.6 → 49.8 kg | Age > 23.6 → 65.8 kg └── Gender = Male → Age ≤ 23.5 → 65.1 kg | Age > 23.5 → 80.7 kg family_history > 0.5 (con antecedente) ├── Height ≤ 1.72 → FCVC ≤ 2.99 → 77.6 kg | FCVC > 2.99 → 96.6 kg └── Height > 1.72 → FAF ≤ 2.0 → 110.2 kg | FAF > 2.0 → 80.2 kg` Tiene sentido: sin antecedente familiar se pesa menos; hombres y mayores pesan más; con antecedente, los altos pesan más y **hacer mucha actividad física (FAF > 2) baja 30 kg** la predicción. (FCVC alto → más peso es una correlación del dataset, no una causa.)
- **Gráfica `3_pred_vs_real_tres_arboles.png`:** tres paneles predicho vs real **con la misma escala**. El completo tiene puntos dispersos (sobreajuste); el podado se acerca a la diagonal; el muy podado muestra **8 franjas horizontales** (sólo puede predecir 8 valores, uno por hoja).
- **Qué hay que explicar (diferente a clasificación):** Aquí **podar mejora el test**: el completo (1576 hojas, casi una fila por hoja) memoriza el ruido y logra R² 0.778; el podado sube a **0.814**. En clasificación el completo era el mejor de los tres; en regresión, no. El muy podado subajusta (R² 0.516), aunque aún explica más de la mitad de la varianza. **Raíz = `family_history_with_overweight ≤ 0.5`**: es el corte que más reduce la varianza del peso (en los datos, casi todas las personas con obesidad tienen antecedente familiar).

#### Tablas de este paso como imagen (1)

![3 · resumen de los tres arboles](imagenes/03_regresion_tablas_3_tabla1_resumen_de_los_tres_arboles.png)

### Resumen de los tres árboles

| modelo     |   profundidad |   n_hojas | raiz_atributo            |     MAE |    RMSE |     R2 |   R2_adj |
|:-----------|--------------:|----------:|:-------------------------|--------:|--------:|-------:|---------:|
| Completo   |            27 |     1,574 | family_history_with_ove… |  6.6092 | 12.4682 | 0.7685 |   0.7574 |
| Podado     |             8 |       116 | family_history_with_ove… |  7.0854 | 11.188  | 0.8136 |   0.8047 |
| Muy podado |             3 |         8 | family_history_with_ove… | 13.8761 | 18.0221 | 0.5162 |   0.4931 |

### Paso 4: Predicho vs real (modelos principales)

> *Ver código e03_regresion.py:279* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L279](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L279)

![4 · pred vs real modelos](imagenes/03_regresion_4_pred_vs_real_modelos.png)

### Datos de la figura: 4_pred_vs_real_modelos

#### Árbol de regresión
RMSE=10.47 kg  R²=0.837  R²adj=0.829

|   Real (kg) |   Predicho (kg) |
|------------:|----------------:|
|      37     |          37     |
|     158.021 |         158.021 |

#### Random Forest
RMSE=8.24 kg  R²=0.899  R²adj=0.894

|   Real (kg) |   Predicho (kg) |
|------------:|----------------:|
|      37     |          37     |
|     158.021 |         158.021 |

#### KNN
RMSE=9.91 kg  R²=0.854  R²adj=0.847

|   Real (kg) |   Predicho (kg) |
|------------:|----------------:|
|      37     |          37     |
|     158.021 |         158.021 |

- **Gráfica `4_pred_vs_real_modelos.png`:** árbol (GridSearch), RF y KNN, misma escala. RF se pega a la diagonal en todo el rango; el árbol muestra escalones; KNN en medio. En los tres la dispersión crece en los pesos altos (hay menos personas ahí).

### Paso 5: Importancia de features (RF de regresión)

> *Ver código e03_regresion.py:299* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L299](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L299)

![5 · importancias rf](imagenes/03_regresion_5_importancias_rf.png)

### Datos de la figura: 5_importancias_rf

#### Importancia de features — Random Forest (regresión de Weight)

| None                           |       valor |
|:-------------------------------|------------:|
| family_history_with_overweight | 0.246216    |
| Height                         | 0.170152    |
| FCVC                           | 0.11476     |
| Age                            | 0.0827581   |
| FAF                            | 0.0787077   |
| CAEC                           | 0.059071    |
| NCP                            | 0.0578208   |
| Gender                         | 0.0569554   |
| TUE                            | 0.0417308   |
| CALC                           | 0.0277678   |
| CH2O                           | 0.0230223   |
| MTRANS_Public_Transportation   | 0.0162897   |
| FAVC                           | 0.0143242   |
| MTRANS_Automobile              | 0.00454437  |
| SCC                            | 0.00446322  |
| MTRANS_Walking                 | 0.000621813 |
| SMOKE                          | 0.000557647 |
| MTRANS_Bike                    | 0.000118903 |
| MTRANS_Motorbike               | 0.000118819 |

- **Gráfica `5_importancias_rf.png`:** **family_history_with_overweight 0.246**, **Height 0.170**, FCVC 0.115, Age 0.083, FAF 0.079, CAEC 0.059, NCP 0.058, Gender 0.057…
- **Lectura:** el antecedente familiar condiciona fuertemente la tendencia de peso y la estatura fija la escala corporal. Es coherente con las raíces de los árboles.

#### Tablas de este paso como imagen (1)

![5 · feature importances](imagenes/03_regresion_tablas_5_tabla1_feature_importances.png)

### feature_importances_

|                                |   importancia |
|:-------------------------------|--------------:|
| family_history_with_overweight |        0.2462 |
| Height                         |        0.1702 |
| FCVC                           |        0.1148 |
| Age                            |        0.0828 |
| FAF                            |        0.0787 |
| CAEC                           |        0.0591 |
| NCP                            |        0.0578 |
| Gender                         |        0.057  |
| TUE                            |        0.0417 |
| CALC                           |        0.0278 |
| CH2O                           |        0.023  |
| MTRANS_Public_Transportation   |        0.0163 |
| FAVC                           |        0.0143 |
| MTRANS_Automobile              |        0.0045 |
| SCC                            |        0.0045 |
| MTRANS_Walking                 |        0.0006 |
| SMOKE                          |        0.0006 |
| MTRANS_Bike                    |        0.0001 |
| MTRANS_Motorbike               |        0.0001 |

### Paso 6: Comparación de métricas y R² ajustado

> *Ver código e03_regresion.py:313* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L313](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e03_regresion.py#L313)

![6 · comparacion metricas](imagenes/03_regresion_6_comparacion_metricas.png)

### Datos de la figura: 6_comparacion_metricas

#### RMSE en test (kg)

| categoria          |   test_RMSE |
|:-------------------|------------:|
| Baseline (media)   |     25.9126 |
| Árbol de regresión |     10.4655 |
| Random Forest      |      8.2356 |
| KNN                |      9.9085 |

#### R² en test

| categoria          |   test_R2 |
|:-------------------|----------:|
| Baseline (media)   |   -0.0001 |
| Árbol de regresión |    0.8369 |
| Random Forest      |    0.899  |
| KNN                |    0.8538 |

#### R² ajustado en test

| categoria          |   test_R2_adj |
|:-------------------|--------------:|
| Baseline (media)   |       -0.0479 |
| Árbol de regresión |        0.8291 |
| Random Forest      |        0.8942 |
| KNN                |        0.8468 |

- **Gráfica `6_comparacion_metricas.png`:** tres paneles de barras: RMSE (más bajo mejor), R² y R² ajustado (más alto mejor), para baseline, árbol, RF y KNN.
- **R² ajustado:** `1 − (1 − R²)(n − 1)/(n − p − 1)` con n = 418 y p = 19 → factor 417/398 ≈ 1.048. Como n ≫ p, la penalización es pequeña: RF 0.8987 → 0.8939; árbol 0.8369 → 0.8291; KNN 0.8538 → 0.8468. Sirve para comparar modelos con distinto número de variables y para no "inflar" el R² añadiendo variables inútiles.

#### Tablas de este paso como imagen (1)

![6 · tabla de resultados](imagenes/03_regresion_tablas_6_tabla1_tabla_de_resultados.png)

### Tabla de resultados

|                    | cv_RMSE   |   test_MAE |   test_RMSE |   test_R2 |   test_R2_adj |
|:-------------------|:----------|-----------:|------------:|----------:|--------------:|
| Baseline (media)   | —         |    21.7064 |     25.9126 |   -0.0001 |       -0.0479 |
| Árbol de regresión | 12.4728   |     6.581  |     10.4655 |    0.8369 |        0.8291 |
| Random Forest      | 9.1178    |     4.8292 |      8.2356 |    0.899  |        0.8942 |
| KNN                | 10.3709   |     5.0484 |      9.9085 |    0.8538 |        0.8468 |

### Paso 7: Conclusiones de regresión (texto)

RF es el mejor (MAE 4.84, RMSE 8.25, R² 0.899, R² adj 0.894). Poda: completo 0.778 → podado
0.814 → postpoda 0.824 → árbol de GridSearch 0.837. KNN (K=3) supera al árbol (0.854). Todas las
raíces: family_history. Sin fuga: el peso se predice sin NObeyesdad.

> **ETAPAS**

## Etapa 04 — Clustering K-Means, paso a paso

**Archivo:** `etapas/e04_clustering.py`. **Figuras:** `figures/04_clustering/`.

**Protocolo:** las 20 features de clasificación **sin la clase**, escaladas con
`StandardScaler` (sobre las 2087 filas: en no supervisado no hay train/test) → elegir K con 5
métodos para K ∈ [2, 12] → K-Means con K = 7 (número de clases reales) y comparar con
`NObeyesdad` → variantes con la clase y sólo numéricas.

**Funciones clave:**

| Función | Qué hace |
| :--- | :--- |
| `datos()` | X (20 features), `y_true` (clases, sólo para evaluar), `X_scaled` |
| `fit_kmeans(k, X)` | `KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)` |
| `knee_point(ks, values)` | detección automática del codo (máxima distancia a la recta entre extremos) |
| `gap_statistic(X, ks, n_refs=10)` | estadístico gap con 10 referencias uniformes |
| `_barrido(X)` / `barrido(variante)` | para cada K: inercia, silueta, CH, DB + gap; devuelve el K óptimo de cada método |
| `analizar_clustering(X, labels)` | tabla de contingencia, mapeo húngaro, matriz de confusión, silueta, ARI, NMI, accuracy de mapeo, pureza |
| `con_clase()` | variantes con la clase ordinal (21 columnas) y one-hot (27 columnas) |
| `solo_numericas()` | variante con las 8 numéricas |
| `_pca_2d(...)` | figura PCA de dos paneles |

---

### Paso 1: Datos escalados

> *Ver código e04_clustering.py:250* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L250](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L250)

- `X_scaled`: (2087, 20). **Advertencia clave:** al estandarizar, `MTRANS_Bike` (7 unos) toma el valor z ≈ **17.2** para esas 7 personas; esas binarias raras pesan muchísimo en la distancia euclídea. Esto explica casi todo lo que pasa en la etapa.

### Paso 2: Barrido de K (tabla)

> *Ver código e04_clustering.py:256* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L256](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L256)

| K | Inercia | Silueta | Calinski-Harabasz | Davies-Bouldin |
| :--- | :--- | :--- | :--- | :--- |
| 2 | 36914 | **0.224** | **272.6** | 2.308 |
| 3 | 33583 | 0.179 | 253.1 | 2.348 |
| 5 | 29460 | 0.193 | 217.0 | 1.681 |
| 7 | 25573 | 0.132 | 219.2 | 1.606 |
| 9 | 22173 | 0.159 | 229.2 | **1.528** |
| 12 | 19082 | 0.187 | 224.0 | 1.532 |

#### Tablas de este paso como imagen (1)

![2 · metricas internas por k](imagenes/04_clustering_tablas_2_tabla1_metricas_internas_por_k.png)

### Métricas internas por K

|   K |   inercia |   silhouette |   calinski_harabasz |   davies_bouldin |
|----:|----------:|-------------:|--------------------:|-----------------:|
|   2 |   36914   |        0.224 |             272.584 |            2.308 |
|   3 |   33583.3 |        0.179 |             253.08  |            2.348 |
|   4 |   31270   |        0.191 |             232.479 |            2.099 |
|   5 |   29460.2 |        0.193 |             216.96  |            1.681 |
|   6 |   27439.7 |        0.131 |             216.904 |            1.668 |
|   7 |   25572.6 |        0.132 |             219.169 |            1.606 |
|   8 |   23991   |        0.145 |             219.726 |            1.624 |
|   9 |   22172.6 |        0.159 |             229.23  |            1.528 |
|  10 |   21550.5 |        0.169 |             216.203 |            1.532 |
|  11 |   19860   |        0.179 |             228.717 |            1.555 |
|  12 |   19082.2 |        0.187 |             223.982 |            1.532 |

### Paso 2.1: Método del codo

> *Ver código e04_clustering.py:262* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L262](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L262)

![2.1 · codo](imagenes/04_clustering_2.1_codo.png)

### Datos de la figura: 2.1_codo

#### Método del codo

|   K |   Inercia (WCSS) |
|----:|-----------------:|
|   2 |          36914   |
|   3 |          33583.3 |
|   4 |          31270   |
|   5 |          29460.2 |
|   6 |          27439.7 |
|   7 |          25572.6 |
|   8 |          23991   |
|   9 |          22172.6 |
|  10 |          21550.5 |
|  11 |          19860   |
|  12 |          19082.2 |

- **Gráfica `2.1_codo.png`:** inercia vs K con línea roja en el codo detectado (**K = 7**).
- **Cómo se lee:** la inercia baja siempre; se busca dónde deja de bajar rápido. Aquí baja de forma **casi uniforme** (no hay codo nítido); la detección automática marca K = 7, pero es un codo débil.

### Paso 2.2: Silueta, Calinski-Harabasz y Davies-Bouldin

> *Ver código e04_clustering.py:277* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L277](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L277)

![2.2 · silueta ch db](imagenes/04_clustering_2.2_silueta_ch_db.png)

### Datos de la figura: 2.2_silueta_ch_db

#### silhouette

|   K |        y |
|----:|---------:|
|   2 | 0.22442  |
|   3 | 0.178876 |
|   4 | 0.191136 |
|   5 | 0.192705 |
|   6 | 0.131053 |
|   7 | 0.131533 |
|   8 | 0.145166 |
|   9 | 0.158847 |
|  10 | 0.16905  |
|  11 | 0.17895  |
|  12 | 0.187275 |

#### calinski_harabasz

|   K |       y |
|----:|--------:|
|   2 | 272.584 |
|   3 | 253.08  |
|   4 | 232.479 |
|   5 | 216.96  |
|   6 | 216.904 |
|   7 | 219.169 |
|   8 | 219.726 |
|   9 | 229.23  |
|  10 | 216.203 |
|  11 | 228.717 |
|  12 | 223.982 |

#### davies_bouldin

|   K |       y |
|----:|--------:|
|   2 | 2.30831 |
|   3 | 2.34777 |
|   4 | 2.09855 |
|   5 | 1.68063 |
|   6 | 1.66776 |
|   7 | 1.60596 |
|   8 | 1.62385 |
|   9 | 1.52793 |
|  10 | 1.53168 |
|  11 | 1.55484 |
|  12 | 1.53206 |

- **Gráfica `2.2_silueta_ch_db.png`:** tres paneles con la curva de cada índice y una línea roja en su óptimo: silueta máx. en **K = 2**, CH máx. en **K = 2**, DB mín. en **K = 9**.
- **Lectura:** la silueta es **baja para todo K** (máx. 0.224; 0.132 en K = 7) → la estructura de clusters es **débil** (no hay grupos bien separados).

### Paso 2.3: Estadístico gap

> *Ver código e04_clustering.py:296* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L296](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L296)

![2.3 · gap](imagenes/04_clustering_2.3_gap.png)

### Datos de la figura: 2.3_gap

#### Estadístico gap

|   K |   gap(K) |        s_k |
|----:|---------:|-----------:|
|   2 |  1.13352 | 0.00639652 |
|   3 |  1.09487 | 0.00711144 |
|   4 |  1.0513  | 0.00678229 |
|   5 |  1.06827 | 0.0044085  |
|   6 |  1.0983  | 0.00552944 |
|   7 |  1.14442 | 0.00579067 |
|   8 |  1.19323 | 0.0035584  |
|   9 |  1.25367 | 0.00502428 |
|  10 |  1.26558 | 0.00564247 |
|  11 |  1.33222 | 0.00432963 |
|  12 |  1.35948 | 0.00618173 |

- **Gráfica `2.3_gap.png`:** gap(K) con barras de error s_K; línea roja en **K = 2**.
- **Cómo se lee:** gap(2) = 1.134 ≥ gap(3) − s(3) = 1.095 − 0.007 → se cumple ya en K = 2 (el criterio elige el **menor** K que lo cumple, aunque la curva vuelva a subir para K grandes).

#### Tablas de este paso como imagen (1)

![2.3 · gap k y s k](imagenes/04_clustering_tablas_2.3_tabla1_gap_k_y_s_k.png)

### gap(K) y s_k

|   K |   gap |   s_k |
|----:|------:|------:|
|   2 | 1.134 | 0.006 |
|   3 | 1.095 | 0.007 |
|   4 | 1.051 | 0.007 |
|   5 | 1.068 | 0.004 |
|   6 | 1.098 | 0.006 |
|   7 | 1.144 | 0.006 |
|   8 | 1.193 | 0.004 |
|   9 | 1.254 | 0.005 |
|  10 | 1.266 | 0.006 |
|  11 | 1.332 | 0.004 |
|  12 | 1.359 | 0.006 |

### Paso 2.4: ¿Coinciden los 5 métodos?

> *Ver código e04_clustering.py:313* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L313](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L313)

| Método | K óptimo |
| :--- | :--- |
| Codo (inercia, kneedle) | 7 |
| Silueta (máx.) | 2 |
| Calinski-Harabasz (máx.) | 2 |
| Davies-Bouldin (mín.) | 9 |
| Gap statistic | 2 |

**No coinciden.** Tres de cinco eligen K = 2; el codo da 7 y DB 9. Cuando los métodos no se ponen
de acuerdo y la silueta es baja, la conclusión es que **no hay una estructura de clusters
clara**. ¿Qué separa el K = 2? **Verificado al preparar este documento:** separa a los **usuarios
de transporte público** (1556 personas, todas en un cluster) del **resto** (531: automóvil,
caminar, moto, bici). No es una partición por nivel de obesidad. (La nota del código dice que
corresponde a `family_history_with_overweight`; ver sección 16.)

#### Tablas de este paso como imagen (1)

![2.4 · k optimo segun cada metodo](imagenes/04_clustering_tablas_2.4_tabla1_k_optimo_segun_cada_metodo.png)

### K óptimo según cada método

|                         |   K óptimo |
|:------------------------|-----------:|
| Codo (inercia)          |          7 |
| Silueta (max)           |          2 |
| Calinski-Harabasz (max) |          2 |
| Davies-Bouldin (min)    |          9 |
| Gap statistic           |          2 |

### Paso 3: K = 7 vs clases reales: métricas

> *Ver código e04_clustering.py:321* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L321](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L321)

- Silueta **0.132**, **ARI 0.088**, **NMI 0.158** → los clusters **no** se corresponden con las 7 clases (ARI ≈ 0 sería azar; 1 sería perfecto).
- **¿Por qué K = 7?** Porque es el número de clases reales (y lo que marca el codo); así se puede comparar cluster ↔ clase.

### Paso 3.1: Tabla de contingencia (clusters × clases)

> *Ver código e04_clustering.py:330* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L330](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L330)

![3.1 · contingencia k7](imagenes/04_clustering_3.1_contingencia_k7.png)

### Datos de la figura: 3.1_contingencia_k7

#### Clases reales (filas) vs clusters K-Means K=7 (columnas)

|                     |   0 |   1 |   2 |   3 |   4 |   5 |   6 |
|:--------------------|----:|----:|----:|----:|----:|----:|----:|
| Insufficient_Weight |   6 |  45 |  45 |  30 |   0 | 141 |   0 |
| Normal_Weight       |  31 |  53 |  31 |  32 |   6 | 125 |   4 |
| Overweight_Level_I  |   9 |  48 |  65 |  91 |   1 |  60 |   2 |
| Overweight_Level_II |   6 |  71 |  94 | 106 |   1 |  12 |   0 |
| Obesity_Type_I      |   2 | 110 | 110 | 122 |   3 |   4 |   0 |
| Obesity_Type_II     |   1 |  51 |  95 | 148 |   0 |   1 |   1 |
| Obesity_Type_III    |   0 | 198 |   1 | 125 |   0 |   0 |   0 |

- **Gráfica `3.1_contingencia_k7.png`:** filas = clase real, columnas = número de cluster (arbitrario), celda = número de personas. Clase \ Cluster 0 1 2 3 4 5 6 Insufficient_Weight 6 45 45 30 0 **141** 0 Normal_Weight 31 53 31 32 6 **125** 4 Overweight_Level_I 9 48 65 **91** 1 60 2 Overweight_Level_II 6 71 94 **106** 1 12 0 Obesity_Type_I 2 110 110 **122** 3 4 0 Obesity_Type_II 1 51 95 **148** 0 1 1 Obesity_Type_III 0 **198** 1 125 0 0 0
- **Cómo se lee:** un buen clustering tendría cada fila concentrada en una columna. Aquí cada clase se reparte en 3–4 clusters, y los clusters 1, 2, 3 mezclan casi todas las clases. Los clusters 0, 4 y 6 son diminutos (55, 11 y 7 personas).

#### Tablas de este paso como imagen (1)

![3.1 · clases reales filas vs clusters k 7 colu](imagenes/04_clustering_tablas_3.1_tabla1_clases_reales_filas_vs_clusters_k_7_colu.png)

### Clases reales (filas) vs clusters K=7 (columnas)

| Clase real          |   0 |   1 |   2 |   3 |   4 |   5 |   6 |
|:--------------------|----:|----:|----:|----:|----:|----:|----:|
| Insufficient_Weight |   6 |  45 |  45 |  30 |   0 | 141 |   0 |
| Normal_Weight       |  31 |  53 |  31 |  32 |   6 | 125 |   4 |
| Overweight_Level_I  |   9 |  48 |  65 |  91 |   1 |  60 |   2 |
| Overweight_Level_II |   6 |  71 |  94 | 106 |   1 |  12 |   0 |
| Obesity_Type_I      |   2 | 110 | 110 | 122 |   3 |   4 |   0 |
| Obesity_Type_II     |   1 |  51 |  95 | 148 |   0 |   1 |   1 |
| Obesity_Type_III    |   0 | 198 |   1 | 125 |   0 |   0 |   0 |

### Paso 3.2: Matriz mapeada con el algoritmo húngaro

> *Ver código e04_clustering.py:339* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L339](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L339)

![3.2 · matriz mapeada k7](imagenes/04_clustering_3.2_matriz_mapeada_k7.png)

### Datos de la figura: 3.2_matriz_mapeada_k7

#### Matriz de confusión — clusters K=7 mapeados a clases (asignación óptima)

|                     |   Insufficient_Weight |   Normal_Weight |   Overweight_Level_I |   Overweight_Level_II |   Obesity_Type_I |   Obesity_Type_II |   Obesity_Type_III |
|:--------------------|----------------------:|----------------:|---------------------:|----------------------:|-----------------:|------------------:|-------------------:|
| Insufficient_Weight |                   141 |               6 |                    0 |                     0 |               45 |                30 |                 45 |
| Normal_Weight       |                   125 |              31 |                    4 |                     6 |               31 |                32 |                 53 |
| Overweight_Level_I  |                    60 |               9 |                    2 |                     1 |               65 |                91 |                 48 |
| Overweight_Level_II |                    12 |               6 |                    0 |                     1 |               94 |               106 |                 71 |
| Obesity_Type_I      |                     4 |               2 |                    0 |                     3 |              110 |               122 |                110 |
| Obesity_Type_II     |                     1 |               1 |                    1 |                     0 |               95 |               148 |                 51 |
| Obesity_Type_III    |                     0 |               0 |                    0 |                     0 |                1 |               125 |                198 |

- **Gráfica `3.2_matriz_mapeada_k7.png`:** matriz de confusión clase real vs "clase asignada al cluster". Mapeo óptimo: {5 → Insufficient, 0 → Normal, 6 → Overweight_I, 4 → Overweight_II, 2 → Obesity_I, 3 → Obesity_II, 1 → Obesity_III}.
- **Resultado:** accuracy con mapeo **0.302**, pureza **0.306** (el azar con 7 clases rondaría 0.14–0.17). Como hay clusters con 7 u 11 personas, a las clases que se les asignan casi no les llegan aciertos.
- **Tabla por clase** (`3.2_tabla1_metricas_por_clase_k_7_sin_la_clase`): misma lectura que en clasificación, con la clase asignada a cada cluster como "predicción". Precision = pureza del cluster asignado a la clase; Recall = cuánto de la clase reúne. Los F1 bajos (0.01–0.46) confirman que los clusters no reproducen las clases.

#### Tablas de este paso como imagen (1)

![3.2 · metricas por clase k 7 sin la clase](imagenes/04_clustering_tablas_3.2_tabla1_metricas_por_clase_k_7_sin_la_clase.png)

### Métricas por clase — K=7 sin la clase

| Clase               |   TP |   FN |   FP |    TN |   Precision |   Recall |    F1 |
|:--------------------|-----:|-----:|-----:|------:|------------:|---------:|------:|
| Insufficient_Weight |  141 |  126 |  202 | 1,618 |       0.411 |    0.528 | 0.462 |
| Normal_Weight       |   31 |  251 |   24 | 1,781 |       0.564 |    0.11  | 0.184 |
| Overweight_Level_I  |    2 |  274 |    5 | 1,806 |       0.286 |    0.007 | 0.014 |
| Overweight_Level_II |    1 |  289 |   10 | 1,787 |       0.091 |    0.003 | 0.007 |
| Obesity_Type_I      |  110 |  241 |  331 | 1,405 |       0.249 |    0.313 | 0.278 |
| Obesity_Type_II     |  148 |  149 |  506 | 1,284 |       0.226 |    0.498 | 0.311 |
| Obesity_Type_III    |  198 |  126 |  378 | 1,385 |       0.344 |    0.611 | 0.44  |

### Paso 3.3: Centroides en unidades originales

> *Ver código e04_clustering.py:352* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L352](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L352)

`scaler.inverse_transform(km.cluster_centers_)` devuelve los centroides a unidades reales:

| Cluster | n | Clase mapeada | Qué lo caracteriza |
| :--- | :--- | :--- | :--- |
| 0 | 55 | Normal_Weight | usuarios que **caminan** (MTRANS_Walking) |
| 1 | 576 | Obesity_Type_III | mujeres (Gender 0.17), antecedente 1.00, transporte público |
| 2 | 441 | Obesity_Type_I | **automóvil**, mayores (32 años) |
| 3 | 654 | Obesity_Type_II | hombres (0.79), antecedente 1.00, transporte público, 104 kg |
| 4 | 11 | Overweight_Level_II | usuarios de **moto** |
| 5 | 343 | Insufficient_Weight | jóvenes, **sin antecedente** (0.12), 56 kg |
| 6 | 7 | Overweight_Level_I | usuarios de **bici** |

**Conclusión:** K-Means separa por **medio de transporte** (bici, moto, caminar, auto vs
público), **antecedente familiar** y **sexo**, no por IMC. Las binarias raras, con z ≈ 17,
atraen centroides propios. Además K-Means asume clusters esféricos en un espacio continuo, y
mezclar one-hot/binarias con continuas rompe esa premisa.

#### Tablas de este paso como imagen (1)

![3.3 · centroides k 7 en unidades originales](imagenes/04_clustering_tablas_3.3_tabla1_centroides_k_7_en_unidades_originales.png)

### Centroides (K=7) en unidades originales

|   Cluster |   n | clase_mapeada       |   Gender |   Age |   Height |   Weight |   BMI_centroide |   family_history_with_overweight |   FAVC |   FCVC |   CAEC |   FAF |   MTRANS_Public_Transportation |
|----------:|----:|:--------------------|---------:|------:|---------:|---------:|----------------:|---------------------------------:|-------:|-------:|-------:|------:|-------------------------------:|
|         0 |  55 | Normal_Weight       |     0.64 | 22    |     1.72 |    70.83 |           23.84 |                             0.67 |   0.55 |   2.47 |   1.38 |  1.6  |                          -0    |
|         1 | 576 | Obesity_Type_III    |     0.17 | 23.42 |     1.64 |    87.44 |           32.42 |                             1    |   0.89 |   2.57 |   1.13 |  0.62 |                           1    |
|         2 | 441 | Obesity_Type_I      |     0.66 | 32.01 |     1.72 |    86.95 |           29.43 |                             0.92 |   0.92 |   2.35 |   1.08 |  1    |                          -0    |
|         3 | 654 | Obesity_Type_II     |     0.79 | 22.29 |     1.78 |   103.99 |           32.95 |                             1    |   0.97 |   2.35 |   1.05 |  1.25 |                           1    |
|         4 |  11 | Overweight_Level_II |     0.82 | 26.09 |     1.69 |    73.09 |           25.65 |                             0.55 |   0.73 |   2.36 |   1.64 |  0.82 |                          -0    |
|         5 | 343 | Insufficient_Weight |     0.3  | 20.32 |     1.64 |    56.3  |           20.97 |                             0.12 |   0.73 |   2.39 |   1.38 |  1.12 |                           0.96 |
|         6 |   7 | Overweight_Level_I  |     1    | 24.71 |     1.75 |    76.71 |           25.09 |                             0.71 |   0.43 |   2.14 |   1.29 |  2    |                           0    |

### Paso 3.4: PCA 2D: clusters vs clases

> *Ver código e04_clustering.py:366* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L366](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L366)

![3.4 · pca 2d](imagenes/04_clustering_3.4_pca_2d.png)

### Datos de la figura: 3.4_pca_2d

#### Clusters K-Means (K=7)

|               |     Valor |
|:--------------|----------:|
| N_puntos      | 2087      |
| PC1_min       |   -4.5963 |
| PC1_max       |    4.2904 |
| PC1_media     |    0      |
| PC2_min       |   -4.5439 |
| PC2_max       |    3.0211 |
| PC2_media     |   -0      |
| correlacion_r |    0      |

#### Clases reales NObeyesdad

|               |     Valor |
|:--------------|----------:|
| N_puntos      | 2087      |
| PC1_min       |   -4.5963 |
| PC1_max       |    4.2904 |
| PC1_media     |    0      |
| PC2_min       |   -4.5439 |
| PC2_max       |    3.0211 |
| PC2_media     |   -0      |
| correlacion_r |    0      |

- **Gráfica `3.4_pca_2d.png`:** dos paneles con los mismos puntos proyectados en PC1–PC2. Izquierda coloreado por cluster, derecha por clase real. **PC1 + PC2 = 25.4 %** de la varianza (13.8 % + 11.6 %).
- **Cómo se lee:** a la izquierda los clusters forman bloques nítidos (una banda diagonal de clusters de transporte público y un bloque abajo a la derecha, el cluster del automóvil); a la derecha las clases están mezcladas en esas mismas zonas. Los colores **no** siguen el mismo patrón → los clusters no son las clases. Ojo: con sólo 25 % de varianza el dibujo es una proyección muy aplanada de 20 dimensiones; sirve para intuir, no para concluir.

### Paso 4: Variantes con la columna de clase (ejercicio académico)

> *Ver código e04_clustering.py:374* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L374](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L374)

Pedido del profesor: comprobar si el clustering "mejora" al incluir la clase.

1. **Ordinal:** se añade `NObeyesdad_ord` = entero 0..6 en `CLASS_ORDER` → 21 columnas.
2. **One-hot:** se añaden 7 columnas `clase_*` → 27 columnas.

Ambas se estandarizan con su propio `StandardScaler` antes de K-Means con K = 7.

### Paso 4.1: Elección de K con clase ordinal (5 métodos)

> *Ver código e04_clustering.py:380* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L380](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L380)

![4.1 · cinco metodos con clase ordinal](imagenes/04_clustering_4.1_cinco_metodos_con_clase_ordinal.png)

### Datos de la figura: 4.1_cinco_metodos_con_clase_ordinal

#### 1. Método del codo

|   K |   Inercia (WCSS) |
|----:|-----------------:|
|   2 |          38983.3 |
|   3 |          34720.8 |
|   4 |          32569.7 |
|   5 |          30449.8 |
|   6 |          28777.4 |
|   7 |          26714.9 |
|   8 |          24902.9 |
|   9 |          23379.8 |
|  10 |          22787.9 |
|  11 |          21493.2 |
|  12 |          19977.9 |

#### 2. Coeficiente de silueta

|   K |   Silueta |
|----:|----------:|
|   2 |  0.213326 |
|   3 |  0.177534 |
|   4 |  0.1838   |
|   5 |  0.131118 |
|   6 |  0.147187 |
|   7 |  0.155492 |
|   8 |  0.158534 |
|   9 |  0.168751 |
|  10 |  0.183674 |
|  11 |  0.182213 |
|  12 |  0.187379 |

#### 3. Calinski-Harabasz

|   K |   Calinski-Harabasz |
|----:|--------------------:|
|   2 |             259.062 |
|   3 |             273.285 |
|   4 |             239.989 |
|   5 |             228.667 |
|   6 |             217.658 |
|   7 |             222.056 |
|   8 |             225.695 |
|   9 |             227.169 |
|  10 |             213.068 |
|  11 |             215.719 |
|  12 |             225.19  |

#### 4. Davies-Bouldin

|   K |   Davies-Bouldin |
|----:|-----------------:|
|   2 |          2.37384 |
|   3 |          2.17649 |
|   4 |          1.76455 |
|   5 |          1.92605 |
|   6 |          1.83604 |
|   7 |          1.64857 |
|   8 |          1.58551 |
|   9 |          1.57494 |
|  10 |          1.69521 |
|  11 |          1.63519 |
|  12 |          1.58879 |

#### 5. Estadístico gap

|   K |   gap(K) |        s_k |
|----:|---------:|-----------:|
|   2 |  1.09282 | 0.00581532 |
|   3 |  1.07648 | 0.00646595 |
|   4 |  1.02957 | 0.00604594 |
|   5 |  1.0549  | 0.00474774 |
|   6 |  1.07153 | 0.00547591 |
|   7 |  1.12501 | 0.00848375 |
|   8 |  1.17687 | 0.00358184 |
|   9 |  1.21999 | 0.00363489 |
|  10 |  1.23037 | 0.00441145 |
|  11 |  1.27175 | 0.00798884 |
|  12 |  1.33531 | 0.00507233 |

- **Gráfica `4.1_cinco_metodos_con_clase_ordinal.png`:** panel 2 × 3 (codo, silueta, CH, DB, gap; el sexto panel vacío). Resultado: codo 5, silueta 2, CH 3, DB 9, gap 2 → **siguen sin coincidir**.

#### Tablas de este paso como imagen (1)

![4.1 · k optimo por metodo con clase ordinal](imagenes/04_clustering_tablas_4.1_tabla1_k_optimo_por_metodo_con_clase_ordinal.png)

### K óptimo por método (con clase ordinal)

|                         |   K óptimo (con clase ordinal) |
|:------------------------|-------------------------------:|
| Codo (inercia)          |                              5 |
| Silueta (max)           |                              2 |
| Calinski-Harabasz (max) |                              3 |
| Davies-Bouldin (min)    |                              9 |
| Gap statistic           |                              2 |

### Paso 4.2: K = 7 con clase: contingencia cruda y matrices mapeadas

> *Ver código e04_clustering.py:411* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L411](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L411)

![4.2 · contingencia cruda con clase](imagenes/04_clustering_4.2_contingencia_cruda_con_clase.png)

### Datos de la figura: 4.2_contingencia_cruda_con_clase

#### Con clase (ordinal) — K=7, tabla de contingencia cruda

|                     |   0 |   1 |   2 |   3 |   4 |   5 |   6 |
|:--------------------|----:|----:|----:|----:|----:|----:|----:|
| Insufficient_Weight | 169 |  28 |   6 |   0 |  43 |   0 |  21 |
| Normal_Weight       | 152 |  39 |  36 |   0 |  25 |   4 |  26 |
| Overweight_Level_I  |  22 | 141 |  10 |   3 |  63 |   2 |  35 |
| Overweight_Level_II |  48 | 134 |   6 |   4 |  94 |   0 |   4 |
| Obesity_Type_I      |   2 | 211 |   5 |  21 | 110 |   0 |   2 |
| Obesity_Type_II     |   1 | 198 |   1 |   1 |  95 |   1 |   0 |
| Obesity_Type_III    |   0 |   0 |   0 | 323 |   1 |   0 |   0 |

#### Con clase (one-hot) — K=7, tabla de contingencia cruda

|                     |   0 |   1 |   2 |   3 |   4 |   5 |   6 |
|:--------------------|----:|----:|----:|----:|----:|----:|----:|
| Insufficient_Weight | 267 |   0 |   0 |   0 |   0 |   0 |   0 |
| Normal_Weight       |   0 |   0 |   0 |   0 |   0 |   0 | 282 |
| Overweight_Level_I  |   0 |   0 | 276 |   0 |   0 |   0 |   0 |
| Overweight_Level_II |   0 |   0 |   0 | 290 |   0 |   0 |   0 |
| Obesity_Type_I      |   0 | 351 |   0 |   0 |   0 |   0 |   0 |
| Obesity_Type_II     |   0 |   0 |   0 |   0 | 296 |   0 |   1 |
| Obesity_Type_III    |   0 |   0 |   0 |   0 |   0 | 324 |   0 |

![4.2 · matrices mapeadas con clase](imagenes/04_clustering_4.2_matrices_mapeadas_con_clase.png)

### Datos de la figura: 4.2_matrices_mapeadas_con_clase

#### Con clase (ordinal) — K=7 mapeado
ARI=0.252  Pureza=0.425

|                     |   Insufficient_Weight |   Normal_Weight |   Overweight_Level_I |   Overweight_Level_II |   Obesity_Type_I |   Obesity_Type_II |   Obesity_Type_III |
|:--------------------|----------------------:|----------------:|---------------------:|----------------------:|-----------------:|------------------:|-------------------:|
| Insufficient_Weight |                   169 |               6 |                   21 |                     0 |               43 |                28 |                  0 |
| Normal_Weight       |                   152 |              36 |                   26 |                     4 |               25 |                39 |                  0 |
| Overweight_Level_I  |                    22 |              10 |                   35 |                     2 |               63 |               141 |                  3 |
| Overweight_Level_II |                    48 |               6 |                    4 |                     0 |               94 |               134 |                  4 |
| Obesity_Type_I      |                     2 |               5 |                    2 |                     0 |              110 |               211 |                 21 |
| Obesity_Type_II     |                     1 |               1 |                    0 |                     1 |               95 |               198 |                  1 |
| Obesity_Type_III    |                     0 |               0 |                    0 |                     0 |                1 |                 0 |                323 |

#### Con clase (one-hot) — K=7 mapeado
ARI=0.999  Pureza=1.000

|                     |   Insufficient_Weight |   Normal_Weight |   Overweight_Level_I |   Overweight_Level_II |   Obesity_Type_I |   Obesity_Type_II |   Obesity_Type_III |
|:--------------------|----------------------:|----------------:|---------------------:|----------------------:|-----------------:|------------------:|-------------------:|
| Insufficient_Weight |                   267 |               0 |                    0 |                     0 |                0 |                 0 |                  0 |
| Normal_Weight       |                     0 |             282 |                    0 |                     0 |                0 |                 0 |                  0 |
| Overweight_Level_I  |                     0 |               0 |                  276 |                     0 |                0 |                 0 |                  0 |
| Overweight_Level_II |                     0 |               0 |                    0 |                   290 |                0 |                 0 |                  0 |
| Obesity_Type_I      |                     0 |               0 |                    0 |                     0 |              351 |                 0 |                  0 |
| Obesity_Type_II     |                     0 |               1 |                    0 |                     0 |                0 |               296 |                  0 |
| Obesity_Type_III    |                     0 |               0 |                    0 |                     0 |                0 |                 0 |                324 |

- **`4.2_contingencia_cruda_con_clase.png`:** tablas de contingencia antes del mapeo (ordinal y one-hot). En one-hot cada fila (clase) cae entera en una columna (cluster), sólo que "corrida" porque la numeración de clusters es arbitraria.
- **`4.2_matrices_mapeadas_con_clase.png`:** tras el mapeo húngaro: la de one-hot es una diagonal casi perfecta.
- **Tablas por clase** de las dos variantes (`4.2_tabla1…2_metricas_por_clase_k_7_con_clase_*`): con one-hot casi todo es 1.000; con ordinal sólo Obesity_Type_III (la clase con el código ordinal extremo) se recupera bien (F1 0.956) y Overweight_Level_II no tiene ningún TP.

#### Tablas de este paso como imagen (2)

![4.2 · metricas por clase k 7 con clase ordinal](imagenes/04_clustering_tablas_4.2_tabla1_metricas_por_clase_k_7_con_clase_ordinal.png)

### Métricas por clase — K=7 con clase ordinal

| Clase               |   TP |   FN |   FP |    TN |   Precision |   Recall |    F1 |
|:--------------------|-----:|-----:|-----:|------:|------------:|---------:|------:|
| Insufficient_Weight |  169 |   98 |  225 | 1,595 |       0.429 |    0.633 | 0.511 |
| Normal_Weight       |   36 |  246 |   28 | 1,777 |       0.562 |    0.128 | 0.208 |
| Overweight_Level_I  |   35 |  241 |   53 | 1,758 |       0.398 |    0.127 | 0.192 |
| Overweight_Level_II |    0 |  290 |    7 | 1,790 |       0     |    0     | 0     |
| Obesity_Type_I      |  110 |  241 |  321 | 1,415 |       0.255 |    0.313 | 0.281 |
| Obesity_Type_II     |  198 |   99 |  553 | 1,237 |       0.264 |    0.667 | 0.378 |
| Obesity_Type_III    |  323 |    1 |   29 | 1,734 |       0.918 |    0.997 | 0.956 |

![4.2 · metricas por clase k 7 con clase one hot](imagenes/04_clustering_tablas_4.2_tabla2_metricas_por_clase_k_7_con_clase_one_hot.png)

### Métricas por clase — K=7 con clase one-hot

| Clase               |   TP |   FN |   FP |    TN |   Precision |   Recall |    F1 |
|:--------------------|-----:|-----:|-----:|------:|------------:|---------:|------:|
| Insufficient_Weight |  267 |    0 |    0 | 1,820 |       1     |    1     | 1     |
| Normal_Weight       |  282 |    0 |    1 | 1,804 |       0.996 |    1     | 0.998 |
| Overweight_Level_I  |  276 |    0 |    0 | 1,811 |       1     |    1     | 1     |
| Overweight_Level_II |  290 |    0 |    0 | 1,797 |       1     |    1     | 1     |
| Obesity_Type_I      |  351 |    0 |    0 | 1,736 |       1     |    1     | 1     |
| Obesity_Type_II     |  296 |    1 |    0 | 1,790 |       1     |    0.997 | 0.998 |
| Obesity_Type_III    |  324 |    0 |    0 | 1,763 |       1     |    1     | 1     |

### Paso 4.3: Tabla comparativa sin/con clase

> *Ver código e04_clustering.py:431* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L431](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L431)

| Variante | Silueta | ARI | NMI | Acc mapeo | Pureza |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Sin clase | 0.132 | 0.088 | 0.158 | 0.302 | 0.306 |
| Con clase (ordinal) | 0.156 | 0.252 | 0.358 | 0.417 | 0.426 |
| Con clase (one-hot) | 0.228 | **0.999** | 0.998 | 0.9995 | 0.9995 |

#### Tablas de este paso como imagen (1)

![4.3 · comparativa de clustering k 7](imagenes/04_clustering_tablas_4.3_tabla1_comparativa_de_clustering_k_7.png)

### Comparativa de clustering (K=7)

|                     |   silueta |    ARI |    NMI |   acc_mapeo |   pureza |
|:--------------------|----------:|-------:|-------:|------------:|---------:|
| sin clase           |    0.1315 | 0.0883 | 0.1575 |      0.3023 |   0.3057 |
| con clase (ordinal) |    0.1555 | 0.252  | 0.3584 |      0.4173 |   0.4255 |
| con clase (one-hot) |    0.2279 | 0.9989 | 0.9984 |      0.9995 |   0.9995 |

### Paso 4.4: PCA 2D con clase ordinal

> *Ver código e04_clustering.py:442* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L442](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L442)

![4.4 · pca 2d con clase ordinal](imagenes/04_clustering_4.4_pca_2d_con_clase_ordinal.png)

### Datos de la figura: 4.4_pca_2d_con_clase_ordinal

#### Clusters K-Means K=7 (con clase ordinal)

|               |     Valor |
|:--------------|----------:|
| N_puntos      | 2087      |
| PC1_min       |   -5.3536 |
| PC1_max       |    3.118  |
| PC1_media     |   -0      |
| PC2_min       |   -4.6882 |
| PC2_max       |    2.3164 |
| PC2_media     |    0      |
| correlacion_r |   -0      |

#### Clases reales NObeyesdad

|               |     Valor |
|:--------------|----------:|
| N_puntos      | 2087      |
| PC1_min       |   -5.3536 |
| PC1_max       |    3.118  |
| PC1_media     |   -0      |
| PC2_min       |   -4.6882 |
| PC2_max       |    2.3164 |
| PC2_media     |    0      |
| correlacion_r |   -0      |

- **Gráfica `4.4_pca_2d_con_clase_ordinal.png`:** igual que 3.4 (PC1 + PC2 = 26.9 %). Se parece un poco más a las clases, pero sigue lejos.

### Paso 4.5: Interpretación

- **Ordinal (mejora moderada):** ARI 0.088 → 0.252. Es **una** columna entre 21: en la distancia euclídea las otras 20 siguen aportando la mayor parte; las binarias raras siguen atrayendo centroides.
- **One-hot (casi perfecta):** ARI 0.999, 2086 de 2087 filas bien. Siete columnas binarias ortogonales estandarizadas pesan más que todas las demás: la inercia mínima se consigue poniendo un centroide en cada "vértice" de clase.
- **¿Por qué el ejercicio no tiene sentido en la práctica?** En un problema no supervisado real **no existen** las etiquetas (si existieran, sería un problema de clasificación). Meter el target como feature es darle la respuesta al algoritmo. Además el mapeo húngaro usa las etiquetas reales. El ejercicio sirve para **ilustrar la diferencia entre supervisado y no supervisado**.

### Paso 5: Variante sólo numéricas (8 features)

> *Ver código e04_clustering.py:452* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L452](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e04_clustering.py#L452)

![5 · metricas solo numericas](imagenes/04_clustering_5_metricas_solo_numericas.png)

### Datos de la figura: 5_metricas_solo_numericas

#### inercia (sólo numéricas)

|   K |        y |
|----:|---------:|
|   2 | 14213.1  |
|   3 | 12614.6  |
|   4 | 11346.4  |
|   5 | 10480.7  |
|   6 |  9676.85 |
|   7 |  9123.3  |
|   8 |  8653.63 |
|   9 |  8338.21 |
|  10 |  7985.28 |
|  11 |  7683.76 |
|  12 |  7376.26 |

#### silhouette (sólo numéricas)

|   K |        y |
|----:|---------:|
|   2 | 0.144971 |
|   3 | 0.149876 |
|   4 | 0.158452 |
|   5 | 0.159789 |
|   6 | 0.164683 |
|   7 | 0.164727 |
|   8 | 0.166637 |
|   9 | 0.177385 |
|  10 | 0.177579 |
|  11 | 0.177037 |
|  12 | 0.185387 |

#### calinski_harabasz (sólo numéricas)

|   K |       y |
|----:|--------:|
|   2 | 364.237 |
|   3 | 337.13  |
|   4 | 327.363 |
|   5 | 308.665 |
|   6 | 301.893 |
|   7 | 287.747 |
|   8 | 276.021 |
|   9 | 260.362 |
|  10 | 251.743 |
|  11 | 243.493 |
|  12 | 238.338 |

#### davies_bouldin (sólo numéricas)

|   K |       y |
|----:|--------:|
|   2 | 2.31836 |
|   3 | 2.05239 |
|   4 | 1.80009 |
|   5 | 1.82379 |
|   6 | 1.67344 |
|   7 | 1.63535 |
|   8 | 1.60966 |
|   9 | 1.61377 |
|  10 | 1.6337  |
|  11 | 1.59567 |
|  12 | 1.59568 |

![5 · matrices solo numericas](imagenes/04_clustering_5_matrices_solo_numericas.png)

### Datos de la figura: 5_matrices_solo_numericas

#### Clases reales vs clusters (K=7, sólo numéricas)

|                     |   0 |   1 |   2 |   3 |   4 |   5 |   6 |
|:--------------------|----:|----:|----:|----:|----:|----:|----:|
| Insufficient_Weight |  85 |   0 |   0 |   7 |  49 | 123 |   3 |
| Normal_Weight       |  98 |   0 |  18 |  26 |  47 |  82 |  11 |
| Overweight_Level_I  |  69 |   5 |  38 |   9 |  52 |  60 |  43 |
| Overweight_Level_II |  55 |   3 |  66 |  19 |  32 |  52 |  63 |
| Obesity_Type_I      |  50 |  25 |  97 |   7 |  55 |  38 |  79 |
| Obesity_Type_II     |   0 |  91 | 119 |   5 |  29 |   0 |  53 |
| Obesity_Type_III    |   0 | 137 |   0 | 187 |   0 |   0 |   0 |

#### Clusters mapeados a clases (asignación óptima)

|                     |   Insufficient_Weight |   Normal_Weight |   Overweight_Level_I |   Overweight_Level_II |   Obesity_Type_I |   Obesity_Type_II |   Obesity_Type_III |
|:--------------------|----------------------:|----------------:|---------------------:|----------------------:|-----------------:|------------------:|-------------------:|
| Insufficient_Weight |                   123 |              85 |                   49 |                     3 |                0 |                 0 |                  7 |
| Normal_Weight       |                    82 |              98 |                   47 |                    11 |               18 |                 0 |                 26 |
| Overweight_Level_I  |                    60 |              69 |                   52 |                    43 |               38 |                 5 |                  9 |
| Overweight_Level_II |                    52 |              55 |                   32 |                    63 |               66 |                 3 |                 19 |
| Obesity_Type_I      |                    38 |              50 |                   55 |                    79 |               97 |                25 |                  7 |
| Obesity_Type_II     |                     0 |               0 |                   29 |                    53 |              119 |                91 |                  5 |
| Obesity_Type_III    |                     0 |               0 |                    0 |                     0 |                0 |               137 |                187 |

- K óptimo por método: codo 6, silueta 12, CH 2, DB 11, gap sin corte (ningún K del rango cumple el criterio) → tampoco coinciden.
- **Gráfica `5_metricas_solo_numericas.png`:** 4 paneles (inercia, silueta, CH, DB) vs K.
- **Gráfica `5_matrices_solo_numericas.png`:** contingencia + matriz mapeada.
- K = 7: silueta 0.165, **ARI 0.156**, NMI 0.262, acc mapeo 0.341, **pureza 0.382** → mejor que con las 20 features (ya no mandan las binarias raras), pero lejos de las clases.
- **Centroides ordenados por IMC:** van de ~22.9 a ~40.0, pero cada cluster mezcla clases contiguas y además se separa por Age (un cluster de 37 años), NCP, FAF y TUE: la estructura "natural" combina peso, hábitos y edad, y no coincide con los cortes fijos de IMC.
- **Tabla por clase** (`5_tabla2_metricas_por_clase_k_7_solo_numericas`); la tabla de centroides pasa a ser `5_tabla3_centroides_solo_numericas_ordenados_por`.

#### Tablas de este paso como imagen (3)

![5 · k optimo por metodo todas las features v](imagenes/04_clustering_tablas_5_tabla1_k_optimo_por_metodo_todas_las_features_v.png)

### K óptimo por método: todas las features vs sólo numéricas

|                         |   K óptimo (todas) | K óptimo (sólo numéricas)   |
|:------------------------|-------------------:|:----------------------------|
| Codo (inercia)          |                  7 | 6                           |
| Silueta (max)           |                  2 | 12                          |
| Calinski-Harabasz (max) |                  2 | 2                           |
| Davies-Bouldin (min)    |                  9 | 11                          |
| Gap statistic           |                  2 | —                           |

![5 · metricas por clase k 7 solo numericas](imagenes/04_clustering_tablas_5_tabla2_metricas_por_clase_k_7_solo_numericas.png)

### Métricas por clase — K=7 sólo numéricas

| Clase               |   TP |   FN |   FP |    TN |   Precision |   Recall |    F1 |
|:--------------------|-----:|-----:|-----:|------:|------------:|---------:|------:|
| Insufficient_Weight |  123 |  144 |  232 | 1,588 |       0.346 |    0.461 | 0.395 |
| Normal_Weight       |   98 |  184 |  259 | 1,546 |       0.275 |    0.348 | 0.307 |
| Overweight_Level_I  |   52 |  224 |  212 | 1,599 |       0.197 |    0.188 | 0.193 |
| Overweight_Level_II |   63 |  227 |  189 | 1,608 |       0.25  |    0.217 | 0.232 |
| Obesity_Type_I      |   97 |  254 |  241 | 1,495 |       0.287 |    0.276 | 0.282 |
| Obesity_Type_II     |   91 |  206 |  170 | 1,620 |       0.349 |    0.306 | 0.326 |
| Obesity_Type_III    |  187 |  137 |   73 | 1,690 |       0.719 |    0.577 | 0.64  |

![5 · centroides solo numericas ordenados por](imagenes/04_clustering_tablas_5_tabla3_centroides_solo_numericas_ordenados_por.png)

### Centroides (sólo numéricas) ordenados por IMC

|   Cluster |   n | clase_mapeada       |   Age |   Height |   Weight |   FCVC |   NCP |   CH2O |   FAF |   TUE |   BMI_centroide |
|----------:|----:|:--------------------|------:|---------:|---------:|-------:|------:|-------:|------:|------:|----------------:|
|         5 | 355 | Insufficient_Weight | 20.61 |     1.74 |    69    |   2.55 |  3.15 |   2.3  |  2.1  |  0.84 |           22.91 |
|         0 | 357 | Normal_Weight       | 20.66 |     1.65 |    65.36 |   2.03 |  3.06 |   1.57 |  0.74 |  0.95 |           23.93 |
|         4 | 264 | Overweight_Level_I  | 21.64 |     1.62 |    70.42 |   2.4  |  1.18 |   1.87 |  0.67 |  0.73 |           26.67 |
|         6 | 252 | Overweight_Level_II | 37.41 |     1.66 |    84.91 |   2.4  |  2.5  |   1.78 |  0.93 |  0.19 |           30.67 |
|         2 | 338 | Obesity_Type_I      | 24.69 |     1.79 |   102.33 |   1.94 |  2.78 |   2.17 |  0.84 |  0.58 |           31.9  |
|         3 | 260 | Obesity_Type_III    | 24.78 |     1.65 |    98.6  |   2.97 |  3.03 |   2.13 |  0.21 |  0.44 |           36.42 |
|         1 | 261 | Obesity_Type_II     | 23.77 |     1.78 |   127.33 |   2.91 |  2.9  |   2.21 |  1.35 |  0.74 |           39.97 |

### Paso 6: Conclusiones del clustering

El clustering no reproduce la etiqueta de obesidad, y **no es un fallo del procedimiento sino un
resultado**: `NObeyesdad` es la discretización de **una sola variable** (el IMC) con cortes fijos
de la OMS, mientras que K-Means busca la partición que minimiza la varianza en las 20 (u 8)
dimensiones a la vez, donde el IMC pesa poco.

> **ETAPAS**

## Etapa 05 — Comparación final, discusión y conclusiones

**Archivo:** `etapas/e05_comparacion.py`. **Figura:** `figures/05_comparacion/1_comparacion_final.png`.

Esta etapa **no entrena nada nuevo** (salvo el caso con 20 % de ruido): **importa** las etapas
02–04 (`from etapas import e02_clasificacion as e02`, etc.) y reutiliza sus modelos de la caché,
incluido el experimento sin SMOTE del paso 8 (`e02.evaluacion_reales()`). Las cifras de sus
textos se **calculan** (f-strings), no están escritas a mano.

---

### Paso 1: Resultados de todos los casos

> *Ver código e05_comparacion.py:131* — [https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e05_comparacion.py#L131](https://github.com/DanGomez2002/ia2-obesidad-py/blob/main/etapas/e05_comparacion.py#L131)

![1 · comparacion final](imagenes/05_comparacion_1_comparacion_final.png)

### Datos de la figura: 1_comparacion_final

#### Clasificación de NObeyesdad

|    |   serie_1 |
|---:|----------:|
|  0 |    0.9378 |

#### Clasificación de NObeyesdad

|    |   serie_2 |
|---:|----------:|
|  0 |    0.9569 |

#### Clasificación de NObeyesdad

|    |   serie_3 |
|---:|----------:|
|  0 |    0.8206 |

#### Clasificación de NObeyesdad

|    |   serie_4 |
|---:|----------:|
|  0 |    0.7464 |

#### Clasificación de NObeyesdad

|    |   serie_5 |
|---:|----------:|
|  0 |    0.8565 |

#### Clasificación de NObeyesdad

|    |   serie_6 |
|---:|----------:|
|  0 |    0.7679 |

#### Clasificación de NObeyesdad

|    |   serie_7 |
|---:|----------:|
|  0 |  0.648325 |

#### Clasificación de NObeyesdad

|    |   serie_8 |
|---:|----------:|
|  0 |  0.789474 |

#### Clasificación de NObeyesdad

|    |   serie_9 |
|---:|----------:|
|  0 |  0.736842 |

#### Clasificación de NObeyesdad

|    |   serie_10 |
|---:|-----------:|
|  0 |   0.767943 |

#### Regresión de Weight

| categoria        |   valor |
|:-----------------|--------:|
| Baseline (media) | -0.0001 |
| Árbol            |  0.8369 |
| Random Forest    |  0.899  |
| KNN              |  0.8538 |

#### Clustering K-Means, K=7
(rojo: con la clase como feature)

| categoria          |     valor |
|:-------------------|----------:|
| Sin clase          | 0.0883186 |
| Sólo numéricas (8) | 0.156434  |
| clase ordinal      | 0.252043  |
| clase one-hot      | 0.998921  |

#### Variante B (sin Weight/Height):
con SMOTE vs sólo filas reales

|    |   serie_1 |
|---:|----------:|
|  0 |    0.8528 |

#### Variante B (sin Weight/Height):
con SMOTE vs sólo filas reales

|    |   serie_2 |
|---:|----------:|
|  0 |  0.195007 |

#### Variante B (sin Weight/Height):
con SMOTE vs sólo filas reales

|    |   serie_3 |
|---:|----------:|
|  0 |  0.421004 |

Tablas: clasificación A, B, B + 20 % de ruido, **B sólo con filas reales (7 y 3 clases)**,
regresión y clustering, más la tabla **"Mejor modelo de cada caso"**:

| Caso | Mejor modelo | Métrica principal | Valor | Otra métrica |
| :--- | :--- | :--- | :--- | :--- |
| Clasificación A (todas) | Random Forest | macro-F1 test | 0.9558 | accuracy 0.9569 |
| Clasificación B (sin W/H) | Random Forest | macro-F1 test | 0.8528 | accuracy 0.8565 |
| Clasificación B + ruido 20 % | Random Forest | macro-F1 test | 0.7801 | accuracy 0.7895 |
| Clasificación B · filas reales · 7 clases | KNN (k=1) | macro-F1 CV 3×10 | 0.1942 | línea base 0.1042 |
| Clasificación B · filas reales · 3 clases | RF con pesos por clase | macro-F1 CV 3×10 | 0.4244 | línea base 0.2616 |
| Regresión de Weight | Random Forest | RMSE test (kg) | 8.2468 | R² 0.8987 |
| Clustering K=7 (sin la clase) | K-Means · sólo numéricas (8) | ARI vs clases | 0.1564 | pureza 0.3824 |

Ojo: en las filas reales "mejor modelo" significa sólo "el de mayor macro-F1"; las diferencias
entre modelos caen dentro de la variación entre folds. Lo importante es la distancia a la línea base.

**Gráfica `1_comparacion_final.png`** (cuatro paneles, colores fijos: azul = árbol, verde = RF,
naranja = KNN, gris = otro/baseline):

1. **Clasificación de NObeyesdad:** accuracy en test de árbol/RF/KNN en A, B y B + 20 % ruido (en el último grupo, dos barras de KNN marcadas k=1 y k=3). Eje y desde 0.5. Se ve que RF (verde) es la barra más alta en los tres grupos y que todo baja de A → B → ruido.
2. **Regresión de Weight:** R² en test de baseline (0.000), árbol (0.837), RF (0.899), KNN (0.854).
3. **Clustering K-Means, K=7:** ARI de sin clase (0.088), sólo numéricas (0.156) en gris, y con clase ordinal (0.252) y one-hot (0.999) en **rojo** (con la clase como feature). La barra roja gigante es la demostración visual de "sólo recupera las clases si se le dan".
4. **Variante B: con SMOTE vs sólo filas reales:** macro-F1 del dataset completo (RF, 0.853) frente al mejor con filas reales en 7 clases (0.194) y en 3 clases (0.424), en morado; la línea negra discontinua sobre cada barra morada es la línea base (0.104 y 0.262). Es la imagen de "cuánto del rendimiento fabricaba SMOTE".

Título general: *"Comparación final: cuánto de la variable objetivo recupera cada enfoque"*.

#### Tablas de este paso como imagen (7)

![1 · clasificacion a todas las features](imagenes/05_comparacion_tablas_1_tabla1_clasificacion_a_todas_las_features.png)

### Clasificación A (todas las features)

|                   |   cv_f1_macro |   test_accuracy |   test_f1_macro |
|:------------------|--------------:|----------------:|----------------:|
| Árbol de decisión |        0.923  |          0.9378 |          0.9362 |
| Random Forest     |        0.9437 |          0.9569 |          0.9558 |
| KNN               |        0.8152 |          0.8206 |          0.8075 |

![1 · clasificacion b sin weight ni height](imagenes/05_comparacion_tablas_1_tabla2_clasificacion_b_sin_weight_ni_height.png)

### Clasificación B (sin Weight ni Height)

|                   |   cv_f1_macro |   test_accuracy |   test_f1_macro |
|:------------------|--------------:|----------------:|----------------:|
| Árbol de decisión |        0.744  |          0.7464 |          0.7401 |
| Random Forest     |        0.8506 |          0.8565 |          0.8528 |
| KNN               |        0.764  |          0.7679 |          0.7567 |

![1 · clasificacion b con 20 de ruido en los h](imagenes/05_comparacion_tablas_1_tabla3_clasificacion_b_con_20_de_ruido_en_los_h.png)

### Clasificación B con 20 % de ruido en los hábitos (test_accuracy_limpio = mismo modelo sin ruido)

|                     |   test_accuracy_limpio |   test_accuracy |   test_f1_macro |
|:--------------------|-----------------------:|----------------:|----------------:|
| Árbol de decisión   |                 0.7464 |          0.6483 |          0.6427 |
| Random Forest       |                 0.8565 |          0.7895 |          0.7801 |
| KNN (k=1)           |                 0.7703 |          0.7368 |          0.725  |
| KNN (k=3, distance) |                 0.7679 |          0.7679 |          0.7564 |

![1 · clasificacion b solo con filas reales si](imagenes/05_comparacion_tablas_1_tabla4_clasificacion_b_solo_con_filas_reales_si.png)

### Clasificación B sólo con filas reales (sin SMOTE), 7 clases (CV 3×10)

|                                |   accuracy |   balanced_accuracy |   f1_macro |   f1_macro_std |
|:-------------------------------|-----------:|--------------------:|-----------:|---------------:|
| Línea base (clase mayoritaria) |     0.5743 |              0.1429 |     0.1042 |         0.0002 |
| Árbol (depth 4, leaf 5)        |     0.5546 |              0.1713 |     0.1564 |         0.0257 |
| Random Forest                  |     0.5758 |              0.1819 |     0.1737 |         0.0206 |
| RF con pesos por clase         |     0.4735 |              0.1994 |     0.195  |         0.0211 |
| KNN (k=15, distance)           |     0.5896 |              0.1693 |     0.152  |         0.0244 |
| KNN (k=1)                      |     0.4633 |              0.1944 |     0.1942 |         0.0253 |

![1 · clasificacion b solo con filas reales si](imagenes/05_comparacion_tablas_1_tabla5_clasificacion_b_solo_con_filas_reales_si.png)

### Clasificación B sólo con filas reales (sin SMOTE), 3 clases (CV 3×10)

|                                |   accuracy |   balanced_accuracy |   f1_macro |   f1_macro_std |
|:-------------------------------|-----------:|--------------------:|-----------:|---------------:|
| Línea base (clase mayoritaria) |     0.6456 |              0.3333 |     0.2616 |         0.0003 |
| Árbol (depth 4, leaf 5)        |     0.631  |              0.384  |     0.3719 |         0.0374 |
| Random Forest                  |     0.6483 |              0.3939 |     0.3824 |         0.0345 |
| RF con pesos por clase         |     0.602  |              0.42   |     0.421  |         0.0318 |
| KNN (k=15, distance)           |     0.6578 |              0.3687 |     0.335  |         0.0305 |
| KNN (k=1)                      |     0.575  |              0.3963 |     0.3957 |         0.028  |

![1 · regresion de weight rmse y mae en kg](imagenes/05_comparacion_tablas_1_tabla6_regresion_de_weight_rmse_y_mae_en_kg.png)

### Regresión de Weight (RMSE y MAE en kg)

|                    | cv_RMSE   |   test_MAE |   test_RMSE |   test_R2 |   test_R2_adj |
|:-------------------|:----------|-----------:|------------:|----------:|--------------:|
| Baseline (media)   | —         |    21.7064 |     25.9126 |   -0.0001 |       -0.0479 |
| Árbol de regresión | 12.4728   |     6.581  |     10.4655 |    0.8369 |        0.8291 |
| Random Forest      | 9.1178    |     4.8292 |      8.2356 |    0.899  |        0.8942 |
| KNN                | 10.3709   |     5.0484 |      9.9085 |    0.8538 |        0.8468 |

![1 · clustering k means con k 7 comparado con](imagenes/05_comparacion_tablas_1_tabla7_clustering_k_means_con_k_7_comparado_con.png)

### Clustering K-Means con K=7 comparado con las clases reales

|                         |   silueta |    ARI |    NMI |   pureza |
|:------------------------|----------:|-------:|-------:|---------:|
| Sin clase (20 features) |    0.1315 | 0.0883 | 0.1575 |   0.3057 |
| Sólo numéricas (8)      |    0.1647 | 0.1564 | 0.2618 |   0.3824 |
| Con clase ordinal       |    0.1555 | 0.252  | 0.3584 |   0.4255 |
| Con clase one-hot       |    0.2279 | 0.9989 | 0.9984 |   0.9995 |

### Paso 2: Discusión (resumen de los argumentos)

- **RF es el mejor en todos los casos supervisados con el dataset completo** (A: 0.957 vs 0.938 árbol y 0.821 KNN; B: 0.857 vs 0.746 y 0.768; regresión: RMSE 8.25 vs 10.47 y 9.91) → reducción de varianza por bagging (en clasificación, además, cada corte elige entre un subconjunto aleatorio de features; el RF de regresión usa todas).
- **El número de árboles no sobreajusta pero deja de compensar** (macro-F1 CV 0.910 con 10 → 0.941 con 100 → 0.946 con 800).
- **La fuga de datos domina la clasificación A** (RF cae 10.0 puntos al quitar Weight/Height). Lo que queda (0.857) *parecería* lo que predicen los hábitos, pero sigue inflado por SMOTE.
- **Sin SMOTE, los hábitos casi no predicen los 7 niveles:** RF B acierta 0.946 en filas sintéticas del test y 0.583 en reales; sólo con filas reales, macro-F1 0.194 frente a 0.104 de la línea base y sin mejora útil de accuracy (0.574); en 3 niveles, 0.424 frente a 0.262. En este escenario RF ya no destaca.
- **KNN depende de la tarea:** en clasificación queda por debajo del árbol en A; en regresión supera al árbol (0.854 vs 0.837) porque promediar vecinos suaviza la predicción.
- **Robustez con 20 % de ruido:** RF pierde 6.7 puntos, árbol 9.8, KNN k=1 3.3, KNN k=3 0 (se queda igual) → k=3 por encima de k=1.
- **Supervisado vs no supervisado:** ARI 0.088 sin la clase, 0.999 con la clase one-hot.
- **Limitaciones:** (a) 77 % sintético (SMOTE) infla las cifras de todos los modelos (medido) y la identificación de filas reales es heurística; (b) un único split de 418 filas con una sola semilla → diferencias de milésimas no son significativas; (c) las métricas de clustering contra las clases miden "parecido con NObeyesdad", no la calidad del clustering en sí.

### Paso 3: Conclusiones generales

1. **Modelo recomendado: Random Forest** (mejor en A, B, regresión y el más robusto al ruido). Con el criterio del 90 % de la mejora bastan **400** árboles en clasificación y **50** en regresión. Con sólo las filas reales ningún modelo destaca con claridad.
2. **El rendimiento alto se apoya en dos artificios:** la fuga del IMC (variante A) y SMOTE (variante B, 85.7 %). La A responde al enunciado; para el problema real, la referencia es el caso sin ninguno de los dos.
3. **Sin fuga ni SMOTE, los hábitos predicen poco:** en 7 clases, macro-F1 0.194 frente a 0.104 de la línea base; en 3 niveles, señal modesta (0.424 vs 0.262) con Age, family_history y FAF como atributos principales. El peso se estima con error típico de 8.2 kg sin usar la clase (dataset completo, con SMOTE).
4. **El clustering no recupera las clases de obesidad** (ARI 0.088).
5. **Cautela al generalizar:** las cifras del dataset completo son optimistas; las de las filas reales son la estimación más honesta, aunque con pocas filas (491, 3 de Obesity_III) y una identificación heurística. Para conclusiones firmes haría falta una muestra real mayor y sin balanceo sintético.

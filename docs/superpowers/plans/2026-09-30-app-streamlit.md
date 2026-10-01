# App web interactiva (Streamlit) — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Una app Streamlit (`streamlit run app.py`) que reproduce, en el mismo orden y con los mismos títulos que `estudio_etapas.html`, solo las gráficas y tablas de cada paso, con controles para cambiar features, hiperparámetros y opciones; añade las iteraciones de K-Means y elimina la postpoda de todo el código.

**Architecture:** Paquete nuevo `webapp/` (no se llama `app/` porque chocaría con `app.py`). Núcleo sin Streamlit y testeable (`webapp/modelos.py`, `webapp/clustering.py`), widgets compartidos (`webapp/controles.py`), un módulo de páginas por etapa (`webapp/etapa01.py` … `etapa05.py`) y `app.py` como shell con la navegación. Los pasos sin parámetros y la etapa 05 reutilizan el código de `etapas/` redirigiendo su salida al gancho `salida.RECEPTOR` que ya existe en `src/salida.py`. El CLI (`main.py`, ▶ de cada etapa) no cambia salvo por la eliminación de la postpoda.

**Tech Stack:** Python 3.12, Streamlit (con `streamlit.testing.v1.AppTest`), scikit-learn, pandas, matplotlib/seaborn, pytest. Entorno: `.venv-1` (Windows).

**Spec:** `docs/superpowers/specs/2026-09-30-app-streamlit-design.md`

**Commits:** cada tarea termina con un paso «Checkpoint» con `git add`/`git commit`. Hazlo **solo si el usuario lo autorizó**; si no, omite ese paso y avisa de qué archivos cambiaron. El repo ya tiene muchos cambios sin commitear: añade solo los archivos de la tarea, nunca `git add -A`.

## Global Constraints

- **Entorno de ejecución.** El plan está escrito con comandos de Windows (`.venv-1/Scripts/python.exe`). **En la sesión en la nube (Linux) hay que sustituir** `.venv-1/Scripts/python.exe` por `.venv/bin/python` y crear antes el entorno: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (no hace falta el `pip install streamlit pytest` de la Task 0 si ya están en `requirements.txt`; la Task 0 solo tiene que añadirlos al archivo). `taskkill ...` de la Task 11 pasa a `pkill -f "streamlit run"`. En Windows el intérprete es `.venv-1/Scripts/python.exe`. Ejecuta con `MPLBACKEND=Agg` si no hay pantalla. Los datos ya están en el repo (`data/obesity_raw.csv`): no se necesita internet para correr nada salvo `pip install`.
- **Cómo ejecutar este plan en la nube:** método *native* (una sola sesión implementa las 12 tareas en orden, sin subagentes por tarea) y al final una revisión independiente de toda la rama. Hacer commits por tarea en una rama `feat/app-streamlit` (no en `main`) y abrir un PR al terminar.
- Se ejecuta desde la raíz del proyecto; todo import es absoluto desde la raíz (`from webapp import ...`, `from etapas import ...`, `from src import ...`, `from config import ...`).
- Semilla por defecto `RANDOM_STATE = 42` (de `config.py`); split 80/20 por defecto; features por defecto = todas.
- Numeración y títulos de los pasos idénticos a `estudio_etapas.html`, salvo los añadidos: `01/2.2`, `01/2.4`, `04/3.5`. Pasos de texto excluidos: `02/6`, `03/7`, `04/4.5`, `04/6`, `05/2`, `05/3`.
- La app no muestra notas ni texto interpretativo: solo controles, figuras y tablas.
- Cero referencias a `ccp_alpha`, `postpoda`, `cost_complexity`, `pruning_path`, `plot_pruning_curve` en ningún `.py` (salvo `tests/` y `docs/`). Los árboles «completo/podado/muy podado» se quedan (prepoda).
- `estudio_etapas.html` y `estudio_etapas.md` no se modifican. No se escribe en `data/`, `figures/` ni `resultados_texto/` desde la app.
- Sin `use_container_width` (deprecado en versiones nuevas de Streamlit); sin nada que requiera red salvo `pip install`.
- El CLI debe seguir funcionando: `python etapas/e0X_*.py --lista` para las 5 etapas.

## Review Focus

1. El usuario deja sin features el paso 1, o solo deja Weight y Height y la variante B se queda vacía: debe verse un aviso claro, no una excepción (Task 5 `controles`, test en Task 7).
2. K-Means con K distinto de 7 (menor: hay clases sin cluster; mayor: hay clusters sin clase) no debe romper el mapeo húngaro ni las métricas (test en Task 4).
3. El algoritmo de Lloyd con `max_iter=1`, con convergencia en la primera iteración o con un cluster que se queda vacío no debe dar NaN ni errores (tests en Task 4).
4. Dos combinaciones distintas de hiperparámetros no deben devolver el mismo resultado cacheado (test en Task 3).
5. `max_depth = 0` significa «sin límite» y `max_features = "todas"` significa `None`; cualquier otro valor se respeta (test en Task 3).

---

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `app.py` (nuevo) | Shell: sidebar con etapas y pasos, cabecera, despacho al render del paso |
| `webapp/__init__.py` (nuevo) | Marca el paquete |
| `webapp/navegacion.py` (nuevo) | `ETAPAS`, `PASOS`: ids y títulos (idénticos al HTML) |
| `webapp/modelos.py` (nuevo) | Núcleo sin Streamlit de clasificación y regresión: split, modelos, barridos, ruido, filas reales |
| `webapp/clustering.py` (nuevo) | Núcleo sin Streamlit de K-Means: matrices, barridos, análisis, PCA, Lloyd, figuras |
| `webapp/ui.py` (nuevo) | `figura`, `tabla`, `subtitulo` de Streamlit |
| `webapp/receptor.py` (nuevo) | Receptor de `salida.RECEPTOR` para reutilizar pasos de `etapas/` |
| `webapp/controles.py` (nuevo) | Formularios y widgets compartidos; estado en `st.session_state` |
| `webapp/etapa01.py` … `etapa05.py` (nuevos) | `RENDER = {id_paso: función}` por etapa |
| `etapas/e02_clasificacion.py`, `etapas/e03_regresion.py`, `src/evaluation.py` (modificar) | Quitar postpoda |
| `requirements.txt` (modificar) | Añadir `streamlit`, `pytest` |
| `pytest.ini`, `tests/*` (nuevos) | Configuración y pruebas |

---

### Task 0: Entorno de pruebas y dependencias

**Files:**
- Modify: `requirements.txt`
- Create: `pytest.ini`, `tests/__init__.py`, `tests/conftest.py`

**Interfaces:**
- Produces: `pytest` ejecutable con `.venv-1/Scripts/python.exe -m pytest`; `streamlit` importable.

- [ ] **Step 1: Instalar dependencias en `.venv-1`**

```bash
cd /c/Users/camil/Documents/ia2-obesidad-py/ia2-obesidad-py
.venv-1/Scripts/python.exe -m pip install "streamlit>=1.40" "pytest>=8"
.venv-1/Scripts/python.exe -c "import streamlit, pytest; from streamlit.testing.v1 import AppTest; print(streamlit.__version__, pytest.__version__)"
```
Expected: imprime las dos versiones sin error.

- [ ] **Step 2: Añadir a `requirements.txt`** (al final del archivo)

```text
# App web interactiva y pruebas
streamlit>=1.40
pytest>=8
```

- [ ] **Step 3: Crear `pytest.ini`**

```ini
[pytest]
testpaths = tests
pythonpath = .
filterwarnings =
    ignore::DeprecationWarning
    ignore::FutureWarning
```

- [ ] **Step 4: Crear `tests/__init__.py` (vacío) y `tests/conftest.py`**

```python
"""Configuración común de las pruebas: backend sin ventanas y raíz del proyecto en sys.path."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
```

- [ ] **Step 5: Comprobar que pytest arranca**

Run: `.venv-1/Scripts/python.exe -m pytest -q`
Expected: `no tests ran` (código 5), sin errores de configuración.

- [ ] **Step 6: Checkpoint**

```bash
git add requirements.txt pytest.ini tests/__init__.py tests/conftest.py
git commit -m "chore: pytest y streamlit como dependencias"
```

---

### Task 1: Eliminar la postpoda

**Files:**
- Create: `tests/test_sin_postpoda.py`
- Modify: `etapas/e02_clasificacion.py` (docstring línea ~20; paso 3.5, líneas ~276-294), `etapas/e03_regresion.py` (docstring líneas ~15 y ~18; paso 3.1, líneas ~260-276; imports), `src/evaluation.py` (`plot_pruning_curve`, líneas ~129-158)
- Delete (artefactos generados): `figures/02_clasificacion/3.5_postpoda_ccp_alpha.*`, `figures/03_regresion/3.1_postpoda_ccp_alpha.*`

**Interfaces:**
- Produces: `e02.etapa.pasos` sin id `3.5`; `e03.etapa.pasos` sin id `3.1`; `src.evaluation` sin `plot_pruning_curve`.

- [ ] **Step 1: Escribir la prueba que falla** — `tests/test_sin_postpoda.py`

```python
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
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_sin_postpoda.py -v`
Expected: 2 FAILED (el primero lista las líneas con postpoda).

- [ ] **Step 3: Quitar el paso 3.5 de `etapas/e02_clasificacion.py`**

Borrar por completo el bloque que empieza en `@etapa.paso("3.5", "Postpoda por coste-complejidad ...` y termina antes de `@etapa.paso("3.6", ...` (decorador, función `paso_3_5_postpoda` y su cuerpo). Y en el docstring del archivo, cambiar la línea

```text
  1    Datos y split                     3.5  Postpoda (ccp_alpha)
```
por
```text
  1    Datos y split
```

- [ ] **Step 4: Quitar el paso 3.1 de `etapas/e03_regresion.py`**

Borrar el bloque desde `@etapa.paso("3.1", "Postpoda por coste-complejidad ...` hasta antes de `@etapa.paso("4", ...` (función `paso_3_1_postpoda`). En el docstring: la línea `- Tres árboles con prepoda, postpoda por ccp_alpha y curva del mejor K de KNN.` pasa a `- Tres árboles con prepoda y curva del mejor K de KNN.`, y la línea

```text
  1    Datos y split                       3.1  Postpoda (ccp_alpha) con R²
```
pasa a
```text
  1    Datos y split
```
En los imports de `e03_regresion.py` quitar `plot_pruning_curve` de `from src.evaluation import (...)` y, si tras el borrado `r2_score` ya no se usa en el archivo (`grep -n r2_score etapas/e03_regresion.py`), quitar también `from sklearn.metrics import r2_score`. En `e02_clasificacion.py` quitar `plot_pruning_curve` del `from src.evaluation import (...)`.

- [ ] **Step 5: Quitar `plot_pruning_curve` de `src/evaluation.py`**

Borrar la función completa `def plot_pruning_curve(alphas, score_train, score_test, ax=None, ylabel="accuracy"):` hasta antes de `def plot_confusion(`. Si `COLOR_TRAIN`/`COLOR_TEST` quedan usadas por `plot_k_curve`/`plot_n_estimators_curve` se mantienen (así es).

- [ ] **Step 6: Borrar los artefactos generados de la postpoda**

```bash
rm -f figures/02_clasificacion/3.5_postpoda_ccp_alpha.* figures/03_regresion/3.1_postpoda_ccp_alpha.*
grep -rn -i "ccp_alpha\|postpoda\|pruning" --include=*.py . --exclude-dir=.venv --exclude-dir=.venv-1 --exclude-dir=tests --exclude-dir=docs
```
Expected: el `grep` no imprime nada.

- [ ] **Step 7: Ejecutar las pruebas y comprobar el CLI**

```bash
.venv-1/Scripts/python.exe -m pytest tests/test_sin_postpoda.py -v
for e in e01_eda e02_clasificacion e03_regresion e04_clustering e05_comparacion; do .venv-1/Scripts/python.exe etapas/$e.py --lista | head -3; done
```
Expected: 2 passed; el CLI lista los pasos de cada etapa (sin 3.5 en la 02 ni 3.1 en la 03) sin errores de import.

- [ ] **Step 8: Checkpoint**

```bash
git add tests/test_sin_postpoda.py etapas/e02_clasificacion.py etapas/e03_regresion.py src/evaluation.py
git commit -m "refactor: eliminar la postpoda (ccp_alpha) de las etapas 02 y 03"
```

---

### Task 2: Registro de navegación idéntico al HTML

**Files:**
- Create: `webapp/__init__.py` (vacío), `webapp/navegacion.py`, `tests/test_navegacion.py`

**Interfaces:**
- Produces: `ETAPAS: dict[str, str]` (id de etapa → nombre) y `PASOS: dict[str, list[tuple[str, str]]]` (id de etapa → lista ordenada de `(id_paso, titulo)`). Usado por `app.py`, por los smoke tests y por las páginas.

- [ ] **Step 1: Escribir la prueba que falla** — `tests/test_navegacion.py`

```python
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
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_navegacion.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'webapp'`.

- [ ] **Step 3: Crear `webapp/__init__.py` (vacío) y `webapp/navegacion.py`**

```python
"""Etapas y pasos de la app, con la numeración y los títulos de estudio_etapas.html."""

ETAPAS = {
    "01": "EDA y preprocesamiento",
    "02": "Clasificación de NObeyesdad",
    "03": "Regresión de Weight",
    "04": "Clustering K-Means",
    "05": "Comparación final",
}

PASOS = {
    "01": [
        ("1", "Carga y EDA mínimo"),
        ("2.1", "Distribución de las 7 clases"),
        ("2.2", "Distribución de Weight"),
        ("2.3", "Chequeo de fuga de datos"),
        ("2.4", "Correlaciones entre numéricas"),
        ("3", "Preprocesamiento → clf.csv y reg.csv"),
    ],
    "02": [
        ("1", "Datos y split"),
        ("2", "Modelos y grillas"),
        ("3", "Variante A (todas las features): tabla de resultados"),
        ("3.1", "Reportes por clase y matrices de confusión (A)"),
        ("3.2", "Importancia de features del RF (A)"),
        ("3.3", "Árbol de GridSearch dibujado (A)"),
        ("3.4", "Tres árboles: completo, podado y muy podado (prepoda)"),
        ("3.6", "Curva del mejor K de KNN (A)"),
        ("3.7", "¿Cuántos árboles necesita el Random Forest?"),
        ("4", "Variante B (sin Weight ni Height)"),
        ("4.1", "Curva del mejor K de KNN (B)"),
        ("5", "Comparación A vs B"),
        ("7", "Robustez ante ruido (jittering) en la variante B"),
        ("8", "Sólo filas reales (sin SMOTE) y sin Weight/Height"),
    ],
    "03": [
        ("1", "Datos y split"),
        ("2", "Modelos, grillas y tabla de resultados"),
        ("2.1", "Curva del mejor K de KNN (regresión)"),
        ("2.2", "¿Cuántos árboles? (regresión)"),
        ("3", "Tres árboles de regresión (prepoda)"),
        ("4", "Predicho vs real (modelos principales)"),
        ("5", "Importancia de features (RF de regresión)"),
        ("6", "Comparación de métricas y R² ajustado"),
    ],
    "04": [
        ("1", "Datos escalados"),
        ("2", "Barrido de K (tabla)"),
        ("2.1", "Método del codo"),
        ("2.2", "Silueta, Calinski-Harabasz y Davies-Bouldin"),
        ("2.3", "Estadístico gap"),
        ("2.4", "¿Coinciden los 5 métodos?"),
        ("3", "K = 7 vs clases reales: métricas"),
        ("3.1", "Tabla de contingencia (clusters × clases)"),
        ("3.2", "Matriz mapeada con el algoritmo húngaro"),
        ("3.3", "Centroides en unidades originales"),
        ("3.4", "PCA 2D: clusters vs clases"),
        ("3.5", "Iteraciones de K-Means"),
        ("4", "Variantes con la columna de clase (ejercicio académico)"),
        ("4.1", "Elección de K con clase ordinal (5 métodos)"),
        ("4.2", "K = 7 con clase: contingencia cruda y matrices mapeadas"),
        ("4.3", "Tabla comparativa sin/con clase"),
        ("4.4", "PCA 2D con clase ordinal"),
        ("5", "Variante sólo numéricas (8 features)"),
    ],
    "05": [
        ("1", "Resultados de todos los casos"),
    ],
}
```

- [ ] **Step 4: Ejecutar las pruebas**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_navegacion.py -v`
Expected: 4 passed. Si `test_titulos_iguales_al_html` falla porque un título del HTML difiere (acentos, `→`), corregir `PASOS` para que sea prefijo del título del HTML; no tocar el HTML.

- [ ] **Step 5: Checkpoint**

```bash
git add webapp/__init__.py webapp/navegacion.py tests/test_navegacion.py
git commit -m "feat(app): registro de etapas y pasos igual al HTML"
```

---

### Task 3: Núcleo de modelos (clasificación y regresión)

**Files:**
- Create: `webapp/modelos.py`, `tests/test_modelos.py`

**Interfaces:**
- Consumes: `src.preprocessing.cargar_clf/cargar_reg`, `src.evaluation.clf_metrics/reg_metrics/results_table`, `etapas.e02_clasificacion` (`_modelos_robustez`, `_con_ruido`, `_modelos_reales`, `filas_reales`, `GRUPOS_3`, `METRICAS_REALES`).
- Produces (usado por `controles` y las páginas 02, 03, 05):
  - Constantes: `MODELOS_CLF = ("Árbol de decisión","Random Forest","KNN")`, `MODELOS_REG = ("Árbol de regresión","Random Forest","KNN")`, `DROP_WH = ["Weight","Height"]`, `HP_CLF`, `HP_REG` (dict nombre → dict de hiperparámetros por defecto).
  - `features_clf() -> list[str]`, `features_reg() -> list[str]`.
  - `split_clf(features, test_size, seed) -> dict(X, y, X_train, X_test, y_train, y_test)`; `split_reg(...)` igual más `n_features`. Cacheados: no mutar los resultados.
  - `construir_clf(nombre, p, seed)`, `construir_reg(nombre, p, seed)` → estimador sin entrenar.
  - `ajustar_clf(nombre, p, features, test_size, seed, con_cv=True) -> dict(modelo, cv_f1_macro, y_pred, test_accuracy, test_f1_macro)`; `ajustar_reg(...) -> dict(modelo, cv_RMSE, y_pred, test_MAE, test_RMSE, test_R2, test_R2_adj)`.
  - `tabla_clf(hp, features, test_size, seed, con_cv=True) -> (res, tabla, split)` con `res = {nombre: dict de ajustar_clf}` y `tabla` con columnas `cv_f1_macro, test_accuracy, test_f1_macro`; `tabla_reg(...)` igual, con fila inicial `"Baseline (media)"` y columnas `cv_RMSE, test_MAE, test_RMSE, test_R2, test_R2_adj`.
  - `curva_k(tarea, features, test_size, seed, ks, weights) -> {"param_knn__n_neighbors": list, "mean_test_score": list}` (formato de `plot_k_curve`; `tarea` es `"clf"` o `"reg"`; en `reg` el score es `-RMSE`).
  - `barrido_arboles(tarea, features, test_size, seed, ns, max_depth, min_samples_leaf) -> DataFrame` indexado por `n_estimators`; columnas `cv_f1_macro, cv_std, test_accuracy, test_f1_macro, tiempo_fit_s` (clf) o `cv_RMSE, cv_std, test_RMSE, test_R2, tiempo_fit_s` (reg).
  - `barrido_ruido(test_size, seed, niveles) -> DataFrame(nivel, modelo, accuracy, f1_macro)`; `niveles` es tupla de fracciones (0.0, 0.05, …).
  - `evaluar_reales(incluir_wh, n_clases, repeticiones, seed) -> DataFrame` (índice = modelos; columnas `accuracy, balanced_accuracy, f1_macro, f1_macro_std`).

- [ ] **Step 1: Escribir las pruebas que fallan** — `tests/test_modelos.py`

```python
import pytest

from webapp import modelos as M

F = tuple(M.features_clf())
FR = tuple(M.features_reg())


def test_split_estratificado_reproducible_y_cacheado():
    a, b = M.split_clf(F, 0.2, 42), M.split_clf(F, 0.2, 42)
    assert a is b
    assert len(a["X_test"]) / len(a["X"]) == pytest.approx(0.2, abs=0.01)
    assert set(a["y_train"]) == set(a["y_test"])


def test_split_reg_no_incluye_weight_ni_nobeyesdad():
    d = M.split_reg(FR, 0.2, 42)
    assert "Weight" not in d["X"].columns and "NObeyesdad" not in d["X"].columns
    assert d["n_features"] == len(FR)


def test_max_depth_cero_es_sin_limite_y_max_features_todas_es_none():
    arbol = M.construir_clf("Árbol de decisión",
                            {"criterion": "gini", "max_depth": 0, "min_samples_leaf": 1}, 42)
    assert arbol.max_depth is None
    rf = M.construir_clf("Random Forest", {"n_estimators": 10, "max_depth": 7,
                                           "min_samples_leaf": 2, "max_features": "todas"}, 42)
    assert rf.max_depth == 7 and rf.max_features is None and rf.min_samples_leaf == 2
    rf2 = M.construir_reg("Random Forest", {"n_estimators": 10, "max_depth": 0,
                                            "min_samples_leaf": 1, "max_features": "sqrt"}, 42)
    assert rf2.max_depth is None and rf2.max_features == "sqrt"


def test_parametros_distintos_dan_resultados_distintos():
    poco = M.ajustar_clf("Árbol de decisión", {"criterion": "gini", "max_depth": 2,
                                               "min_samples_leaf": 1}, F, 0.2, 42, con_cv=False)
    mucho = M.ajustar_clf("Árbol de decisión", {"criterion": "gini", "max_depth": 0,
                                                "min_samples_leaf": 1}, F, 0.2, 42, con_cv=False)
    assert poco["modelo"].get_depth() == 2
    assert mucho["modelo"].get_depth() > 2
    assert mucho["test_accuracy"] > poco["test_accuracy"]


def test_tabla_clf_tiene_los_tres_modelos_y_las_columnas_esperadas():
    res, tabla, split = M.tabla_clf(M.HP_CLF, F, 0.2, 42, con_cv=False)
    assert list(tabla.index) == list(M.MODELOS_CLF)
    assert {"cv_f1_macro", "test_accuracy", "test_f1_macro"} <= set(tabla.columns)
    assert tabla["test_accuracy"].between(0, 1).all()
    assert len(res["KNN"]["y_pred"]) == len(split["y_test"])


def test_tabla_reg_incluye_baseline_y_el_rf_lo_supera():
    _, tabla, _ = M.tabla_reg(M.HP_REG, FR, 0.2, 42, con_cv=False)
    assert tabla.index[0] == "Baseline (media)"
    assert {"cv_RMSE", "test_MAE", "test_RMSE", "test_R2", "test_R2_adj"} <= set(tabla.columns)
    assert tabla.loc["Random Forest", "test_R2"] > tabla.loc["Baseline (media)", "test_R2"]


def test_curva_k_tiene_el_formato_de_plot_k_curve():
    r = M.curva_k("clf", F, 0.2, 42, (1, 3, 5), "uniform")
    assert r["param_knn__n_neighbors"] == [1, 3, 5]
    assert len(r["mean_test_score"]) == 3
    rr = M.curva_k("reg", FR, 0.2, 42, (1, 3), "distance")
    assert all(s < 0 for s in rr["mean_test_score"])   # -RMSE


def test_barrido_arboles_clf_y_reg():
    df = M.barrido_arboles("clf", F, 0.2, 42, (10, 25), 0, 1)
    assert list(df.index) == [10, 25]
    assert {"cv_f1_macro", "cv_std", "test_f1_macro", "tiempo_fit_s"} <= set(df.columns)
    dr = M.barrido_arboles("reg", FR, 0.2, 42, (10, 25), 0, 1)
    assert {"cv_RMSE", "cv_std", "test_RMSE", "test_R2", "tiempo_fit_s"} <= set(dr.columns)


def test_barrido_ruido_incluye_el_nivel_cero_y_cuatro_modelos():
    df = M.barrido_ruido(0.2, 42, (0.0, 0.05))
    assert sorted(df["nivel"].unique()) == [0, 5]
    assert df["modelo"].nunique() == 4


def test_evaluar_reales_devuelve_la_linea_base_y_metricas():
    t = M.evaluar_reales(False, 3, 1, 42)
    assert "Línea base (clase mayoritaria)" in t.index
    assert {"accuracy", "balanced_accuracy", "f1_macro", "f1_macro_std"} <= set(t.columns)
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_modelos.py -v`
Expected: FAIL `ModuleNotFoundError`/`ImportError` de `webapp.modelos`.

- [ ] **Step 3: Escribir `webapp/modelos.py`**

```python
"""Núcleo de clasificación y regresión de la app (sin Streamlit, para poder probarlo).

Todo se cachea por parámetros con lru_cache: dos combinaciones distintas de hiperparámetros
nunca comparten resultado. Los objetos devueltos son compartidos: no se deben modificar.
"""
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (KFold, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.data import TARGET_CLF, TARGET_REG
from src.evaluation import clf_metrics, reg_metrics, results_table
from src.preprocessing import cargar_clf, cargar_reg
from etapas import e02_clasificacion as e02

MODELOS_CLF = ("Árbol de decisión", "Random Forest", "KNN")
MODELOS_REG = ("Árbol de regresión", "Random Forest", "KNN")
DROP_WH = ["Weight", "Height"]

HP_CLF = {
    "Árbol de decisión": {"criterion": "gini", "max_depth": 10, "min_samples_leaf": 1},
    "Random Forest": {"n_estimators": 200, "max_depth": 0, "min_samples_leaf": 1,
                      "max_features": "sqrt"},
    "KNN": {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"},
}
HP_REG = {
    "Árbol de regresión": {"criterion": "squared_error", "max_depth": 8, "min_samples_leaf": 5},
    "Random Forest": {"n_estimators": 200, "max_depth": 0, "min_samples_leaf": 1,
                      "max_features": "todas"},
    "KNN": {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"},
}


def _prof(v):
    """0 (o None) = sin límite de profundidad."""
    return None if not v else int(v)


def _mf(v):
    return None if v == "todas" else v


def features_clf():
    return [c for c in cargar_clf().columns if c != TARGET_CLF]


def features_reg():
    return [c for c in cargar_reg().columns if c != TARGET_REG]


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _split_clf(features, test_size, seed):
    df = cargar_clf()
    X, y = df[list(features)], df[TARGET_CLF].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    return dict(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def split_clf(features, test_size, seed):
    return _split_clf(tuple(features), float(test_size), int(seed))


@lru_cache(maxsize=32)
def _split_reg(features, test_size, seed):
    df = cargar_reg()
    X, y = df[list(features)], df[TARGET_REG]
    strata = pd.qcut(y, q=10, labels=False)   # deciles de peso, como en la etapa 03
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=strata, random_state=seed)
    return dict(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                n_features=X.shape[1])


def split_reg(features, test_size, seed):
    return _split_reg(tuple(features), float(test_size), int(seed))


# ----------------------------------------------------------------------------
# Construcción de modelos
# ----------------------------------------------------------------------------
def _knn(cls, p):
    return Pipeline([("scaler", StandardScaler()),
                     ("knn", cls(n_neighbors=int(p["n_neighbors"]), weights=p["weights"],
                                 metric=p["metric"]))])


def construir_clf(nombre, p, seed):
    if nombre == "Árbol de decisión":
        return DecisionTreeClassifier(criterion=p["criterion"], max_depth=_prof(p["max_depth"]),
                                      min_samples_leaf=int(p["min_samples_leaf"]),
                                      random_state=seed)
    if nombre == "Random Forest":
        return RandomForestClassifier(n_estimators=int(p["n_estimators"]),
                                      max_depth=_prof(p["max_depth"]),
                                      min_samples_leaf=int(p["min_samples_leaf"]),
                                      max_features=_mf(p["max_features"]),
                                      random_state=seed, n_jobs=-1)
    if nombre == "KNN":
        return _knn(KNeighborsClassifier, p)
    raise KeyError(nombre)


def construir_reg(nombre, p, seed):
    if nombre == "Árbol de regresión":
        return DecisionTreeRegressor(criterion=p["criterion"], max_depth=_prof(p["max_depth"]),
                                     min_samples_leaf=int(p["min_samples_leaf"]),
                                     random_state=seed)
    if nombre == "Random Forest":
        return RandomForestRegressor(n_estimators=int(p["n_estimators"]),
                                     max_depth=_prof(p["max_depth"]),
                                     min_samples_leaf=int(p["min_samples_leaf"]),
                                     max_features=_mf(p["max_features"]),
                                     random_state=seed, n_jobs=-1)
    if nombre == "KNN":
        return _knn(KNeighborsRegressor, p)
    raise KeyError(nombre)


# ----------------------------------------------------------------------------
# Entrenamiento (cacheado por parámetros)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=128)
def _ajustar_clf(nombre, p_json, features, test_size, seed, con_cv):
    d = _split_clf(features, test_size, seed)
    modelo = construir_clf(nombre, json.loads(p_json), seed)
    cv = float("nan")
    if con_cv:
        cv = float(cross_val_score(
            modelo, d["X_train"], d["y_train"], scoring="f1_macro",
            cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean())
    modelo.fit(d["X_train"], d["y_train"])
    y_pred = modelo.predict(d["X_test"])
    return dict(modelo=modelo, cv_f1_macro=cv, y_pred=y_pred,
                **{f"test_{k}": v for k, v in clf_metrics(d["y_test"], y_pred).items()})


def ajustar_clf(nombre, p, features, test_size, seed, con_cv=True):
    return _ajustar_clf(nombre, json.dumps(p, sort_keys=True), tuple(features),
                        float(test_size), int(seed), bool(con_cv))


@lru_cache(maxsize=128)
def _ajustar_reg(nombre, p_json, features, test_size, seed, con_cv):
    d = _split_reg(features, test_size, seed)
    modelo = construir_reg(nombre, json.loads(p_json), seed)
    cv = float("nan")
    if con_cv:
        cv = float(-cross_val_score(
            modelo, d["X_train"], d["y_train"], scoring="neg_root_mean_squared_error",
            cv=KFold(5, shuffle=True, random_state=seed)).mean())
    modelo.fit(d["X_train"], d["y_train"])
    y_pred = modelo.predict(d["X_test"])
    return dict(modelo=modelo, cv_RMSE=cv, y_pred=y_pred,
                **{f"test_{k}": v for k, v in
                   reg_metrics(d["y_test"], y_pred, n_features=d["n_features"]).items()})


def ajustar_reg(nombre, p, features, test_size, seed, con_cv=True):
    return _ajustar_reg(nombre, json.dumps(p, sort_keys=True), tuple(features),
                        float(test_size), int(seed), bool(con_cv))


def tabla_clf(hp, features, test_size, seed, con_cv=True):
    """(res, tabla, split): los tres modelos entrenados con los hiperparámetros `hp`."""
    res = {n: ajustar_clf(n, hp[n], features, test_size, seed, con_cv) for n in MODELOS_CLF}
    tabla = results_table({n: {"cv_f1_macro": r["cv_f1_macro"],
                               "test_accuracy": r["test_accuracy"],
                               "test_f1_macro": r["test_f1_macro"]} for n, r in res.items()})
    return res, tabla, split_clf(features, test_size, seed)


def tabla_reg(hp, features, test_size, seed, con_cv=True):
    """(res, tabla, split) con la fila 'Baseline (media)' al principio."""
    d = split_reg(features, test_size, seed)
    dummy = DummyRegressor(strategy="mean").fit(d["X_train"], d["y_train"])
    filas = {"Baseline (media)": {
        "cv_RMSE": float("nan"),
        **{f"test_{k}": v for k, v in reg_metrics(
            d["y_test"], dummy.predict(d["X_test"]), n_features=d["n_features"]).items()}}}
    res = {n: ajustar_reg(n, hp[n], features, test_size, seed, con_cv) for n in MODELOS_REG}
    cols = ["cv_RMSE", "test_MAE", "test_RMSE", "test_R2", "test_R2_adj"]
    for n, r in res.items():
        filas[n] = {c: r[c] for c in cols}
    return res, results_table(filas), d


# ----------------------------------------------------------------------------
# Barridos
# ----------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _curva_k(tarea, features, test_size, seed, ks, weights):
    if tarea == "clf":
        d = _split_clf(features, test_size, seed)
        cls, scoring = KNeighborsClassifier, "f1_macro"
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    else:
        d = _split_reg(features, test_size, seed)
        cls, scoring = KNeighborsRegressor, "neg_root_mean_squared_error"
        cv = KFold(5, shuffle=True, random_state=seed)
    scores = []
    for k in ks:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("knn", cls(n_neighbors=int(k), weights=weights))])
        scores.append(float(cross_val_score(pipe, d["X_train"], d["y_train"],
                                            scoring=scoring, cv=cv).mean()))
    return {"param_knn__n_neighbors": list(ks), "mean_test_score": scores}


def curva_k(tarea, features, test_size, seed, ks, weights):
    return _curva_k(tarea, tuple(features), float(test_size), int(seed), tuple(ks), weights)


@lru_cache(maxsize=16)
def _barrido_arboles(tarea, features, test_size, seed, ns, max_depth, min_samples_leaf):
    clf = tarea == "clf"
    d = (_split_clf if clf else _split_reg)(features, test_size, seed)
    cv = (StratifiedKFold if clf else KFold)(5, shuffle=True, random_state=seed)
    filas = []
    for n in ns:
        kw = dict(n_estimators=int(n), max_depth=_prof(max_depth),
                  min_samples_leaf=int(min_samples_leaf), random_state=seed, n_jobs=-1)
        rf = RandomForestClassifier(**kw) if clf else RandomForestRegressor(**kw)
        scoring = "f1_macro" if clf else "neg_root_mean_squared_error"
        s = cross_val_score(rf, d["X_train"], d["y_train"], cv=cv, scoring=scoring)
        t0 = time.perf_counter()
        rf.fit(d["X_train"], d["y_train"])
        t_fit = time.perf_counter() - t0
        if clf:
            m = clf_metrics(d["y_test"], rf.predict(d["X_test"]))
            filas.append({"n_estimators": n, "cv_f1_macro": s.mean(), "cv_std": s.std(),
                          "test_accuracy": m["accuracy"], "test_f1_macro": m["f1_macro"],
                          "tiempo_fit_s": t_fit})
        else:
            m = reg_metrics(d["y_test"], rf.predict(d["X_test"]), n_features=d["n_features"])
            filas.append({"n_estimators": n, "cv_RMSE": -s.mean(), "cv_std": s.std(),
                          "test_RMSE": m["RMSE"], "test_R2": m["R2"], "tiempo_fit_s": t_fit})
    return pd.DataFrame(filas).set_index("n_estimators")


def barrido_arboles(tarea, features, test_size, seed, ns, max_depth, min_samples_leaf):
    return _barrido_arboles(tarea, tuple(features), float(test_size), int(seed),
                            tuple(ns), int(max_depth), int(min_samples_leaf))


@lru_cache(maxsize=8)
def _barrido_ruido(test_size, seed, niveles):
    d = _split_clf(tuple(features_clf()), test_size, seed)
    X_tr, X_te = d["X_train"].drop(columns=DROP_WH), d["X_test"].drop(columns=DROP_WH)
    filas = []
    for nl in niveles:
        Xa, Xb = e02._con_ruido(X_tr, X_te, nl) if nl > 0 else (X_tr, X_te)
        for nombre, modelo in e02._modelos_robustez().items():
            modelo.fit(Xa, d["y_train"])
            p = modelo.predict(Xb)
            filas.append({"nivel": int(round(nl * 100)), "modelo": nombre,
                          "accuracy": accuracy_score(d["y_test"], p),
                          "f1_macro": f1_score(d["y_test"], p, average="macro")})
    return pd.DataFrame(filas)


def barrido_ruido(test_size, seed, niveles):
    return _barrido_ruido(float(test_size), int(seed), tuple(float(n) for n in niveles))


@lru_cache(maxsize=16)
def _evaluar_reales(incluir_wh, n_clases, repeticiones, seed):
    df = cargar_clf()
    X, y = df[features_clf()], df[TARGET_CLF].astype(str)
    real = e02.filas_reales(X)
    X, y = X[real], y[real]
    if not incluir_wh:
        X = X.drop(columns=DROP_WH)
    if n_clases == 3:
        y = y.map(e02.GRUPOS_3)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=repeticiones, random_state=seed)
    filas = {}
    for nombre, modelo in e02._modelos_reales().items():
        s = cross_validate(modelo, X, y, cv=cv, scoring=e02.METRICAS_REALES, n_jobs=-1)
        filas[nombre] = {**{m: s[f"test_{m}"].mean() for m in e02.METRICAS_REALES},
                         "f1_macro_std": s["test_f1_macro"].std()}
    return pd.DataFrame(filas).T


def evaluar_reales(incluir_wh, n_clases, repeticiones, seed):
    return _evaluar_reales(bool(incluir_wh), int(n_clases), int(repeticiones), int(seed))
```

- [ ] **Step 4: Ejecutar las pruebas**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_modelos.py -v`
Expected: 10 passed (tarda 1-2 minutos por los barridos y la CV repetida).

- [ ] **Step 5: Checkpoint**

```bash
git add webapp/modelos.py tests/test_modelos.py
git commit -m "feat(app): núcleo de modelos parametrizable para clasificación y regresión"
```

---

### Task 4: Núcleo de clustering (K-Means, análisis y algoritmo de Lloyd)

**Files:**
- Create: `webapp/clustering.py`, `tests/test_clustering.py`

**Interfaces:**
- Consumes: `etapas.e04_clustering` (`gap_statistic`, `knee_point`, `_heatmap_ct`, `_heatmap_cm`, `analizar_clustering`, `datos`, `km7` solo en tests), `src.preprocessing` (`CLASS_ORDER`, `NUMERIC_COLS`, `cargar_clf`).
- Produces (usado por la página 04 y la 05):
  - `matriz(variante) -> dict(X, X_scaled, scaler, y_true)`; `variante` ∈ `"base"` (20 features sin clase), `"num"` (8 numéricas), `"ord"` (base + clase ordinal), `"oh"` (base + clase one-hot).
  - `ajustar(variante, k, n_init, init, seed) -> KMeans` entrenado (`init` ∈ `"k-means++"`, `"random"`).
  - `analizar(variante, labels) -> dict(ct, cm, mapping, metricas)`; `metricas` tiene `silueta, ARI, NMI, acc_mapeo, pureza`. Funciona con cualquier K.
  - `barrido(variante, k_min, k_max, n_init, init, seed) -> dict(metrics: DataFrame indexado por K con inercia, silhouette, calinski_harabasz, davies_bouldin; ks: list[int])`.
  - `gap(variante, k_min, k_max, n_refs, seed) -> (DataFrame gap/s_k, k_gap|None)`.
  - `resumen_k(b, g) -> Series` con los 5 métodos (`"Codo (inercia)"`, `"Silueta (max)"`, `"Calinski-Harabasz (max)"`, `"Davies-Bouldin (min)"`, `"Gap statistic"`), nombre `"K óptimo"`.
  - `pca2d(variante) -> (Z, pca)`; `fig_pca(variante, labels, titulo) -> Figure`.
  - `lloyd(X, k, init="k-means++", seed=42, max_iter=50, centros_iniciales=None) -> (pasos, convergio)`; cada paso es `dict(centroides, etiquetas, inercia, movimiento)`.
  - `lloyd_cacheado(variante, k, init, seed, max_iter) -> (pasos, convergio, Z, Zc_all)`.
  - `fig_iteracion(Z, Zc_all, pasos, i, k) -> Figure`.
  - `fig_cinco_metodos(b, g) -> Figure`.
  - `heatmap_ct(ct, ax, titulo)`, `heatmap_cm(cm, ax, titulo)` (re-exportados de la etapa 04).

- [ ] **Step 1: Escribir las pruebas que fallan** — `tests/test_clustering.py`

```python
import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from webapp import clustering as CL

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.2, random_state=0)


def test_lloyd_la_inercia_no_crece_y_converge():
    pasos, convergio = CL.lloyd(X, 4, "k-means++", 0, 50)
    inercias = [p["inercia"] for p in pasos]
    assert all(b <= a + 1e-9 for a, b in zip(inercias, inercias[1:]))
    assert convergio and len(pasos) >= 2
    assert np.array_equal(pasos[-1]["etiquetas"], pasos[-2]["etiquetas"])


def test_lloyd_coincide_con_sklearn_desde_los_mismos_centros():
    pasos, _ = CL.lloyd(X, 4, "random", 1, 200)
    km = KMeans(n_clusters=4, init=pasos[0]["centroides"], n_init=1, max_iter=500,
                tol=0, algorithm="lloyd").fit(X)
    assert adjusted_rand_score(pasos[-1]["etiquetas"], km.labels_) == pytest.approx(1.0)
    assert pasos[-1]["inercia"] == pytest.approx(km.inertia_, rel=1e-6)


def test_lloyd_con_max_iter_1_devuelve_un_solo_paso_sin_converger():
    pasos, convergio = CL.lloyd(X, 4, "random", 0, 1)
    assert len(pasos) == 1 and convergio is False


def test_lloyd_converge_en_la_primera_iteracion_si_los_centros_ya_son_estables():
    pasos0, _ = CL.lloyd(X, 4, "k-means++", 0, 200)
    pasos, convergio = CL.lloyd(X, 4, max_iter=50, centros_iniciales=pasos0[-1]["centroides"])
    assert convergio and len(pasos) == 2


def test_lloyd_un_cluster_vacio_conserva_su_centroide():
    centros = np.vstack([X[0], X[1], [1e6, 1e6]])      # el tercero no captura ningún punto
    pasos, _ = CL.lloyd(X, 3, max_iter=5, centros_iniciales=centros)
    for p in pasos:
        assert np.isfinite(p["centroides"]).all() and np.isfinite(p["inercia"])
    assert np.allclose(pasos[-1]["centroides"][2], [1e6, 1e6])


@pytest.mark.parametrize("k", [3, 7, 10])
def test_analizar_acepta_cualquier_k(k):
    km = CL.ajustar("base", k, 3, "k-means++", 42)
    a = CL.analizar("base", km.labels_)
    assert a["cm"].shape == (7, 7)
    assert 0 <= a["metricas"]["acc_mapeo"] <= 1
    assert 0 <= a["metricas"]["pureza"] <= 1


def test_analizar_con_k7_coincide_con_la_etapa_04():
    from etapas import e04_clustering as e04
    mio = CL.analizar("base", CL.ajustar("base", 7, 10, "k-means++", 42).labels_)["metricas"]
    orig = e04.analizar_clustering(e04.datos()["X_scaled"], e04.km7().labels_)["metricas"]
    for clave, valor in orig.items():
        assert mio[clave] == pytest.approx(valor)


def test_matriz_dimensiones_de_las_cuatro_variantes():
    assert CL.matriz("base")["X_scaled"].shape[1] == 20
    assert CL.matriz("num")["X_scaled"].shape[1] == 8
    assert CL.matriz("ord")["X_scaled"].shape[1] == 21
    assert CL.matriz("oh")["X_scaled"].shape[1] == 27


def test_barrido_gap_y_resumen():
    b = CL.barrido("num", 2, 6, 3, "k-means++", 42)
    assert b["ks"] == [2, 3, 4, 5, 6] and list(b["metrics"].index) == b["ks"]
    g = CL.gap("num", 2, 6, 3, 42)
    r = CL.resumen_k(b, g)
    assert list(r.index) == ["Codo (inercia)", "Silueta (max)", "Calinski-Harabasz (max)",
                             "Davies-Bouldin (min)", "Gap statistic"]


def test_figuras_se_generan():
    km = CL.ajustar("base", 4, 3, "k-means++", 42)
    assert CL.fig_pca("base", km.labels_, "t") is not None
    pasos, conv, Z, Zc = CL.lloyd_cacheado("base", 4, "k-means++", 42, 5)
    assert CL.fig_iteracion(Z, Zc, pasos, len(pasos) - 1, 4) is not None
    b = CL.barrido("base", 2, 5, 2, "k-means++", 42)
    assert CL.fig_cinco_metodos(b, CL.gap("base", 2, 5, 3, 42)) is not None
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_clustering.py -v`
Expected: FAIL `ImportError: cannot import name 'clustering' from 'webapp'`.

- [ ] **Step 3: Escribir `webapp/clustering.py`**

```python
"""Núcleo de clustering de la app (sin Streamlit): matrices, barridos, análisis, PCA y Lloyd."""
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, confusion_matrix,
                             davies_bouldin_score, normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE
from src.data import TARGET_CLF
from src.preprocessing import CLASS_ORDER, NUMERIC_COLS, cargar_clf
from etapas.e04_clustering import _heatmap_cm as heatmap_cm
from etapas.e04_clustering import _heatmap_ct as heatmap_ct
from etapas.e04_clustering import gap_statistic, knee_point

VARIANTES = ("base", "num", "ord", "oh")


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _base():
    df = cargar_clf()
    y = pd.Categorical(df[TARGET_CLF], categories=CLASS_ORDER)
    return df.drop(columns=[TARGET_CLF]), y


@lru_cache(maxsize=4)
def matriz(variante):
    """'base' = 20 features sin la clase; 'num' = 8 numéricas; 'ord' = base + clase ordinal;
    'oh' = base + clase one-hot. Siempre escaladas con StandardScaler."""
    X, y = _base()
    if variante == "base":
        Xv = X
    elif variante == "num":
        Xv = X[NUMERIC_COLS]
    elif variante == "ord":
        Xv = X.assign(NObeyesdad_ord=pd.Categorical(y, categories=CLASS_ORDER, ordered=True).codes)
    elif variante == "oh":
        Xv = pd.concat([X, pd.get_dummies(y, prefix="clase", dtype="int8")], axis=1)
    else:
        raise KeyError(variante)
    scaler = StandardScaler()
    return dict(X=Xv, X_scaled=scaler.fit_transform(Xv), scaler=scaler, y_true=y)


@lru_cache(maxsize=64)
def ajustar(variante, k, n_init, init, seed):
    return KMeans(n_clusters=int(k), n_init=int(n_init), init=init,
                  random_state=int(seed)).fit(matriz(variante)["X_scaled"])


def analizar(variante, labels):
    """Contingencia, mapeo húngaro cluster->clase, matriz de confusión y métricas.
    Funciona con cualquier K: los clusters que sobran quedan 'sin_clase' (cuentan como error)."""
    m = matriz(variante)
    y_true = m["y_true"]
    ct = pd.crosstab(pd.Series(y_true, name="Clase real"),
                     pd.Series(labels, name="Cluster")).reindex(CLASS_ORDER)
    filas, cols = linear_sum_assignment(-ct.values)
    mapping = {int(ct.columns[c]): ct.index[r] for r, c in zip(filas, cols)}
    pred = pd.Series(labels).map(mapping).fillna("sin_clase").to_numpy()
    cm = confusion_matrix(np.asarray(y_true.astype(str)), pred, labels=CLASS_ORDER)
    return {
        "ct": ct, "cm": cm, "mapping": mapping,
        "metricas": {
            "silueta": silhouette_score(m["X_scaled"], labels),
            "ARI": adjusted_rand_score(y_true, labels),
            "NMI": normalized_mutual_info_score(y_true, labels),
            "acc_mapeo": float(np.trace(cm) / len(y_true)),
            "pureza": float(ct.max(axis=0).sum() / ct.values.sum()),
        },
    }


# ----------------------------------------------------------------------------
# Elección de K
# ----------------------------------------------------------------------------
@lru_cache(maxsize=16)
def barrido(variante, k_min, k_max, n_init, init, seed):
    X = matriz(variante)["X_scaled"]
    ks = list(range(int(k_min), int(k_max) + 1))
    filas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=int(n_init), init=init, random_state=int(seed)).fit(X)
        filas.append({"K": k, "inercia": km.inertia_,
                      "silhouette": silhouette_score(X, km.labels_),
                      "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
                      "davies_bouldin": davies_bouldin_score(X, km.labels_)})
    return dict(metrics=pd.DataFrame(filas).set_index("K"), ks=ks)


@lru_cache(maxsize=16)
def gap(variante, k_min, k_max, n_refs, seed):
    ks = list(range(int(k_min), int(k_max) + 1))
    return gap_statistic(matriz(variante)["X_scaled"], ks, n_refs=int(n_refs), seed=int(seed))


def resumen_k(b, g):
    m = b["metrics"]
    return pd.Series({
        "Codo (inercia)": knee_point(b["ks"], m["inercia"].values),
        "Silueta (max)": int(m["silhouette"].idxmax()),
        "Calinski-Harabasz (max)": int(m["calinski_harabasz"].idxmax()),
        "Davies-Bouldin (min)": int(m["davies_bouldin"].idxmin()),
        "Gap statistic": g[1],
    }, name="K óptimo")


def fig_cinco_metodos(b, g):
    """Panel 2x3 con los cinco métodos de elección de K (como el paso 4.1 de la etapa 04)."""
    m, (gap_df, k_gap) = b["metrics"], g
    r = resumen_k(b, g)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    paneles = [
        (axes[0, 0], "inercia", r["Codo (inercia)"], "codo", "1. Método del codo", "Inercia (WCSS)"),
        (axes[0, 1], "silhouette", r["Silueta (max)"], "max", "2. Coeficiente de silueta", "Silueta"),
        (axes[0, 2], "calinski_harabasz", r["Calinski-Harabasz (max)"], "max",
         "3. Calinski-Harabasz", "Calinski-Harabasz"),
        (axes[1, 0], "davies_bouldin", r["Davies-Bouldin (min)"], "min",
         "4. Davies-Bouldin", "Davies-Bouldin"),
    ]
    for ax, col, best, sentido, titulo, ylabel in paneles:
        ax.plot(m.index, m[col], marker="o", color="steelblue")
        ax.axvline(best, c="r", ls="--", label=f"{sentido}: K={best}")
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        ax.set_title(titulo)
        ax.legend()
    ax = axes[1, 1]
    ax.errorbar(gap_df.index, gap_df["gap"], yerr=gap_df["s_k"], marker="o", capsize=3,
                color="steelblue")
    if k_gap is not None:
        ax.axvline(k_gap, c="r", ls="--", label=f"gap: K={k_gap}")
        ax.legend()
    ax.set_xlabel("K")
    ax.set_ylabel("gap(K)")
    ax.set_title("5. Estadístico gap")
    axes[1, 2].axis("off")
    return fig


# ----------------------------------------------------------------------------
# PCA 2D
# ----------------------------------------------------------------------------
@lru_cache(maxsize=4)
def pca2d(variante):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(matriz(variante)["X_scaled"]), pca


def fig_pca(variante, labels, titulo):
    Z, pca = pca2d(variante)
    y_true = matriz(variante)["y_true"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=labels, palette="tab10", s=12, ax=axes[0],
                    legend="full")
    axes[0].set_title(titulo)
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y_true.astype(str), hue_order=CLASS_ORDER,
                    palette="tab10", s=12, ax=axes[1])
    axes[1].set_title("Clases reales NObeyesdad")
    r = pca.explained_variance_ratio_
    for ax in axes:
        ax.set_xlabel(f"PC1 ({r[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({r[1] * 100:.1f}%)")
        ax.legend(fontsize=7, markerscale=1.5)
    return fig


# ----------------------------------------------------------------------------
# Algoritmo de Lloyd paso a paso
# ----------------------------------------------------------------------------
def lloyd(X, k, init="k-means++", seed=RANDOM_STATE, max_iter=50, centros_iniciales=None):
    """Paso i = (centroides c_i, asignación de cada punto a su c_i más cercano, inercia).
    El paso siguiente recalcula cada centroide como la media de su grupo (un cluster vacío
    conserva su centroide). Termina cuando las etiquetas no cambian (convergió) o al llegar a
    `max_iter` pasos. Devuelve (pasos, convergio); cada paso es un dict con centroides,
    etiquetas, inercia y movimiento (desplazamiento máximo de un centroide respecto al paso
    anterior)."""
    X = np.asarray(X, dtype=float)
    if centros_iniciales is not None:
        centros = np.array(centros_iniciales, dtype=float)
    elif init == "k-means++":
        centros = np.array(kmeans_plusplus(X, k, random_state=seed)[0], dtype=float)
    else:
        rng = np.random.default_rng(seed)
        centros = X[rng.choice(len(X), size=k, replace=False)].astype(float)
    k = len(centros)
    pasos, previas, convergio = [], None, False
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2)
        etiquetas = d2.argmin(axis=1)
        movimiento = (float(np.linalg.norm(centros - pasos[-1]["centroides"], axis=1).max())
                      if pasos else 0.0)
        pasos.append(dict(centroides=centros.copy(), etiquetas=etiquetas,
                          inercia=float(d2[np.arange(len(X)), etiquetas].sum()),
                          movimiento=movimiento))
        if previas is not None and np.array_equal(etiquetas, previas):
            convergio = True
            break
        previas = etiquetas
        nuevos = centros.copy()
        for j in range(k):
            miembros = X[etiquetas == j]
            if len(miembros):
                nuevos[j] = miembros.mean(axis=0)
        centros = nuevos
    return pasos, convergio


@lru_cache(maxsize=16)
def lloyd_cacheado(variante, k, init, seed, max_iter):
    """Lloyd sobre una variante + la proyección PCA 2D de los puntos y de los centroides."""
    pasos, convergio = lloyd(matriz(variante)["X_scaled"], int(k), init, int(seed), int(max_iter))
    Z, pca = pca2d(variante)
    return pasos, convergio, Z, [pca.transform(p["centroides"]) for p in pasos]


def fig_iteracion(Z, Zc_all, pasos, i, k):
    """Izquierda: puntos coloreados por cluster y centroides (X) con su trayectoria hasta el
    paso i. Derecha: inercia por iteración con el paso actual marcado."""
    paso = pasos[i]
    vmax = max(len(Zc_all[0]) - 1, 1)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax.scatter(Z[:, 0], Z[:, 1], c=paso["etiquetas"], cmap="tab20", vmin=0, vmax=vmax,
               s=10, alpha=0.6)
    for j in range(len(Zc_all[0])):
        tray = np.array([zc[j] for zc in Zc_all[:i + 1]])
        ax.plot(tray[:, 0], tray[:, 1], "-", color="black", lw=1, alpha=0.6)
    actual = Zc_all[i]
    ax.scatter(actual[:, 0], actual[:, 1], c=range(len(actual)), cmap="tab20", vmin=0, vmax=vmax,
               marker="X", s=240, edgecolors="black", linewidths=1.6, zorder=3)
    ax.set_xlim(Z[:, 0].min() - 0.5, Z[:, 0].max() + 0.5)
    ax.set_ylim(Z[:, 1].min() - 0.5, Z[:, 1].max() + 0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"K = {k} · iteración {i} de {len(pasos) - 1}")

    ine = [p["inercia"] for p in pasos]
    ax2.plot(range(len(ine)), ine, marker="o", color="steelblue")
    ax2.scatter([i], [ine[i]], s=120, color="red", zorder=3)
    ax2.set_xlabel("Iteración")
    ax2.set_ylabel("Inercia (WCSS)")
    ax2.set_title("Inercia por iteración")
    ax2.grid(alpha=0.3)
    return fig
```

- [ ] **Step 4: Ejecutar las pruebas**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_clustering.py -v`
Expected: 13 passed (los de `analizar`/`barrido` tardan unos segundos).

- [ ] **Step 5: Checkpoint**

```bash
git add webapp/clustering.py tests/test_clustering.py
git commit -m "feat(app): núcleo de clustering con algoritmo de Lloyd paso a paso"
```

---

### Task 5: Infraestructura de UI, controles y shell `app.py` (con smoke test en rojo)

**Files:**
- Create: `webapp/ui.py`, `webapp/receptor.py`, `webapp/controles.py`, `webapp/etapa01.py` … `webapp/etapa05.py` (esqueletos con `RENDER = {}`), `app.py`, `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `webapp.navegacion` (Task 2), `webapp.modelos` (Task 3).
- Produces:
  - `ui.figura(fig)`, `ui.tabla(df_o_serie, titulo=None, decimales=4)`, `ui.subtitulo(texto)`.
  - `receptor.ejecutar(fn)`: corre un paso de `etapas/` mostrando sus figuras y tablas en Streamlit (stdout descartado, sin escribir archivos).
  - `controles`: `leer(clave, defaults)`, `formulario(clave, construir, defaults, etiqueta="Entrenar")`, `defaults_split(tarea)`, `params_split(tarea)`, `form_split(tarea)`, `features_b(p)`, `params_hp(tarea)`, `form_hp(tarea)`, `form_arboles(tarea)`, `form_curva_k(tarea)`, `form_n_arboles(tarea)`, `form_ruido()`, `form_reales()`, `form_clu(rango=False, k=False)`, `form_lloyd()`. `tarea` ∈ `"clf"`, `"reg"`.
  - Cada `webapp/etapaNN.py` expone `RENDER: dict[str, Callable[[], None]]` (id de paso → función sin argumentos).
  - `app.py`: sidebar con `st.radio(key="etapa")` y `st.radio(key=f"paso_{etapa}")`; cabecera `st.subheader(f"{paso} · {titulo}")`.

- [ ] **Step 1: Escribir el smoke test** — `tests/test_app_smoke.py`

```python
"""Cada paso de la app se renderiza con sus parámetros por defecto sin excepciones ni errores."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from webapp.navegacion import PASOS

RAIZ = Path(__file__).resolve().parents[1]
CASOS = [(e, i) for e, lista in PASOS.items() for i, _ in lista]


def abrir(etapa, paso, **estado):
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=600)
    at.session_state["etapa"] = etapa
    at.session_state[f"paso_{etapa}"] = paso
    for clave, valor in estado.items():
        at.session_state[clave] = valor
    return at.run()


@pytest.mark.parametrize("etapa,paso", CASOS, ids=[f"{e}-{i}" for e, i in CASOS])
def test_paso_se_renderiza(etapa, paso):
    at = abrir(etapa, paso)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.subheader[0].value.startswith(paso)
```

- [ ] **Step 2: Escribir `webapp/ui.py`**

```python
"""Salida de la app: figuras y tablas en Streamlit."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def figura(fig):
    st.pyplot(fig)
    plt.close(fig)


def subtitulo(texto):
    st.markdown(f"##### {texto}")


def tabla(df, titulo=None, decimales=4):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.round(decimales).copy()
    for c in df.columns:               # evita columnas de tipos mezclados (Arrow)
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    if titulo:
        subtitulo(titulo)
    st.dataframe(df)
```

- [ ] **Step 3: Escribir `webapp/receptor.py`**

```python
"""Conecta los pasos de etapas/ (que escriben con src.salida) con Streamlit.

src/salida.py ya prevé un RECEPTOR: si está definido, tabla() y mostrar() le entregan lo
que producen en lugar de imprimir/guardar. Aquí se implementa para Streamlit. Los títulos
y las notas de los pasos se descartan (la app solo muestra figuras y tablas).
"""
import contextlib
import io

from src import salida
from webapp import ui


class ReceptorStreamlit:
    def titulo(self, texto):
        pass

    def subtitulo(self, texto):
        ui.subtitulo(texto)

    def nota(self, texto):
        pass

    def tabla(self, df, titulo=None):
        ui.tabla(df, titulo)

    def figura(self, fig, nombre=None):
        ui.figura(fig)


@contextlib.contextmanager
def activo():
    # OPCIONES se toca directamente (no con salida.configurar): configurar(mostrar=True)
    # lanzaría un subproceso de prueba de QtAgg dentro del servidor de Streamlit.
    previo, opciones = salida.RECEPTOR, dict(salida.OPCIONES)
    salida.OPCIONES.update(mostrar=False, guardar=False, graficar_tablas=False)
    salida.RECEPTOR = ReceptorStreamlit()
    try:
        yield
    finally:
        salida.RECEPTOR = previo
        salida.OPCIONES.update(opciones)


def ejecutar(fn):
    """Ejecuta un paso de etapas/ y muestra sus figuras y tablas; su stdout se descarta."""
    with activo(), contextlib.redirect_stdout(io.StringIO()):
        fn()
```

- [ ] **Step 4: Escribir `webapp/controles.py`**

```python
"""Formularios y widgets compartidos. Los parámetros aplicados de cada formulario viven en
st.session_state['params::<clave>'] y los comparten todos los pasos que usan esa clave."""
import copy

import streamlit as st

from webapp import modelos as M

DEF_CLU = {"k_min": 2, "k_max": 12, "n_init": 10, "init": "k-means++", "seed": 42, "k": 7}
DEF_LLOYD = {"k": 7, "init": "k-means++", "seed": 42, "max_iter": 30}
DEF_CURVA_K = {"k_min": 1, "k_max": 31, "paso": 2, "weights": "uniform"}
DEF_N_ARBOLES = {"ns": [10, 25, 50, 100, 200], "max_depth": 0, "min_samples_leaf": 1}
DEF_ARBOLES = {
    "clf": {"Completo": {"max_depth": 0, "min_samples_leaf": 1},
            "Podado": {"max_depth": 6, "min_samples_leaf": 5},
            "Muy podado": {"max_depth": 3, "min_samples_leaf": 10}},
    "reg": {"Completo": {"max_depth": 0, "min_samples_leaf": 1},
            "Podado": {"max_depth": 8, "min_samples_leaf": 5},
            "Muy podado": {"max_depth": 3, "min_samples_leaf": 20}},
}
DEF_RUIDO = {"max": 20, "paso": 5}
DEF_REALES = {"incluir_wh": False, "repeticiones": 3}
CRITERIOS = {"clf": ["gini", "entropy", "log_loss"], "reg": ["squared_error", "friedman_mse"]}
N_ARBOLES_OPCIONES = [10, 25, 50, 100, 200, 400, 800]


def _clave(c):
    return f"params::{c}"


def leer(clave, defaults):
    """Parámetros vigentes de `clave`: los últimos aplicados o, si no hay, los de por defecto."""
    return copy.deepcopy(st.session_state.get(_clave(clave), defaults))


def formulario(clave, construir, defaults, etiqueta="Entrenar"):
    """Dibuja un st.form. `construir(p)` crea los widgets (con `p` como valores iniciales) y
    devuelve un dict. Al pulsar el botón se guardan y se devuelven los nuevos parámetros;
    mientras tanto se devuelven los últimos aplicados."""
    guardado = leer(clave, defaults)
    with st.form(f"form::{clave}"):
        nuevo = construir(guardado)
        enviado = st.form_submit_button(etiqueta)
    if enviado:
        st.session_state[_clave(clave)] = nuevo
        return nuevo
    return guardado


def _parar(nivel, mensaje):
    getattr(st, nivel)(mensaje)
    st.stop()


# ----------------------------------------------------------------------------
# Split y features
# ----------------------------------------------------------------------------
def _todas(tarea):
    return M.features_clf() if tarea == "clf" else M.features_reg()


def defaults_split(tarea):
    return {"features": _todas(tarea), "test_size": 0.2, "seed": 42}


def params_split(tarea):
    """Split vigente, sin dibujar nada. Para si no hay ninguna feature elegida."""
    p = leer(f"split_{tarea}", defaults_split(tarea))
    if not p["features"]:
        _parar("error", "Elige al menos una feature en el paso 1 de esta etapa.")
    return p


def form_split(tarea):
    todas = _todas(tarea)

    def construir(p):
        return {
            "features": st.multiselect("Features", todas, key=f"split_{tarea}_features",
                                       default=[f for f in p["features"] if f in todas]),
            "test_size": st.slider("Tamaño del test", 0.1, 0.5, float(p["test_size"]), 0.05,
                                   key=f"split_{tarea}_test"),
            "seed": int(st.number_input("Semilla", 0, 9999, int(p["seed"]),
                                        key=f"split_{tarea}_seed")),
        }

    p = formulario(f"split_{tarea}", construir, defaults_split(tarea), "Aplicar")
    if not p["features"]:
        _parar("error", "Elige al menos una feature.")
    return p


def features_b(p):
    """Features de la variante B (sin Weight ni Height). Avisa y para si no queda ninguna."""
    fb = [f for f in p["features"] if f not in M.DROP_WH]
    if not fb:
        _parar("warning", "La variante B (sin Weight ni Height) se queda sin features: "
                          "añade otras en el paso 1.")
    return fb


# ----------------------------------------------------------------------------
# Hiperparámetros
# ----------------------------------------------------------------------------
def params_hp(tarea):
    return leer(f"hp_{tarea}", M.HP_CLF if tarea == "clf" else M.HP_REG)


def form_hp(tarea):
    nombres = M.MODELOS_CLF if tarea == "clf" else M.MODELOS_REG
    defaults = M.HP_CLF if tarea == "clf" else M.HP_REG
    crit, mf, pesos = CRITERIOS[tarea], ["sqrt", "log2", "todas"], ["uniform", "distance"]
    metricas = ["euclidean", "manhattan", "chebyshev"]

    def construir(p):
        a, rf, kn = nombres
        k = f"hp_{tarea}"
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**{a}**")
            arbol = {
                "criterion": st.selectbox("criterion", crit, index=crit.index(p[a]["criterion"]),
                                          key=f"{k}_a_crit"),
                "max_depth": st.slider("max_depth (0 = sin límite)", 0, 30,
                                       int(p[a]["max_depth"]), key=f"{k}_a_prof"),
                "min_samples_leaf": st.slider("min_samples_leaf", 1, 50,
                                              int(p[a]["min_samples_leaf"]), key=f"{k}_a_hoja"),
            }
        with c2:
            st.markdown(f"**{rf}**")
            bosque = {
                "n_estimators": st.slider("n_estimators", 10, 800, int(p[rf]["n_estimators"]),
                                          10, key=f"{k}_r_n"),
                "max_depth": st.slider("max_depth (0 = sin límite)", 0, 30,
                                       int(p[rf]["max_depth"]), key=f"{k}_r_prof"),
                "min_samples_leaf": st.slider("min_samples_leaf", 1, 50,
                                              int(p[rf]["min_samples_leaf"]), key=f"{k}_r_hoja"),
                "max_features": st.selectbox("max_features", mf,
                                             index=mf.index(p[rf]["max_features"]),
                                             key=f"{k}_r_mf"),
            }
        with c3:
            st.markdown("**KNN**")
            vecinos = {
                "n_neighbors": st.slider("n_neighbors (K)", 1, 51, int(p[kn]["n_neighbors"]),
                                         key=f"{k}_k_n"),
                "weights": st.selectbox("weights", pesos, index=pesos.index(p[kn]["weights"]),
                                        key=f"{k}_k_w"),
                "metric": st.selectbox("metric", metricas, index=metricas.index(p[kn]["metric"]),
                                       key=f"{k}_k_m"),
            }
        return {a: arbol, rf: bosque, kn: vecinos}

    with st.expander("Hiperparámetros de los modelos", expanded=True):
        return formulario(f"hp_{tarea}", construir, defaults)


def form_arboles(tarea):
    def construir(p):
        out = {}
        for col, nombre in zip(st.columns(3), ("Completo", "Podado", "Muy podado")):
            with col:
                st.markdown(f"**{nombre}**")
                out[nombre] = {
                    "max_depth": st.slider("max_depth (0 = sin límite)", 0, 30,
                                           int(p[nombre]["max_depth"]),
                                           key=f"arb_{tarea}_{nombre}_prof"),
                    "min_samples_leaf": st.slider("min_samples_leaf", 1, 50,
                                                  int(p[nombre]["min_samples_leaf"]),
                                                  key=f"arb_{tarea}_{nombre}_hoja"),
                }
        return out

    with st.expander("Parámetros de los tres árboles", expanded=True):
        return formulario(f"arboles_{tarea}", construir, DEF_ARBOLES[tarea])


def form_curva_k(tarea):
    pesos = ["uniform", "distance"]

    def construir(p):
        lo, hi = st.slider("Rango de K", 1, 51, (int(p["k_min"]), int(p["k_max"])),
                           key=f"curvak_{tarea}_rango")
        return {"k_min": lo, "k_max": hi,
                "paso": st.selectbox("Paso entre valores de K", [1, 2],
                                     index=[1, 2].index(int(p["paso"])), key=f"curvak_{tarea}_paso"),
                "weights": st.selectbox("weights", pesos, index=pesos.index(p["weights"]),
                                        key=f"curvak_{tarea}_w")}

    with st.expander("Parámetros de la curva de K", expanded=True):
        return formulario(f"curvak_{tarea}", construir, DEF_CURVA_K, "Calcular")


def form_n_arboles(tarea):
    def construir(p):
        return {"ns": st.multiselect("Números de árboles", N_ARBOLES_OPCIONES, default=p["ns"],
                                     key=f"narb_{tarea}_ns"),
                "max_depth": st.slider("max_depth (0 = sin límite)", 0, 30, int(p["max_depth"]),
                                       key=f"narb_{tarea}_prof"),
                "min_samples_leaf": st.slider("min_samples_leaf", 1, 50,
                                              int(p["min_samples_leaf"]), key=f"narb_{tarea}_hoja")}

    with st.expander("Parámetros del barrido de árboles", expanded=True):
        p = formulario(f"narb_{tarea}", construir, DEF_N_ARBOLES, "Calcular")
    if len(p["ns"]) < 2:
        _parar("warning", "Elige al menos dos números de árboles.")
    return p


def form_ruido():
    def construir(p):
        return {"max": st.select_slider("Ruido máximo (% de la desviación típica)",
                                        [5, 10, 15, 20, 30, 40], value=int(p["max"]),
                                        key="ruido_max"),
                "paso": st.selectbox("Paso", [5, 10], index=[5, 10].index(int(p["paso"])),
                                     key="ruido_paso")}

    with st.expander("Parámetros del ruido", expanded=True):
        return formulario("ruido", construir, DEF_RUIDO, "Calcular")


def form_reales():
    def construir(p):
        return {"incluir_wh": st.checkbox("Incluir Weight y Height (variante A)",
                                          value=bool(p["incluir_wh"]), key="reales_wh"),
                "repeticiones": st.slider("Repeticiones de la CV de 3 folds", 1, 10,
                                          int(p["repeticiones"]), key="reales_rep")}

    with st.expander("Parámetros de la evaluación", expanded=True):
        return formulario("reales", construir, DEF_REALES, "Calcular")


# ----------------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------------
def form_clu(rango=False, k=False):
    inits = ["k-means++", "random"]

    def construir(p):
        out = dict(p)
        c = st.columns(4)
        if rango:
            out["k_min"], out["k_max"] = c[0].slider("Rango de K", 2, 15,
                                                     (int(p["k_min"]), int(p["k_max"])),
                                                     key="clu_rango")
        if k:
            out["k"] = c[0].slider("K", 2, 15, int(p["k"]), key="clu_k")
        out["n_init"] = c[1].slider("n_init", 1, 20, int(p["n_init"]), key="clu_ninit")
        out["init"] = c[2].selectbox("init", inits, index=inits.index(p["init"]), key="clu_init")
        out["seed"] = int(c[3].number_input("Semilla", 0, 9999, int(p["seed"]), key="clu_seed"))
        return out

    with st.expander("Parámetros de K-Means", expanded=True):
        p = formulario("clu", construir, DEF_CLU, "Calcular")
    if rango and p["k_max"] - p["k_min"] < 2:
        _parar("warning", "El rango de K debe cubrir al menos 3 valores.")
    return p


def params_clu():
    return leer("clu", DEF_CLU)


def form_lloyd():
    inits = ["k-means++", "random"]

    def construir(p):
        c = st.columns(4)
        return {"k": c[0].slider("K", 2, 12, int(p["k"]), key="lloyd_k"),
                "init": c[1].selectbox("Inicialización", inits, index=inits.index(p["init"]),
                                       key="lloyd_init"),
                "seed": int(c[2].number_input("Semilla", 0, 9999, int(p["seed"]),
                                              key="lloyd_seed")),
                "max_iter": c[3].slider("max_iter", 1, 100, int(p["max_iter"]),
                                        key="lloyd_maxiter")}

    with st.expander("Parámetros de las iteraciones", expanded=True):
        return formulario("lloyd", construir, DEF_LLOYD, "Calcular")
```

- [ ] **Step 5: Crear los esqueletos de las páginas** (`webapp/etapa01.py`, `etapa02.py`, `etapa03.py`, `etapa04.py`, `etapa05.py`), cada uno con exactamente:

```python
"""Páginas de la etapa (se rellenan en su tarea)."""

RENDER = {}
```

- [ ] **Step 6: Escribir `app.py`**

```python
"""App web interactiva del proyecto IA2 (Estimation of Obesity Levels, UCI 544).

Ejecutar desde la raíz del proyecto:   streamlit run app.py
Las etapas y los pasos tienen la misma numeración y títulos que estudio_etapas.html.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import streamlit as st

from src import salida
from webapp import etapa01, etapa02, etapa03, etapa04, etapa05
from webapp.navegacion import ETAPAS, PASOS

# Sin ventanas ni archivos: elige el backend Agg y aplica el mismo estilo que el CLI
salida.configurar(mostrar=False, guardar=False, graficar_tablas=False)
st.set_page_config(page_title="Proyecto IA2 — Obesidad", layout="wide")

RENDER = {"01": etapa01.RENDER, "02": etapa02.RENDER, "03": etapa03.RENDER,
          "04": etapa04.RENDER, "05": etapa05.RENDER}

st.sidebar.title("Proyecto IA2")
etapa = st.sidebar.radio("Etapa", list(ETAPAS), key="etapa",
                         format_func=lambda e: f"{e} · {ETAPAS[e]}")
titulos = dict(PASOS[etapa])
paso = st.sidebar.radio("Paso", list(titulos), key=f"paso_{etapa}",
                        format_func=lambda i: f"{i} · {titulos[i]}")

st.header(f"Etapa {etapa} — {ETAPAS[etapa]}")
st.subheader(f"{paso} · {titulos[paso]}")

render = RENDER[etapa].get(paso)
if render is None:
    st.error("Paso no implementado.")
else:
    render()
```

- [ ] **Step 7: Comprobar que el shell funciona y que el smoke test está en rojo**

```bash
.venv-1/Scripts/python.exe -m pytest tests/test_app_smoke.py -q -x 2>&1 | tail -5
```
Expected: `FAILED tests/test_app_smoke.py::test_paso_se_renderiza[01-1]` con `assert not at.error` (mensaje «Paso no implementado.»). Eso confirma que el shell, la navegación y el harness funcionan y que los pasos aún faltan. Si falla por una excepción de import o de `AppTest` en vez de por `at.error`, corregirlo antes de seguir.

- [ ] **Step 8: Checkpoint**

```bash
git add app.py webapp/ui.py webapp/receptor.py webapp/controles.py webapp/etapa0*.py tests/test_app_smoke.py
git commit -m "feat(app): shell de Streamlit, controles compartidos y smoke test por paso"
```

---

### Task 6: Etapa 01 — EDA y preprocesamiento

**Files:**
- Modify: `webapp/etapa01.py`

**Interfaces:**
- Consumes: `etapas.e01_eda` (`df_raw`, `df_bmi`, `paso_2_1_clases`, `paso_2_3_fuga`), `webapp.receptor.ejecutar`, `webapp.ui`.
- Produces: `RENDER` con los ids `"1","2.1","2.2","2.3","2.4","3"`.

- [ ] **Step 1: Ejecutar el smoke test de la etapa y verlo fallar**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_app_smoke.py -k "01-" -q`
Expected: 6 FAILED («Paso no implementado»).

- [ ] **Step 2: Escribir `webapp/etapa01.py`**

```python
"""Etapa 01 — EDA y preprocesamiento."""
import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE
from etapas import e01_eda as e01
from src.data import TARGET_CLF, TARGET_REG
from src.preprocessing import CLASS_ORDER, build_datasets
from webapp import receptor, ui


def paso_1():
    df = e01.df_raw()
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", df.shape[0])
    c2.metric("Columnas", df.shape[1])
    c3.metric("Filas duplicadas", int(df.duplicated().sum()))
    n = st.slider("Filas a mostrar", 5, 50, 5, key="e01_head")
    ui.tabla(df.head(n), "Primeras filas", 3)
    ui.tabla(df.describe().T, "Estadísticos de las columnas numéricas", 3)
    ui.tabla(pd.DataFrame({"tipo": df.dtypes.astype(str), "faltantes": df.isna().sum()}),
             "Tipos y valores faltantes por columna")
    buf = io.StringIO()
    df.info(buf=buf)
    with st.expander("df.info()"):
        st.code(buf.getvalue())
    cat = list(df.select_dtypes(exclude="number").columns)
    col = st.selectbox("Columna categórica", cat, key="e01_cat")
    ui.tabla(df[col].value_counts().rename("n"), f"Valores de {col}")


def paso_2_2():
    df = e01.df_raw()
    bins = st.slider("Número de bins del histograma", 10, 100, 40, key="e01_bins")
    ui.tabla(df[TARGET_REG].describe().round(2), "Weight (kg)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[TARGET_REG], bins=bins, kde=True, ax=axes[0])
    axes[0].set_title("Histograma de Weight (kg)")
    sns.boxplot(data=df, x=TARGET_CLF, y=TARGET_REG, order=CLASS_ORDER, ax=axes[1])
    axes[1].set_title("Weight por clase de NObeyesdad")
    axes[1].tick_params(axis="x", rotation=45)
    ui.figura(fig)


def paso_2_3():
    receptor.ejecutar(e01.paso_2_3_fuga)
    d = e01.df_bmi()
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    filas = {str(cols): cross_val_score(DecisionTreeClassifier(random_state=RANDOM_STATE),
                                        d[cols], d[TARGET_CLF], cv=cv).mean()
             for cols in (["BMI"], ["Weight", "Height"], ["Weight"], ["Height"])}
    ui.tabla(pd.Series(filas, name="accuracy (CV 5)"),
             "Accuracy de un árbol usando sólo estas columnas")


def paso_2_4():
    metodo = st.selectbox("Método de correlación", ["pearson", "spearman", "kendall"],
                          key="e01_corr")
    num = e01.df_raw().select_dtypes("number")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(method=metodo), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"Correlación ({metodo}) entre numéricas")
    ui.figura(fig)


def paso_3():
    df_clf, df_reg = build_datasets(e01.df_raw())
    c1, c2 = st.columns(2)
    c1.metric("clf.csv (filas × columnas)", f"{df_clf.shape[0]} × {df_clf.shape[1]}")
    c2.metric("reg.csv (filas × columnas)", f"{df_reg.shape[0]} × {df_reg.shape[1]}")
    ui.tabla(df_clf.head(), "clf.csv — primeras filas", 3)
    ui.tabla(df_reg.head(), "reg.csv — primeras filas", 3)


RENDER = {
    "1": paso_1,
    "2.1": lambda: receptor.ejecutar(e01.paso_2_1_clases),
    "2.2": paso_2_2,
    "2.3": paso_2_3,
    "2.4": paso_2_4,
    "3": paso_3,
}
```

- [ ] **Step 3: Ejecutar el smoke test de la etapa**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_app_smoke.py -k "01-" -q`
Expected: 6 passed.

- [ ] **Step 4: Comprobar que la app no escribió archivos**

Run: `git status --short data figures resultados_texto | head`
Expected: sin cambios nuevos respecto a antes de la tarea (el paso 3 no guarda CSV y el receptor desactiva el guardado).

- [ ] **Step 5: Checkpoint**

```bash
git add webapp/etapa01.py
git commit -m "feat(app): etapa 01 con los pasos 2.2 y 2.4 que faltaban en el HTML"
```

---

### Task 7: Etapa 02 — Clasificación

**Files:**
- Modify: `webapp/etapa02.py`, `tests/test_app_smoke.py` (añadir la prueba de la variante B vacía)

**Interfaces:**
- Consumes: `webapp.modelos` (Task 3), `webapp.controles` (Task 5), `webapp.ui`, `etapas.e02_clasificacion` (`_con_ruido`, `_modelos_robustez` vía `modelos`), `src.evaluation` (`metricas_por_clase`, `plot_confusion`, `plot_k_curve`, `plot_n_estimators_curve`, `tree_summary`).
- Produces: `RENDER` con los ids `"1","2","3","3.1","3.2","3.3","3.4","3.6","3.7","4","4.1","5","7","8"`.

- [ ] **Step 1: Añadir la prueba de la variante B sin features a `tests/test_app_smoke.py`**

```python
def test_variante_b_sin_features_muestra_un_aviso_y_no_falla():
    solo_wh = {"features": ["Weight", "Height"], "test_size": 0.2, "seed": 42}
    for paso in ("4", "4.1", "7"):
        at = abrir("02", paso, **{"params::split_clf": solo_wh})
        assert not at.exception, [e.value for e in at.exception]
        if paso != "7":    # el paso 7 siempre usa todas las features menos Weight/Height
            assert at.warning, f"el paso {paso} debería avisar"


def test_sin_features_se_avisa_con_un_error_claro():
    vacio = {"features": [], "test_size": 0.2, "seed": 42}
    at = abrir("02", "3", **{"params::split_clf": vacio})
    assert not at.exception
    assert at.error and "al menos una feature" in at.error[0].value
```

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_app_smoke.py -k "02- or variante_b or sin_features" -q`
Expected: FAILED (páginas sin implementar).

- [ ] **Step 2: Escribir `webapp/etapa02.py`**

```python
"""Etapa 02 — Clasificación de NObeyesdad (variantes A y B)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.tree import plot_tree

from src.evaluation import (metricas_por_clase, plot_confusion, plot_k_curve,
                            plot_n_estimators_curve, tree_summary)
from src.preprocessing import CLASS_ORDER
from webapp import controles as C
from webapp import modelos as M
from webapp import ui


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def _entrenar(variante, hp):
    sp = C.params_split("clf")
    feats = list(sp["features"]) if variante == "A" else C.features_b(sp)
    with st.spinner("Entrenando…"):
        return M.tabla_clf(hp, feats, sp["test_size"], sp["seed"])


def _features(variante):
    sp = C.params_split("clf")
    return (list(sp["features"]) if variante == "A" else C.features_b(sp)), sp


def _reportes_y_matrices(res, split, sufijo):
    y_test = split["y_test"]
    for nombre, r in res.items():
        rep = pd.DataFrame(classification_report(y_test, r["y_pred"], labels=CLASS_ORDER,
                                                 digits=3, output_dict=True)).T
        ui.tabla(rep.loc[CLASS_ORDER, ["precision", "recall", "f1-score"]],
                 f"classification_report por clase — {nombre} ({sufijo})", 3)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (nombre, r) in zip(axes, res.items()):
        plot_confusion(y_test, r["y_pred"], CLASS_ORDER, f"{nombre} ({sufijo})", ax=ax)
    ui.figura(fig)
    for nombre, r in res.items():
        cm = confusion_matrix(y_test, r["y_pred"], labels=CLASS_ORDER)
        ui.tabla(metricas_por_clase(cm, CLASS_ORDER), f"Métricas por clase — {nombre} ({sufijo})", 3)


def _importancias(res, split, color, titulo):
    rf = res["Random Forest"]["modelo"]
    imp = pd.Series(rf.feature_importances_, index=split["X_train"].columns).sort_values(
        ascending=False)
    ui.tabla(imp.rename("importancia"), "feature_importances_ (reducción media de impureza)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color=color)
    ax.set_title(titulo)
    ui.figura(fig)


def _curva_k(variante):
    p = C.form_curva_k("clf")
    feats, sp = _features(variante)
    ks = tuple(range(p["k_min"], p["k_max"] + 1, p["paso"]))
    with st.spinner("Calculando la curva…"):
        cv = M.curva_k("clf", feats, sp["test_size"], sp["seed"], ks, p["weights"])
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_k_curve(cv, ax=ax)
    ui.figura(fig)
    ui.tabla(pd.Series(cv["mean_test_score"], index=cv["param_knn__n_neighbors"],
                       name="macro-F1 (CV)"), "Score por K", 4)


# ----------------------------------------------------------------------------
# Pasos
# ----------------------------------------------------------------------------
def paso_1():
    sp = C.form_split("clf")
    d = M.split_clf(sp["features"], sp["test_size"], sp["seed"])
    ui.tabla(pd.DataFrame({"filas": [len(d["X_train"]), len(d["X_test"])],
                           "features": [d["X_train"].shape[1]] * 2}, index=["train", "test"]),
             "Split estratificado")
    ui.tabla((d["y_train"].value_counts(normalize=True).reindex(CLASS_ORDER) * 100)
             .rename("% en train"), "Distribución de clases en train (%)", 1)


def paso_2():
    hp = C.form_hp("clf")
    ui.tabla(pd.DataFrame(hp).T.astype(str), "Hiperparámetros vigentes")


def paso_3():
    hp = C.form_hp("clf")
    _, t, _ = _entrenar("A", hp)
    ui.tabla(t, "Variante A — macro-F1 en CV y métricas en test")


def paso_3_1():
    hp = C.form_hp("clf")
    res, _, split = _entrenar("A", hp)
    _reportes_y_matrices(res, split, "A")


def paso_3_2():
    hp = C.form_hp("clf")
    res, _, split = _entrenar("A", hp)
    _importancias(res, split, "steelblue", "Importancia de features — Random Forest (A)")


def paso_3_3():
    hp = C.form_hp("clf")
    res, _, split = _entrenar("A", hp)
    nivel = st.slider("Niveles a dibujar", 1, 6, 3, key="e02_niv33")
    arbol = res["Árbol de decisión"]["modelo"]
    ui.tabla(pd.DataFrame({"profundidad": [arbol.get_depth()], "hojas": [arbol.get_n_leaves()]}),
             "Árbol de decisión (A)", 0)
    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(arbol, feature_names=split["X_train"].columns, class_names=list(arbol.classes_),
              max_depth=nivel, filled=True, fontsize=8, ax=ax)
    ax.set_title(f"Árbol de decisión (variante A) — dibujo truncado a {nivel} niveles")
    ui.figura(fig)


def paso_3_4():
    sp = C.params_split("clf")
    p = C.form_arboles("clf")
    nivel = st.slider("Niveles a dibujar (Completo y Podado)", 1, 6, 3, key="e02_niv34")
    feats = list(sp["features"])
    d = M.split_clf(feats, sp["test_size"], sp["seed"])
    arboles = {n: M.ajustar_clf("Árbol de decisión", {"criterion": "gini", **q}, feats,
                                sp["test_size"], sp["seed"], con_cv=False)["modelo"]
               for n, q in p.items()}
    resumen = []
    for nombre, arbol in arboles.items():
        s = tree_summary(arbol, d["X"].columns, nombre)
        trunc = None if nombre == "Muy podado" else nivel
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(arbol, feature_names=d["X"].columns, class_names=list(arbol.classes_),
                  filled=True, max_depth=trunc, fontsize=8, ax=ax)
        ax.set_title(f"Árbol {nombre} — profundidad {s['profundidad']}, {s['n_hojas']} hojas, "
                     f"raíz: {s['raiz_atributo']} ≤ {s['raiz_umbral']}"
                     + (f"  (dibujo truncado a {nivel} niveles)" if trunc else ""))
        ui.figura(fig)
        resumen.append({**s,
                        "acc_train": accuracy_score(d["y_train"], arbol.predict(d["X_train"])),
                        "acc_test": accuracy_score(d["y_test"], arbol.predict(d["X_test"])),
                        "f1_macro_test": f1_score(d["y_test"], arbol.predict(d["X_test"]),
                                                  average="macro")})
    cols = ["modelo", "profundidad", "n_hojas", "raiz_atributo", "acc_train", "acc_test",
            "f1_macro_test"]
    ui.tabla(pd.DataFrame(resumen)[cols].set_index("modelo"), "Resumen de los tres árboles", 3)

    preds = {n: a.predict(d["X_test"]) for n, a in arboles.items()}
    vmax = max(pd.crosstab(d["y_test"], p_).values.max() for p_ in preds.values())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (nombre, y_pred) in zip(axes, preds.items()):
        plot_confusion(d["y_test"], y_pred, CLASS_ORDER,
                       f"Árbol {nombre} — acc test {accuracy_score(d['y_test'], y_pred):.3f}",
                       ax=ax, vmin=0, vmax=vmax)
    ui.figura(fig)
    for nombre, y_pred in preds.items():
        cm = confusion_matrix(d["y_test"], y_pred, labels=CLASS_ORDER)
        ui.tabla(metricas_por_clase(cm, CLASS_ORDER), f"Métricas por clase — árbol {nombre}", 3)


def paso_3_6():
    C.form_hp("clf")      # el K de KNN de los demás pasos se edita aquí también
    _curva_k("A")


def paso_3_7():
    p = C.form_n_arboles("clf")
    feats, sp = _features("A")
    hp_rf = C.params_hp("clf")["Random Forest"]
    with st.spinner("Entrenando los bosques…"):
        df = M.barrido_arboles("clf", feats, sp["test_size"], sp["seed"], sorted(p["ns"]),
                               p["max_depth"], p["min_samples_leaf"])
    ui.tabla(df, "Random Forest: rendimiento y coste según el número de árboles", 4)
    fig, _ = plot_n_estimators_curve(
        df, "cv_f1_macro", "test_f1_macro", "macro-F1",
        "Random Forest: macro-F1 según el número de árboles (variante A)",
        col_std="cv_std", n_proyecto=int(hp_rf["n_estimators"]), mejor="max")
    ui.figura(fig)


def paso_4():
    hp = C.form_hp("clf")
    res, t, split = _entrenar("B", hp)
    ui.tabla(t, "Variante B — macro-F1 en CV y métricas en test")
    _reportes_y_matrices(res, split, "B")
    _importancias(res, split, "indianred",
                  "Importancia de features — Random Forest (B, sin Weight/Height)")


def paso_4_1():
    C.form_hp("clf")
    _curva_k("B")


def paso_5():
    hp = C.params_hp("clf")
    _, tA, _ = _entrenar("A", hp)
    _, tB, _ = _entrenar("B", hp)
    comp = pd.concat({"A (todas)": tA, "B (sin Weight/Height)": tB}, axis=0)
    ui.tabla(comp, "Comparación A vs B")
    fig, ax = plt.subplots(figsize=(8, 4))
    datos = comp.reset_index().rename(columns={"level_0": "variante", "level_1": "modelo"})
    sns.barplot(data=datos, x="modelo", y="test_f1_macro", hue="variante", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Macro-F1 en test por modelo y variante")
    ui.figura(fig)


def paso_7():
    p = C.form_ruido()
    sp = C.params_split("clf")
    niveles = tuple(x / 100 for x in range(0, p["max"] + 1, p["paso"]))
    with st.spinner("Entrenando con ruido…"):
        df = M.barrido_ruido(sp["test_size"], sp["seed"], niveles)
    for metrica, titulo in (("accuracy", "Accuracy en test por nivel de ruido"),
                            ("f1_macro", "Macro-F1 en test por nivel de ruido")):
        ui.tabla(df.pivot(index="nivel", columns="modelo", values=metrica)
                 .rename_axis("Nivel de ruido (%)"), titulo)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ancho = df.pivot(index="nivel", columns="modelo", values="accuracy")
    for marcador, modelo in zip(["o", "s", "^", "D"], ancho.columns):
        ax.plot(ancho.index, ancho[modelo], marker=marcador, lw=2.2, label=modelo)
    ax.set_title("Degradación del accuracy vs nivel de ruido en hábitos (variante B)")
    ax.set_xlabel("Nivel de perturbación gaussiana (% de desviación estándar)")
    ax.set_ylabel("Accuracy en test")
    ax.set_xticks(list(ancho.index))
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(title="Modelo", loc="center left", bbox_to_anchor=(1.01, 0.5))
    ui.figura(fig)


def paso_8():
    p = C.form_reales()
    sp = C.params_split("clf")
    with st.spinner("Validación cruzada sobre las filas reales…"):
        t7 = M.evaluar_reales(p["incluir_wh"], 7, p["repeticiones"], sp["seed"])
        t3 = M.evaluar_reales(p["incluir_wh"], 3, p["repeticiones"], sp["seed"])
    ui.tabla(t7, "Sólo filas reales · 7 clases (CV de 3 folds repetida)")
    ui.tabla(t3, "Sólo filas reales · 3 clases agrupadas (CV de 3 folds repetida)")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, t, azar, titulo in [(axes[0], t7, 1 / 7, "7 clases"),
                                (axes[1], t3, 1 / 3, "3 clases agrupadas")]:
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["balanced_accuracy"], 0.4, label="balanced accuracy", color="#4c78a8")
        ax.bar(x + 0.2, t["f1_macro"], 0.4, yerr=t["f1_macro_std"], capsize=3,
               label="macro-F1 (± desv. entre folds)", color="#f58518")
        ax.axhline(azar, ls="--", color="gray", lw=1.2,
                   label=f"azar en balanced acc. (1/{round(1 / azar)})")
        ax.set_xticks(x, [n.replace(" (clase mayoritaria)", "") for n in t.index],
                      rotation=20, ha="right")
        ax.set_title(f"Filas reales · {titulo} · {'con' if p['incluir_wh'] else 'sin'} Weight/Height")
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("métrica")
    ui.figura(fig)


RENDER = {
    "1": paso_1, "2": paso_2, "3": paso_3, "3.1": paso_3_1, "3.2": paso_3_2, "3.3": paso_3_3,
    "3.4": paso_3_4, "3.6": paso_3_6, "3.7": paso_3_7, "4": paso_4, "4.1": paso_4_1,
    "5": paso_5, "7": paso_7, "8": paso_8,
}
```

- [ ] **Step 3: Ejecutar el smoke test de la etapa y las pruebas de casos límite**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_app_smoke.py -k "02- or variante_b or sin_features" -q`
Expected: 16 passed (14 pasos + 2 pruebas de features). El paso 7 de la variante B vacía no falla porque usa todas las features menos Weight/Height. Si algún paso lanza excepción, corregir el módulo (no relajar la prueba).

- [ ] **Step 4: Verificar que cambiar parámetros cambia el resultado (a mano)**

```bash
.venv-1/Scripts/python.exe - <<'E'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=600)
at.session_state["etapa"] = "02"; at.session_state["paso_02"] = "3"
at.session_state["params::hp_clf"] = {
  "Árbol de decisión": {"criterion": "gini", "max_depth": 2, "min_samples_leaf": 1},
  "Random Forest": {"n_estimators": 20, "max_depth": 0, "min_samples_leaf": 1, "max_features": "sqrt"},
  "KNN": {"n_neighbors": 1, "weights": "uniform", "metric": "euclidean"}}
at.run(); poco = at.dataframe[-1].value
at.session_state["params::hp_clf"]["Árbol de decisión"]["max_depth"] = 0
at.run(); mucho = at.dataframe[-1].value
print(poco.loc["Árbol de decisión", "test_accuracy"], "<", mucho.loc["Árbol de decisión", "test_accuracy"])
E
```
Expected: imprime dos números con el primero menor que el segundo.

- [ ] **Step 5: Checkpoint**

```bash
git add webapp/etapa02.py tests/test_app_smoke.py
git commit -m "feat(app): etapa 02 interactiva (features, hiperparámetros, ruido y filas reales)"
```

---

### Task 8: Etapa 03 — Regresión

**Files:**
- Modify: `webapp/etapa03.py`

**Interfaces:**
- Consumes: `webapp.modelos` (`tabla_reg`, `ajustar_reg`, `split_reg`, `curva_k`, `barrido_arboles`), `webapp.controles`, `webapp.ui`, `src.evaluation` (`plot_k_curve`, `plot_n_estimators_curve`, `reg_metrics`, `tree_summary`).
- Produces: `RENDER` con los ids `"1","2","2.1","2.2","3","4","5","6"`.

- [ ] **Step 1: Ejecutar el smoke test de la etapa y verlo fallar**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_app_smoke.py -k "03-" -q`
Expected: 8 FAILED.

- [ ] **Step 2: Escribir `webapp/etapa03.py`**

```python
"""Etapa 03 — Regresión de Weight."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.tree import plot_tree

from src.evaluation import plot_k_curve, plot_n_estimators_curve, reg_metrics, tree_summary
from webapp import controles as C
from webapp import modelos as M
from webapp import ui


def _entrenar(hp):
    sp = C.params_split("reg")
    with st.spinner("Entrenando…"):
        return M.tabla_reg(hp, sp["features"], sp["test_size"], sp["seed"])


def _pred_vs_real(y_test, preds, titulos):
    """Paneles predicho vs real con la misma escala en todos los ejes."""
    lo = min(y_test.min(), min(p.min() for p in preds.values())) - 2
    hi = max(y_test.max(), max(p.max() for p in preds.values())) + 2
    fig, axes = plt.subplots(1, len(preds), figsize=(6 * len(preds), 5))
    for ax, (nombre, y_pred) in zip(axes, preds.items()):
        ax.scatter(y_test, y_pred, s=12, alpha=0.5, color="steelblue")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Real (kg)")
        ax.set_ylabel("Predicho (kg)")
        ax.set_title(titulos[nombre])
    ui.figura(fig)


def paso_1():
    sp = C.form_split("reg")
    d = M.split_reg(sp["features"], sp["test_size"], sp["seed"])
    ui.tabla(pd.DataFrame({"filas": [len(d["X_train"]), len(d["X_test"])],
                           "features": [d["n_features"]] * 2}, index=["train", "test"]),
             "Split (estratificado por deciles de Weight)")
    ui.tabla(d["y_train"].describe().round(2), "Weight en train")


def paso_2():
    hp = C.form_hp("reg")
    _, t, _ = _entrenar(hp)
    ui.tabla(t, "Resultados: RMSE de CV y métricas en test (MAE, RMSE, R², R²_adj)")


def paso_2_1():
    C.form_hp("reg")
    p = C.form_curva_k("reg")
    sp = C.params_split("reg")
    ks = tuple(range(p["k_min"], p["k_max"] + 1, p["paso"]))
    with st.spinner("Calculando la curva…"):
        cv = M.curva_k("reg", sp["features"], sp["test_size"], sp["seed"], ks, p["weights"])
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_k_curve(cv, ylabel="RMSE (CV)", negate=True, ax=ax)
    ui.figura(fig)
    ui.tabla(pd.Series([-s for s in cv["mean_test_score"]], index=cv["param_knn__n_neighbors"],
                       name="RMSE (CV)"), "RMSE por K", 3)


def paso_2_2():
    p = C.form_n_arboles("reg")
    sp = C.params_split("reg")
    hp_rf = C.params_hp("reg")["Random Forest"]
    with st.spinner("Entrenando los bosques…"):
        df = M.barrido_arboles("reg", sp["features"], sp["test_size"], sp["seed"],
                               sorted(p["ns"]), p["max_depth"], p["min_samples_leaf"])
    ui.tabla(df, "Random Forest (regresión): error y coste según el número de árboles", 4)
    fig, _ = plot_n_estimators_curve(
        df, "cv_RMSE", "test_RMSE", "RMSE (kg)", "Random Forest: RMSE según el número de árboles",
        col_std="cv_std", n_proyecto=int(hp_rf["n_estimators"]), mejor="min")
    ui.figura(fig)


def paso_3():
    sp = C.params_split("reg")
    p = C.form_arboles("reg")
    nivel = st.slider("Niveles a dibujar (Completo y Podado)", 1, 6, 3, key="e03_niv")
    feats = list(sp["features"])
    d = M.split_reg(feats, sp["test_size"], sp["seed"])
    arboles = {n: M.ajustar_reg("Árbol de regresión", {"criterion": "squared_error", **q}, feats,
                                sp["test_size"], sp["seed"], con_cv=False)["modelo"]
               for n, q in p.items()}
    resumen, preds, titulos = [], {}, {}
    for nombre, arbol in arboles.items():
        s = tree_summary(arbol, d["X"].columns, nombre)
        m = reg_metrics(d["y_test"], arbol.predict(d["X_test"]), n_features=d["n_features"])
        resumen.append({**s, **m})
        trunc = None if nombre == "Muy podado" else nivel
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(arbol, max_depth=trunc, feature_names=d["X"].columns, filled=True,
                  rounded=True, ax=ax, fontsize=9 if trunc is None else 8)
        ax.set_title(f"Árbol {nombre} — profundidad {s['profundidad']}, {s['n_hojas']} hojas, "
                     f"raíz: {s['raiz_atributo']} ≤ {s['raiz_umbral']}"
                     + (f" (truncado a {nivel} niveles)" if trunc else ""))
        ui.figura(fig)
        preds[nombre] = arbol.predict(d["X_test"])
        titulos[nombre] = (f"Árbol {nombre}\nRMSE={m['RMSE']:.2f} kg  R²={m['R2']:.3f}  "
                           f"R²adj={m['R2_adj']:.3f}")
    cols = ["modelo", "profundidad", "n_hojas", "raiz_atributo", "MAE", "RMSE", "R2", "R2_adj"]
    ui.tabla(pd.DataFrame(resumen)[cols].set_index("modelo"), "Resumen de los tres árboles")
    _pred_vs_real(d["y_test"], preds, titulos)


def paso_4():
    hp = C.form_hp("reg")
    res, _, d = _entrenar(hp)
    modelos = {n: r for n, r in res.items()}
    titulos = {n: (f"{n}\nRMSE={r['test_RMSE']:.2f} kg  R²={r['test_R2']:.3f}  "
                   f"R²adj={r['test_R2_adj']:.3f}") for n, r in modelos.items()}
    _pred_vs_real(d["y_test"], {n: r["y_pred"] for n, r in modelos.items()}, titulos)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (nombre, r) in zip(axes, modelos.items()):
        sns.histplot(d["y_test"] - r["y_pred"], bins=40, kde=True, ax=ax, color="steelblue")
        ax.axvline(0, c="r", ls="--", lw=1.2)
        ax.set_title(f"Residuos — {nombre}")
        ax.set_xlabel("Real − Predicho (kg)")
        ax.set_ylabel("Frecuencia")
    ui.figura(fig)


def paso_5():
    hp = C.form_hp("reg")
    res, _, d = _entrenar(hp)
    rf = res["Random Forest"]["modelo"]
    imp = pd.Series(rf.feature_importances_, index=d["X_train"].columns).sort_values(
        ascending=False)
    ui.tabla(imp.rename("importancia"), "feature_importances_")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color="steelblue")
    ax.set_title("Importancia de features — Random Forest (regresión de Weight)")
    ax.set_xlabel("Importancia (reducción de impureza)")
    ui.figura(fig)


def paso_6():
    hp = C.form_hp("reg")
    _, t, _ = _entrenar(hp)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    t["test_RMSE"].plot.bar(ax=axes[0], color="steelblue", title="RMSE en test (kg)")
    t["test_R2"].plot.bar(ax=axes[1], color="indianred", title="R² en test")
    t["test_R2_adj"].plot.bar(ax=axes[2], color="forestgreen", title="R² ajustado en test")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    ui.figura(fig)
    ui.tabla(t, "Tabla de resultados")


RENDER = {"1": paso_1, "2": paso_2, "2.1": paso_2_1, "2.2": paso_2_2, "3": paso_3,
          "4": paso_4, "5": paso_5, "6": paso_6}
```

- [ ] **Step 3: Ejecutar el smoke test de la etapa**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_app_smoke.py -k "03-" -q`
Expected: 8 passed.

- [ ] **Step 4: Checkpoint**

```bash
git add webapp/etapa03.py
git commit -m "feat(app): etapa 03 interactiva con residuos y R² ajustado"
```

---

### Task 9: Etapa 04 — Clustering e iteraciones de K-Means

**Files:**
- Modify: `webapp/etapa04.py`

**Interfaces:**
- Consumes: `webapp.clustering` (Task 4), `webapp.controles` (`form_clu`, `form_lloyd`, `params_clu`), `webapp.ui`, `src.evaluation.metricas_por_clase`, `src.preprocessing` (`CLASS_ORDER`, `NUMERIC_COLS`).
- Produces: `RENDER` con los ids `"1","2","2.1","2.2","2.3","2.4","3","3.1","3.2","3.3","3.4","3.5","4","4.1","4.2","4.3","4.4","5"`.

- [ ] **Step 1: Ejecutar el smoke test de la etapa y verlo fallar**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_app_smoke.py -k "04-" -q`
Expected: 18 FAILED.

- [ ] **Step 2: Escribir `webapp/etapa04.py`**

```python
"""Etapa 04 — Clustering K-Means."""
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.evaluation import metricas_por_clase
from src.preprocessing import CLASS_ORDER, NUMERIC_COLS
from webapp import clustering as CL
from webapp import controles as C
from webapp import ui

N_REFS = 10   # muestras de referencia del estadístico gap


def _b(variante, p):
    return CL.barrido(variante, p["k_min"], p["k_max"], p["n_init"], p["init"], p["seed"])


def _g(variante, p, n_refs=N_REFS):
    return CL.gap(variante, p["k_min"], p["k_max"], n_refs, p["seed"])


def _km(variante, p):
    return CL.ajustar(variante, p["k"], p["n_init"], p["init"], p["seed"])


def _con_gap(variante, p, n_refs=N_REFS):
    with st.spinner("Calculando el estadístico gap (tarda unos segundos)…"):
        return _g(variante, p, n_refs)


# ----------------------------------------------------------------------------
# 1–2. Datos y elección de K
# ----------------------------------------------------------------------------
def paso_1():
    m = CL.matriz("base")
    c1, c2 = st.columns(2)
    c1.metric("Filas", m["X_scaled"].shape[0])
    c2.metric("Features (sin la clase)", m["X_scaled"].shape[1])
    ui.tabla(pd.DataFrame(m["X_scaled"], columns=m["X"].columns).head(),
             "Primeras filas escaladas (StandardScaler)", 3)


def paso_2():
    p = C.form_clu(rango=True)
    ui.tabla(_b("base", p)["metrics"], "Métricas internas por K", 3)


def paso_2_1():
    p = C.form_clu(rango=True)
    b = _b("base", p)
    codo = CL.knee_point(b["ks"], b["metrics"]["inercia"].values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(b["metrics"].index, b["metrics"]["inercia"], marker="o")
    ax.axvline(codo, c="r", ls="--", label=f"codo detectado: K={codo}")
    ax.set_xlabel("K")
    ax.set_ylabel("Inercia (WCSS)")
    ax.set_title("Método del codo")
    ax.legend()
    ui.figura(fig)


def paso_2_2():
    p = C.form_clu(rango=True)
    m = _b("base", p)["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for ax, (col, mejor, sentido) in zip(axes, [
            ("silhouette", int(m["silhouette"].idxmax()), "max"),
            ("calinski_harabasz", int(m["calinski_harabasz"].idxmax()), "max"),
            ("davies_bouldin", int(m["davies_bouldin"].idxmin()), "min")]):
        ax.plot(m.index, m[col], marker="o")
        ax.axvline(mejor, c="r", ls="--", label=f"{sentido} en K={mejor}")
        ax.set_xlabel("K")
        ax.set_title(col)
        ax.legend()
    ui.figura(fig)


def paso_2_3():
    p = C.form_clu(rango=True)
    n_refs = st.slider("Muestras de referencia (gap)", 3, 20, N_REFS, key="e04_refs")
    gap_df, k_gap = _con_gap("base", p, n_refs)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(gap_df.index, gap_df["gap"], yerr=gap_df["s_k"], marker="o", capsize=3)
    if k_gap is not None:
        ax.axvline(k_gap, c="r", ls="--", label=f"gap: K={k_gap}")
        ax.legend()
    ax.set_xlabel("K")
    ax.set_ylabel("gap(K)")
    ax.set_title("Estadístico gap")
    ui.figura(fig)
    ui.tabla(gap_df, "gap(K) y s_k", 3)


def paso_2_4():
    p = C.form_clu(rango=True)
    g = _con_gap("base", p)
    ui.tabla(CL.resumen_k(_b("base", p), g), "K óptimo según cada método")


# ----------------------------------------------------------------------------
# 3. K vs clases reales
# ----------------------------------------------------------------------------
def paso_3():
    p = C.form_clu(k=True)
    a = CL.analizar("base", _km("base", p).labels_)
    ui.tabla(pd.Series(a["metricas"], name=f"K = {p['k']}"), "Silueta, ARI, NMI, accuracy con mapeo y pureza")


def paso_3_1():
    p = C.form_clu(k=True)
    ct = CL.analizar("base", _km("base", p).labels_)["ct"]
    ui.tabla(ct, f"Clases reales (filas) vs clusters K={p['k']} (columnas)", 0)
    fig, ax = plt.subplots(figsize=(9, 6))
    CL.heatmap_ct(ct, ax, f"Clases reales (filas) vs clusters K-Means K={p['k']} (columnas)")
    ui.figura(fig)


def paso_3_2():
    p = C.form_clu(k=True)
    a = CL.analizar("base", _km("base", p).labels_)
    fig, ax = plt.subplots(figsize=(9, 7))
    CL.heatmap_cm(a["cm"], ax, f"Matriz de confusión — clusters K={p['k']} mapeados a clases")
    ui.figura(fig)
    ui.tabla(pd.Series(a["mapping"], name="clase asignada").rename_axis("Cluster"),
             "Mapeo cluster → clase (algoritmo húngaro)")
    ui.tabla(metricas_por_clase(a["cm"], CLASS_ORDER), f"Métricas por clase — K={p['k']}", 3)


def _centroides(variante, p, columnas, ordenar=False):
    m, km = CL.matriz(variante), _km(variante, p)
    a = CL.analizar(variante, km.labels_)
    c = pd.DataFrame(m["scaler"].inverse_transform(km.cluster_centers_), columns=m["X"].columns)
    c.index.name = "Cluster"
    c.insert(0, "clase_mapeada", [a["mapping"].get(i, "sin_clase") for i in c.index])
    c.insert(0, "n", pd.Series(km.labels_).value_counts().sort_index().values)
    c["BMI_centroide"] = c["Weight"] / c["Height"] ** 2
    extra = [] if "BMI_centroide" in columnas else ["BMI_centroide"]
    c = c[["n", "clase_mapeada", *columnas, *extra]]
    return c.sort_values("BMI_centroide") if ordenar else c


def paso_3_3():
    p = C.form_clu(k=True)
    cols = ["Gender", "Age", "Height", "Weight", "BMI_centroide", "family_history_with_overweight",
            "FAVC", "FCVC", "CAEC", "FAF", "MTRANS_Public_Transportation"]
    ui.tabla(_centroides("base", p, cols), f"Centroides (K={p['k']}) en unidades originales", 2)


def paso_3_4():
    p = C.form_clu(k=True)
    ui.figura(CL.fig_pca("base", _km("base", p).labels_, f"Clusters K-Means (K={p['k']})"))


def paso_3_5():
    p = C.form_lloyd()
    with st.spinner("Calculando las iteraciones…"):
        pasos, convergio, Z, Zc = CL.lloyd_cacheado("base", p["k"], p["init"], p["seed"],
                                                     p["max_iter"])
    if convergio:
        st.success(f"Convergió en {len(pasos) - 1} iteraciones.")
    else:
        st.info(f"No convergió en {p['max_iter']} iteraciones (se muestran las {len(pasos)} calculadas).")
    i = st.slider("Iteración", 0, len(pasos) - 1, 0, key="e04_iter")
    espacio = st.empty()
    if st.button("▶ Reproducir", key="e04_play"):
        for j in range(len(pasos)):
            with espacio.container():
                ui.figura(CL.fig_iteracion(Z, Zc, pasos, j, p["k"]))
            time.sleep(0.6)
    else:
        with espacio.container():
            ui.figura(CL.fig_iteracion(Z, Zc, pasos, i, p["k"]))
    cambian = [0] + [int((pasos[j]["etiquetas"] != pasos[j - 1]["etiquetas"]).sum())
                     for j in range(1, len(pasos))]
    ui.tabla(pd.DataFrame({"inercia": [s["inercia"] for s in pasos],
                           "movimiento_max_centroide": [s["movimiento"] for s in pasos],
                           "puntos_que_cambian": cambian}).rename_axis("iteración"),
             "Evolución por iteración", 3)


# ----------------------------------------------------------------------------
# 4. Con la columna de clase
# ----------------------------------------------------------------------------
def paso_4():
    filas = {"sin clase": "base", "con clase (ordinal)": "ord", "con clase (one-hot)": "oh",
             "sólo numéricas": "num"}
    ui.tabla(pd.DataFrame({n: CL.matriz(v)["X_scaled"].shape for n, v in filas.items()},
                          index=["filas", "columnas"]).T, "Dimensiones de cada variante", 0)


def paso_4_1():
    p = C.form_clu(rango=True)
    g = _con_gap("ord", p)
    b = _b("ord", p)
    ui.figura(CL.fig_cinco_metodos(b, g))
    ui.tabla(CL.resumen_k(b, g).rename("K óptimo (con clase ordinal)"),
             "K óptimo por método (con clase ordinal)")


def paso_4_2():
    p = C.form_clu(k=True)
    an = {nombre: CL.analizar(v, _km(v, p).labels_)
          for nombre, v in (("ordinal", "ord"), ("one-hot", "oh"))}
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, (nombre, a) in zip(axes, an.items()):
        CL.heatmap_ct(a["ct"], ax, f"Con clase ({nombre}) — K={p['k']}, contingencia cruda")
    ui.figura(fig)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, (nombre, a) in zip(axes, an.items()):
        CL.heatmap_cm(a["cm"], ax, f"Con clase ({nombre}) — K={p['k']} mapeado\n"
                      f"ARI={a['metricas']['ARI']:.3f}  Pureza={a['metricas']['pureza']:.3f}")
    ui.figura(fig)
    for nombre, a in an.items():
        ui.tabla(metricas_por_clase(a["cm"], CLASS_ORDER),
                 f"Métricas por clase — K={p['k']} con clase {nombre}", 3)


def paso_4_3():
    p = C.form_clu(k=True)
    filas = {"sin clase": "base", "con clase (ordinal)": "ord", "con clase (one-hot)": "oh"}
    t = pd.DataFrame({n: CL.analizar(v, _km(v, p).labels_)["metricas"]
                      for n, v in filas.items()}).T[["silueta", "ARI", "NMI", "acc_mapeo", "pureza"]]
    ui.tabla(t, f"Comparativa de clustering (K={p['k']})")


def paso_4_4():
    p = C.form_clu(k=True)
    ui.figura(CL.fig_pca("ord", _km("ord", p).labels_,
                         f"Clusters K-Means K={p['k']} (con clase ordinal)"))


# ----------------------------------------------------------------------------
# 5. Sólo numéricas
# ----------------------------------------------------------------------------
def paso_5():
    p = C.form_clu(rango=True, k=True)
    b = _b("num", p)
    resumen = pd.concat([CL.resumen_k(_b("base", p), _con_gap("base", p)).rename("K óptimo (todas)"),
                         CL.resumen_k(b, _con_gap("num", p)).rename("K óptimo (sólo numéricas)")],
                        axis=1)
    ui.tabla(resumen, "K óptimo por método: todas las features vs sólo numéricas")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, col in zip(axes, ["inercia", "silhouette", "calinski_harabasz", "davies_bouldin"]):
        ax.plot(b["metrics"].index, b["metrics"][col], marker="o")
        ax.set_xlabel("K")
        ax.set_title(f"{col} (sólo numéricas)")
    ui.figura(fig)
    a = CL.analizar("num", _km("num", p).labels_)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    CL.heatmap_ct(a["ct"], axes[0], f"Clases reales vs clusters (K={p['k']}, sólo numéricas)")
    CL.heatmap_cm(a["cm"], axes[1], "Clusters mapeados a clases (asignación óptima)")
    ui.figura(fig)
    ui.tabla(metricas_por_clase(a["cm"], CLASS_ORDER), f"Métricas por clase — K={p['k']} sólo numéricas", 3)
    ui.tabla(pd.Series(a["metricas"]), "Métricas del clustering")
    ui.tabla(_centroides("num", p, NUMERIC_COLS, ordenar=True),
             "Centroides (sólo numéricas) ordenados por IMC", 2)


RENDER = {
    "1": paso_1, "2": paso_2, "2.1": paso_2_1, "2.2": paso_2_2, "2.3": paso_2_3, "2.4": paso_2_4,
    "3": paso_3, "3.1": paso_3_1, "3.2": paso_3_2, "3.3": paso_3_3, "3.4": paso_3_4,
    "3.5": paso_3_5, "4": paso_4, "4.1": paso_4_1, "4.2": paso_4_2, "4.3": paso_4_3,
    "4.4": paso_4_4, "5": paso_5,
}
```

- [ ] **Step 3: Ejecutar el smoke test de la etapa**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_app_smoke.py -k "04-" -q`
Expected: 18 passed (los pasos con gap tardan ~20-60 s la primera vez; después usan la caché).

- [ ] **Step 4: Comprobar las iteraciones de K-Means con parámetros distintos**

```bash
.venv-1/Scripts/python.exe - <<'E'
from streamlit.testing.v1 import AppTest
for k, init, mi in [(3, "random", 1), (7, "k-means++", 30), (12, "random", 100)]:
    at = AppTest.from_file("app.py", default_timeout=600)
    at.session_state["etapa"] = "04"; at.session_state["paso_04"] = "3.5"
    at.session_state["params::lloyd"] = {"k": k, "init": init, "seed": 42, "max_iter": mi}
    at.run()
    assert not at.exception and not at.error, (k, init, mi)
    print(k, init, mi, "->", (at.success or at.info)[0].value, "| filas:", len(at.dataframe[-1].value))
E
```
Expected: tres líneas, sin excepciones; con `max_iter=1` el mensaje es «No convergió…» y la tabla tiene 1 fila.

- [ ] **Step 5: Checkpoint**

```bash
git add webapp/etapa04.py
git commit -m "feat(app): etapa 04 interactiva con las iteraciones de K-Means (paso 3.5)"
```

---

### Task 10: Etapa 05 — Comparación final con los parámetros activos

**Files:**
- Modify: `webapp/etapa05.py`

**Interfaces:**
- Consumes: `etapas.e05_comparacion` (`_ruido`, `paso_1_resultados`, `_estado`, `NIVEL_RUIDO`), `src.cache.obtener`, `webapp.modelos`, `webapp.clustering`, `webapp.controles` (`params_split`, `params_hp`, `params_clu`), `webapp.receptor.ejecutar`.
- Produces: `RENDER = {"1": paso_1}`. Recalcula las tablas con los hiperparámetros, la semilla y el tamaño de test vigentes de las etapas 02 y 03, y con la configuración de K-Means vigente (K fijo en 7).

- [ ] **Step 1: Ejecutar el smoke test y verlo fallar**

Run: `.venv-1/Scripts/python.exe -m pytest tests/test_app_smoke.py -k "05-" -q`
Expected: 1 FAILED.

- [ ] **Step 2: Escribir `webapp/etapa05.py`**

```python
"""Etapa 05 — Comparación final de todos los casos con los parámetros activos."""
import pandas as pd
import streamlit as st

from etapas import e05_comparacion as e05
from src import cache
from webapp import clustering as CL
from webapp import controles as C
from webapp import modelos as M
from webapp import receptor


def _resultados():
    """Dict con el formato que espera etapas/e05_comparacion.py (resultados())."""
    spc, hpc = C.params_split("clf"), C.params_hp("clf")
    spr, hpr = C.params_split("reg"), C.params_hp("reg")
    pc = C.params_clu()
    todas = tuple(M.features_clf())
    sin_wh = tuple(f for f in todas if f not in M.DROP_WH)

    _, clf_A, _ = M.tabla_clf(hpc, todas, spc["test_size"], spc["seed"])
    _, clf_B, _ = M.tabla_clf(hpc, sin_wh, spc["test_size"], spc["seed"])
    ruido = cache.obtener(f"cmp_ruido_{int(e05.NIVEL_RUIDO * 100)}", e05._ruido)
    _, reg, _ = M.tabla_reg(hpr, tuple(M.features_reg()), spr["test_size"], spr["seed"])

    def metricas(variante):
        km = CL.ajustar(variante, 7, pc["n_init"], pc["init"], pc["seed"])
        return CL.analizar(variante, km.labels_)["metricas"]

    clu = pd.DataFrame({
        "Sin clase (20 features)": metricas("base"),
        "Sólo numéricas (8)": metricas("num"),
        "Con clase ordinal": metricas("ord"),
        "Con clase one-hot": metricas("oh"),
    }).T[["silueta", "ARI", "NMI", "pureza"]].astype(float)

    reales = {"B7": M.evaluar_reales(False, 7, 3, spc["seed"]),
              "B3": M.evaluar_reales(False, 3, 3, spc["seed"])}
    return dict(clf_A=clf_A, clf_B=clf_B, ruido=ruido, reg=reg, clu=clu, reales=reales,
                barrido_rf=None)


def paso_1():
    with st.spinner("Calculando todos los casos con los parámetros activos…"):
        res = _resultados()
    e05._estado["res"] = res
    try:
        receptor.ejecutar(e05.paso_1_resultados)
    finally:
        e05._estado.pop("res", None)


RENDER = {"1": paso_1}
```

- [ ] **Step 3: Ejecutar el smoke test y una comprobación de que usa los parámetros activos**

```bash
.venv-1/Scripts/python.exe -m pytest tests/test_app_smoke.py -k "05-" -q
.venv-1/Scripts/python.exe - <<'E'
from streamlit.testing.v1 import AppTest
def tabla_a(at):
    return next(d.value for d in at.dataframe if "Árbol de decisión" in d.value.index and "test_accuracy" in d.value.columns)
at = AppTest.from_file("app.py", default_timeout=900)
at.session_state["etapa"] = "05"; at.session_state["paso_05"] = "1"
at.run(); base = tabla_a(at).loc["Árbol de decisión", "test_accuracy"]
at.session_state["params::hp_clf"] = {
  "Árbol de decisión": {"criterion": "gini", "max_depth": 2, "min_samples_leaf": 1},
  "Random Forest": {"n_estimators": 20, "max_depth": 0, "min_samples_leaf": 1, "max_features": "sqrt"},
  "KNN": {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"}}
at.run(); nuevo = tabla_a(at).loc["Árbol de decisión", "test_accuracy"]
print(base, "->", nuevo); assert nuevo < base
E
```
Expected: `1 passed` y la segunda comprobación imprime dos números con el segundo menor (árbol de profundidad 2).

- [ ] **Step 4: Checkpoint**

```bash
git add webapp/etapa05.py
git commit -m "feat(app): etapa 05 recalcula la comparación con los parámetros activos"
```

---

### Task 11: Verificación final

**Files:**
- Modify: ninguno (solo si la verificación encuentra fallos)

- [ ] **Step 1: Suite completa**

Run: `.venv-1/Scripts/python.exe -m pytest -q 2>&1 | tail -15`
Expected: todo en verde (≈ 4 de navegación, 2 de postpoda, 10 de modelos, 13 de clustering, 54 de smoke + 2 de casos límite). Si algo falla, usar superpowers:systematic-debugging y corregir la causa, no la prueba.

- [ ] **Step 2: El CLI sigue funcionando**

```bash
for e in e01_eda e02_clasificacion e03_regresion e04_clustering e05_comparacion; do echo "== $e"; echo v | MPLBACKEND=Agg .venv-1/Scripts/python.exe etapas/$e.py 2>&1 | head -4; done
git status --short data figures resultados_texto | head
```
Expected: cada etapa muestra su menú de pasos; `git status` no muestra archivos nuevos en `data/`, `figures/`, `resultados_texto/` causados por la app o por estas pruebas.

- [ ] **Step 3: Arrancar la app de verdad y comprobar que responde**

```bash
.venv-1/Scripts/python.exe -m streamlit run app.py --server.headless true --server.port 8765 > "$TEMP/streamlit.log" 2>&1 &
sleep 12; curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765/_stcore/health; curl -s http://localhost:8765/_stcore/health; echo; tail -5 "$TEMP/streamlit.log"
```
Expected: `200` y `ok`, sin trazas de error en el log. Luego detener el servidor (`taskkill //F //IM streamlit.exe` o cerrar el proceso de Python que escucha en 8765).

- [ ] **Step 4: Checklist manual para el usuario** (la revisión visual no se puede automatizar)

Pedir al usuario que abra `http://localhost:8765` (o que ejecute `streamlit run app.py`) y compruebe: el orden y los títulos de la barra lateral coinciden con el HTML; en 02/3.4 y 03/3 cambiar profundidades y pulsar «Entrenar» cambia los árboles; en 04/3.5 el slider y el botón ▶ muestran cómo se mueven los centroides; en 01/2.2, 01/2.4 y 03/4 aparecen las gráficas que faltaban en el HTML.

- [ ] **Step 5: Checkpoint final**

```bash
git status --short
git add -u webapp tests app.py
git commit -m "chore(app): verificación final de la app interactiva"
```
(solo si había cambios por la verificación y el usuario autorizó los commits).

---

## Self-review (hecha al escribir el plan)

- **Cobertura del spec:** requisitos 1–6 → Tasks 2 (orden/títulos), 5–10 (solo figuras y tablas, controles), 6 y 8 (pasos faltantes: 01/2.2, 01/2.4, 03 residuos y 01/2.3 con su figura), 9 (iteraciones K-Means = paso 3.5), 1 (postpoda). Tecnología Streamlit y dependencia en `requirements.txt` → Task 0. Pruebas → tests por tarea + smoke. Fuera de alcance (HTML/MD intactos, sin despliegue) respetado en Global Constraints.
- **Placeholders:** ninguno; el único bloque de «nota para el implementador» de la Task 9 se resuelve en su Step 3 con el código final de `_centroides`.
- **Consistencia de tipos:** `M.tabla_clf/tabla_reg` devuelven `(res, tabla, split)` y así se consumen en 02, 03 y 05; `CL.barrido` devuelve `dict(metrics, ks)` y `CL.gap` una tupla `(df, k)`, consumidos así en `resumen_k`, `fig_cinco_metodos` y la página 04; `controles.form_*` devuelven los dicts que lee cada página; `RENDER` usa los ids exactos de `navegacion.PASOS`.
- **Review Focus:** cada línea tiene su prueba (features vacías → Task 7; K≠7 → Task 4 `test_analizar_acepta_cualquier_k`; Lloyd → Task 4; caché por parámetros → Task 3; `max_depth=0`/`max_features` → Task 3).

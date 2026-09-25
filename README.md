# IA2 — Entrega 1: Estimation of Obesity Levels (versión Python, sin Jupyter)

Proyecto académico de Inteligencia Artificial 2 sobre el dataset *Estimation of Obesity Levels
Based on Eating Habits and Physical Condition* (UCI ML Repository, id 544): **clasificación**
multiclase de `NObeyesdad`, **regresión** de `Weight` y **clustering** K-Means, con árboles de
decisión, random forest y KNN.

Es la misma entrega que la versión en notebooks (`../ia2-obesidad`), reescrita como un proyecto
Python normal: 4 scripts (uno por etapa) divididos en pasos numerados que se ejecutan desde un
menú, desde VS Code o desde la terminal. Las cifras son idénticas a las de los notebooks
(misma semilla y mismos splits).

**→ Para saber dónde hacer clic y qué paso ejecutar para ver cada gráfica, lee [`GUIA.md`](GUIA.md).**

## Instalación rápida

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py            # menú interactivo
python run_all.py         # todo el proyecto sin ventanas (~1 min); PNG en figures/
```

## Resumen de resultados (test, 418 filas)

| Tarea | Modelo | Métrica principal | Otras |
|---|---|---|---|
| Clasificación A (todas las features) | Random Forest | **acc 0.957** | macro-F1 0.956 |
| Clasificación B (sin Weight/Height) | Random Forest | **acc 0.857** | macro-F1 0.853 |
| Clasificación B con 20 % de ruido | Random Forest | acc 0.790 | árbol 0.648, KNN k=1 0.737, k=3 0.768 |
| Regresión de Weight | Random Forest | **RMSE 8.25 kg** | MAE 4.84, R² 0.899, R²adj 0.894 |
| Clustering K=7 (sin clase) | K-Means | **ARI 0.088** | silueta 0.132, pureza 0.306 |
| Clustering K=7 (clase ordinal) | K-Means | ARI 0.252 | pureza 0.426 |
| Clustering K=7 (clase one-hot) | K-Means | ARI 0.999 | pureza 0.9995 |
| Clustering K=7 (sólo numéricas) | K-Means | ARI 0.156 | pureza 0.382 |

Elección de K (sin la clase): codo → 7, silueta → 2, Calinski-Harabasz → 2, Davies-Bouldin → 9,
gap → 2. Los métodos no coinciden: la estructura de clusters es débil.

## Hallazgos clave

- `NObeyesdad` ≈ f(Weight, Height) vía IMC: un árbol con sólo esas dos columnas acierta 0.954 en
  CV. Por eso la clasificación se presenta en dos variantes (A con todo, B sin Weight/Height) y
  la regresión excluye `NObeyesdad`.
- ~77 % del dataset es sintético (SMOTE): explica que KNN elija K=1 en la variante A. Con ruido
  realista (etapa 02, paso 7) K=3 supera a K=1 y Random Forest es el más robusto.
- Los tres árboles de clasificación comparten la raíz `Weight ≤ 99.536`; los de regresión,
  `family_history_with_overweight ≤ 0.5`.
- K-Means no recupera las clases: las binarias raras (`MTRANS_Bike`, `Motorbike`) dominan la
  distancia euclídea estandarizada. Con la clase en one-hot se recupera al 99.95 %, lo que
  ilustra la diferencia entre aprendizaje supervisado y no supervisado.

## Estructura

Ver la sección 5 de [`GUIA.md`](GUIA.md).

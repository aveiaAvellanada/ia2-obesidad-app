# App web interactiva (Streamlit) — diseño

Fecha: 2026-09-30 · Estado: aprobado en conversación, pendiente de revisión del spec escrito.

## Objetivo

Versión de `estudio_etapas.html` como aplicación web interactiva. Muestra solo gráficas y
tablas (sin notas, porque dependían de un entrenamiento fijo) y permite modificar los
parámetros de decisión de cada paso: features, hiperparámetros, número de árboles, K, etc.

## Requisitos (del usuario)

1. Mismo orden, mismos títulos y misma numeración (1, 2.1, 2.3, 3…) que el HTML, etapa por etapa.
2. Sin notas ni pasos de texto (conclusiones, discusión, interpretación).
3. Incluir los pasos que el código tiene y el HTML omitió (ver abajo).
4. Controles interactivos por paso: features, hiperparámetros, opciones.
5. Nuevo: visualización de las iteraciones de K-Means (formación de la nube de puntos en clusters).
6. Eliminar la postpoda (ccp_alpha) de todo el código.

## Decisiones

- **Tecnología:** Streamlit (`streamlit run app.py`). Se instala en `.venv-1` y se añade a `requirements.txt`.
- **Enfoque:** paquete nuevo `webapp/` (no `app/`, que chocaría con `app.py`) con una función de
  render por paso, que recibe parámetros. Reutiliza `src/` (datos, preprocesamiento,
  `plot_confusion`, `plot_k_curve`, métricas) y las definiciones de modelos de `etapas/`. Los
  pasos sin parámetros y la etapa 05 reutilizan el código de `etapas/` a través del gancho
  `salida.RECEPTOR` que ya existe en `src/salida.py`. El CLI (`main.py`, ▶ de cada etapa) sigue funcionando.
  Se descartó refactorizar `etapas/` para parametrizarlo (riesgo sobre el código de clase) y
  capturar las figuras de los pasos actuales (no aceptan parámetros).
- **Sin GridSearch por defecto:** se entrena con los hiperparámetros elegidos; botón «Entrenar»;
  resultados cacheados por combinación de parámetros.

## Estructura

```
app.py                  # entrada; sidebar con etapas/pasos; despacha al render
webapp/
  __init__.py
  navegacion.py         # árbol etapa -> pasos (ids y títulos idénticos al HTML)
  modelos.py            # núcleo sin Streamlit: clasificación y regresión
  clustering.py         # núcleo sin Streamlit: K-Means, análisis, PCA y Lloyd
  ui.py, receptor.py    # salida a Streamlit y gancho de salida.RECEPTOR
  controles.py          # widgets compartidos (features, split, semilla, hiperparámetros)
  etapa01.py … etapa05.py
tests/test_app_smoke.py # streamlit.testing: abre cada paso con parámetros por defecto
```

## Navegación y contenido

Sidebar: 5 etapas y, debajo, sus pasos con la numeración del HTML. Área principal: controles +
figuras + tablas del paso. Pasos de texto fuera: 2.6/3.6/3.7/4.5/4.6 y los de discusión y
conclusiones generales de la etapa 05.

Pasos y gráficas que el código tiene y el HTML no mostraba (verificado comparando los PNG de
`figures/` con los `data-src` del HTML y los `@etapa.paso` con los `<article class="step">`):
- Etapa 01: pasos 2.2 (distribución de Weight) y 2.4 (correlaciones); además la figura del paso 2.3.
- Etapa 03: la figura de residuos del paso 4.
- Etapa 04: el paso nuevo 3.5 (iteraciones de K-Means).

El orden y los títulos se comprueban con un test que compara el registro con el HTML.

## Controles

- **Clasificación (02) y regresión (03):** multiselect de features con atajos «todas» y
  «sin Weight/Height» (variantes A y B); tamaño de test; semilla.
  - Árbol: `criterion`, `max_depth`, `min_samples_leaf`.
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
  - KNN: K, `weights`, métrica (con StandardScaler en pipeline, como ahora).
  - 3.4: profundidad y `min_samples_leaf` editables para los tres árboles (prepoda).
  - 3.6: rango de K; 3.7: lista de números de árboles; 7: ruido 0–20 %; 8: filas reales sin SMOTE.
- **Clustering (04):** rango de K, `n_init`, inicialización, semilla; escalado con StandardScaler.
- **Etapa 05:** recalcula tablas y figura comparativa con los parámetros activos de las etapas
  anteriores (o los valores por defecto si no se ha entrenado nada).

## Iteraciones de K-Means (paso nuevo 3.5, etapa 04)

PCA 2D con puntos coloreados por cluster y centroides. Slider de iteración + botón ▶ de
autoplay. Implementa el algoritmo de Lloyd paso a paso (asignación → recálculo de centroides),
mostrando la inercia por iteración hasta converger. Controles: K, inicialización (aleatoria /
k-means++), semilla, `max_iter`. Las iteraciones se calculan una vez por combinación de
parámetros y se guardan como lista de (centroides, etiquetas, inercia).

## Postpoda (eliminar)

- `etapas/e02_clasificacion.py`: paso 3.5 y su entrada en el docstring.
- `etapas/e03_regresion.py`: paso 3.1 y las menciones en el docstring.
- `src/evaluation.py`: `plot_pruning_curve`.
- Cualquier otra referencia a `ccp_alpha` / `cost_complexity` (verificar con grep).
- Se mantienen los árboles «completo, podado y muy podado»: son prepoda.

## Fuera de alcance

- `estudio_etapas.html` y `.md` no se modifican (estáticos, generados).
- No se despliega la app; uso local.

## Pruebas

- `tests/test_app_smoke.py`: cada paso se renderiza con parámetros por defecto sin excepción.
- Comprobación de que no queda `ccp_alpha` en el código y de que las etapas del CLI siguen ejecutando.
- Verificación manual de las iteraciones de K-Means y de que cambiar parámetros cambia los resultados.

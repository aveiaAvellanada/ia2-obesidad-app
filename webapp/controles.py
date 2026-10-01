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

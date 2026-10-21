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

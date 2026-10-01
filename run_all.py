"""Ejecuta las 5 etapas completas, sin abrir ventanas, y guarda todas las figuras en figures/.

    python run_all.py                 # usa la caché de modelos si existe
    python run_all.py --limpiar-cache # recalcula todo desde cero (~2-3 min)

Es la forma de reproducir el proyecto entero de una vez (equivalente al run_all.sh
de la versión en notebooks).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import cache, salida

if __name__ == "__main__":
    salida.configurar(mostrar=False, guardar=True)
    if "--limpiar-cache" in sys.argv:
        cache.limpiar()
    t0 = time.time()
    from etapas import e01_eda, e02_clasificacion, e03_regresion, e04_clustering, e05_comparacion
    for mod in (e01_eda, e02_clasificacion, e03_regresion, e04_clustering, e05_comparacion):
        mod.etapa.ejecutar_todo()
    print(f"\nProyecto completo ejecutado en {time.time() - t0:.0f} s. Figuras en figures/.")

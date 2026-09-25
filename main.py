"""MENÚ PRINCIPAL del proyecto — punto de entrada para ver cualquier gráfica.

Cómo usarlo (ver GUIA.md para capturas de dónde hacer clic):
  1. Abre este archivo en VS Code y pulsa el botón ▶ "Run Python File" (arriba a la derecha),
     o en una terminal:  python main.py
  2. Elige la etapa (1-4) y luego el paso (ej. 3.4). Se abre una ventana con la gráfica y se
     guarda un PNG en figures/. Cierra la ventana para continuar.

Atajos directos sin menú:
  python main.py 02 3.4          -> etapa 02, paso 3.4
  python main.py 04 2.1 2.2 2.3  -> varios pasos seguidos
  python main.py todo            -> igual que run_all.py (todo, sin ventanas)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import cache, salida

ETAPAS = {}


def _cargar_etapas():
    """Importa las 4 etapas (import perezoso: así el menú arranca rápido)."""
    if not ETAPAS:
        from etapas import e01_eda, e02_clasificacion, e03_regresion, e04_clustering
        for mod in (e01_eda, e02_clasificacion, e03_regresion, e04_clustering):
            ETAPAS[mod.etapa.numero] = mod.etapa
    return ETAPAS


def menu_principal():
    salida.configurar(mostrar=True, guardar=True)
    etapas = _cargar_etapas()
    while True:
        print("\n" + "=" * 70)
        print("  IA2 — Estimation of Obesity Levels (UCI 544): menú principal")
        print("=" * 70)
        for num, et in etapas.items():
            print(f"  [{num[-1]}] {num} — {et.nombre}   ({len(et.pasos)} pasos)")
        print("  [t] Ejecutar TODO el proyecto en orden (sin ventanas, guarda PNG en figures/)")
        v = "ON " if salida.OPCIONES["mostrar"] else "OFF"
        g = "ON " if salida.OPCIONES["guardar"] else "OFF"
        b = "ON " if salida.OPCIONES["graficar_tablas"] else "OFF"
        print(f"  [v] Ventanas de gráficas: {v}     [g] Guardar PNG: {g}     [c] Limpiar caché")
        print(f"  [b] Gráfica de cada tabla (figures/<etapa>/tablas/): {b}")
        print("  [q] Salir")
        eleccion = input("\nElige: ").strip().lower()
        if eleccion in ("q", ""):
            print("Hasta luego.")
            return
        if eleccion == "t":
            ejecutar_todo()
        elif eleccion == "v":
            salida.configurar(mostrar=not salida.OPCIONES["mostrar"])
        elif eleccion == "g":
            salida.configurar(guardar=not salida.OPCIONES["guardar"])
        elif eleccion == "b":
            # No pasa por configurar() para no repetir la prueba del backend gráfico
            salida.OPCIONES["graficar_tablas"] = not salida.OPCIONES["graficar_tablas"]
        elif eleccion == "c":
            cache.limpiar()
        elif eleccion in ("1", "2", "3", "4"):
            etapas["0" + eleccion].menu()
        else:
            print("Opción no válida.")


def ejecutar_todo():
    """Corre las 4 etapas completas sin abrir ventanas (sólo guarda PNG)."""
    mostrar_antes = salida.OPCIONES["mostrar"]
    salida.configurar(mostrar=False, guardar=True)
    for et in _cargar_etapas().values():
        et.ejecutar_todo()
    salida.configurar(mostrar=mostrar_antes)
    print("\nListo. Todas las figuras están en figures/<etapa>/")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        menu_principal()
    elif args[0] == "todo":
        ejecutar_todo()
    else:
        # python main.py 02 3.4 [3.5 ...]
        salida.configurar(mostrar=True, guardar=True)
        et = _cargar_etapas()[args[0].zfill(2)]
        for paso_id in args[1:] or [p.id for p in et.pasos]:
            et.ejecutar(paso_id)

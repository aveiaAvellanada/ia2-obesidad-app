"""Mini-framework para organizar cada etapa en PASOS numerados.

Cada etapa (e01_eda.py, e02_clasificacion.py, ...) crea una `Etapa` y registra
funciones con el decorador `@etapa.paso("3.4", "título", figuras=4)`.
La numeración coincide con las secciones de los notebooks originales,
para que "ir a 3.4" signifique lo mismo en todos lados.

Cualquier paso puede ejecutarse solo: si necesita modelos entrenados, la etapa
los calcula bajo demanda (y los deja en caché). Ver `cli()` para las opciones
de línea de comandos y `menu()` para el menú interactivo.
"""
import argparse
import inspect
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src import salida, cache

# Raíz del proyecto (src/pasos.py está en <raiz>/src/), para mostrar rutas relativas
# como "etapas/e02_clasificacion.py:412" en vez de una ruta absoluta.
_RAIZ_PROYECTO = Path(__file__).resolve().parents[1]


@dataclass
class Paso:
    id: str
    titulo: str
    fn: Callable
    figuras: int = 0

    @property
    def ubicacion(self) -> str:
        """'etapas/e02_clasificacion.py:412 (hasta 431)': archivo (relativo a la raíz
        del proyecto) y líneas donde empieza (el decorador @etapa.paso) y termina la
        función de este paso, obtenidas en tiempo de ejecución con
        inspect.getsourcelines() — nunca se escriben a mano, así que no se
        desactualizan si el archivo cambia. 'archivo:línea' va sin espacios para que
        la terminal integrada de VS Code lo detecte como enlace clickeable (Ctrl+clic)."""
        try:
            archivo = Path(inspect.getfile(self.fn)).resolve().relative_to(_RAIZ_PROYECTO)
            lineas, inicio = inspect.getsourcelines(self.fn)
            return f"{archivo.as_posix()}:{inicio} (hasta {inicio + len(lineas) - 1})"
        except (OSError, TypeError, ValueError):
            return ""

    @property
    def etiqueta(self):
        figs = f"  ({self.figuras} figura{'s' if self.figuras != 1 else ''})" if self.figuras else ""
        ubi = f"  {self.ubicacion}" if self.ubicacion else ""
        return f"[{self.id:>4}] {self.titulo}{figs}{ubi}"


@dataclass
class Etapa:
    numero: str          # "01", "02", ...
    nombre: str          # "EDA y preprocesamiento"
    slug: str            # nombre corto sin acentos para la carpeta de figuras: "eda", "clasificacion"
    descripcion: str = ""
    pasos: list = field(default_factory=list)

    # ---- registro ----
    def paso(self, id: str, titulo: str, figuras: int = 0):
        def decorador(fn):
            self.pasos.append(Paso(id, titulo, fn, figuras))
            return fn
        return decorador

    @property
    def carpeta_figuras(self):
        return f"{self.numero}_{self.slug}"

    def figura(self, fig, nombre: str, datos=None):
        """Atajo: guarda/muestra una figura en la carpeta de esta etapa."""
        salida.mostrar(fig, nombre, self.carpeta_figuras, datos=datos)

    # ---- ejecución ----
    def buscar(self, id: str) -> Paso:
        for p in self.pasos:
            if p.id == id:
                return p
        raise KeyError(f"No existe el paso '{id}' en la etapa {self.numero}. Pasos: "
                       + ", ".join(p.id for p in self.pasos))

    def ejecutar(self, id: str):
        p = self.buscar(id)
        # Registra el paso en curso: las tablas lo usan para nombrar sus PNG
        # (figures/<etapa>/tablas/<paso>_tablaN_<titulo>.png)
        salida.fijar_contexto(self.carpeta_figuras, p.id)
        salida.titulo(f"Etapa {self.numero} — {self.nombre}   ·   Paso {p.id}: {p.titulo}"
                      + (f"\n{p.ubicacion}" if p.ubicacion else ""))
        t0 = time.time()
        p.fn()
        print(f"\n   (paso {p.id} terminado en {time.time() - t0:.1f} s)")

    def ejecutar_todo(self):
        salida.titulo(f"Etapa {self.numero} — {self.nombre}: ejecutando los {len(self.pasos)} pasos")
        for p in self.pasos:
            self.ejecutar(p.id)

    def listar(self):
        print(f"\nEtapa {self.numero} — {self.nombre}")
        if self.descripcion:
            print("  " + self.descripcion)
        for p in self.pasos:
            print("  " + p.etiqueta)

    # ---- menú interactivo de la etapa ----
    def menu(self):
        while True:
            self.listar()
            print("\n  [t] ejecutar TODOS los pasos    [v] volver / salir")
            eleccion = input("Elige un paso (ej. 3.4): ").strip().lower()
            if eleccion in ("v", "q", ""):
                return
            if eleccion == "t":
                self.ejecutar_todo()
                continue
            try:
                self.ejecutar(eleccion)
            except KeyError as e:
                print(e)


def cli(etapa: Etapa):
    """Línea de comandos común a todas las etapas.

    python etapas/e02_clasificacion.py              -> menú interactivo de la etapa
    python etapas/e02_clasificacion.py --todo       -> ejecuta todos los pasos
    python etapas/e02_clasificacion.py --paso 3.4   -> sólo ese paso (se pueden pasar varios)
    python etapas/e02_clasificacion.py --lista      -> lista de pasos
    Por defecto sólo MUESTRA: figuras en ventana y tablas/texto en la consola, sin escribir
    archivos. Opciones: --guardar (además guarda PNG en figures/ y tablas en
    resultados_texto/)  --sin-ventanas (no abre ventanas)  --limpiar-cache
    """
    ap = argparse.ArgumentParser(description=f"Etapa {etapa.numero} — {etapa.nombre}")
    ap.add_argument("--paso", nargs="+", metavar="ID", help="id(s) de paso a ejecutar, ej. 3.4")
    ap.add_argument("--todo", action="store_true", help="ejecutar todos los pasos en orden")
    ap.add_argument("--lista", action="store_true", help="listar los pasos y salir")
    ap.add_argument("--sin-ventanas", action="store_true", help="no abrir ventanas de gráficas")
    ap.add_argument("--guardar", action="store_true",
                    help="guardar también PNG en figures/ y tablas en resultados_texto/")
    ap.add_argument("--sin-guardar", action="store_true",
                    help="(por defecto ya no se guarda; se mantiene por compatibilidad)")
    ap.add_argument("--limpiar-cache", action="store_true", help="borrar modelos cacheados antes de empezar")
    args = ap.parse_args()

    salida.configurar(mostrar=not args.sin_ventanas, guardar=args.guardar and not args.sin_guardar)
    if args.limpiar_cache:
        cache.limpiar()
    if args.lista:
        etapa.listar()
    elif args.todo:
        etapa.ejecutar_todo()
    elif args.paso:
        for id in args.paso:
            etapa.ejecutar(id)
    else:
        etapa.menu()
    return 0

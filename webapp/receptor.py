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

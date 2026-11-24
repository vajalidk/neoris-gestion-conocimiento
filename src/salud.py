"""
Diagnóstico del sistema (incidente 1: falta de memoria en el servidor).

Idea: el sistema nunca debe dejar de funcionar.
  - Mide la memoria y el tiempo de cada operación.
  - Si la memoria pasa del 80% del límite, entra en "modo ligero"
    (búsqueda solo por palabras, chatbot sin modelo de lenguaje y grafo sin animación).
  - Si una operación falla, se registra el error y se usa una respuesta de respaldo.
  - Desde la pestaña Diagnóstico se puede liberar memoria o reconstruir el índice.
"""
import gc
import os
import time
from collections import deque
from datetime import datetime

import psutil

LIMITE_MB = int(os.getenv("NEORIS_LIMITE_MB", "800"))  # Streamlit Cloud da cerca de 1 GB


class Salud:
    def __init__(self):
        self.eventos = deque(maxlen=50)  # bitácora de las últimas operaciones
        self.forzar_ligero = False       # para probar el modo ligero a mano

    def memoria_mb(self):
        return psutil.Process().memory_info().rss / 1024 ** 2

    def modo_ligero(self):
        return self.forzar_ligero or self.memoria_mb() > 0.8 * LIMITE_MB

    def registrar(self, tipo, mensaje, ms=0.0):
        self.eventos.appendleft({"hora": datetime.now().strftime("%H:%M:%S"), "tipo": tipo,
                                 "mensaje": mensaje, "ms": round(ms, 1), "memoria_mb": round(self.memoria_mb())})

    def ejecutar(self, nombre, funcion, respaldo=None):
        """Corre una operación midiendo tiempo. Si falla, registra el error y regresa el respaldo."""
        inicio = time.perf_counter()
        try:
            resultado = funcion()
            self.registrar("ok", nombre, (time.perf_counter() - inicio) * 1000)
            return resultado
        except MemoryError:
            gc.collect()
            self.forzar_ligero = True
            self.registrar("error", f"{nombre}: sin memoria, se activó el modo ligero")
        except Exception as e:
            self.registrar("error", f"{nombre}: {type(e).__name__}: {e}")
        return respaldo

    def liberar(self):
        antes = self.memoria_mb()
        gc.collect()
        self.registrar("acción", f"Memoria liberada: {antes:.0f} MB -> {self.memoria_mb():.0f} MB")

    def estado(self):
        mem = psutil.virtual_memory()
        return {"proceso_mb": round(self.memoria_mb()), "limite_mb": LIMITE_MB,
                "uso_limite": min(self.memoria_mb() / LIMITE_MB, 1.0), "sistema_pct": mem.percent,
                "modo": "Ligero" if self.modo_ligero() else "Normal",
                "errores": sum(1 for e in self.eventos if e["tipo"] == "error")}

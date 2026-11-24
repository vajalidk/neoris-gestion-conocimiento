"""
Evaluación del sistema de recomendación de documentos - Neoris

Este script NO modifica el recomendador. Solo mide qué tan buenos son sus resultados.

Etapas:
  1. Definir la consulta y cuántos documentos recomienda el sistema (top K)
  2. Definir la verdad de referencia (qué documentos SÍ son relevantes)
  3. Pedirle al sistema sus recomendaciones
  4. Armar la tabla verdad vs. recuperado para los 42 documentos
  5. Calcular Precision, Recall y F1-score con scikit-learn
  6. Generar la matriz de confusión
  7. Graficar los resultados

Uso:  python src/evaluacion.py
"""
import matplotlib
matplotlib.use("Agg")  # para guardar las gráficas sin abrir ventanas
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             f1_score, precision_score, recall_score)

from busqueda_semantica import buscar, documentos  # el recomendador de la Actividad 3, sin cambios
from utilidades import CARPETA_RESULTADOS

# ---------------------------------------------------------------
# 1. Consulta a evaluar
# ---------------------------------------------------------------
CONSULTA = "qué hacer cuando falla un servidor en la nube"
K = 5  # el sistema recomienda los 5 documentos más parecidos

# ---------------------------------------------------------------
# 2. Verdad de referencia (ground truth)
#    Necesidad de información: "necesito documentos que me ayuden a detectar,
#    resolver o prevenir la falla de un servidor en la nube y recuperar el servicio".
#    Revisé los 42 documentos a mano. Solo estos 4 responden esa necesidad:
# ---------------------------------------------------------------
RELEVANTES = {
    "INF-001": "Reporte de la caída del servidor por falta de memoria y cómo se resolvió",
    "INF-002": "Respaldos de bases de datos para recuperar información si algo falla",
    "INF-005": "Monitoreo y alertas para detectar a tiempo que un servidor está fallando",
    "INF-006": "Recuperación ante desastres: cómo levantar los servicios si falla la nube",
}
# Ejemplos de documentos que NO son relevantes aunque hablen de servidores o nube:
#   INF-007 ataque de fuerza bruta: el servidor no falló, es un tema de seguridad
#   INF-008 control de costos de la nube: no ayuda a resolver una falla
#   INF-003 configuración de máquinas virtuales: es para crearlas, no para cuando fallan


def linea(titulo):
    print("\n" + "=" * 72 + f"\n{titulo}\n" + "=" * 72)


linea("1. CONSULTA EVALUADA")
print(f'Consulta: "{CONSULTA}"')
print(f"Documentos en el corpus: {len(documentos)} | El sistema recomienda: top {K}")

linea("2. VERDAD DE REFERENCIA (documentos relevantes)")
for doc_id, motivo in RELEVANTES.items():
    print(f"  {doc_id}: {motivo}")

# ---------------------------------------------------------------
# 3. Recomendaciones del sistema
# ---------------------------------------------------------------
linea(f"3. DOCUMENTOS RECOMENDADOS POR EL SISTEMA (top {K})")
recomendados = buscar(CONSULTA, top=K)
ids_recuperados = [doc["id"] for doc, _ in recomendados]
for pos, (doc, score) in enumerate(recomendados, 1):
    marca = "relevante" if doc["id"] in RELEVANTES else "NO relevante"
    print(f"  {pos}. {doc['id']} | similitud {score:.3f} | {marca} | {doc['titulo']}")

# ---------------------------------------------------------------
# 4. Tabla verdad vs. recuperado (1 = sí, 0 = no) para los 42 documentos
# ---------------------------------------------------------------
y_true = [1 if d["id"] in RELEVANTES else 0 for d in documentos]      # ¿es relevante?
y_pred = [1 if d["id"] in ids_recuperados else 0 for d in documentos]  # ¿el sistema lo recomendó?


def tipo(v, p):
    return {(1, 1): "VP", (0, 1): "FP", (1, 0): "FN", (0, 0): "VN"}[(v, p)]


linea("4. TABLA: VERDAD DE REFERENCIA VS. RECUPERADO")
print(f"{'Documento':<11}{'Relevante':<11}{'Recuperado':<12}{'Resultado'}")
for d, v, p in zip(documentos, y_true, y_pred):
    print(f"{d['id']:<11}{'Sí' if v else 'No':<11}{'Sí' if p else 'No':<12}{tipo(v, p)}")
print("VP = verdadero positivo, FP = falso positivo, FN = falso negativo, VN = verdadero negativo")

# ---------------------------------------------------------------
# 5. Métricas con scikit-learn
# ---------------------------------------------------------------
precision = precision_score(y_true, y_pred)  # de lo que recomendó, cuánto sirve
recall = recall_score(y_true, y_pred)        # de lo que sirve, cuánto encontró
f1 = f1_score(y_true, y_pred)                # equilibrio entre las dos

linea("5. MÉTRICAS")
print(f"Precision: {precision:.3f}  ({precision:.0%})")
print(f"Recall:    {recall:.3f}  ({recall:.0%})")
print(f"F1-score:  {f1:.3f}  ({f1:.0%})")

# ---------------------------------------------------------------
# 6. Matriz de confusión
# ---------------------------------------------------------------
matriz = confusion_matrix(y_true, y_pred, labels=[1, 0])
vp, fn, fp, vn = matriz.ravel()
linea("6. MATRIZ DE CONFUSIÓN")
print(f"                     Recuperado: Sí   Recuperado: No")
print(f"Relevante: Sí        VP = {vp:<12} FN = {fn}")
print(f"Relevante: No        FP = {fp:<12} VN = {vn}")

# ---------------------------------------------------------------
# 7. Gráficas
# ---------------------------------------------------------------
CARPETA_RESULTADOS.mkdir(exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

# 7a. Barras con las tres métricas
nombres = ["Precision", "Recall", "F1-score"]
valores = [precision, recall, f1]
barras = ax1.bar(nombres, valores, color=["#2E6FD8", "#17A589", "#E0A100"])
ax1.set_ylim(0, 1.05)
ax1.set_title("Métricas del recomendador")
ax1.set_ylabel("Valor (0 a 1)")
for barra, valor in zip(barras, valores):
    ax1.text(barra.get_x() + barra.get_width() / 2, valor + 0.02, f"{valor:.2f}", ha="center", fontweight="bold")

# 7b. Matriz de confusión
ConfusionMatrixDisplay(matriz, display_labels=["Relevante", "No relevante"]).plot(
    ax=ax2, cmap="Blues", colorbar=False, values_format="d")
ax2.set_xlabel("Recomendado por el sistema")
ax2.set_ylabel("Verdad de referencia")
ax2.set_xticklabels(["Sí", "No"]); ax2.set_yticklabels(["Relevante", "No relevante"])
ax2.set_title("Matriz de confusión")

fig.suptitle(f'Evaluación de la consulta: "{CONSULTA}" (top {K})', fontsize=11)
plt.tight_layout()
plt.savefig(CARPETA_RESULTADOS / "evaluacion_metricas.png", dpi=150)

# 7c. Extra: cómo cambian las métricas si el sistema recomienda más o menos documentos
ks = list(range(1, 11))
p_k, r_k, f_k = [], [], []
for k in ks:
    ids_k = [doc["id"] for doc, _ in buscar(CONSULTA, top=k)]
    pred_k = [1 if d["id"] in ids_k else 0 for d in documentos]
    p_k.append(precision_score(y_true, pred_k, zero_division=0))
    r_k.append(recall_score(y_true, pred_k))
    f_k.append(f1_score(y_true, pred_k))

plt.figure(figsize=(7, 4.2))
plt.plot(ks, p_k, "o-", label="Precision", color="#2E6FD8")
plt.plot(ks, r_k, "s-", label="Recall", color="#17A589")
plt.plot(ks, f_k, "^-", label="F1-score", color="#E0A100")
plt.axvline(K, color="gray", linestyle="--", linewidth=1)
plt.text(K + 0.1, 0.05, f"K usado = {K}", color="gray")
plt.xticks(ks); plt.ylim(0, 1.05)
plt.xlabel("Número de documentos recomendados (K)")
plt.ylabel("Valor")
plt.title("Precision y Recall según cuántos documentos recomienda")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CARPETA_RESULTADOS / "evaluacion_por_k.png", dpi=150)

linea("7. MÉTRICAS SEGÚN K")
print(f"{'K':<4}{'Precision':<12}{'Recall':<10}{'F1'}")
for k, p, r, f in zip(ks, p_k, r_k, f_k):
    print(f"{k:<4}{p:<12.2f}{r:<10.2f}{f:.2f}")

print("\nGráficas guardadas en resultados/evaluacion_metricas.png y resultados/evaluacion_por_k.png")

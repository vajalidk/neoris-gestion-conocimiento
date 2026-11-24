"""
Evaluación final del sistema de Neoris.

Compara el sistema del avance (TF-IDF simple) contra el sistema final (motor híbrido):
  1. Recuperación: Precision y Recall en 10 consultas de prueba (datos/consultas_prueba.json).
     El corte de resultados se ajustó con otras 15 consultas (datos/consultas_validacion.json),
     así las consultas de prueba solo se usan para medir.
  2. Clasificación: Precision y Recall con validación cruzada de 5 partes repetida 5 veces
     (con 100 documentos, una sola partición cambia mucho el resultado)
  3. Recomendaciones: qué tan seguido los documentos sugeridos son de la misma categoría

Uso:  python src/evaluacion_final.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline, make_union

from motor import Motor, dividir, normalizar
from utilidades import CARPETA_RESULTADOS, PALABRAS_VACIAS, RAIZ

K = 5          # documentos que regresa el sistema por consulta
K_TEMA = 3     # en consultas de dos temas, documentos por tema


def recuperados_final(motor, consulta):
    temas = dividir(consulta)
    if len(temas) == 1:
        return [r["id"] for r in motor.buscar(consulta, K)]
    return list(dict.fromkeys(r["id"] for t in temas for r in motor.buscar(t, K_TEMA)))


def recuperados_avance(vec, X, ids, consulta):
    from sklearn.metrics.pairwise import cosine_similarity
    s = cosine_similarity(vec.transform([consulta]), X)[0]
    return [ids[i] for i in s.argsort()[::-1][:K] if s[i] > 0]


def metricas(ids, relevantes, recuperados):
    """Vectores 1/0 sobre los 100 documentos, igual que en la Actividad 4."""
    y_true = [int(i in relevantes) for i in ids]
    y_pred = [int(i in recuperados) for i in ids]
    return (precision_score(y_true, y_pred, zero_division=0), recall_score(y_true, y_pred), f1_score(y_true, y_pred))


def evaluar():
    motor = Motor()
    ids, textos = motor.ids, [d["texto"] for d in motor.docs]
    etiquetas = [d["categoria"] for d in motor.docs]

    # Sistema del avance: TF-IDF con palabras exactas (como busqueda_semantica.py)
    vec_avance = TfidfVectorizer(strip_accents="unicode", stop_words=PALABRAS_VACIAS, sublinear_tf=True)
    X_avance = vec_avance.fit_transform(textos)

    # ---------- 1. Recuperación ----------
    pruebas = json.loads((RAIZ / "datos" / "consultas_prueba.json").read_text(encoding="utf-8"))
    filas = []
    for p in pruebas:
        rec_a = recuperados_avance(vec_avance, X_avance, ids, p["consulta"])
        rec_f = recuperados_final(motor, p["consulta"])
        pa, ra, fa = metricas(ids, p["relevantes"], rec_a)
        pf, rf, ff = metricas(ids, p["relevantes"], rec_f)
        filas.append({**p, "recuperados_avance": rec_a, "recuperados_final": rec_f,
                      "avance": [round(pa, 3), round(ra, 3), round(fa, 3)], "final": [round(pf, 3), round(rf, 3), round(ff, 3)],
                      "no_encontrados_final": [r for r in p["relevantes"] if r not in rec_f]})
    prom = lambda clave, j: round(float(np.mean([f[clave][j] for f in filas])), 3)
    recuperacion = {"avance": {"precision": prom("avance", 0), "recall": prom("avance", 1), "f1": prom("avance", 2)},
                    "final": {"precision": prom("final", 0), "recall": prom("final", 1), "f1": prom("final", 2)},
                    "consultas": filas}

    # Consultas de validación (las que se usaron para ajustar el corte)
    validacion = json.loads((RAIZ / "datos" / "consultas_validacion.json").read_text(encoding="utf-8"))
    pv = [metricas(ids, v["relevantes"], recuperados_final(motor, v["consulta"])) for v in validacion]
    recuperacion["validacion"] = {"precision": round(float(np.mean([x[0] for x in pv])), 3),
                                  "recall": round(float(np.mean([x[1] for x in pv])), 3), "consultas": len(validacion)}

    # ---------- 2. Clasificación (validación cruzada) ----------
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    medidas = ["precision_macro", "recall_macro", "f1_macro", "accuracy"]
    modelos = {
        "avance": make_pipeline(TfidfVectorizer(strip_accents="unicode", stop_words=PALABRAS_VACIAS, ngram_range=(1, 2),
                                                sublinear_tf=True), LogisticRegression(max_iter=2000, C=10)),
        "final": make_pipeline(make_union(TfidfVectorizer(strip_accents="unicode", stop_words=PALABRAS_VACIAS, ngram_range=(1, 2), sublinear_tf=True),
                                          TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=normalizar, sublinear_tf=True)),
                               LogisticRegression(max_iter=2000, C=10)),
    }
    clasificacion = {}
    for nombre, modelo in modelos.items():
        r = cross_validate(modelo, textos, etiquetas, cv=cv, scoring=medidas)
        clasificacion[nombre] = {m.replace("_macro", ""): round(float(r[f"test_{m}"].mean()), 3) for m in medidas}

    # ---------- 3. Recomendaciones ----------
    aciertos = []
    for i, d in enumerate(motor.docs):
        sim = [motor.ids.index(r["id"]) for r in motor.similares(d["id"], 3)]
        aciertos.append(np.mean([etiquetas[j] == etiquetas[i] for j in sim]))
    casos = {semilla: [(r["id"], round(r["score"], 3), r["terminos"][:4]) for r in motor.similares(semilla, 3)]
             for semilla in ["INF-001", "SOP-009", "POL-002"]}
    recomendaciones = {"precision_misma_categoria": round(float(np.mean(aciertos)), 3), "casos": casos}

    resultados = {"recuperacion": recuperacion, "clasificacion": clasificacion, "recomendaciones": recomendaciones}
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    (CARPETA_RESULTADOS / "evaluacion_final.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    graficar(resultados)
    return resultados


def graficar(r):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    nombres = ["Precision", "Recall", "F1-score"]
    x = np.arange(3)
    for ax, datos, titulo in [(a1, r["recuperacion"], "Recuperación (10 consultas, top 5)"),
                              (a2, r["clasificacion"], "Clasificación (validación cruzada 5x5)")]:
        va = [datos["avance"]["precision"], datos["avance"]["recall"], datos["avance"]["f1"]]
        vf = [datos["final"]["precision"], datos["final"]["recall"], datos["final"]["f1"]]
        b1 = ax.bar(x - 0.2, va, 0.4, label="Avance", color="#AAB4C8")
        b2 = ax.bar(x + 0.2, vf, 0.4, label="Sistema final", color="#2E6FD8")
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}", ha="center", fontsize=9)
        ax.set_xticks(x, nombres); ax.set_ylim(0, 1.1); ax.set_title(titulo); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(CARPETA_RESULTADOS / "evaluacion_final.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    r = evaluar()
    print("=" * 70 + "\nRECUPERACIÓN (promedio de 10 consultas, top 5)\n" + "=" * 70)
    print(f"{'Consulta':<62}{'Tipo':<22}{'P avance':>9}{'P final':>9}{'R avance':>9}{'R final':>9}")
    for f in r["recuperacion"]["consultas"]:
        print(f"{f['consulta'][:60]:<62}{f['tipo'][:20]:<22}{f['avance'][0]:>9.2f}{f['final'][0]:>9.2f}{f['avance'][1]:>9.2f}{f['final'][1]:>9.2f}")
    for s in ["avance", "final"]:
        m = r["recuperacion"][s]
        print(f"{s.upper():<8} Precision {m['precision']:.3f} | Recall {m['recall']:.3f} | F1 {m['f1']:.3f}")
    v = r["recuperacion"]["validacion"]
    print(f"(Validación, {v['consultas']} consultas usadas para ajustar el corte: Precision {v['precision']:.3f} | Recall {v['recall']:.3f})")
    print("\n" + "=" * 70 + "\nCLASIFICACIÓN (validación cruzada de 5 partes, repetida 5 veces)\n" + "=" * 70)
    for s, m in r["clasificacion"].items():
        print(f"{s.upper():<8} Precision {m['precision']:.3f} | Recall {m['recall']:.3f} | F1 {m['f1']:.3f} | Exactitud {m['accuracy']:.3f}")
    print("\n" + "=" * 70 + "\nRECOMENDACIONES (documentos parecidos)\n" + "=" * 70)
    print(f"Recomendaciones de la misma categoría: {r['recomendaciones']['precision_misma_categoria']:.1%}")
    for semilla, recs in r["recomendaciones"]["casos"].items():
        print(f"  {semilla} -> " + " | ".join(f"{i} ({s:.2f}) {t}" for i, s, t in recs))
    print("\nResultados guardados en resultados/evaluacion_final.json y resultados/evaluacion_final.png")

"""
Chatbot de Neoris: responde preguntas usando solo los documentos del repositorio.

Cómo funciona (recuperación de contexto):
  1. Reglas simples para saludos y ayuda.
  2. Divide la pregunta si trae varios temas (incidente 3).
  3. Busca los documentos más parecidos con el motor.
  4. Elige las oraciones de esos documentos que más se parecen a la pregunta y arma la respuesta
     citando de qué documento sale cada parte.
  5. Opcional: si hay una clave de API (ANTHROPIC_API_KEY), un modelo de lenguaje redacta la respuesta
     con esos mismos documentos. Si falla o no hay clave, se usa la respuesta del paso 4 (incidente 1).
"""
import os
import re

from sklearn.metrics.pairwise import cosine_similarity

from motor import dividir, expandir, normalizar

MINIMO = 0.12  # si el mejor documento no llega a este puntaje, el bot dice que no sabe
SALUDOS = r"^(hola|buenas|buenos dias|buenas tardes|buenas noches|hey|que tal)\b"
AYUDA = r"\b(ayuda|que puedes hacer|como funcionas|que sabes)\b"


def responder(motor, pregunta, ligero=False):
    """Regresa un diccionario con el texto de la respuesta, las fuentes y el modo usado."""
    p = normalizar(pregunta.strip())
    if re.search(SALUDOS, p) and len(p.split()) <= 3:
        return {"texto": "¡Hola! Soy el asistente de conocimiento de Neoris. Pregúntame sobre soporte, "
                         "desarrollo, infraestructura, proyectos o políticas internas.", "fuentes": [], "modo": "regla"}
    if re.search(AYUDA, p):
        return {"texto": f"Busco en los {len(motor.docs)} documentos de Neoris y te respondo con lo que dicen. Ejemplos:\n"
                         "- ¿Qué hago si un usuario está bloqueado?\n- ¿Cómo se respaldan las bases de datos?\n"
                         "- ¿Cuál es la política de trabajo remoto?", "fuentes": [], "modo": "regla"}

    partes, fuentes = [], []
    for tema in dividir(pregunta):
        docs = motor.buscar(tema, k=2 if ligero else 3)
        if not docs or docs[0]["score"] < MINIMO:
            partes.append(f"Sobre *\"{tema}\"* no encontré información en los documentos. "
                          "Intenta con otras palabras o revisa la pestaña de búsqueda.")
            continue
        fuentes += docs
        partes.append(_extracto(motor, tema, docs))

    texto, modo = "\n\n".join(partes), "recuperación"
    if fuentes and not ligero:
        generado = _con_modelo(pregunta, fuentes)
        if generado:
            texto, modo = generado, "modelo de lenguaje"
    return {"texto": texto, "fuentes": fuentes, "modo": modo, "temas": dividir(pregunta)}


SOLUCION = r"\b(soluci|resolv|corrig|recomienda|para evitar|reinici|renov|configur|asign|instal|desbloque|agreg|aument|activ|verifica|debe|deben|hay que|se prueba|se levantan)"


def _extracto(motor, tema, docs):
    """Arma la respuesta con el documento principal: la oración más parecida a la pregunta
    y las oraciones que dicen qué hacer (solución o recomendación)."""
    principal = docs[0]
    cuerpo = principal["texto"][len(principal["titulo"]):]  # sin el título
    oraciones = [o for o in re.split(r"(?<=[.!?])\s+", cuerpo.strip()) if len(o) > 20]
    q = motor.vec_palabras.transform([expandir(tema)])
    puntajes = cosine_similarity(q, motor.vec_palabras.transform(oraciones))[0]
    elegidas = {int(puntajes.argmax())}
    elegidas |= {i for i, o in enumerate(oraciones) if re.search(SOLUCION, normalizar(o))}
    elegidas = sorted(elegidas)[:3]  # máximo 3 oraciones, en el orden del documento
    lineas = [f"**Según {principal['id']} ({principal['titulo']}):**"] + [f"- {oraciones[i]}" for i in elegidas]
    # Si otro documento también es muy parecido, se sugiere revisarlo
    extra = [d for d in docs[1:] if d["score"] >= 0.6 * principal["score"]]
    if extra:
        lineas.append("\nTambién te puede servir: " +", ".join(f"{d['id']} ({d['titulo']})" for d in extra))
    return "\n".join(lineas)


def _con_modelo(pregunta, fuentes):
    """Respuesta redactada por un modelo de lenguaje, solo con el contexto recuperado."""
    clave = os.getenv("ANTHROPIC_API_KEY")
    if not clave:
        return None
    try:
        import anthropic
        contexto = "\n\n".join(f"[{d['id']}] {d['texto']}" for d in fuentes)
        msg = anthropic.Anthropic(api_key=clave).messages.create(
            model=os.getenv("NEORIS_MODELO", "claude-haiku-5-5"), max_tokens=400,
            system="Eres el asistente de conocimiento de Neoris. Responde en español, breve, usando SOLO los "
                   "documentos dados y citando su id entre corchetes. Si no están en los documentos, dilo.",
            messages=[{"role": "user", "content": f"Documentos:\n{contexto}\n\nPregunta: {pregunta}"}])
        return msg.content[0].text
    except Exception:
        return None  # si falla el modelo, el sistema sigue con la respuesta por recuperación

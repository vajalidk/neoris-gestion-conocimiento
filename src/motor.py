"""
Motor del sistema de gestión del conocimiento de Neoris.

Hace cuatro cosas con el mismo índice:
  1. Búsqueda híbrida (palabras con raíz + pedazos de letras + sinónimos)
  2. Recomendaciones (documentos parecidos y por historial de consultas)
  3. Clasificación automática de documentos nuevos
  4. División de consultas con dos temas

Incidentes que atiende:
  - Incidente 2 (consultas mal interpretadas): raíces de palabras, pedazos de letras
    para faltas de ortografía, diccionario de sinónimos y embeddings opcionales.
  - Incidente 3 (dos temas en una consulta): se divide la consulta y se busca cada tema.
"""
import os
import re
import unicodedata

import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from utilidades import NOMBRES_CATEGORIAS, PALABRAS_VACIAS, cargar_documentos

RAIZ_PALABRA = SnowballStemmer("spanish").stem
VACIAS = set(PALABRAS_VACIAS) | {"que", "hacer", "hago", "necesito", "quiero", "como", "cual", "cuales"}
UMBRAL_CLASIFICACION = 0.45  # si la confianza es menor, va a revisión manual (los documentos correctos salen con 50% a 75%)
PESO_PALABRAS, PESO_LETRAS = 0.7, 0.3
CORTE_RELATIVO = 0.55  # solo se muestran documentos con al menos 55% del puntaje del mejor (ajustado con datos/consultas_validacion.json)

# Incidente 2: palabras que usa la gente -> palabras que usan los documentos de Neoris
SINONIMOS = {
    "hackear": "ataque acceso seguridad", "hackeo": "ataque acceso seguridad", "hacker": "ataque seguridad",
    "home office": "trabajo remoto casa", "desde casa": "trabajo remoto", "remoto": "trabajar casa",
    "bd": "base datos", "backup": "respaldo", "respaldar": "respaldo",
    "caido": "caida falla responde", "cayo": "caida falla responde", "se cae": "caida falla",
    "no funciona": "falla error", "truena": "falla error", "lento": "tarda rendimiento",
    "inhabilitado": "bloqueado", "acceder": "entrar iniciar sesion", "login": "iniciar sesion",
    "password": "contrasena", "clave": "contrasena", "bug": "error corrigio",
    "deploy": "despliegue desplegar", "server": "servidor", "down": "caida", "memory": "memoria",
    "costo": "costos presupuesto", "vacaciones": "vacaciones permisos", "sueldo": "recursos humanos",
}


def normalizar(texto):
    """Minúsculas y sin acentos."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def tokens(texto):
    """Palabras importantes reducidas a su raíz: 'servidores' y 'servidor' -> 'servidor'."""
    return [RAIZ_PALABRA(p) for p in re.findall(r"[a-z0-9]+", normalizar(texto)) if p not in VACIAS and len(p) > 1]


def expandir(consulta):
    """Agrega sinónimos del diccionario a la consulta."""
    texto = normalizar(consulta)
    extra = [v for k, v in SINONIMOS.items() if re.search(rf"\b{k}\b", texto)]
    return " ".join([consulta] + extra)


def dividir(consulta):
    """Incidente 3: separa una consulta con varios temas.
    'la cuenta está bloqueada y también venció el SSL' -> ['la cuenta está bloqueada', 'venció el SSL']"""
    partes = re.split(r"\s*(?:;|\?|\by tambi[eé]n\b|\badem[aá]s\b|\by por otro lado\b|\bpor otro lado\b|\botra cosa\b)\s*",
                      consulta, flags=re.I)
    partes = [p.strip(" ,.") for p in partes if len(tokens(p)) >= 2]
    return partes if len(partes) > 1 else [consulta]


class Motor:
    def __init__(self, documentos=None):
        self.docs = documentos or cargar_documentos()
        self.ids = [d["id"] for d in self.docs]
        textos = [d["texto"] for d in self.docs]

        # Índice 1: palabras con raíz (para el significado)
        self.vec_palabras = TfidfVectorizer(analyzer=tokens, sublinear_tf=True)
        self.X_palabras = self.vec_palabras.fit_transform(textos)
        # Índice 2: pedazos de 3 a 5 letras (aguanta faltas de ortografía)
        self.vec_letras = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                          preprocessor=normalizar, sublinear_tf=True)
        self.X_letras = self.vec_letras.fit_transform(textos)
        # Similitud entre documentos (para recomendaciones), se calcula una sola vez
        self.sim_docs = self._similitud(self.X_palabras, self.X_letras, self.X_palabras, self.X_letras)

        # Clasificador: regresión logística sobre las palabras con raíz
        self.clf = LogisticRegression(max_iter=2000, C=10).fit(self.X_palabras, [d["categoria"] for d in self.docs])
        self.embeddings = cargar_embeddings(textos)  # opcional, None si no está disponible

    @staticmethod
    def _similitud(Ap, Al, Bp, Bl):
        return PESO_PALABRAS * cosine_similarity(Ap, Bp) + PESO_LETRAS * cosine_similarity(Al, Bl)

    def _vector_consulta(self, consulta, ligero=False):
        texto = expandir(consulta)
        qp = self.vec_palabras.transform([texto])
        if ligero:  # modo ligero: solo palabras, menos cálculo
            return cosine_similarity(qp, self.X_palabras)[0]
        s = self._similitud(qp, self.vec_letras.transform([texto]), self.X_palabras, self.X_letras)[0]
        if self.embeddings is not None:
            s = 0.5 * s + 0.5 * self.embeddings.similitud(texto)
        return s

    def terminos_comunes(self, texto, i, maximo=5):
        """Palabras que comparten la consulta y el documento: explican por qué se recomendó."""
        raices = set(tokens(expandir(texto)))
        vistas, comunes = set(), []
        for palabra in re.findall(r"\w+", self.docs[i]["texto"]):
            r = RAIZ_PALABRA(normalizar(palabra))
            if r in raices and r not in vistas:
                vistas.add(r); comunes.append(palabra.lower())
        return comunes[:maximo]

    def buscar(self, consulta, k=5, minimo=0.08, relativo=CORTE_RELATIVO, ligero=False):
        """Regresa hasta k documentos parecidos con su puntaje y los términos en común.
        Se descartan los que no llegan al puntaje mínimo ni al 55% del mejor (menos ruido)."""
        s = self._vector_consulta(consulta, ligero)
        orden = s.argsort()[::-1][:k]
        corte = max(minimo, relativo * s[orden[0]])
        return [self._resultado(i, s[i], consulta) for i in orden if s[i] >= corte]

    def buscar_temas(self, consulta, k=5, ligero=False):
        """Incidente 3: un resultado por cada tema de la consulta."""
        return {tema: self.buscar(tema, k, ligero=ligero) for tema in dividir(consulta)}

    def similares(self, doc_id, k=3):
        """Recomendación por similitud semántica entre documentos."""
        i = self.ids.index(doc_id)
        orden = [j for j in self.sim_docs[i].argsort()[::-1] if j != i][:k]
        return [self._resultado(j, self.sim_docs[i][j], self.docs[i]["texto"]) for j in orden]

    def por_historial(self, consultas, vistos=(), k=3):
        """Recomendación por historial: junta las últimas consultas y los documentos vistos."""
        if not consultas and not vistos:
            return []
        s = np.zeros(len(self.docs))
        for c in consultas[-5:]:
            s += self._vector_consulta(c)
        for v in vistos:
            s += self.sim_docs[self.ids.index(v)]
        s /= len(consultas[-5:]) + len(vistos)
        contexto = " ".join(list(consultas[-5:]) + [self.docs[self.ids.index(v)]["titulo"] for v in vistos])
        orden = [i for i in s.argsort()[::-1] if self.ids[i] not in vistos][:k]
        return [self._resultado(i, s[i], contexto) for i in orden]

    def clasificar(self, texto):
        """Asigna categoría. Si la confianza es baja, lo marca para revisión manual."""
        prob = self.clf.predict_proba(self.vec_palabras.transform([texto]))[0]
        orden = prob.argsort()[::-1]
        return {
            "categoria": NOMBRES_CATEGORIAS[self.clf.classes_[orden[0]]],
            "confianza": float(prob[orden[0]]),
            "revision_manual": bool(prob[orden[0]] < UMBRAL_CLASIFICACION),
            "probabilidades": {NOMBRES_CATEGORIAS[self.clf.classes_[j]]: float(prob[j]) for j in orden},
        }

    def _resultado(self, i, score, texto_base):
        d = self.docs[i]
        return {"id": d["id"], "titulo": d["titulo"], "categoria": NOMBRES_CATEGORIAS[d["categoria"]],
                "texto": d["texto"], "score": float(score), "terminos": self.terminos_comunes(texto_base, i)}


def cargar_embeddings(textos):
    """Incidente 2 (extensión): embeddings con un modelo preentrenado multilingüe.
    Solo se activan con NEORIS_EMBEDDINGS=1 y si sentence-transformers está instalado.
    Si algo falla, el sistema sigue funcionando con TF-IDF."""
    if os.getenv("NEORIS_EMBEDDINGS") != "1":
        return None
    try:
        from sentence_transformers import SentenceTransformer
        modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        E = modelo.encode(textos, normalize_embeddings=True)

        class Emb:
            def similitud(self, consulta):
                return E @ modelo.encode([consulta], normalize_embeddings=True)[0]
        return Emb()
    except Exception:
        return None

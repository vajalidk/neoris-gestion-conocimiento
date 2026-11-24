"""
Sistema de Gestión del Conocimiento con IA - Neoris
Interfaz en Streamlit que integra: búsqueda y recomendaciones, chatbot, clasificación,
grafo de conocimiento, métricas y diagnóstico del sistema.

Ejecutar:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.append(str(Path(__file__).parent / "src"))
import grafo  # noqa: E402
from chatbot import responder  # noqa: E402
from motor import Motor, dividir  # noqa: E402
from salud import Salud  # noqa: E402

st.set_page_config(page_title="Neoris · Gestión del Conocimiento", page_icon="📚", layout="wide")


# ---------- Recursos compartidos (se crean una sola vez y se reutilizan) ----------
@st.cache_resource(show_spinner="Construyendo el índice de documentos...")
def cargar_motor():
    return Motor()


@st.cache_resource
def cargar_salud():
    return Salud()


@st.cache_resource
def cargar_grafo(_motor):
    return grafo.construir(_motor)


@st.cache_data
def cargar_metricas():
    ruta = Path(__file__).parent / "resultados" / "evaluacion_final.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


salud = cargar_salud()
motor = salud.ejecutar("Cargar índice", cargar_motor)
if motor is None:
    st.error("No se pudo cargar el índice. Revisa la pestaña Diagnóstico o recarga la página.")
    st.stop()
ligero = salud.modo_ligero()

# Memoria de la sesión: aquí se comparten datos entre los componentes
for clave, valor in {"historial": [], "vistos": [], "chat": [], "foco": None, "ultima": ""}.items():
    st.session_state.setdefault(clave, valor)


def ver_documento(doc_id):
    if doc_id not in st.session_state.vistos:
        st.session_state.vistos.append(doc_id)
    st.session_state.foco = doc_id


def tarjeta(r, clave):
    """Muestra un documento recomendado y la información usada para recomendarlo."""
    with st.container(border=True):
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{r['id']} · {r['titulo']}**  \n{r['categoria']} · similitud **{r['score']:.3f}**")
        if r["terminos"]:
            c1.caption("Términos en común: " + ", ".join(r["terminos"]))
        c2.button("Ver", key=clave, on_click=ver_documento, args=(r["id"],), use_container_width=True)


# ---------- Barra lateral ----------
with st.sidebar:
    st.title("Neoris")
    st.caption("Sistema de Gestión del Conocimiento con IA")
    st.metric("Documentos en el repositorio", len(motor.docs))
    estado = salud.estado()
    st.progress(estado["uso_limite"], text=f"Memoria: {estado['proceso_mb']} MB · modo {estado['modo']}")
    st.subheader("Tu historial")
    if st.session_state.historial:
        for c in st.session_state.historial[-5:][::-1]:
            st.write(f"• {c}")
        if st.button("Borrar historial"):
            st.session_state.historial, st.session_state.vistos, st.session_state.ultima = [], [], ""
            st.rerun()
    else:
        st.caption("Aún no has buscado nada.")

tabs = st.tabs(["Buscar y recomendar", "Chatbot", "Clasificar documento", "Grafo de conocimiento", "Métricas", "Diagnóstico"])

# ---------- 1. Búsqueda y recomendaciones ----------
with tabs[0]:
    st.header("Buscar y recomendar documentos")
    consulta = st.text_input("Escribe lo que necesitas", placeholder="Ej. el servidor se cayó por falta de memoria")
    if consulta:
        if consulta != st.session_state.ultima:  # se guarda una sola vez, aunque la página se recargue
            st.session_state.ultima = consulta
            st.session_state.historial.append(consulta)
        temas = salud.ejecutar("Búsqueda", lambda: motor.buscar_temas(consulta, ligero=ligero), respaldo={})
        if len(temas) > 1:
            st.info(f"Detecté {len(temas)} temas en tu consulta. Busqué cada uno por separado.")
        for t, (tema, resultados) in enumerate(temas.items()):
            if len(temas) > 1:
                st.subheader(f"Tema {t + 1}: {tema}")
            if not resultados:
                st.warning("No encontré documentos parecidos. Intenta con otras palabras.")
            for n, r in enumerate(resultados):
                tarjeta(r, f"b{t}{n}")
            mostrados = {r["id"] for r in resultados}
            otros = [r for r in motor.buscar(tema, k=8, relativo=0) if r["id"] not in mostrados]
            if otros:
                with st.expander("Otros posibles (menos parecidos)"):
                    for r in otros:
                        st.write(f"{r['id']} · {r['titulo']} · similitud {r['score']:.3f}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Documento seleccionado")
        if st.session_state.vistos:
            doc_id = st.session_state.vistos[-1]
            d = motor.docs[motor.ids.index(doc_id)]
            st.markdown(f"**{d['id']} · {d['titulo']}**")
            st.write(d["texto"][len(d["titulo"]):])
            st.markdown("**Documentos relacionados** (similitud semántica)")
            for n, r in enumerate(motor.similares(doc_id)):
                tarjeta(r, f"s{n}")
        else:
            st.caption("Da clic en «Ver» en un resultado para abrirlo aquí.")
    with col2:
        st.subheader("Recomendado para ti")
        recs = motor.por_historial(st.session_state.historial, st.session_state.vistos)
        if recs:
            st.caption(f"Basado en tus últimas {min(len(st.session_state.historial), 5)} consultas "
                       f"y {len(st.session_state.vistos)} documentos vistos.")
            for n, r in enumerate(recs):
                tarjeta(r, f"h{n}")
        else:
            st.caption("Cuando busques o abras documentos, aquí verás recomendaciones según tu historial.")

# ---------- 2. Chatbot ----------
with tabs[1]:
    st.header("Chatbot de conocimiento")
    st.caption("Responde con lo que dicen los documentos de Neoris y te dice de dónde sacó la respuesta.")
    for m in st.session_state.chat:
        with st.chat_message(m["rol"]):
            st.markdown(m["texto"])
            if m.get("fuentes"):
                with st.expander("Documentos consultados"):
                    st.dataframe(pd.DataFrame([{"Documento": f["id"], "Título": f["titulo"], "Similitud": round(f["score"], 3)}
                                               for f in m["fuentes"]]), hide_index=True, use_container_width=True)
    if pregunta := st.chat_input("Pregunta algo, por ejemplo: ¿qué hago si un usuario está bloqueado?"):
        st.session_state.chat.append({"rol": "user", "texto": pregunta})
        st.session_state.historial.append(pregunta)
        r = salud.ejecutar("Chatbot", lambda: responder(motor, pregunta, ligero),
                           respaldo={"texto": "Tuve un problema al responder. Intenta de nuevo.", "fuentes": [], "modo": "respaldo"})
        st.session_state.chat.append({"rol": "assistant", "texto": r["texto"] + f"\n\n*Modo: {r['modo']}*", "fuentes": r["fuentes"]})
        st.rerun()

# ---------- 3. Clasificación ----------
with tabs[2]:
    st.header("Clasificar un documento nuevo")
    archivo = st.file_uploader("Sube un .txt o pega el texto abajo", type="txt")
    texto = st.text_area("Texto del documento", value=archivo.read().decode("utf-8") if archivo else "", height=150,
                         placeholder="Ej. El cliente no puede entrar al sistema y su cuenta aparece bloqueada.")
    if st.button("Clasificar", type="primary") and texto.strip():
        res = salud.ejecutar("Clasificación", lambda: motor.clasificar(texto))
        if res:
            if res["revision_manual"]:
                st.warning(f"Confianza baja ({res['confianza']:.0%}). Se recomienda revisión manual. "
                           f"Categoría más probable: **{res['categoria']}**")
            else:
                st.success(f"Categoría: **{res['categoria']}** (confianza {res['confianza']:.0%})")
            st.bar_chart(pd.Series(res["probabilidades"], name="Probabilidad"), horizontal=True)
            st.markdown("**Documentos parecidos en el repositorio**")
            for n, r in enumerate(motor.buscar(texto, k=3)):
                tarjeta(r, f"c{n}")

# ---------- 4. Grafo ----------
with tabs[3]:
    st.header("Grafo de conocimiento")
    G = cargar_grafo(motor)
    info = grafo.resumen(G)
    st.caption(f"{info['nodos']} nodos y {info['conexiones']} conexiones: Neoris, {info.get('categoría', 0)} categorías, "
               f"{info.get('documento', 0)} documentos y {info.get('concepto', 0)} conceptos. "
               "Arrastra los nodos, haz zoom y pasa el mouse para ver detalles.")
    c1, c2, c3 = st.columns([2, 1, 2])
    opciones = ["(todo el grafo)"] + sorted(G.nodes)
    foco = st.session_state.foco if st.session_state.foco in G else "(todo el grafo)"
    elegido = c1.selectbox("Explorar alrededor de", opciones, index=opciones.index(foco))
    profundidad = c2.slider("Niveles", 1, 2, 1)
    tipos = c3.multiselect("Mostrar", ["organización", "categoría", "concepto", "documento"],
                           default=["organización", "categoría", "concepto", "documento"])
    H = grafo.filtrar(G, None if elegido.startswith("(") else elegido, profundidad, tipos)
    if ligero and H.number_of_nodes() > 40:
        st.info("Modo ligero: el grafo se muestra sin animación para ahorrar memoria.")
    pagina = salud.ejecutar("Grafo", lambda: grafo.html(H, fisica=not ligero))
    if pagina:
        components.html(pagina, height=640)
    st.markdown("**Leyenda:** ⭐ Neoris · ▭ categoría · ● documento (color de su categoría) · ◆ concepto · "
                "línea gris gruesa = documentos similares · línea punteada = relación de la ontología")

# ---------- 5. Métricas ----------
with tabs[4]:
    st.header("Evaluación del sistema")
    met = cargar_metricas()
    if not met:
        st.info("Corre `python src/evaluacion_final.py` para generar las métricas.")
    else:
        rec, clf = met["recuperacion"], met["clasificacion"]
        c = st.columns(4)
        c[0].metric("Precision búsqueda", f"{rec['final']['precision']:.2f}", f"{rec['final']['precision'] - rec['avance']['precision']:+.2f} vs avance")
        c[1].metric("Recall búsqueda", f"{rec['final']['recall']:.2f}", f"{rec['final']['recall'] - rec['avance']['recall']:+.2f} vs avance")
        c[2].metric("Precision clasificación", f"{clf['final']['precision']:.2f}")
        c[3].metric("Recall clasificación", f"{clf['final']['recall']:.2f}")
        st.image(str(Path(__file__).parent / "resultados" / "evaluacion_final.png"))
        st.subheader("Consultas de prueba")
        st.dataframe(pd.DataFrame([{"Consulta": f["consulta"], "Tipo": f["tipo"], "Relevantes": ", ".join(f["relevantes"]),
                                    "Recuperados": ", ".join(f["recuperados_final"]), "Precision": f["final"][0],
                                    "Recall": f["final"][1]} for f in rec["consultas"]]), hide_index=True, use_container_width=True)
        st.caption(f"Recomendaciones de documentos parecidos que son de la misma categoría: "
                   f"{met['recomendaciones']['precision_misma_categoria']:.0%}")

# ---------- 6. Diagnóstico ----------
with tabs[5]:
    st.header("Diagnóstico del sistema")
    st.caption("Herramientas para que el sistema siga funcionando aunque el servidor tenga poca memoria.")
    e = salud.estado()
    c = st.columns(4)
    c[0].metric("Memoria del proceso", f"{e['proceso_mb']} MB", f"límite {e['limite_mb']} MB", delta_color="off")
    c[1].metric("Memoria del servidor", f"{e['sistema_pct']:.0f}%")
    c[2].metric("Modo", e["modo"])
    c[3].metric("Errores recientes", e["errores"])
    b = st.columns(3)
    if b[0].button("Liberar memoria", use_container_width=True):
        st.cache_data.clear(); salud.liberar(); st.rerun()
    if b[1].button("Reconstruir índice", use_container_width=True):
        cargar_motor.clear(); cargar_grafo.clear(); salud.registrar("acción", "Índice reconstruido"); st.rerun()
    salud.forzar_ligero = b[2].toggle("Forzar modo ligero", value=salud.forzar_ligero,
                                      help="Simula poca memoria: búsqueda solo por palabras y grafo sin animación.")
    st.subheader("Bitácora de operaciones")
    if salud.eventos:
        st.dataframe(pd.DataFrame(list(salud.eventos)), hide_index=True, use_container_width=True)
    else:
        st.caption("Todavía no hay operaciones registradas.")

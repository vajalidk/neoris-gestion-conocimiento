"""
Grafo de conocimiento de Neoris (amplía la ontología de la Fase 1).

Nodos:   Neoris, 5 categorías (áreas), 100 documentos y 15 conceptos del dominio.
Aristas: Neoris -tiene área-> categoría, categoría -contiene-> documento,
         documento -menciona-> concepto, documento -similar a- documento,
         concepto -relación de la ontología-> concepto.
"""
import re

import networkx as nx
from pyvis.network import Network

from motor import normalizar

COLORES = {"Soporte técnico": "#E4572E", "Desarrollo de software": "#2E6FD8", "Infraestructura y nube": "#17A589",
           "Gestión de proyectos": "#7B4FC9", "Políticas internas": "#D9A400"}
COLOR_CONCEPTO, COLOR_NEORIS = "#F28C28", "#0F1B33"
SIMILITUD_MINIMA = 0.22  # dos documentos se conectan si se parecen al menos esto

# Concepto del dominio -> palabras que lo indican en el texto (sin acentos)
CONCEPTOS = {
    "Cliente": ["cliente"], "Ticket": ["ticket"], "Incidente o falla": ["incidente", "caida", "falla", "fallo", "ataque", "error"],
    "Servidor": ["servidor"], "Nube": ["nube", "maquina virtual", "region"], "SAP": ["sap"],
    "Base de datos": ["base de datos", "bases de datos"], "Seguridad": ["seguridad", "firewall", "certificado", "vpn", "antivirus"],
    "Usuario y acceso": ["usuario", "sesion", "contrasena", "bloquead", "accesos"],
    "API e integración": ["api", "integracion", "interfaz"], "Código y pruebas": ["codigo", "prueba", "repositorio"],
    "Proyecto": ["proyecto"], "Riesgo": ["riesgo"], "Empleado": ["empleado", "consultor", "mentor"],
    "Lección aprendida": ["lecciones", "aprendimos", "retrospectiva"],
}
# Relaciones entre conceptos tomadas de la ontología (neoris-ontologia.owl)
RELACIONES = [
    ("Cliente", "Ticket", "abre"), ("Ticket", "SAP", "afecta a"), ("Incidente o falla", "Servidor", "ocurre en"),
    ("Servidor", "Nube", "vive en"), ("Base de datos", "Servidor", "corre en"), ("Seguridad", "Servidor", "protege"),
    ("Usuario y acceso", "Seguridad", "depende de"), ("API e integración", "SAP", "conecta con"),
    ("Cliente", "Proyecto", "contrata"), ("Proyecto", "Riesgo", "tiene"), ("Proyecto", "Lección aprendida", "genera"),
    ("Empleado", "Proyecto", "trabaja en"), ("Código y pruebas", "API e integración", "implementa"),
    ("Ticket", "Incidente o falla", "reporta"),
]


def construir(motor):
    """Arma el grafo completo a partir de los documentos y el índice del motor."""
    G = nx.Graph()
    G.add_node("Neoris", tipo="organización", label="Neoris", color=COLOR_NEORIS, shape="star", size=40,
               title="Neoris: consultora de TI")
    for cat, color in COLORES.items():
        G.add_node(cat, tipo="categoría", label=cat, color=color, shape="box", size=30, title=f"Categoría: {cat}")
        G.add_edge("Neoris", cat, label="tiene área")
    for c in CONCEPTOS:
        G.add_node(c, tipo="concepto", label=c, color=COLOR_CONCEPTO, shape="diamond", size=18, title=f"Concepto: {c}")
    for a, b, rel in RELACIONES:
        G.add_edge(a, b, label=rel, dashes=True)

    for i, d in enumerate(motor.docs):
        cat = motor._resultado(i, 0, "")["categoria"]
        G.add_node(d["id"], tipo="documento", label=d["id"], color=COLORES[cat], shape="dot", size=14,
                   title=f"{d['id']}: {d['titulo']}\nCategoría: {cat}", categoria=cat)
        G.add_edge(cat, d["id"], label="contiene")
        texto = normalizar(d["texto"])
        for c, claves in CONCEPTOS.items():
            if any(re.search(rf"\b{k}", texto) for k in claves):  # \b: palabra completa, "api" no cuenta dentro de "rapido"
                G.add_edge(d["id"], c, label="menciona", color="#C9CED8")
        j = max((j for j in range(len(motor.docs)) if j != i), key=lambda j: motor.sim_docs[i][j])
        if motor.sim_docs[i][j] >= SIMILITUD_MINIMA:
            G.add_edge(d["id"], motor.ids[j], label=f"similar ({motor.sim_docs[i][j]:.2f})", color="#8A94A6", width=2)
    return G


def filtrar(G, foco=None, profundidad=1, tipos=("organización", "categoría", "concepto", "documento")):
    """Deja solo los tipos de nodo elegidos y, si hay foco, solo sus vecinos."""
    H = G.subgraph([n for n, a in G.nodes(data=True) if a["tipo"] in tipos])
    if foco and foco in H:
        H = nx.ego_graph(H, foco, radius=profundidad)
    return H


def html(H, fisica=True, alto="620px"):
    """Convierte el grafo a una página interactiva (zoom, arrastrar nodos, ver detalles al pasar el mouse)."""
    net = Network(height=alto, width="100%", bgcolor="#ffffff", font_color="#0F1B33", cdn_resources="in_line")
    net.from_nx(H)
    for e in net.edges:  # las etiquetas de relación se ven al pasar el mouse, no saturan el dibujo
        e["title"] = e.pop("label", "")
    if fisica:  # se acomoda solo antes de mostrarse y queda centrado
        net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -6000, "springLength": 120, "damping": 0.4},'
                        ' "stabilization": {"enabled": true, "iterations": 300, "fit": true}},'
                        ' "interaction": {"hover": true, "navigationButtons": true}}')
    else:  # modo ligero: posiciones calculadas una vez, sin animación
        pos = nx.spring_layout(H, seed=1, scale=450)
        for n in net.nodes:
            n["x"], n["y"] = float(pos[n["id"]][0]), float(pos[n["id"]][1])
        net.toggle_physics(False)
    # Al terminar de acomodarse se detiene la animación, y se centra cada vez que cambia de tamaño
    # (por ejemplo, al abrir la pestaña del grafo, que antes estaba oculta)
    return net.generate_html().replace(
        "network = new vis.Network(container, data, options);",
        "network = new vis.Network(container, data, options);\n"
        "network.once('stabilizationIterationsDone', function () { network.setOptions({physics: false}); network.fit(); });\n"
        "new ResizeObserver(function () { network.redraw(); network.fit(); }).observe(container);")


def resumen(G):
    tipos = nx.get_node_attributes(G, "tipo")
    conteo = {t: sum(1 for v in tipos.values() if v == t) for t in set(tipos.values())}
    return {"nodos": G.number_of_nodes(), "conexiones": G.number_of_edges(), **conteo}

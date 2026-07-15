"""
server.py — ALEPH
=================
Servidor local que sirve galaxia.html y expone el buscador semántico y RAG.

Endpoints:
    GET  /                          → galaxia.html
    GET  /api/buscar?q=&top=&tipo=  → recuperación semántica (no generativa)
    GET  /api/consultar?q=&top=     → RAG: recuperación + respuesta generativa
"""

import os
import json
import requests
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

BASE_DIR        = Path(__file__).parent

# BACKEND=local (default) usa Ollama corriendo en esta máquina — gratis, sin límite, requiere GPU/CPU propia.
# BACKEND=cloud usa Groq (generación) + Nomic (embeddings) — para desplegar sin Ollama (ej. Render free tier).
BACKEND         = os.environ.get("BACKEND", "local")

OLLAMA_EMBED    = "http://localhost:11434/api/embeddings"
OLLAMA_GENERATE = "http://localhost:11434/api/generate"
MODELO_EMBED    = "nomic-embed-text"
MODELO_GEN      = os.environ.get("MODELO_GEN", "llama3.2:3b")

NOMIC_EMBED_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
NOMIC_API_KEY   = os.environ.get("NOMIC_API_KEY")
MODELO_EMBED_NOMIC = "nomic-embed-text-v1.5"

GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY")
MODELO_GEN_CLOUD = os.environ.get("MODELO_GEN_CLOUD", "llama-3.3-70b-versatile")

app = Flask(__name__)
CORS(app)


def get_embedding(text: str) -> list:
    if BACKEND == "cloud":
        resp = requests.post(
            NOMIC_EMBED_URL,
            headers={"Authorization": f"Bearer {NOMIC_API_KEY}"},
            json={"model": MODELO_EMBED_NOMIC, "texts": [text.strip()], "task_type": "search_query"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    resp = requests.post(OLLAMA_EMBED, json={"model": MODELO_EMBED, "prompt": text.strip()}, timeout=60)
    resp.raise_for_status()
    return resp.json()["embedding"]


def get_supabase_docs(emb: list, top: int, tipo: str | None) -> list:
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    result = sb.rpc("buscar_normas", {
        "query_embedding": emb,
        "top_k":           top,
        "filtro_tipo":     tipo,
    }).execute()
    return result.data


def get_chunks(emb: list, top: int) -> list:
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    result = sb.rpc("buscar_chunks", {
        "query_embedding": emb,
        "top_k":           top,
    }).execute()
    return result.data


def build_rag_prompt(query: str, chunks: list) -> str:
    """Construye el prompt agrupando chunks por norma."""
    from collections import defaultdict
    por_norma = defaultdict(list)
    for c in chunks:
        por_norma[c["nombre_norma"]].append(c)

    fragmentos = []
    for nombre, norma_chunks in por_norma.items():
        c0    = norma_chunks[0]
        tipo  = c0.get("tipo_norma", "")
        anio  = str(c0.get("anio", "") or "")
        titulo = c0.get("titulo_resumido") or ""

        encabezado = f"Norma: {nombre} ({tipo})"
        if anio and anio not in ("nan", "None", ""):
            try:
                encabezado += f" (año {int(float(anio))})"
            except ValueError:
                pass

        partes = [encabezado]
        if titulo and titulo != nombre:
            partes.append(f"  Título: {titulo}")

        for c in norma_chunks:
            partes.append(f"  ---\n  {c['texto']}")

        fragmentos.append("\n".join(partes))

    contexto = "\n\n".join(fragmentos)
    total_docs = len(por_norma)
    return (
        "Sos un asistente legal especializado en normativa registral argentina.\n"
        "Tenés acceso a fragmentos de documentos normativos. Respondé la consulta "
        "basándote ÚNICAMENTE en esos fragmentos.\n"
        f"- Antes de responder, revisá las {total_docs} normas a continuación UNA POR UNA "
        "— no respondas solo con la última que leíste.\n"
        "- Incluí TODAS las normas relevantes que encuentres, no te limites a una o dos.\n"
        "- Nombrá siempre cada norma por su nombre real entre corchetes: [Ley 8236], [OSC 14-25], "
        "etc. NUNCA la llames \"Documento 1\", \"Documento 2\" ni por ningún número — esos números "
        "no existen fuera de este prompt y no significan nada para quien lee la respuesta.\n"
        "- Usá la respuesta 'El corpus normativo disponible no contiene información específica "
        "sobre este tema.' ÚNICAMENTE si, después de revisar TODOS los documentos, ninguno se "
        "relaciona ni parcialmente con la consulta. Si al menos un documento aporta algo, "
        "respondé con eso — no rechaces la consulta.\n"
        "- No uses conocimiento propio fuera del contexto.\n"
        "Respondé en español, de forma clara y estructurada.\n\n"
        f"FRAGMENTOS NORMATIVOS:\n{contexto}\n\n"
        f"CONSULTA: {query}\n\n"
        "RESPUESTA:"
    )


def get_completion(prompt: str) -> str:
    if BACKEND == "cloud":
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": MODELO_GEN_CLOUD,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    resp = requests.post(
        OLLAMA_GENERATE,
        json={"model": MODELO_GEN, "prompt": prompt, "stream": True},
        timeout=300,
        stream=True,
    )
    resp.raise_for_status()
    partes = []
    for linea in resp.iter_lines():
        if linea:
            chunk = json.loads(linea)
            partes.append(chunk.get("response", ""))
            if chunk.get("done"):
                break
    return "".join(partes).strip()


@app.route("/")
def index():
    return send_file(BASE_DIR / "galaxia.html")


@app.route("/api/buscar")
def buscar():
    q    = request.args.get("q", "").strip()
    top  = int(request.args.get("top", 8))
    tipo = request.args.get("tipo") or None

    if not q:
        return jsonify([])

    try:
        emb = get_embedding(q)
    except Exception as e:
        origen = "Nomic" if BACKEND == "cloud" else "Ollama"
        return jsonify({"error": f"{origen} (embeddings) no disponible: {e}"}), 503

    try:
        return jsonify(get_supabase_docs(emb, top, tipo))
    except Exception as e:
        return jsonify({"error": f"Supabase no disponible: {e}"}), 503


@app.route("/api/consultar")
def consultar():
    q   = request.args.get("q", "").strip()
    top = int(request.args.get("top", 5))

    print(f"\n{'='*60}")
    print(f"  CONSULTA: {q}")
    print(f"{'='*60}")

    if not q:
        return jsonify({"error": "Consulta vacía"}), 400

    try:
        emb = get_embedding(q)
        print(f"  [OK] Embedding generado ({len(emb)} dims)")
    except Exception as e:
        print(f"  [ERROR] Embedding: {e}")
        origen = "Nomic" if BACKEND == "cloud" else "Ollama"
        return jsonify({"error": f"{origen} (embeddings) no disponible: {e}"}), 503

    # top*2 chunks para cubrir varios fragmentos por norma sin saturar el contexto del modelo
    try:
        chunks = get_chunks(emb, top * 2)
        print(f"  [OK] Chunks recuperados: {len(chunks)}")
    except Exception as e:
        print(f"  [ERROR] Supabase (chunks): {e}")
        return jsonify({"error": f"Supabase no disponible: {e}"}), 503

    if not chunks:
        return jsonify({"respuesta": "No encontré normativa relevante para esa consulta.", "fuentes": []})

    print(f"\n  --- Chunks recuperados ---")
    for c in chunks:
        sim = round(c.get("similitud", 0) * 100, 1)
        print(f"  {sim:5.1f}%  {c.get('nombre_norma')} [chunk {c.get('chunk_idx')}]  |  {c['texto'][:80]}")

    prompt = build_rag_prompt(q, chunks)

    print(f"\n  --- Prompt enviado al modelo ---")
    print(prompt)
    print(f"  --- Fin prompt ---\n")

    try:
        respuesta = get_completion(prompt)
        print(f"  --- Respuesta del modelo ---")
        print(respuesta)
        print(f"  --- Fin respuesta ---\n")
    except Exception as e:
        origen = "Groq" if BACKEND == "cloud" else "Ollama"
        print(f"  [ERROR] {origen} generación: {e}")
        return jsonify({"error": f"{origen} (generación) no disponible: {e}"}), 503

    # fuentes únicas por norma (la de mayor similitud)
    vistos: dict = {}
    for c in chunks:
        nombre = c.get("nombre_norma", "")
        if nombre not in vistos or c.get("similitud", 0) > vistos[nombre].get("similitud", 0):
            vistos[nombre] = c
    fuentes = [
        {
            "nombre_norma":    c.get("nombre_norma"),
            "tipo_norma":      c.get("tipo_norma"),
            "titulo_resumido": c.get("titulo_resumido"),
            "similitud":       round(c.get("similitud", 0) * 100, 1),
            "url_pdf":         c.get("url_pdf"),
        }
        for c in vistos.values()
    ]

    return jsonify({"respuesta": respuesta, "fuentes": fuentes})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("=" * 50)
    print("  ALEPH — Servidor")
    print(f"  Backend: {BACKEND}")
    print(f"  Modelo generación: {MODELO_GEN_CLOUD if BACKEND == 'cloud' else MODELO_GEN}")
    print(f"  http://localhost:{port}")
    print("=" * 50)
    if BACKEND == "cloud" and not (GROQ_API_KEY and NOMIC_API_KEY):
        print("  [ADVERTENCIA] BACKEND=cloud pero falta GROQ_API_KEY y/o NOMIC_API_KEY en el .env")
    app.run(host="0.0.0.0", port=port, debug=False)

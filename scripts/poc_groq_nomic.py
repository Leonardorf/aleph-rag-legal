"""
POC: reemplazar Ollama local por Groq (generación) + Nomic API (embeddings)
para poder desplegar ALEPH en un hosting sin GPU (ej. Render free tier).

Valida dos cosas antes de tocar server.py en serio:
  1. Que el embedding de Nomic hosteado sea compatible (similitud coseno alta)
     con el que genera Ollama local para la misma consulta.
  2. Que Groq genere una respuesta RAG razonable, midiendo velocidad.

No modifica nada del proyecto real — solo lee de Supabase.
"""
import os
import sys
import json
import time
import numpy as np
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

SUPABASE_URL   = os.environ.get("SUPABASE_URL")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
NOMIC_API_KEY  = os.environ.get("NOMIC_API_KEY")

QUERY = "cancelación de inhibición"
TOP   = 5

def fail(msg):
    print(f"\n[FALTA] {msg}")
    sys.exit(1)

if not SUPABASE_URL or not SUPABASE_KEY:
    fail("SUPABASE_URL / SUPABASE_KEY no están en el .env del proyecto")
if not GROQ_API_KEY:
    fail("GROQ_API_KEY no está en el .env — conseguila en https://console.groq.com/keys")
if not NOMIC_API_KEY:
    fail("NOMIC_API_KEY no está en el .env — conseguila en https://atlas.nomic.ai/")


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_ollama_local(text):
    resp = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text.strip()},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def embed_nomic_api(text, task_type="search_query"):
    resp = requests.post(
        "https://api-atlas.nomic.ai/v1/embedding/text",
        headers={"Authorization": f"Bearer {NOMIC_API_KEY}"},
        json={"model": "nomic-embed-text-v1.5", "texts": [text], "task_type": task_type},
        timeout=30,
    )
    if not resp.ok:
        print(f"  [ERROR Nomic] {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
    data = resp.json()
    return data["embeddings"][0]


def buscar_chunks(emb, top_k):
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    result = sb.rpc("buscar_chunks", {"query_embedding": emb, "top_k": top_k}).execute()
    return result.data


def groq_completion(prompt, model="llama-3.3-70b-versatile"):
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        timeout=60,
    )
    if not resp.ok:
        print(f"  [ERROR Groq] {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def build_prompt(query, chunks):
    from collections import defaultdict
    por_norma = defaultdict(list)
    for c in chunks:
        por_norma[c["nombre_norma"]].append(c)
    fragmentos = []
    for i, (nombre, cs) in enumerate(por_norma.items(), 1):
        partes = [f"[DOC {i}] {cs[0].get('tipo_norma','')}: {nombre}"]
        for c in cs:
            partes.append(f"  ---\n  {c['texto']}")
        fragmentos.append("\n".join(partes))
    contexto = "\n\n".join(fragmentos)
    total = len(por_norma)
    return (
        "Sos un asistente legal especializado en normativa registral argentina.\n"
        "Tenés acceso a fragmentos de documentos normativos. Respondé la consulta "
        "basándote ÚNICAMENTE en esos fragmentos.\n"
        f"- Antes de responder, revisá los {total} documentos numerados [DOC 1] a [DOC {total}] "
        "UNO POR UNO — no respondas solo con lo último que leíste.\n"
        "- Incluí TODOS los documentos relevantes que encuentres, no te limites a uno o dos.\n"
        "- Citá cada norma por su nombre entre corchetes.\n"
        "- No uses conocimiento propio fuera del contexto.\n"
        "Respondé en español, de forma clara y estructurada.\n\n"
        f"FRAGMENTOS NORMATIVOS:\n{contexto}\n\n"
        f"CONSULTA: {query}\n\nRESPUESTA:"
    )


print("=" * 60)
print(f"  QUERY: {QUERY}")
print("=" * 60)

print("\n[1/4] Embedding local (Ollama nomic-embed-text) — baseline...")
t0 = time.time()
vec_local = embed_ollama_local(QUERY)
print(f"  OK — {len(vec_local)} dims, {time.time()-t0:.2f}s")

print("\n[2/4] Embedding hosteado (Nomic API, task_type=search_query)...")
t0 = time.time()
vec_nomic = embed_nomic_api(QUERY)
print(f"  OK — {len(vec_nomic)} dims, {time.time()-t0:.2f}s")

sim = cosine(vec_local, vec_nomic) if len(vec_local) == len(vec_nomic) else None
if sim is None:
    print(f"  [ALERTA] dimensiones distintas: local={len(vec_local)} vs nomic={len(vec_nomic)} — NO son compatibles, habría que re-embeber el corpus.")
else:
    print(f"  Similitud coseno local vs hosteado: {sim:.4f}  ({'compatible, se puede usar directo' if sim > 0.9 else 'diverge bastante, revisar task_type/prefijo'})")

print("\n[3/4] Recuperando chunks de Supabase con el embedding de Nomic...")
t0 = time.time()
chunks = buscar_chunks(vec_nomic, TOP * 2)
print(f"  OK — {len(chunks)} chunks, {time.time()-t0:.2f}s")
for c in chunks[:8]:
    print(f"    {round(c.get('similitud',0)*100,1):5.1f}%  {c.get('nombre_norma')} [chunk {c.get('chunk_idx')}]")

print("\n[4/4] Generando respuesta con Groq...")
prompt = build_prompt(QUERY, chunks)
t0 = time.time()
respuesta = groq_completion(prompt)
dt = time.time() - t0
print(f"  OK — {dt:.2f}s\n")
print("-" * 60)
print(respuesta)
print("-" * 60)

print("\nResumen:")
print(f"  Similitud embeddings local vs Nomic: {sim:.4f}" if sim is not None else "  Embeddings NO comparables (distinta dimensión)")
print(f"  Tiempo Groq (generación): {dt:.2f}s  (vs ~87-222s con Ollama local)")

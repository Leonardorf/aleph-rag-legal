"""
04_buscar.py
============
Buscador semántico de prueba desde la consola.
Permite validar que los embeddings funcionan ANTES de construir el frontend.

Uso:
    python 04_buscar.py
    python 04_buscar.py "cancelación de inhibición"
    python 04_buscar.py "certificados en papel" --tipo "Resolución Administrativa"
    python 04_buscar.py "zona de frontera" --top 5

También funciona en modo offline usando el JSON local
(sin necesidad de Supabase), útil para desarrollo.
"""

import os
import sys
import json
import argparse
import requests
import numpy as np
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

DATA_DIR    = Path(__file__).parent.parent / "data"
OLLAMA_URL  = "http://localhost:11434/api/embeddings"

def get_embedding(text: str, modelo: str = "nomic-embed-text") -> np.ndarray:
    resp = requests.post(OLLAMA_URL, json={"model": modelo, "prompt": text.strip()}, timeout=60)
    resp.raise_for_status()
    return np.array(resp.json()["embedding"], dtype=np.float32)

def detectar_modelo(vectores: list) -> str:
    """Lee el modelo usado al generar los embeddings."""
    return vectores[0].get("modelo_embedding", "nomic-embed-text") if vectores else "nomic-embed-text"

def similitud_coseno(a: np.ndarray, b: list) -> float:
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def buscar_local(query: str, top_k: int = 10,
                 filtro_tipo: str = None, filtro_jur: str = None) -> list[dict]:
    """Búsqueda offline usando el JSON local."""
    json_path = DATA_DIR / "normas_vectors.json"
    if not json_path.exists():
        print(f"[ERROR] {json_path} no encontrado. Corré 02_embeddings.py primero.")
        return []

    with open(json_path, encoding="utf-8") as f:
        normas = json.load(f)

    modelo = detectar_modelo(normas)
    print(f"[...] Generando embedding con {modelo} para: '{query}'")
    q_emb = get_embedding(query, modelo)

    resultados = []
    for n in normas:
        if filtro_tipo and n.get("tipo_norma") != filtro_tipo:
            continue
        if filtro_jur and n.get("jurisdiccion") != filtro_jur:
            continue
        sim = similitud_coseno(q_emb, n["embedding_combined"])
        resultados.append({**n, "similitud": sim})

    resultados.sort(key=lambda x: x["similitud"], reverse=True)
    return resultados[:top_k]

def buscar_supabase(query: str, top_k: int = 10,
                    filtro_tipo: str = None, filtro_jur: str = None) -> list[dict]:
    """Búsqueda usando la función RPC de Supabase."""
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL o SUPABASE_KEY no definidos")
        sb = create_client(url, key)
    except Exception as e:
        print(f"[WARN] Supabase no disponible ({e}). Usando modo offline.")
        return buscar_local(query, top_k, filtro_tipo, filtro_jur)

    print(f"[...] Generando embedding para: '{query}'")
    q_emb = get_embedding(query).tolist()

    resp = sb.rpc("buscar_normas", {
        "query_embedding": q_emb,
        "top_k":           top_k,
        "filtro_tipo":     filtro_tipo,
    }).execute()
    return resp.data

def mostrar_resultados(resultados: list[dict], query: str):
    print(f"\n{'='*60}")
    print(f"  Resultados para: \"{query}\"")
    print(f"{'='*60}\n")
    if not resultados:
        print("  Sin resultados.")
        return
    for i, r in enumerate(resultados, 1):
        sim = r.get("similitud", 0)
        filled = int(sim * 20)
        barra = "#" * filled + "." * (20 - filled)
        print(f"  {i:2}. {r.get('nombre_norma', 'N/D')}")
        print(f"      Tipo: {r.get('tipo_norma','')}  |  "
              f"Jur: {r.get('jurisdiccion','')}  |  "
              f"Similitud: {sim:.3f} [{barra}]")
        titulo = r.get("titulo_resumido", "")
        if titulo and titulo != r.get("nombre_norma", ""):
            print(f"      {titulo}")
        resumen = str(r.get("texto_resumido", "") or "")
        if resumen and resumen.lower() not in ("nan", "none", ""):
            print(f"      >> {resumen[:120]}...")
        url = r.get("url_pdf", "")
        if url:
            print(f"      PDF: {url}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Buscador semántico ALEPH")
    parser.add_argument("query", nargs="?", help="Consulta en lenguaje natural")
    parser.add_argument("--top",  type=int, default=5, help="Cantidad de resultados")
    parser.add_argument("--tipo", type=str, default=None, help="Filtrar por tipo de norma")
    parser.add_argument("--jur",  type=str, default=None, help="Filtrar por jurisdicción")
    parser.add_argument("--local", action="store_true", help="Forzar modo offline (sin Supabase)")
    args = parser.parse_args()

    if args.query:
        queries = [args.query]
    else:
        # Modo interactivo
        print("ALEPH — Buscador semántico de normativa registral y de derecho inmobiliario")
        print("Escribí tu consulta (o 'salir' para terminar):\n")
        queries = []
        while True:
            q = input("  > ").strip()
            if q.lower() in ("salir", "exit", "q"):
                break
            if q:
                queries.append(q)
                break  # una sola consulta en modo interactivo por ahora

    for q in queries:
        if args.local:
            resultados = buscar_local(q, args.top, args.tipo, args.jur)
        else:
            resultados = buscar_supabase(q, args.top, args.tipo, args.jur)
        mostrar_resultados(resultados, q)

if __name__ == "__main__":
    main()

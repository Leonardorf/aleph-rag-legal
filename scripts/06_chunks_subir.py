"""
06_chunks_subir.py
==================
Toma los textos ya extraídos por 05_ocr_textos.py (data/textos_completos/*.txt),
los divide en chunks solapados, genera un embedding por chunk con Ollama y los
sube a la tabla `norma_chunks` de Supabase.

Este script NO importa fitz/easyocr/torch — sólo lee .txt locales y habla por
red con Ollama y Supabase. Mantenerlo separado de la extracción OCR evita
combinar en un mismo proceso "carga de modelos de ML + subida a la nube",
que en esta máquina disparaba picos de memoria muy grandes.

Uso:
    python 06_chunks_subir.py              # continúa donde quedó
    python 06_chunks_subir.py --reset      # vacía la tabla y arranca de cero
    python 06_chunks_subir.py --chunk-size 1000 --overlap 200

Prerequisito:
    - Corré 05_ocr_textos.py primero
    - Ollama corriendo con nomic-embed-text
    - Tabla norma_chunks creada en Supabase (SQL al final de este archivo)

SQL a ejecutar UNA VEZ en Supabase → SQL Editor:
──────────────────────────────────────────────────────────────────────────────
    create extension if not exists vector;

    create table if not exists norma_chunks (
        id              serial primary key,
        nombre_norma    text not null,
        tipo_norma      text,
        anio            text,
        jurisdiccion    text,
        titulo_resumido text,
        url_pdf         text,
        chunk_idx       integer not null,
        texto           text not null,
        embedding       vector(768),
        created_at      timestamptz default now()
    );

    create index if not exists idx_norma_chunks_embedding
        on norma_chunks using ivfflat (embedding vector_cosine_ops)
        with (lists = 10);

    create or replace function buscar_chunks(
        query_embedding vector(768),
        top_k           int default 8
    )
    returns table (
        nombre_norma    text,
        tipo_norma      text,
        anio            text,
        jurisdiccion    text,
        titulo_resumido text,
        url_pdf         text,
        chunk_idx       integer,
        texto           text,
        similitud       float
    )
    language sql stable as $$
        select nombre_norma, tipo_norma, anio, jurisdiccion,
               titulo_resumido, url_pdf, chunk_idx, texto,
               1 - (embedding <=> query_embedding) as similitud
        from norma_chunks
        order by embedding <=> query_embedding
        limit top_k;
    $$;
──────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import gc
import time
import json
import argparse
import requests
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

DATA_DIR   = Path(__file__).parent.parent / "data"
TEXTOS_DIR = DATA_DIR / "textos_completos"

OLLAMA_URL         = "http://localhost:11434/api/embeddings"
MODELO_EMBED       = "nomic-embed-text"
CHUNK_SIZE_DEFAULT = 1000
OVERLAP_DEFAULT    = 200


def nombre_a_archivo(nombre_norma: str) -> str:
    return re.sub(r"[^\w\-]", "_", nombre_norma)[:80]


def chunkar(texto: str, size: int, overlap: int) -> list[str]:
    """Divide en chunks respetando párrafos cuando es posible."""
    chunks = []
    inicio = 0
    n = len(texto)
    while inicio < n:
        fin = min(inicio + size, n)
        if fin < n:
            for sep in ("\n\n", "\n", ". "):
                corte = texto.rfind(sep, inicio + overlap, fin)
                if corte > inicio:
                    fin = corte + len(sep)
                    break

        chunk = texto[inicio:fin].strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        if fin >= n:
            break
        inicio = fin - overlap

    return chunks


def get_embedding(text: str, retries: int = 3) -> list:
    for intento in range(retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": MODELO_EMBED, "prompt": text.strip()},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            if intento < retries - 1:
                time.sleep(3 * (intento + 1))
            else:
                raise e


def main():
    parser = argparse.ArgumentParser(description="Chunking + embeddings + subida para ALEPH")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    parser.add_argument("--overlap",    type=int, default=OVERLAP_DEFAULT)
    parser.add_argument("--reset",      action="store_true",
                         help="Vaciar la tabla antes de empezar (por defecto continúa donde quedó)")
    args = parser.parse_args()

    print("=" * 60)
    print("ALEPH — Chunks + Embeddings + Subida")
    print(f"  chunk_size={args.chunk_size}  overlap={args.overlap}")
    print("=" * 60)

    json_path = DATA_DIR / "normas_vectors.json"
    if not json_path.exists():
        print(f"[ERROR] No se encontró {json_path}. Corré 02_embeddings.py primero.")
        return
    with open(json_path, encoding="utf-8") as f:
        normas = json.load(f)
    print(f"[OK] {len(normas)} normas en el JSON")

    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    if args.reset:
        print("\n[1/3] Vaciando tabla norma_chunks...")
        sb.table("norma_chunks").delete().neq("id", 0).execute()
        print("[OK] Tabla vaciada")
        ya_procesadas = set()
    else:
        print("\n[1/3] Revisando normas ya subidas...")
        existentes = sb.table("norma_chunks").select("nombre_norma").execute()
        ya_procesadas = {row["nombre_norma"] for row in existentes.data}
        print(f"[OK] {len(ya_procesadas)} normas ya tienen chunks subidos, se saltean")

    print("\n[2/3] Chunkeando, generando embeddings y subiendo...")
    total_chunks = 0
    errores      = 0
    total_normas = len(normas)

    for i, n in enumerate(normas, 1):
        nombre   = n["nombre_norma"]
        print(f"\n[{i}/{total_normas}] {nombre}")

        if nombre in ya_procesadas:
            print("  [OK] Ya subida, se salta")
            continue

        txt_path = TEXTOS_DIR / f"{nombre_a_archivo(nombre)}.txt"
        if not txt_path.exists():
            print(f"  [!] No hay texto extraído ({txt_path.name}). Corré 05_ocr_textos.py primero.")
            errores += 1
            continue

        texto = txt_path.read_text(encoding="utf-8").strip()
        if not texto:
            print(f"  [--] Texto vacío: {nombre}")
            continue

        chunks = chunkar(texto, args.chunk_size, args.overlap)
        print(f"  {nombre}: {len(chunks)} chunks  ({len(texto):,} chars)")

        meta = {
            "nombre_norma":    nombre,
            "tipo_norma":      n.get("tipo_norma", ""),
            "anio":            str(n.get("anio", "") or ""),
            "jurisdiccion":    n.get("jurisdiccion", ""),
            "titulo_resumido": n.get("titulo_resumido", ""),
            "url_pdf":         n.get("url_pdf", ""),
        }

        encabezado = f"{meta['tipo_norma']} {nombre}".strip()

        for idx, chunk in enumerate(chunks):
            try:
                # Se le agrega el nombre/tipo de norma como prefijo SOLO para el
                # embedding, así el vector "sabe" a qué norma pertenece (si no,
                # una consulta como "¿qué establece la Ley 17801?" no encuentra
                # los propios chunks de la Ley 17801, porque su texto de artículo
                # no menciona el número de ley).
                emb = get_embedding(f"{encabezado}: {chunk}")
                sb.table("norma_chunks").insert(
                    {**meta, "chunk_idx": idx, "texto": chunk, "embedding": emb}
                ).execute()
                total_chunks += 1
            except Exception as e:
                print(f"  [!] Error chunk {idx}: {e}")
                errores += 1
            time.sleep(0.03)

        gc.collect()

    print(f"\n{'='*60}")
    print(f"[OK] {total_chunks} chunks subidos, {errores} errores.")
    print(f"     Cubrís el texto completo de {len(normas)} normas.")
    print("=" * 60)


if __name__ == "__main__":
    main()

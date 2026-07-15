"""
03_cargar_supabase.py
=====================
Carga los embeddings generados en Supabase usando la extensión pgvector.
Supabase tiene plan gratuito suficiente para este corpus.

Prerequisitos:
    1. Crear proyecto en https://supabase.com (gratis)
    2. Ejecutar el SQL de setup (ver abajo o el archivo setup_supabase.sql)
    3. Obtener SUPABASE_URL y SUPABASE_KEY desde Settings > API

Variables de entorno necesarias (en .env):
    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_KEY=eyJ...  (service_role key, NO la anon key)

Uso:
    python 03_cargar_supabase.py

SQL de setup (ejecutar UNA VEZ en el SQL Editor de Supabase):
─────────────────────────────────────────────────────────────
    -- Habilitar extensión pgvector
    create extension if not exists vector;

    -- Tabla principal
    create table if not exists normas_drp (
        id                  serial primary key,
        id_norma            integer unique not null,
        nombre_norma        text not null,
        tipo_norma          text,
        numero_norma        text,
        anio                text,
        jurisdiccion        text,
        titulo_resumido     text,
        texto_resumido      text,
        url_pdf             text,
        vigente             boolean default true,
        weight_nombre       float default 0.7,
        weight_descripcion  float default 0.3,
        embedding_nombre    vector(1536),
        embedding_descripcion vector(1536),
        embedding_combined  vector(1536),
        created_at          timestamptz default now()
    );

    -- Índice IVFFlat para búsqueda vectorial rápida
    -- (para corpus pequeño <10k normas, no es estrictamente necesario)
    create index if not exists idx_embedding_combined
        on normas_drp using ivfflat (embedding_combined vector_cosine_ops)
        with (lists = 10);

    -- Función de búsqueda semántica (la usará la API)
    create or replace function buscar_normas(
        query_embedding vector(1536),
        top_k           int default 10,
        filtro_tipo     text default null,
        filtro_jur      text default null
    )
    returns table (
        id_norma        integer,
        nombre_norma    text,
        tipo_norma      text,
        jurisdiccion    text,
        titulo_resumido text,
        texto_resumido  text,
        url_pdf         text,
        similitud       float
    )
    language sql stable as $$
        select
            id_norma, nombre_norma, tipo_norma, jurisdiccion,
            titulo_resumido, texto_resumido, url_pdf,
            1 - (embedding_combined <=> query_embedding) as similitud
        from normas_drp
        where vigente = true
          and (filtro_tipo is null or tipo_norma = filtro_tipo)
          and (filtro_jur  is null or jurisdiccion = filtro_jur)
        order by embedding_combined <=> query_embedding
        limit top_k;
    $$;
─────────────────────────────────────────────────────────────
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

try:
    from supabase import create_client
    SUPABASE_OK = True
except ImportError:
    SUPABASE_OK = False
    print("[WARN] supabase-py no instalado. Ejecutá: pip install supabase")

DATA_DIR = Path(__file__).parent.parent / "data"

def main():
    print("=" * 60)
    print("ALEPH — Carga en Supabase (pgvector)")
    print("=" * 60)

    if not SUPABASE_OK:
        return

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("[ERROR] Definí SUPABASE_URL y SUPABASE_KEY en el archivo .env")
        return

    json_path = DATA_DIR / "normas_vectors.json"
    if not json_path.exists():
        print(f"[ERROR] No se encontró {json_path}")
        print("        Primero corré 02_embeddings.py")
        return

    with open(json_path, encoding="utf-8") as f:
        vectores = json.load(f)
    print(f"[OK] {len(vectores)} normas cargadas desde {json_path}")

    supabase = create_client(url, key)

    errores = 0
    for v in tqdm(vectores, desc="Insertando en Supabase"):
        fila = {
            "id_norma":              v["id_norma"],
            "nombre_norma":          v["nombre_norma"],
            "tipo_norma":            v["tipo_norma"],
            "numero_norma":          v["numero_norma"],
            "anio":                  v["anio"],
            "jurisdiccion":          v["jurisdiccion"],
            "titulo_resumido":       v["titulo_resumido"],
            "texto_resumido":        v["texto_resumido"],
            "url_pdf":               v["url_pdf"],
            "vigente":               v["vigente"],
            "weight_nombre":         v["weight_nombre"],
            "weight_descripcion":    v["weight_descripcion"],
            "embedding_nombre":      v["embedding_nombre"],
            "embedding_descripcion": v["embedding_descripcion"],
            "embedding_combined":    v["embedding_combined"],
        }
        try:
            supabase.table("normas_drp").upsert(fila, on_conflict="id_norma").execute()
        except Exception as e:
            print(f"  [!] Error en id_norma={v['id_norma']}: {e}")
            errores += 1
        time.sleep(0.05)

    print(f"\n[OK] Carga completa. {len(vectores) - errores} normas insertadas, {errores} errores.")

if __name__ == "__main__":
    main()

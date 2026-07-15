"""
02_embeddings.py
================
Genera embeddings semánticos para cada norma del dataset usando Ollama
(100% local, sin costo, sin API key).

Modelos recomendados (instalar con `ollama pull <modelo>`):
    - nomic-embed-text   → 768 dims, rápido, bueno para español
    - mxbai-embed-large  → 1024 dims, mejor calidad jurídica
    - bge-m3             → 1024 dims, multilingüe, muy bueno

Prerequisito:
    - Ollama corriendo: `ollama serve`
    - Modelo descargado: `ollama pull nomic-embed-text`
    - Haber corrido 01_scraper.py → ../data/normas_raw.csv

Uso:
    python 02_embeddings.py
    python 02_embeddings.py --modelo mxbai-embed-large

Salida:
    ../data/normas_embeddings.csv   (sin vectores, para inspección)
    ../data/normas_vectors.json     (con vectores, para carga en BD)
"""

import json
import time
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"

OLLAMA_URL    = "http://localhost:11434/api/embeddings"
MODELO_DEFAULT = "nomic-embed-text"

WEIGHT_NOMBRE      = 0.7
WEIGHT_DESCRIPCION = 0.3

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_embedding(text: str, modelo: str, retries: int = 3) -> list[float]:
    """Obtiene embedding desde Ollama con reintentos."""
    text = text.strip()
    if not text:
        return None  # se resuelve después con dimensión correcta

    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": modelo,
                "prompt": text
            }, timeout=60)
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            wait = 3 * (attempt + 1)
            print(f"  [!] Error (intento {attempt+1}): {e}. Esperando {wait}s...")
            time.sleep(wait)
    print("  [!!] Falló definitivamente, devolviendo ceros.")
    return None

def detectar_dimension(modelo: str) -> int:
    """Detecta la dimensión del modelo haciendo un embedding de prueba."""
    emb = get_embedding("test", modelo)
    if emb:
        return len(emb)
    raise RuntimeError(f"No se pudo conectar a Ollama con el modelo {modelo}. "
                       f"¿Está corriendo `ollama serve`?")

def normalizar(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-8)

def combinar_embeddings(emb_nombre: list, emb_desc: list) -> list[float]:
    """Combinación ponderada 0.7/0.3 igual que GINA, renormalizada."""
    n = normalizar(np.array(emb_nombre, dtype=np.float32))
    d = normalizar(np.array(emb_desc,   dtype=np.float32))
    combined = WEIGHT_NOMBRE * n + WEIGHT_DESCRIPCION * d
    return normalizar(combined).tolist()

def construir_texto_nombre(row: pd.Series) -> str:
    partes = [row["nombre_norma"]]
    if row.get("jurisdiccion"):
        partes.append(f"Jurisdicción: {row['jurisdiccion']}")
    if row.get("tema") and str(row["tema"]).strip():
        partes.append(f"Tema: {row['tema']}")
    # Subtítulo descriptivo (lo que va después del –)
    titulo = str(row.get("titulo_resumido", ""))
    if "–" in titulo:
        subtitulo = titulo.split("–", 1)[-1].strip()
        if subtitulo:
            partes.append(subtitulo)
    return " | ".join(partes)

def construir_texto_descripcion(row: pd.Series) -> str:
    desc = str(row.get("texto_resumido", "")).strip()
    if len(desc) < 50:
        desc = str(row.get("titulo_resumido", "")).strip()
    return desc

def verificar_ollama(modelo: str):
    """Verifica que Ollama esté corriendo y el modelo disponible."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        modelos = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        modelo_base = modelo.split(":")[0]
        if modelo_base not in modelos:
            print(f"[WARN] El modelo '{modelo}' no está descargado.")
            print(f"       Ejecutá: ollama pull {modelo}")
            print(f"       Modelos disponibles: {modelos}")
            return False
        return True
    except Exception:
        print("[ERROR] No se pudo conectar a Ollama.")
        print("        Asegurate de que esté corriendo: ollama serve")
        return False

# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generador de embeddings con Ollama")
    parser.add_argument("--modelo", default=MODELO_DEFAULT,
                        help=f"Modelo Ollama a usar (default: {MODELO_DEFAULT})")
    args = parser.parse_args()
    modelo = args.modelo

    print("=" * 60)
    print(f"ALEPH — Embeddings con Ollama ({modelo})")
    print("=" * 60)

    if not verificar_ollama(modelo):
        return

    csv_entrada = DATA_DIR / "normas_raw.csv"
    if not csv_entrada.exists():
        print(f"[ERROR] No se encontró {csv_entrada}")
        print("        Primero corré 01_scraper.py")
        return

    df = pd.read_csv(csv_entrada)
    print(f"[OK] {len(df)} normas cargadas")

    # Detectar dimensión del modelo
    print(f"[...] Detectando dimensión del modelo {modelo}...")
    dim = detectar_dimension(modelo)
    vector_cero = [0.0] * dim
    print(f"[OK] Dimensión: {dim}")
    print(f"[OK] Embeddings por norma: 2 (nombre + descripcion) -> 1 combinado")
    print(f"[OK] Total llamadas a Ollama: {len(df) * 2}")
    print()

    vectores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generando embeddings"):
        texto_nombre = construir_texto_nombre(row)
        texto_desc   = construir_texto_descripcion(row)

        emb_nombre = get_embedding(texto_nombre, modelo) or vector_cero
        emb_desc   = get_embedding(texto_desc,   modelo) or vector_cero
        emb_combined = combinar_embeddings(emb_nombre, emb_desc)

        vectores.append({
            "id_norma":              int(row["id_norma"]),
            "nombre_norma":          row["nombre_norma"],
            "tipo_norma":            row["tipo_norma"],
            "numero_norma":          str(row.get("numero_norma", "")),
            "anio":                  str(row.get("anio", "")),
            "jurisdiccion":          row.get("jurisdiccion", ""),
            "tema":                  str(row.get("tema", "")),
            "titulo_resumido":       row.get("titulo_resumido", ""),
            "texto_resumido":        row.get("texto_resumido", ""),
            "url_pdf":               row.get("url_pdf", ""),
            "vigente":               True,
            "modelo_embedding":      modelo,
            "embedding_dim":         dim,
            "texto_embedding_nombre": texto_nombre,
            "texto_embedding_desc":   texto_desc,
            "weight_nombre":         WEIGHT_NOMBRE,
            "weight_descripcion":    WEIGHT_DESCRIPCION,
            "embedding_nombre":      emb_nombre,
            "embedding_descripcion": emb_desc,
            "embedding_combined":    emb_combined,
        })

    # Guardar JSON completo (con vectores)
    out_json = DATA_DIR / "normas_vectors.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(vectores, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Vectores guardados en: {out_json}")

    # Guardar CSV sin vectores (para inspección)
    cols = [k for k in vectores[0].keys() if not k.startswith("embedding_")]
    df_out = pd.DataFrame([{k: v[k] for k in cols} for v in vectores])
    out_csv = DATA_DIR / "normas_embeddings.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] CSV sin vectores:    {out_csv}")
    print(f"\n[OK] Completado. {len(vectores)} normas vectorizadas con {modelo}.")
    print(f"     Dimensión de vectores: {dim}")
    print(f"     Costo: $0.00 USD")

if __name__ == "__main__":
    main()

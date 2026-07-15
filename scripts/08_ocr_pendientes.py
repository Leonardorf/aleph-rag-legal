"""
08_ocr_pendientes.py
=====================
Corre OCR (easyocr) sobre los PDFs que 07_fix_duplicados.py descargó pero
no pudo leer digitalmente (escaneados), y sube sus chunks a Supabase.
Uso puntual, una sola vez.
"""

import os
import re
import gc
import json
import time
from pathlib import Path

import fitz
import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
PDFS_DIR   = BASE_DIR / "pdfs"
TEXTOS_DIR = DATA_DIR / "textos_completos"

OLLAMA_URL   = "http://localhost:11434/api/embeddings"
MODELO_EMBED = "nomic-embed-text"
CHUNK_SIZE   = 1000
OVERLAP      = 200

PENDIENTES = [
    "OSC 396 Anexo VII", "OSC 396 Anexo IV", "OSC 396 Anexo II",
    "OSC 396 Anexo VI", "OSC 396 Anexo VIII", "OSC 396 Anexo V",
    "OSC 396 Anexo III", "DTR 1 (UIF)", "DTR 2 (UIF)", "DTR 5 (UIF)",
]


def nombre_a_archivo(nombre: str) -> str:
    return re.sub(r"[^\w\-]", "_", nombre)[:80]


def extraer_texto_ocr(pdf_path: Path, reader) -> str:
    doc = fitz.open(str(pdf_path))
    partes = []
    for num_pag in range(doc.page_count):
        pix = doc[num_pag].get_pixmap(dpi=100, colorspace=fitz.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        resultado = reader.readtext(img, detail=0, paragraph=True)
        partes.append("\n".join(resultado))
    doc.close()
    return "\n\n".join(partes).strip()


def chunkar(texto: str, size: int, overlap: int) -> list:
    chunks, inicio, n = [], 0, len(texto)
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
            resp = requests.post(OLLAMA_URL, json={"model": MODELO_EMBED, "prompt": text.strip()}, timeout=60)
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            if intento < retries - 1:
                time.sleep(3 * (intento + 1))
            else:
                raise e


def main():
    with open(DATA_DIR / "normas_vectors.json", encoding="utf-8") as f:
        normas = json.load(f)
    por_nombre = {n["nombre_norma"]: n for n in normas}

    print("Cargando modelo easyocr...")
    import easyocr
    reader = easyocr.Reader(["es"], verbose=False)

    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    total_chunks, errores = 0, 0
    for nombre in PENDIENTES:
        n = por_nombre.get(nombre)
        if not n:
            print(f"[!] '{nombre}' no está en normas_vectors.json")
            continue
        pdf_path = PDFS_DIR / f"{nombre_a_archivo(nombre)}.pdf"
        txt_path = TEXTOS_DIR / f"{nombre_a_archivo(nombre)}.txt"
        print(f"\n{nombre}")
        if not pdf_path.exists():
            print(f"  [!] No existe el PDF: {pdf_path}")
            errores += 1
            continue

        if txt_path.exists():
            texto = txt_path.read_text(encoding="utf-8").strip()
        else:
            print("  [OCR] extrayendo...")
            texto = extraer_texto_ocr(pdf_path, reader)
            if texto:
                txt_path.write_text(texto, encoding="utf-8")
            gc.collect()

        if not texto:
            print("  [!] OCR no devolvió texto")
            errores += 1
            continue

        chunks = chunkar(texto, CHUNK_SIZE, OVERLAP)
        meta = {
            "nombre_norma":    nombre,
            "tipo_norma":      n.get("tipo_norma", ""),
            "anio":            str(n.get("anio", "") or ""),
            "jurisdiccion":    n.get("jurisdiccion", ""),
            "titulo_resumido": n.get("titulo_resumido", ""),
            "url_pdf":         n.get("url_pdf", ""),
        }
        encabezado = f"{meta['tipo_norma']} {nombre}".strip()
        print(f"  {len(chunks)} chunks ({len(texto):,} chars)")
        for idx, chunk in enumerate(chunks):
            try:
                emb = get_embedding(f"{encabezado}: {chunk}")
                sb.table("norma_chunks").insert({**meta, "chunk_idx": idx, "texto": chunk, "embedding": emb}).execute()
                total_chunks += 1
            except Exception as e:
                print(f"    [!] Error chunk {idx}: {e}")
                errores += 1
            time.sleep(0.03)

    print(f"\n[OK] {total_chunks} chunks subidos, {errores} errores")


if __name__ == "__main__":
    main()

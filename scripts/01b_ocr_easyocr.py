"""
01b_ocr_easyocr.py
==================
Aplica OCR con easyocr a los documentos sin texto en normas_raw.csv.
Actualiza el CSV con el texto extraído y re-genera embeddings + galaxia.

Uso:
    python 01b_ocr_easyocr.py
"""

import re
import ssl
import csv
import fitz
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Fix SSL para descarga de modelos easyocr
ssl._create_default_https_context = ssl._create_unverified_context

import easyocr

DATA_DIR = Path(__file__).parent.parent / "data"
PDFS_DIR = Path(__file__).parent.parent / "pdfs"

CSV_IN  = DATA_DIR / "normas_raw.csv"
CSV_OUT = DATA_DIR / "normas_raw.csv"

MAX_CHARS  = 8000
DPI        = 150
MAX_PAGINAS = 4  # máximo de páginas a procesar por doc

def nombre_a_pdf(nombre_norma: str) -> Path:
    safe = re.sub(r"[^\w\-]", "_", nombre_norma)[:80] + ".pdf"
    return PDFS_DIR / safe

def extraer_texto_easyocr(reader: easyocr.Reader, pdf_path: Path) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        texto_total = ""
        for num_pag in range(min(MAX_PAGINAS, doc.page_count)):
            pix = doc[num_pag].get_pixmap(dpi=DPI, colorspace=fitz.csRGB)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            resultado = reader.readtext(img_np, detail=0, paragraph=True)
            texto_total += "\n".join(resultado) + "\n"
            if len(texto_total) >= MAX_CHARS:
                break
        doc.close()
        return texto_total[:MAX_CHARS].strip()
    except Exception as e:
        print(f"  [!] Error en {pdf_path.name}: {e}")
        return ""

def resumir_texto(texto: str, max_chars: int = 600) -> str:
    lineas = [l.strip() for l in texto.split("\n") if len(l.strip()) > 40]
    return " ".join(lineas[:6])[:max_chars]

def main():
    print("=" * 60)
    print("OCR con easyocr — documentos sin texto")
    print("=" * 60)

    with open(CSV_IN, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    sin_texto = [r for r in rows if not r.get("texto_completo", "").strip()]
    print(f"Documentos sin texto: {len(sin_texto)} de {len(rows)}")

    if not sin_texto:
        print("Nada que procesar.")
        return

    print("Cargando modelo easyocr (español)...")
    reader = easyocr.Reader(["es"], verbose=False)
    print("Modelo listo.\n")

    actualizados = 0
    for row in tqdm(rows, desc="Procesando"):
        if row.get("texto_completo", "").strip():
            continue  # ya tiene texto

        pdf_path = nombre_a_pdf(row["nombre_norma"])
        if not pdf_path.exists():
            tqdm.write(f"  [!] PDF no encontrado: {pdf_path.name}")
            continue

        texto = extraer_texto_easyocr(reader, pdf_path)
        if texto:
            row["texto_completo"] = texto
            row["texto_resumido"] = resumir_texto(texto)
            actualizados += 1
            tqdm.write(f"  [OK] {row['nombre_norma']} — {len(texto)} chars")
        else:
            tqdm.write(f"  [--] {row['nombre_norma']} — sin texto extraíble")

    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] {actualizados} documentos actualizados en {CSV_OUT}")
    print("\nSiguiente paso: correr 02_embeddings.py y 05_galaxia.py")

if __name__ == "__main__":
    main()

"""
05_ocr_textos.py
================
Extrae el texto completo de cada norma (digital o vía OCR) y lo guarda en
archivos locales de texto. No toca la red ni Supabase — sólo lee PDFs y
escribe .txt. El paso de subida a Supabase corre por separado en
06_chunks_subir.py, para no combinar en un mismo proceso "OCR + subida a
la nube" (esa combinación disparaba picos de memoria muy grandes en esta
máquina, probablemente por el antivirus/EDR corporativo).

Uso:
    python 05_ocr_textos.py            # saltea los que ya tienen .txt
    python 05_ocr_textos.py --force    # reprocesa todos
    python 05_ocr_textos.py --ocr      # fuerza OCR en todos los docs
"""

import re
import json
import argparse
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

DATA_DIR   = Path(__file__).parent.parent / "data"
PDFS_DIR   = Path(__file__).parent.parent / "pdfs"
TEXTOS_DIR = DATA_DIR / "textos_completos"

MIN_CHARS_POR_PAG = 80


def nombre_a_archivo(nombre_norma: str) -> str:
    return re.sub(r"[^\w\-]", "_", nombre_norma)[:80]


def nombre_a_pdf(nombre_norma: str) -> Path:
    return PDFS_DIR / f"{nombre_a_archivo(nombre_norma)}.pdf"


def extraer_texto_digital(pdf_path: Path) -> tuple[str, bool]:
    """Extrae texto con PyMuPDF. Devuelve (texto, es_imagen)."""
    try:
        doc = fitz.open(str(pdf_path))
        paginas = [page.get_text() for page in doc]
        doc.close()
        texto = "\n".join(paginas)
        texto = re.sub(r"[ \t]{3,}", " ", texto)
        texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
        promedio = len(texto) / max(len(paginas), 1)
        return texto, promedio < MIN_CHARS_POR_PAG
    except Exception as e:
        print(f"  [!] Error leyendo {pdf_path.name}: {e}")
        return "", True


def extraer_texto_ocr(pdf_path: Path, reader) -> str:
    """Aplica easyocr a todas las páginas del PDF."""
    try:
        doc = fitz.open(str(pdf_path))
        partes = []
        for num_pag in range(doc.page_count):
            pix = doc[num_pag].get_pixmap(dpi=100, colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            resultado = reader.readtext(img, detail=0, paragraph=True)
            partes.append("\n".join(resultado))
        doc.close()
        return "\n\n".join(partes).strip()
    except Exception as e:
        print(f"  [!] OCR error en {pdf_path.name}: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Extracción de texto (digital/OCR) para ALEPH")
    parser.add_argument("--force", action="store_true", help="Reprocesar aunque ya exista el .txt")
    parser.add_argument("--ocr",   action="store_true", help="Forzar OCR en todos los documentos")
    args = parser.parse_args()

    TEXTOS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "normas_vectors.json", encoding="utf-8") as f:
        normas = json.load(f)
    print(f"[OK] {len(normas)} normas en el JSON")

    reader = None
    total = len(normas)
    hechos = 0
    saltados = 0
    errores = 0

    for i, n in enumerate(normas, 1):
        nombre = n["nombre_norma"]
        pdf_path = nombre_a_pdf(nombre)
        txt_path = TEXTOS_DIR / f"{nombre_a_archivo(nombre)}.txt"
        print(f"\n[{i}/{total}] {nombre}")

        if txt_path.exists() and not args.force:
            print("  [OK] Ya extraído, se salta")
            saltados += 1
            continue

        if not pdf_path.exists():
            print(f"  [!] PDF no encontrado: {pdf_path.name}")
            errores += 1
            continue

        texto, es_imagen = extraer_texto_digital(pdf_path)

        if es_imagen or args.ocr:
            print("  [OCR] extrayendo con easyocr...")
            if reader is None:
                import easyocr
                print("  Cargando modelo easyocr (primera vez)...")
                reader = easyocr.Reader(["es"], verbose=False)
            texto = extraer_texto_ocr(pdf_path, reader)

        if not texto:
            print(f"  [--] Sin texto: {nombre}")
            errores += 1
            continue

        txt_path.write_text(texto, encoding="utf-8")
        print(f"  [OK] {len(texto):,} chars guardados en {txt_path.name}")
        hechos += 1

    print(f"\n{'='*60}")
    print(f"[OK] {hechos} extraídos, {saltados} saltados, {errores} errores.")
    print("=" * 60)


if __name__ == "__main__":
    main()

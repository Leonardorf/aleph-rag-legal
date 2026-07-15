"""
test_ocr_qwen.py
================
Prueba de OCR con qwen2.5vl via Ollama API sobre un PDF.

Uso:
    python test_ocr_qwen.py                        # usa OSC_576.pdf
    python test_ocr_qwen.py pdfs/DTR_1.pdf         # PDF específico
"""

import sys
import json
import base64
import fitz
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO     = "qwen2.5vl:7b"
DPI        = 150

PROMPT = (
    "Extraé todo el texto de este documento tal como aparece, "
    "respetando párrafos y estructura. "
    "No agregues interpretaciones ni comentarios, solo el texto del documento."
)

def pdf_pagina_a_base64(pdf_path: Path, num_pag: int = 0) -> str:
    doc = fitz.open(str(pdf_path))
    pix = doc[num_pag].get_pixmap(dpi=DPI)
    doc.close()
    return base64.b64encode(pix.tobytes("png")).decode()

def ocr_qwen(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    num_paginas = min(doc.page_count, 3)
    doc.close()

    texto_total = ""
    for i in range(num_paginas):
        print(f"  Procesando página {i+1}/{num_paginas}...")
        b64 = pdf_pagina_a_base64(pdf_path, i)
        resp = requests.post(OLLAMA_URL, json={
            "model":  MODELO,
            "prompt": PROMPT,
            "images": [b64],
            "stream": True,
        }, timeout=300, stream=True)
        resp.raise_for_status()
        fragmentos = []
        for linea in resp.iter_lines():
            if linea:
                chunk = json.loads(linea)
                fragmentos.append(chunk.get("response", ""))
                if chunk.get("done"):
                    break
        texto_total += "".join(fragmentos) + "\n"

    return texto_total.strip()

def main():
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pdfs/OSC_576.pdf")
    if not pdf_path.exists():
        print(f"No encontrado: {pdf_path}")
        sys.exit(1)

    print(f"PDF: {pdf_path}")
    print(f"Modelo: {MODELO}")
    print("-" * 50)

    texto = ocr_qwen(pdf_path)
    print(texto)
    print("-" * 50)
    print(f"Total chars: {len(texto)}")

if __name__ == "__main__":
    main()

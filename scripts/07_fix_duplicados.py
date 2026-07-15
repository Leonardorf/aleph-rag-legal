"""
07_fix_duplicados.py
=====================
Recuperación puntual de normas que quedaron "pisadas" entre sí porque
compartían un `nombre_norma` genérico (Dictamen, OSC 396, Decreto, DTR N).
El bug de origen (extracción de número en 01_scraper.py) ya está corregido
para scrapeos futuros; este script repara los datos ya existentes:

1. Detecta grupos de normas_vectors.json con nombre_norma repetido.
2. Les asigna un nombre único (a partir de titulo_resumido) para los casos
   Dictamen / OSC 396 / Decreto, o descarga+compara contenido para los DTR
   (donde dos URLs distintas pueden ser el mismo documento republicado).
3. Descarga los PDFs nuevos, extrae el texto y sube los chunks faltantes
   a Supabase (`norma_chunks`).
4. Borra las filas huérfanas con el nombre genérico viejo cuando ya fueron
   reemplazadas por el conjunto desambiguado.
5. Actualiza normas_vectors.json y normas_raw.csv para que reflejen los
   nombres corregidos de forma permanente.
"""

import os
import re
import gc
import json
import time
import hashlib
from pathlib import Path
from collections import defaultdict

import requests
import fitz  # PyMuPDF
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
PDFS_DIR   = BASE_DIR / "pdfs"
TEXTOS_DIR = DATA_DIR / "textos_completos"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

OLLAMA_URL   = "http://localhost:11434/api/embeddings"
MODELO_EMBED = "nomic-embed-text"
CHUNK_SIZE   = 1000
OVERLAP      = 200


def nombre_a_archivo(nombre: str) -> str:
    return re.sub(r"[^\w\-]", "_", nombre)[:80]


def descargar(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, verify=False)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"    [!] No se pudo descargar {url}: {e}")
        return False


def extraer_texto_digital(pdf_path: Path) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        paginas = [p.get_text() for p in doc]
        doc.close()
        texto = "\n".join(paginas)
        texto = re.sub(r"[ \t]{3,}", " ", texto)
        texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
        return texto
    except Exception as e:
        print(f"    [!] Error leyendo {pdf_path.name}: {e}")
        return ""


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
    print("=" * 70)
    print("Recuperación de normas con nombre_norma duplicado")
    print("=" * 70)

    json_path = DATA_DIR / "normas_vectors.json"
    with open(json_path, encoding="utf-8") as f:
        normas = json.load(f)

    grupos = defaultdict(list)
    for n in normas:
        grupos[n["nombre_norma"]].append(n)
    dup_groups = {k: v for k, v in grupos.items() if len(v) > 1}
    print(f"[OK] {len(dup_groups)} grupos con nombre repetido: {list(dup_groups.keys())}")

    cambios = {}          # id(entry) -> nuevo nombre_norma
    a_eliminar = set()    # id(entry) de entradas duplicadas reales a quitar del JSON
    orfanas_supabase = set()  # nombre_norma genérico viejo a borrar de norma_chunks

    # ── Grupos de reemplazo total (nombre genérico sin ningún dato útil) ──────
    reglas_regex = {
        "Dictamen": r"dictamen\s+(\d+)",
        "OSC 396":  r"anexo\s+([ivxlcdm]+|\d+)",
        "Decreto":  r"decreto\s+(\d+)",
    }
    formatos = {
        "Dictamen": "Dictamen {}",
        "OSC 396":  "OSC 396 Anexo {}",
        "Decreto":  "Decreto {}",
    }

    for base, patron in reglas_regex.items():
        entries = dup_groups.get(base, [])
        if not entries:
            continue
        nuevos_nombres = set()
        for n in entries:
            m = re.search(patron, n["titulo_resumido"], re.IGNORECASE)
            if not m:
                print(f"  [!] No pude desambiguar '{base}': {n['titulo_resumido']!r}")
                continue
            nuevo = formatos[base].format(m.group(1).upper())
            cambios[id(n)] = nuevo
            nuevos_nombres.add(nuevo)
        if len(nuevos_nombres) == len(entries):
            orfanas_supabase.add(base)
        else:
            print(f"  [!] '{base}' no quedó totalmente desambiguado, no se borrará la fila vieja")

    # ── Grupos DTR: puede ser duplicado real o documento distinto ────────────
    tmp_dir = BASE_DIR / "_tmp_dtr_check"
    for base in [k for k in dup_groups if k.startswith("DTR")]:
        entries = dup_groups[base]
        if len(entries) != 2:
            continue
        urls = [e["url_pdf"] for e in entries]
        if urls[0] == urls[1]:
            print(f"  [=] '{base}': mismo PDF repetido dos veces, se elimina la copia extra")
            a_eliminar.add(id(entries[1]))
            continue

        txt_cache = TEXTOS_DIR / f"{nombre_a_archivo(base)}.txt"
        hash_cache = None
        if txt_cache.exists():
            hash_cache = hashlib.md5(txt_cache.read_text(encoding="utf-8")[:2000].encode()).hexdigest()

        tmp_dir.mkdir(exist_ok=True)
        hashes = []
        for i, e in enumerate(entries):
            tmp_pdf = tmp_dir / f"{nombre_a_archivo(base)}_{i}.pdf"
            ok = descargar(e["url_pdf"], tmp_pdf)
            texto = extraer_texto_digital(tmp_pdf) if ok else ""
            hashes.append(hashlib.md5(texto[:2000].encode()).hexdigest() if texto else None)
            time.sleep(0.3)

        if hashes[0] and hashes[0] == hashes[1]:
            print(f"  [=] '{base}': las dos URL tienen el mismo contenido, se elimina la copia extra")
            a_eliminar.add(id(entries[1]))
            continue

        # contenidos distintos: el que coincide con lo ya cacheado conserva el nombre
        idx_conocido = next((i for i, h in enumerate(hashes) if h and h == hash_cache), None)
        if idx_conocido is None:
            idx_conocido = 0  # no había caché o no coincidió con ninguno: el primero queda como está
        idx_nuevo = 1 - idx_conocido
        etiqueta = "UIF" if "uif" in entries[idx_nuevo]["url_pdf"].lower() else "2"
        nuevo_nombre = f"{base} ({etiqueta})"
        cambios[id(entries[idx_nuevo])] = nuevo_nombre
        print(f"  [+] '{base}': son documentos distintos, el nuevo queda como '{nuevo_nombre}'")

    if not cambios and not a_eliminar:
        print("\n[OK] No hay nada para corregir.")
        return

    # ── Aplicar cambios en memoria ────────────────────────────────────────────
    nuevas_normas = []
    entradas_a_procesar = []  # (entry, nombre_nuevo) que hay que descargar/OCR/subir
    for n in normas:
        if id(n) in a_eliminar:
            continue
        if id(n) in cambios:
            nuevo_nombre = cambios[id(n)]
            n["nombre_norma"] = nuevo_nombre
            entradas_a_procesar.append(n)
        nuevas_normas.append(n)

    print(f"\n[OK] {len(entradas_a_procesar)} normas quedaron con nombre corregido y deben subirse")
    print(f"[OK] {len(a_eliminar)} entradas duplicadas reales se quitan del dataset")

    # ── Descargar PDF + extraer texto para las entradas corregidas ───────────
    print("\n[1/3] Descargando PDFs y extrayendo texto...")
    for n in entradas_a_procesar:
        nombre = n["nombre_norma"]
        pdf_path = PDFS_DIR / f"{nombre_a_archivo(nombre)}.pdf"
        txt_path = TEXTOS_DIR / f"{nombre_a_archivo(nombre)}.txt"
        print(f"  - {nombre}")
        if not pdf_path.exists():
            descargar(n["url_pdf"], pdf_path)
            time.sleep(0.3)
        if not txt_path.exists() and pdf_path.exists():
            texto = extraer_texto_digital(pdf_path)
            if texto:
                txt_path.write_text(texto, encoding="utf-8")
            else:
                print(f"    [!] Sin texto extraíble (podría necesitar OCR): {nombre}")

    # ── Subir a Supabase ──────────────────────────────────────────────────────
    print("\n[2/3] Chunkeando, generando embeddings y subiendo a Supabase...")
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    existentes = sb.table("norma_chunks").select("nombre_norma").execute()
    ya_subidas = {row["nombre_norma"] for row in existentes.data}

    total_chunks, errores = 0, 0
    for n in entradas_a_procesar:
        nombre = n["nombre_norma"]
        if nombre in ya_subidas:
            print(f"  - {nombre}: ya estaba subida, se salta")
            continue
        txt_path = TEXTOS_DIR / f"{nombre_a_archivo(nombre)}.txt"
        if not txt_path.exists():
            print(f"  - {nombre}: sin .txt, no se puede subir")
            errores += 1
            continue
        texto = txt_path.read_text(encoding="utf-8").strip()
        if not texto:
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
        print(f"  - {nombre}: {len(chunks)} chunks")
        for idx, chunk in enumerate(chunks):
            try:
                emb = get_embedding(f"{encabezado}: {chunk}")
                sb.table("norma_chunks").insert({**meta, "chunk_idx": idx, "texto": chunk, "embedding": emb}).execute()
                total_chunks += 1
            except Exception as e:
                print(f"    [!] Error chunk {idx}: {e}")
                errores += 1
            time.sleep(0.03)
        gc.collect()

    print(f"[OK] {total_chunks} chunks subidos, {errores} errores")

    # ── Borrar filas huérfanas con el nombre genérico viejo ──────────────────
    print("\n[3/3] Limpiando filas huérfanas...")
    for nombre_viejo in orfanas_supabase:
        try:
            sb.table("norma_chunks").delete().eq("nombre_norma", nombre_viejo).execute()
            print(f"  - Borrado '{nombre_viejo}' de norma_chunks")
        except Exception as e:
            print(f"  [!] No se pudo borrar '{nombre_viejo}': {e}")
        for carpeta, ext in [(PDFS_DIR, "pdf"), (TEXTOS_DIR, "txt")]:
            p = carpeta / f"{nombre_a_archivo(nombre_viejo)}.{ext}"
            if p.exists():
                p.unlink()
                print(f"  - Borrado archivo local {p.name}")

    # ── Persistir el dataset corregido ───────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(nuevas_normas, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] {json_path.name} actualizado ({len(nuevas_normas)} normas)")

    raw_path = DATA_DIR / "normas_raw.csv"
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        url_a_nombre = {n["url_pdf"]: n["nombre_norma"] for n in nuevas_normas}
        urls_eliminadas = {n["url_pdf"] for n in normas if id(n) in a_eliminar}
        df = df[~df["url_pdf"].isin(urls_eliminadas)]
        df["nombre_norma"] = df["url_pdf"].map(url_a_nombre).fillna(df["nombre_norma"])
        df.to_csv(raw_path, index=False, encoding="utf-8")
        print(f"[OK] {raw_path.name} actualizado ({len(df)} filas)")

    if tmp_dir.exists():
        for f in tmp_dir.glob("*"):
            f.unlink()
        tmp_dir.rmdir()

    print("\n" + "=" * 70)
    print("Listo.")
    print("=" * 70)


if __name__ == "__main__":
    main()

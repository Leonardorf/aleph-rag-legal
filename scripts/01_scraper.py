"""
01_scraper.py
=============
Scraper para un portal de normativa registral provincial.
Descarga todos los PDFs y construye un CSV con los metadatos extraídos
del nombre del archivo y del texto interno del PDF.

Uso:
    python 01_scraper.py

Salida:
    ../data/normas_raw.csv
    ../pdfs/  (PDFs descargados)
"""

import re
import time
import warnings
import requests
import urllib3
import fitz          # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://jusmendoza.gob.ar"
NORMATIVA_URL = (
    "https://jusmendoza.gob.ar/direccion-de-registros-publicos-y-archivo-judicial"
    "-%c2%b7-inicio/normativa-direccion-de-registros-publicos-1-3-y-4-cj/"
)

PDFS_DIR = Path(__file__).parent.parent / "pdfs"
DATA_DIR = Path(__file__).parent.parent / "data"
PDFS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── Clasificadores ────────────────────────────────────────────────────────────

TIPO_PATTERNS = [
    (r"resoluc[ií]on\s+administrativa|\bra\s+n[°º]",   "Resolución Administrativa"),
    (r"resoluc[ií]on\s+conjunta|resoluci.n\s+conjunta", "Resolución Conjunta"),
    (r"resoluc[ií]on|resoluci.n",                       "Resolución"),
    (r"\bdtr\b|disposici.n\s+t[eé]cnico",               "DTR"),
    (r"\bosc\b|orden\s+de\s+servicio\s+c",              "OSC"),
    (r"\bosa\b|orden\s+de\s+servicio\s+a",              "OSA"),
    (r"\bphe\s+osc\b",                                  "OSC"),
    (r"orden\s+de\s+servicio",                          "OSC"),
    (r"\bdictamen\b",                                   "Dictamen"),
    (r"acordada",                                       "Acordada"),
    (r"decreto",                                        "Decreto"),
    (r"\bley\b",                                        "Ley"),
    (r"circular",                                       "Circular"),
    (r"disposici[oó]n",                                 "Disposición"),
    (r"instrucci[oó]n",                                 "Instrucción"),
    (r"anexo",                                          "Anexo"),
    (r"plano",                                          "Plano"),
]

JURISDICCION_PATTERNS = [
    (r"\bnacional\b",   "Nacional"),
    (r"\bprovincial\b", "Provincial"),
    (r"\bnaciona[l]?\b|\bley\s+\d{5,}\b", "Nacional"),
]

def clasificar_tipo(texto: str) -> str:
    t = texto.lower()
    for patron, tipo in TIPO_PATTERNS:
        if re.search(patron, t):
            return tipo
    return "Otro"

def inferir_jurisdiccion(titulo: str, tipo: str) -> str:
    t = titulo.lower()
    # OSC, OSA y DTR son siempre normativa interna del organismo
    if tipo in ("OSC", "OSA", "DTR", "Dictamen",
                "Resolución Administrativa", "Resolución Conjunta"):
        return "DRP"
    # Leyes con número > 10000 son nacionales
    m = re.search(r"ley\s+(?:n[°º]?\s*)?(\d+)", t)
    if m and int(m.group(1)) > 10000:
        return "Nacional"
    if re.search(r"ley\s+(?:n[°º]?\s*)?8\d{3}", t):
        return "Provincial"
    if "acordada" in tipo.lower():
        return "Provincial"
    if "decreto" in tipo.lower() or "resolución" in tipo.lower():
        return "Provincial"
    for patron, jur in JURISDICCION_PATTERNS:
        if re.search(patron, t):
            return jur
    return "Provincial"

def extraer_numero(titulo: str, tipo: str) -> str:
    patrones = [
        r"n[°º]\s*([\w/\-]+)",
        r"n[°º]\s*(\d+[\w/\-]*)",
        rf"{re.escape(tipo.lower())}\s+n?[°º]?\.?\s*(\d{{1,5}})" if tipo and tipo != "Otro" else None,
        r"(\d{2,5}/\d{2,4})",
        r"(\d{4,6})",
        r"\b(\d{1,4})\b",
    ]
    for p in patrones:
        if p is None:
            continue
        m = re.search(p, titulo, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def extraer_anexo(titulo: str) -> str:
    """Detecta un sufijo 'Anexo <número/romano>' para distinguir documentos
    que comparten el mismo número de norma (ej. OSC 396 Anexo II vs Anexo VII)."""
    m = re.search(r"anexo\s+([ivxlcdm]+|\d+)", titulo, re.IGNORECASE)
    return f"Anexo {m.group(1).upper()}" if m else ""

def extraer_anio(titulo: str, numero: str) -> str:
    # Intentar del número (ej. 04/26 → 2026)
    m = re.search(r"/(\d{2,4})$", numero)
    if m:
        y = m.group(1)
        return f"20{y}" if len(y) == 2 else y
    # Intentar del título
    m = re.search(r"\b(20\d{2}|19\d{2})\b", titulo)
    if m:
        return m.group(1)
    return ""

def generar_nombre_norma(tipo: str, numero: str, anio: str) -> str:
    if not numero:
        return tipo
    if anio and "/" not in numero:
        return f"{tipo} {numero}/{anio}"
    return f"{tipo} {numero}"

# ── Extracción de texto PDF ───────────────────────────────────────────────────

def extraer_texto_pdf(pdf_path: Path, max_chars: int = 8000) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        texto = ""
        for page in doc:
            texto += page.get_text()
            if len(texto) >= max_chars:
                break
        doc.close()
        # Limpieza básica
        texto = re.sub(r"\s{3,}", "\n", texto)
        texto = re.sub(r"\n{3,}", "\n\n", texto)
        return texto[:max_chars].strip()
    except Exception as e:
        print(f"  [!] Error leyendo PDF: {e}")
        return ""

def resumir_texto(texto: str, max_chars: int = 600) -> str:
    """Toma los primeros párrafos con contenido real como resumen."""
    lineas = [l.strip() for l in texto.split("\n") if len(l.strip()) > 40]
    resumen = " ".join(lineas[:6])
    return resumen[:max_chars]

# ── Scraping ──────────────────────────────────────────────────────────────────

def obtener_links_pdfs(url: str) -> list[dict]:
    """
    Parsea la página y devuelve lista de {titulo, url_pdf}.
    Si el sitio no está disponible, usa el listado hardcodeado como fallback.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith(".pdf") or "/wp-content/uploads/" in href:
                titulo = a.get_text(strip=True)
                if not titulo:
                    titulo = href.split("/")[-1].replace("-", " ").replace(".pdf", "")
                if not href.startswith("http"):
                    href = BASE_URL + href
                links.append({"titulo": titulo, "url_pdf": href})
        print(f"[scraper] {len(links)} PDFs encontrados en la página.")
        return links
    except Exception as e:
        print(f"[scraper] No se pudo acceder a la página ({e}). Usando listado hardcodeado.")
        return LINKS_HARDCODEADOS

# Listado hardcodeado como fallback (copiado de la página)
LINKS_HARDCODEADOS = [
    # ── Resoluciones Administrativas ──────────────────────────────────────────
    {"titulo": "Resolución Administrativa RA N° 05/26 – Informes digitales de titularidad, inhibición y estado jurídico del inmueble",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2026/05/Resolucion-Administrativa-RA-N%C2%B0-05-24-Informes-digitales-de-titularidad-inhibicion-y-estado-juridico-del-inmueble.pdf"},
    {"titulo": "Resolución Administrativa RA N° 04/26 – Principio de Rogación – Aplicación",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2026/04/Resolucion-Administrativa-RA-N%C2%B0-04-26.pdf"},
    {"titulo": "Resolución Administrativa RA N° 03/26 – Cancelación de Inhibición con firma digital",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2026/04/Resolucion-Administrativa-RA-N%C2%B0-03.26-Cancelacion-de-inhibicion-con-firma-digital.pdf"},
    {"titulo": "Resolución Administrativa RA N° 03/24 – Implementación Certificados en soporte papel",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2025/10/Resolucion-Administrativa-RA-N%C2%B0-03-24-Implementacion-Certificados-en-soporte-papel.pdf"},
    # ── Resoluciones Conjuntas ────────────────────────────────────────────────
    {"titulo": "Resolución Conjunta N° 02/2025 – Informe de Titularidad Provincial",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2025/09/Resolucion-Conjunta-N%C2%B0-0225-Informe-de-Titularidad-Provincial.pdf"},
    {"titulo": "Resolución Conjunta N° 01/2025 – Informe Titularidad Negativo",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2025/08/Resolucion-Conjunta-N%C2%B0-01-2025-Informe-Titularidad-Negativo.pdf"},
    {"titulo": "Resolución Conjunta N° 01/2024 – Infraestructura de Datos Espaciales de Mendoza",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/11/Resolucion-Conjunta-N%C2%B0-01-2024-Infraestructura-de-Datos-Espaciales-de-Mendoza.pdf"},
    # ── Acordadas ─────────────────────────────────────────────────────────────
    {"titulo": "Acordada N° 31766 – Comisión Destrucción de Expedientes",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/12/Acordada-N%C2%B0-31766-Comision-Destruccion-de-Expedientes.pdf"},
    # ── Leyes ─────────────────────────────────────────────────────────────────
    {"titulo": "Ley N° 17801 – Régimen Nacional Registral",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Ley-N%C2%B0-17801-Regimen-Nacional-Registral.pdf"},
    {"titulo": "Ley 8236 – Reglamentación Provincial de la Ley 17.801",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Ley-8236-Reglamentacion-Provincial-de-la-Ley-17.801.pdf"},
    {"titulo": "Ley 6279 – Colaboración del Colegio Notarial a los Registros de Propiedad Inmueble",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Ley-6279-Colaboracion-del-Colegio-Notarial-a-los-Registros-de-Propiedad-Inmueble.pdf"},
    # ── Resoluciones / Decretos ───────────────────────────────────────────────
    {"titulo": "Resolución 166 – Zonas de Frontera",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Resolucion-166-2009-Zonas-de-Fronteras.pdf"},
    {"titulo": "Decreto 253 – 2018 – Zona de Seguridad de Frontera",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Decreto-253-2018-Zona-de-Seguridad-de-Frontera.pdf"},
    {"titulo": "Plano de Mendoza. Modificación 2018",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Plano-de-Mendoza.-Modificacion-2018.pdf"},
    {"titulo": "Anexo 1 – Decreto 283. Zona frontera desarrollo y seguridad",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Anexo-1.-Decreto-283.-Zona-frontera-desarrollo-y-seguridad.pdf"},
    {"titulo": "Anexo 2 – Decreto 253 – Denominación y descripción de areas",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Anexo-2.-Decreto-253.-Denominacion-y-descripcion-de-areas.pdf"},
    # ── DTR — Disposiciones Técnico Registrales ───────────────────────────────
    {"titulo": "DTR N° 1 – UIF",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/uif-dtr-1-1.pdf",
     "tema": "UIF"},
    {"titulo": "DTR N° 1 – UIF (versión 2)",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/uif-dtr-1.pdf",
     "tema": "UIF"},
    {"titulo": "DTR N° 2 – UIF – Modificación DTR Conjunta N° 1",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Disposicion-tecnico-registral-conjunta-N%C2%B0-2.pdf",
     "tema": "UIF"},
    {"titulo": "DTR N° 2 – UIF – Modificación DTR 1",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/uif-dtr-2.pdf",
     "tema": "UIF"},
    {"titulo": "DTR N° 3 – Hipoteca – Cómputo plazo de caducidad",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/hipoteca-dtr-3.pdf",
     "tema": "Hipoteca"},
    {"titulo": "DTR N° 4 – Inscripción provisional – Plazo solicitud de prórroga",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Disposicion-tecnico-registral-conjunta-N%C2%B0-4.pdf",
     "tema": "Inscripción provisional"},
    {"titulo": "DTR N° 5 – UIF – Beneficiarios finales",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Disposicion-tecnico-registral-conjunta-N%C2%B0-5.pdf",
     "tema": "UIF"},
    {"titulo": "DTR N° 5 – UIF – Beneficiarios finales (versión 2)",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/uif-dtr-5.pdf",
     "tema": "UIF"},
    # ── OSC — Órdenes de Servicio Conjuntas ───────────────────────────────────
    {"titulo": "OSC N° 333 – Medidas cautelares – Anotación de litis",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/medidas-cautelares-os-333.pdf",
     "tema": "Medidas cautelares"},
    {"titulo": "OSC N° 396 Anexo II – Mandatos – Registración",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/mandatos-os-396-anexo-II.pdf",
     "tema": "Mandatos"},
    {"titulo": "OSC N° 396 Anexo III – Sociedades – Sección IV Ley 19550",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/sociedades-os-396-anexo-III.pdf",
     "tema": "Sociedades"},
    {"titulo": "OSC N° 396 Anexo IV – Hipoteca – Cláusula de actualización",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/hipoteca-os-396-anexo-IV.pdf",
     "tema": "Hipoteca"},
    {"titulo": "OSC N° 396 Anexo V – Sinceramiento fiscal",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/sinceramiento-fiscal-os-396-anexo-V.pdf",
     "tema": "Sinceramiento fiscal"},
    {"titulo": "OSC N° 396 Anexo VI – Medidas cautelares – Consorcio PH, sociedades, divorcio, usufructo y nulidad",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/medidas-cautelares-os-396-anexo-VI.pdf",
     "tema": "Medidas cautelares"},
    {"titulo": "OSC N° 396 Anexo VII – Accesoriedad",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/03/Accesoriedad-Orden-de-Servicio-N%C2%B0-396-Anexo-VII.pdf",
     "tema": "Accesoriedad"},
    {"titulo": "OSC N° 396 Anexo VIII – Segundo testimonio",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/segundo-testimonio-os-396-anexo-VIII.pdf",
     "tema": "Segundo testimonio"},
    {"titulo": "OSC N° 402 – Medidas cautelares – Modificación anotación de litis",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Orden-de-Servicio-N%C2%B0-402.pdf",
     "tema": "Medidas cautelares"},
    {"titulo": "OSC N° 488 – Sinceramiento fiscal",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Orden-de-Servicio-N%C2%B0-488.pdf",
     "tema": "Sinceramiento fiscal"},
    {"titulo": "OSC N° 526 – Tracto abreviado – Caja forense",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/tracto-abreviado-os-526.pdf",
     "tema": "Tracto abreviado"},
    {"titulo": "OSC N° 575 – Sucesiones – Partición y adjudicación",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/sucesiones-os-575.pdf",
     "tema": "Sucesiones"},
    {"titulo": "OSC N° 576 – Identidad de género – Ley 26743",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Orden-de-Servicio-N%C2%B0-576.pdf",
     "tema": "Identidad de género"},
    {"titulo": "OSC N° 577 – Servidumbres – Servidumbres administrativas",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/02/Orden-de-Servicio-N%C2%B0-577.pdf",
     "tema": "Servidumbres"},
    {"titulo": "OSC N° 2-24 – Fideicomiso – Calificación Registral",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/fideicomiso-OS-C-2-24.pdf",
     "tema": "Fideicomiso"},
    {"titulo": "OSC N° 3-24 – Tierras rurales – Calificación de Tierras Rurales",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/05/tierras-rurales-calificacion-de-tierras-rurales-OS-C-3-24.pdf",
     "tema": "Tierras rurales"},
    {"titulo": "OSC N° 4-24 – Dominio público – Incorporación de bienes del dominio público del estado a la base de titulares",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/08/Dominio-publico-del-estado-a-la-base-de-los-titulares-O.S.C-4-24.pdf",
     "tema": "Dominio público"},
    {"titulo": "OSC N° 5-24 – Tierras rurales – Calificación de Tierras Rurales",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/06/tierras-rurales-calificacion-de-tierras-rurales-OSC-5-24.pdf",
     "tema": "Tierras rurales"},
    {"titulo": "OSC N° 8-24 – Decomiso y extinción de dominio – Calificación de decomiso",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/08/O.S.C.-08-24.pdf",
     "tema": "Decomiso"},
    {"titulo": "OSC N° 9-25 – Adecuación PH Especial – No se califica inhibición",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2025/02/O.S.C.9-25.pdf",
     "tema": "Propiedad horizontal"},
    {"titulo": "OSC N° 10-25 – Modificación OSC N° 2-24 – Calificación registral fideicomiso",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2025/02/O.S.C.10-25.pdf",
     "tema": "Fideicomiso"},
    {"titulo": "OSC N° 12-25 – Sucesiones – Ampliación de supuestos OSC N° 575",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2025/05/Ampliacion-de-presunpuestos-O.S.C.12-25.pdf",
     "tema": "Sucesiones"},
    {"titulo": "OSC N° 13-25 – Publicidad cartular – Modificación de OSC N° 07-2024 – Títulos notariales, judiciales y administrativos",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2025/08/Publicidad-Cartular-OSC-13-25.pdf",
     "tema": "Publicidad cartular"},
    {"titulo": "OSC N° 14-25 – Mandatos – Calificación Registral",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2025/10/OSC-N%C2%B0-14-25.pdf",
     "tema": "Mandatos"},
    {"titulo": "OSC N° 15-25 – Vivienda – Modificación Orden de Servicio Vivienda (arts. 244 a 256 CCCN)",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2026/01/O.S.C.15-25-2.pdf",
     "tema": "Vivienda"},
    {"titulo": "OSC N° 16-25 – Modificación de Reglamento PH – Calificación registral de modificación de Reglamento de Copropiedad y Administración",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2025/12/O.S.C.16-25.pdf",
     "tema": "Propiedad horizontal"},
    {"titulo": "OSC N° 17-25 – Particiones extrajudiciales – Divorcio. No exigencia de aportes Caja Forense y conformidades profesionales",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2025/12/O.S.C.17-25.pdf",
     "tema": "Particiones extrajudiciales"},
    {"titulo": "OSC N° 19-26 – Publicidad cartular – Modificación de OSC N° 07-2024 – Vivienda",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2026/06/O.S.C.19-26.pdf",
     "tema": "Publicidad cartular"},
    {"titulo": "PHE OSC N° 01-23 – Rúbrica libros – Rúbrica libros consorcio propiedad horizontal",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/rubrica-libros-os-c-01-2023.pdf",
     "tema": "Propiedad horizontal"},
    # ── OSA — Órdenes de Servicio Administrativas ─────────────────────────────
    {"titulo": "OSA N° 1-23 – UIF – Modificación DTR 5",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/uif-os-A-1-23.pdf",
     "tema": "UIF"},
    {"titulo": "OSA N° 87-25 – Mandatos – Implementación Módulo Poderes – Sayges",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2025/08/O.S.A.87-25.pdf",
     "tema": "Mandatos"},
    {"titulo": "OSA N° 100-25 – Mandatos – Implementación Módulo Poderes",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2025/10/O.S.A.100-25.pdf",
     "tema": "Mandatos"},
    {"titulo": "OSA N° 117-26 – UIF Procedimiento",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2026/04/O.S.A.117-26-1.pdf",
     "tema": "UIF"},
    # ── Dictámenes ────────────────────────────────────────────────────────────
    {"titulo": "Dictamen N° 8 – Revocación acto administrativo – Reproducción de acto jurídico – Falta de firma",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-8-Revocacion-acto-administrativo-Reproduccion.pdf",
     "tema": "Revocación acto administrativo"},
    {"titulo": "Dictamen N° 11 – Acción de nulidad – Cosa juzgada irrita",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-11-accion-de-nulidad-cosa-juzgada.pdf",
     "tema": "Acción de nulidad"},
    {"titulo": "Dictamen N° 12 – Revocación acto administrativo – Reproducción – Ausencia causal de nulidad",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-12-revocacion-acto-administrativo-reproduccion.pdf",
     "tema": "Revocación acto administrativo"},
    {"titulo": "Dictamen N° 15 – Vivienda – Desafectación – Recurso rechaza",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-15-vivienda-desafectacion-rechazo.pdf",
     "tema": "Vivienda"},
    {"titulo": "Dictamen N° 16 – Donación a la municipalidad – Dominio público",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-16-donacion-a-la-municipalidad-dominio-publico.pdf",
     "tema": "Dominio público"},
    {"titulo": "Dictamen N° 17 – Revocación acto administrativo – Reproducción – Falta impresión dígito pulgar",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-17-revocacion-acto-administrativo-reproduccion.pdf",
     "tema": "Revocación acto administrativo"},
    {"titulo": "Dictamen N° 18 – Sucesiones – Adjudicación",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-18-sucesiones-adjudicacion.pdf",
     "tema": "Sucesiones"},
    {"titulo": "Dictamen N° 19 – Rectificación de asiento – Recurso rechaza",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-19-rectificacion-de-asiento-rechaza.pdf",
     "tema": "Rectificación"},
    {"titulo": "Dictamen N° 20 – Fideicomiso – Revocación acto administrativo – Recurso admite parcialmente",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-20-fideicomiso-revocacion-acto-adminsitrativo-admite-parcial.pdf",
     "tema": "Fideicomiso"},
    {"titulo": "Dictamen N° 21 – Régimen patrimonial del matrimonio – Asentimiento conyugal – Recurso rechaza",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-21-regimen-patrimonial-del-matrimonio-asentimiento-rechaza.pdf",
     "tema": "Régimen patrimonial"},
    {"titulo": "Dictamen N° 23 – Prescripción adquisitiva – Adquisición previa",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-23-prescripcion-adquisitiva-adquisicion-previa.pdf",
     "tema": "Prescripción adquisitiva"},
    {"titulo": "Dictamen N° 24 – Revocación acto administrativo – Sin reproducción",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-24-revocacion-acto-administrativo-sin-reproduccion.pdf",
     "tema": "Revocación acto administrativo"},
    {"titulo": "Dictamen N° 25 – Protocolización de expediente administrativo – Revocación de donación",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-25-protocolizacion-expte-administrativo-revocacion-de-donacion.pdf",
     "tema": "Donación"},
    {"titulo": "Dictamen N° 26 – Principio de especialidad – Superficie ad corpus",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-26-principio-de-especialidad-superficie-ad-corpus.pdf",
     "tema": "Principio de especialidad"},
    {"titulo": "Dictamen N° 27 – Acción de nulidad – Afectación a terceros",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-27-accion-de-nulidad-afectacion-a-terceros.pdf",
     "tema": "Acción de nulidad"},
    {"titulo": "Dictamen N° 28 – Rectificación de escritura pública – Error en la fecha",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-28-rectificacion-escritura-publica-error-en-la-fecha.pdf",
     "tema": "Rectificación"},
    {"titulo": "Dictamen N° 29 – Abandono de dominio – Dominio privado del Estado",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-29-abandono-de-dominio-dominio-privado-del-Estado-1.pdf",
     "tema": "Dominio público"},
    {"titulo": "Dictamen N° 31 – Fideicomiso – Título causa",
     "url_pdf": "https://jusmendoza.gob.ar/wp-content/uploads/2024/04/Dictamen-31-fideicomiso-titulo-causa.pdf",
     "tema": "Fideicomiso"},
]

# ── Pipeline principal ────────────────────────────────────────────────────────

def descargar_pdf(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, verify=False)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  [!] No se pudo descargar {url}: {e}")
        return False

NOMBRES_USADOS: dict[str, str] = {}  # nombre_norma -> url_pdf, para garantizar unicidad


def desambiguar_nombre(nombre: str, url_pdf: str) -> str:
    """Si el nombre ya fue asignado a un PDF distinto, le agrega un sufijo
    numérico para evitar que dos normas distintas se pisen entre sí."""
    if nombre not in NOMBRES_USADOS or NOMBRES_USADOS[nombre] == url_pdf:
        NOMBRES_USADOS.setdefault(nombre, url_pdf)
        return nombre
    contador = 2
    candidato = f"{nombre} ({contador})"
    while candidato in NOMBRES_USADOS and NOMBRES_USADOS[candidato] != url_pdf:
        contador += 1
        candidato = f"{nombre} ({contador})"
    NOMBRES_USADOS.setdefault(candidato, url_pdf)
    return candidato


def procesar_link(item: dict) -> dict | None:
    titulo    = item["titulo"]
    url_pdf   = item["url_pdf"]
    tipo      = clasificar_tipo(titulo)
    numero    = extraer_numero(titulo, tipo)
    anio      = extraer_anio(titulo, numero)
    nombre    = generar_nombre_norma(tipo, numero, anio)
    anexo     = extraer_anexo(titulo)
    if anexo and anexo.lower() not in nombre.lower():
        nombre = f"{nombre} {anexo}"
    nombre    = desambiguar_nombre(nombre, url_pdf)
    jur       = inferir_jurisdiccion(titulo, tipo)

    # Nombre seguro para el archivo
    safe_name = re.sub(r"[^\w\-]", "_", nombre)[:80] + ".pdf"
    pdf_path  = PDFS_DIR / safe_name

    descargado = descargar_pdf(url_pdf, pdf_path)
    texto_completo = ""
    texto_resumido = ""
    if descargado:
        texto_completo = extraer_texto_pdf(pdf_path)
        texto_resumido = resumir_texto(texto_completo)
        time.sleep(0.5)   # cortesía al servidor

    return {
        "id_norma":         None,
        "nombre_norma":     nombre,
        "tipo_norma":       tipo,
        "numero_norma":     numero,
        "anio":             anio,
        "jurisdiccion":     jur,
        "tema":             item.get("tema", ""),
        "titulo_resumido":  titulo,
        "texto_resumido":   texto_resumido,
        "texto_completo":   texto_completo,
        "url_pdf":          url_pdf,
        "archivo_local":    str(pdf_path) if descargado else "",
        "vigente":          True,
    }

def main():
    print("=" * 60)
    print("ALEPH — Scraper y extractor de texto")
    print("=" * 60)

    links = obtener_links_pdfs(NORMATIVA_URL)
    registros = []

    for item in tqdm(links, desc="Procesando PDFs"):
        r = procesar_link(item)
        if r:
            registros.append(r)

    df = pd.DataFrame(registros)
    df["id_norma"] = range(1, len(df) + 1)
    cols = ["id_norma"] + [c for c in df.columns if c != "id_norma"]
    df = df[cols]

    out = DATA_DIR / "normas_raw.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"\n[OK] {len(df)} normas procesadas.")
    print(f"[OK] CSV guardado en: {out}")
    print("\nResumen por tipo:")
    print(df["tipo_norma"].value_counts().to_string())
    print("\nResumen por jurisdicción:")
    print(df["jurisdiccion"].value_counts().to_string())
    print("\nPrimeras filas:")
    print(df[["nombre_norma", "tipo_norma", "jurisdiccion", "anio"]].to_string(index=False))

if __name__ == "__main__":
    main()

"""
Genera coordenadas 3D para la visualización galaxia usando UMAP.
Salida: data/galaxia.json
"""

import json
import numpy as np
import umap

with open("../data/normas_vectors.json", encoding="utf-8") as f:
    data = json.load(f)

# Usar embedding_combined; fallback a embedding_nombre
vectors = []
valid = []
for norma in data:
    emb = norma.get("embedding_combined") or norma.get("embedding_nombre")
    if emb and len(emb) > 0:
        vectors.append(emb)
        valid.append(norma)

print(f"Normas con vector: {len(valid)}")

X = np.array(vectors, dtype=np.float32)

reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=min(15, len(valid) - 1))
coords = reducer.fit_transform(X)

# Normalizar a rango [-100, 100] para Three.js
for i in range(3):
    mn, mx = coords[:, i].min(), coords[:, i].max()
    coords[:, i] = (coords[:, i] - mn) / (mx - mn) * 200 - 100

COLORES = {
    "OSC": "#4fc3f7",
    "OSA": "#81c784",
    "DTR": "#ffb74d",
    "Ley": "#ce93d8",
    "Decreto": "#f48fb1",
    "Dictamen": "#80cbc4",
    "Acordada": "#fff176",
    "Resolución": "#ff8a65",
    "Resolución Administrativa": "#a5d6a7",
    "Resolución Conjunta": "#90caf9",
    "Plano": "#bcaaa4",
}
COLOR_DEFAULT = "#e0e0e0"

puntos = []
for i, norma in enumerate(valid):
    tipo = norma.get("tipo_norma", "")
    puntos.append({
        "id": norma.get("id_norma"),
        "nombre": norma.get("nombre_norma", ""),
        "tipo": tipo,
        "anio": str(norma.get("anio", "")),
        "tema": norma.get("tema", ""),
        "titulo": norma.get("titulo_resumido", "") or norma.get("nombre_norma", ""),
        "resumen": norma.get("texto_resumido", ""),
        "url": norma.get("url_pdf", ""),
        "color": COLORES.get(tipo, COLOR_DEFAULT),
        "x": float(coords[i, 0]),
        "y": float(coords[i, 1]),
        "z": float(coords[i, 2]),
    })

out = {"puntos": puntos, "tipos": list(COLORES.keys()), "colores": COLORES}
with open("../data/galaxia.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"Guardado: data/galaxia.json ({len(puntos)} puntos)")

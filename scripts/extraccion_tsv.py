from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from pyproj import Transformer
from tqdm import tqdm


# =========================
# RUTAS
# =========================

BANDS_DIR = Path("../imagen")
POLYGONS_PATH = Path("../vectorial/poligonos_entrenamiento.geojson")
OUTPUT_PATH = Path("../dataset/dataset_armero.tsv")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================
# BANDAS A USAR
# =========================

BANDS = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]

SCALE_FACTOR = 10000.0


# =========================
# BUSCAR ARCHIVO DE CADA BANDA
# =========================

def find_band_file(bands_dir: Path, band: str) -> Path:
    """
    Busca archivos .jp2 que contengan el nombre de la banda.
    Ejemplos válidos:
    T18NWL_20240123T152649_B02_10m.jp2
    B02_10m.jp2
    """

    candidates = list(bands_dir.rglob(f"*_{band}_*.jp2"))

    if not candidates:
        candidates = list(bands_dir.rglob(f"*{band}*.jp2"))

    # Evitar confundir B08 con B8A
    if band == "B08":
        candidates = [c for c in candidates if "B8A" not in c.name]

    if not candidates:
        raise FileNotFoundError(f"No encontré archivo para la banda {band} en {bands_dir}")

    # Preferir 10m cuando exista
    candidates_10m = [c for c in candidates if "10m" in c.name]
    if candidates_10m:
        return candidates_10m[0]

    # Luego 20m
    candidates_20m = [c for c in candidates if "20m" in c.name]
    if candidates_20m:
        return candidates_20m[0]

    return candidates[0]


band_files = {band: find_band_file(BANDS_DIR, band) for band in BANDS}

print("Bandas encontradas:")
for band, path in band_files.items():
    print(f"{band}: {path}")


# =========================
# USAR B02 COMO REFERENCIA
# =========================

reference_band = band_files["B02"]

with rasterio.open(reference_band) as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_width = ref.width
    ref_height = ref.height

print("\nCRS de referencia:", ref_crs)
print("Tamaño de referencia:", ref_width, ref_height)


# =========================
# LEER POLÍGONOS
# =========================

polygons = gpd.read_file(POLYGONS_PATH)

if "clase" not in polygons.columns:
    raise ValueError("El archivo de polígonos debe tener una columna llamada 'clase'.")

if polygons.crs is None:
    raise ValueError("El GeoJSON no tiene CRS definido. Revisa la exportación desde QGIS.")

polygons = polygons.to_crs(ref_crs)

print("\nPolígonos cargados:")
print(polygons[["id", "clase"]])

print("\nConteo de polígonos por clase:")
print(polygons["clase"].value_counts())


# =========================
# TRANSFORMADOR A LAT/LON
# =========================

transformer_to_wgs84 = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)


# =========================
# ABRIR BANDAS COMO VRT
# Esto permite alinear bandas de 20m con las de 10m.
# =========================

datasets = {}
vrts = {}

for band, path in band_files.items():
    src = rasterio.open(path)

    vrt = WarpedVRT(
        src,
        crs=ref_crs,
        transform=ref_transform,
        width=ref_width,
        height=ref_height,
        resampling=Resampling.bilinear,
    )

    datasets[band] = src
    vrts[band] = vrt


# =========================
# EXTRAER PÍXELES
# =========================

rows = []

with rasterio.open(reference_band) as ref:

    for _, poly in tqdm(polygons.iterrows(), total=len(polygons), desc="Extrayendo píxeles"):

        geom = poly.geometry
        clase = poly["clase"]

        if geom is None or geom.is_empty:
            continue

        minx, miny, maxx, maxy = geom.bounds

        window = from_bounds(
            minx,
            miny,
            maxx,
            maxy,
            transform=ref_transform,
        )

        window = window.round_offsets().round_lengths()

        if window.width <= 0 or window.height <= 0:
            continue

        window_transform = ref.window_transform(window)

        mask = geometry_mask(
            [geom],
            out_shape=(int(window.height), int(window.width)),
            transform=window_transform,
            invert=True,
        )

        if not mask.any():
            continue

        band_arrays = {}

        for band in BANDS:
            arr = vrts[band].read(1, window=window).astype("float32")
            arr = arr / SCALE_FACTOR
            band_arrays[band] = arr

        pixel_rows, pixel_cols = np.where(mask)

        for r, c in zip(pixel_rows, pixel_cols):

            values = {}
            valid_pixel = True

            for band in BANDS:
                value = band_arrays[band][r, c]

                if np.isnan(value) or value <= 0:
                    valid_pixel = False
                    break

                values[band] = value

            if not valid_pixel:
                continue

            x, y = rasterio.transform.xy(
                window_transform,
                r,
                c,
                offset="center",
            )

            lon, lat = transformer_to_wgs84.transform(x, y)

            rows.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    **values,
                    "clase": clase,
                }
            )


# =========================
# EXPORTAR TSV
# =========================

df = pd.DataFrame(rows)

if df.empty:
    raise ValueError(
        "El dataset quedó vacío. Revisa que los polígonos estén encima de las bandas correctas."
    )

columns = ["latitude", "longitude"] + BANDS + ["clase"]
df = df[columns]

df.to_csv(OUTPUT_PATH, sep="\t", index=False)

print("\nDataset creado correctamente:")
print(OUTPUT_PATH)

print("\nPrimeras filas:")
print(df.head())

print("\nConteo de píxeles por clase:")
print(df["clase"].value_counts())


# =========================
# CERRAR ARCHIVOS
# =========================

for vrt in vrts.values():
    vrt.close()

for src in datasets.values():
    src.close()
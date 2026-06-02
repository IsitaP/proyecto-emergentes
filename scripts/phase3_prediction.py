from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "dataset_armero.tsv"
IMAGE_DIR = BASE_DIR / "imagen"
OUTPUT_DIR = BASE_DIR / "phase3_outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_RASTER = OUTPUT_DIR / "classified_armero_ann.tif"
OUTPUT_AREAS = OUTPUT_DIR / "area_by_class.csv"


# ============================================================
# BANDAS USADAS EN EL DATASET
# ============================================================

FEATURES = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]


# ============================================================
# RUTAS DE LAS BANDAS
# Ajustadas a tus nombres de archivo
# ============================================================

BAND_PATHS = {
    "B02": IMAGE_DIR / "T18NWL_20240123T152649_B02_10m.jp2",
    "B03": IMAGE_DIR / "T18NWL_20240123T152649_B03_10m.jp2",
    "B04": IMAGE_DIR / "T18NWL_20240123T152649_B04_10m.jp2",
    "B05": IMAGE_DIR / "T18NWL_20240123T152649_B05_20m.jp2",
    "B06": IMAGE_DIR / "T18NWL_20240123T152649_B06_20m.jp2",
    "B07": IMAGE_DIR / "T18NWL_20240123T152649_B07_20m.jp2",
    "B08": IMAGE_DIR / "T18NWL_20240123T152649_B08_10m.jp2",
    "B8A": IMAGE_DIR / "T18NWL_20240123T152649_B8A_20m.jp2",
    "B11": IMAGE_DIR / "T18NWL_20240123T152649_B11_20m.jp2",
    "B12": IMAGE_DIR / "T18NWL_20240123T152649_B12_20m.jp2",
}


# ============================================================
# MAPEO DE CLASES
# Estos códigos serán los valores del GeoTIFF final
# ============================================================

CLASS_MAPPING = {
    "Vegetacion": 1,
    "Pastos": 2,
    "Lahar": 3,
    "Urbano": 4,
}

INVERSE_MAPPING = {
    1: "Vegetacion",
    2: "Pastos",
    3: "Lahar",
    4: "Urbano",
}


# ============================================================
# FUNCIÓN: VERIFICAR ARCHIVOS
# ============================================================

def verify_files():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {DATASET_PATH}")

    for band_name, band_path in BAND_PATHS.items():
        if not band_path.exists():
            raise FileNotFoundError(f"No se encontró la banda {band_name}: {band_path}")


# ============================================================
# FUNCIÓN: ENTRENAR MODELO ANN FINAL
# ============================================================

def train_final_model():
    print("Cargando dataset...")
    df = pd.read_csv(DATASET_PATH, sep="\t")

    print("Columnas encontradas:")
    print(df.columns.tolist())

    missing_columns = [col for col in FEATURES + ["clase"] if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Faltan columnas en el dataset: {missing_columns}")

    df["clase"] = df["clase"].astype(str).str.strip()

    print("\nClases encontradas:")
    print(df["clase"].value_counts())

    X = df[FEATURES]
    y = df["clase"].map(CLASS_MAPPING)

    if y.isnull().any():
        print("Clases no reconocidas:")
        print(df.loc[y.isnull(), "clase"].unique())
        raise ValueError("Hay clases en el dataset que no están en CLASS_MAPPING.")

    print("\nEntrenando modelo ANN final con todo el dataset...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ann", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        ))
    ])

    model.fit(X, y)

    print("Modelo ANN entrenado correctamente.")
    return model


# ============================================================
# FUNCIÓN: GENERAR RASTER CLASIFICADO
# ============================================================

def generate_classified_raster(model):
    print("\nAbriendo banda de referencia B02...")

    reference_path = BAND_PATHS["B02"]

    with rasterio.open(reference_path) as ref_src:
        ref_profile = ref_src.profile.copy()
        ref_crs = ref_src.crs
        ref_transform = ref_src.transform
        ref_width = ref_src.width
        ref_height = ref_src.height

        print("CRS:", ref_crs)
        print("Tamaño referencia:", ref_width, "x", ref_height)
        print("Transform:", ref_transform)

        output_profile = ref_profile.copy()
        output_profile.update({
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
            "compress": "lzw"
        })

        # Abrimos todas las bandas y las alineamos a la grilla de B02
        sources = {}
        vrts = {}

        try:
            for band_name in FEATURES:
                src = rasterio.open(BAND_PATHS[band_name])
                sources[band_name] = src

                vrt = WarpedVRT(
                    src,
                    crs=ref_crs,
                    transform=ref_transform,
                    width=ref_width,
                    height=ref_height,
                    resampling=Resampling.bilinear
                )

                vrts[band_name] = vrt

            print("\nGenerando raster clasificado...")
            print("Archivo de salida:", OUTPUT_RASTER)

            with rasterio.open(OUTPUT_RASTER, "w", **output_profile) as dst:

                total_windows = sum(1 for _ in ref_src.block_windows(1))
                current_window = 0

                for _, window in ref_src.block_windows(1):
                    current_window += 1

                    if current_window % 50 == 0 or current_window == 1:
                        print(f"Procesando ventana {current_window}/{total_windows}")

                    band_arrays = []

                    for band_name in FEATURES:
                        data = vrts[band_name].read(1, window=window).astype("float32")

                        # Sentinel-2 L2A suele venir escalado por 10000.
                        # Tu TSV tiene valores entre 0 y 1, por eso dividimos.
                        data = data / 10000.0

                        band_arrays.append(data)

                    stack = np.stack(band_arrays, axis=-1)

                    rows, cols, n_bands = stack.shape
                    pixels = stack.reshape(-1, n_bands)

                    valid_mask = np.all(np.isfinite(pixels), axis=1)
                    valid_mask = valid_mask & np.all(pixels > 0, axis=1)

                    predicted = np.zeros((pixels.shape[0],), dtype=np.uint8)

                    if np.any(valid_mask):
                        valid_pixels = pd.DataFrame(
                            pixels[valid_mask],
                            columns=FEATURES
                        )

                        predicted[valid_mask] = model.predict(valid_pixels).astype(np.uint8)

                    classified_window = predicted.reshape(rows, cols)

                    dst.write(classified_window, 1, window=window)

            print("\nGeoTIFF clasificado generado correctamente:")
            print(OUTPUT_RASTER)

        finally:
            for vrt in vrts.values():
                vrt.close()

            for src in sources.values():
                src.close()


# ============================================================
# FUNCIÓN: CALCULAR ÁREAS POR CLASE
# ============================================================

def calculate_area_by_class():
    print("\nCalculando áreas por clase...")

    with rasterio.open(OUTPUT_RASTER) as src:
        raster = src.read(1)
        transform = src.transform

        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)
        pixel_area_m2 = pixel_width * pixel_height

        print("Área por píxel en m²:", pixel_area_m2)

    values, counts = np.unique(raster, return_counts=True)

    rows = []

    for value, count in zip(values, counts):
        value = int(value)

        if value == 0:
            continue

        area_m2 = count * pixel_area_m2
        area_ha = area_m2 / 10000
        area_km2 = area_m2 / 1_000_000

        rows.append({
            "Class_Code": value,
            "Class_Name": INVERSE_MAPPING.get(value, "Unknown"),
            "Pixel_Count": int(count),
            "Area_m2": round(area_m2, 2),
            "Area_ha": round(area_ha, 4),
            "Area_km2": round(area_km2, 6)
        })

    area_df = pd.DataFrame(rows)
    area_df.to_csv(OUTPUT_AREAS, index=False)

    print("\nTabla de áreas:")
    print(area_df)

    print("\nArchivo de áreas generado:")
    print(OUTPUT_AREAS)


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def main():
    print("==========================================")
    print("PHASE 3 - FULL SCENE PREDICTION")
    print("Modelo final: ANN")
    print("Normalización: StandardScaler")
    print("==========================================")

    verify_files()

    model = train_final_model()

    generate_classified_raster(model)

    calculate_area_by_class()

    print("\n==========================================")
    print("PHASE 3 COMPLETADA")
    print("Archivos generados:")
    print(f"- {OUTPUT_RASTER}")
    print(f"- {OUTPUT_AREAS}")
    print("==========================================")


if __name__ == "__main__":
    main()
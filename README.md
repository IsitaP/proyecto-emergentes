# GeoAI Armero - Clasificación de Coberturas con Sentinel-2

## Descripción del Proyecto

Este proyecto corresponde a la primera fase del trabajo final de **GeoAI**, cuyo objetivo es construir una base de datos geoespacial para el análisis y clasificación supervisada de coberturas del suelo en la zona de **Armero Viejo, Tolima, Colombia**.

La región de estudio fue seleccionada debido a su importancia histórica, ambiental y geoespacial, ya que corresponde al área afectada por los lahares generados durante la erupción del volcán Nevado del Ruiz en 1985. Desde el enfoque de GeoAI, esta zona permite analizar diferentes tipos de cobertura terrestre usando imágenes satelitales Sentinel-2.

El proyecto utiliza datos de teledetección, polígonos de entrenamiento y procesamiento en Python para construir un dataset tabular que posteriormente podrá ser usado para entrenar y comparar modelos de aprendizaje automático.

---

## Objetivo General

Construir un dataset de entrenamiento a partir de imágenes satelitales Sentinel-2 y polígonos digitalizados manualmente, con el fin de clasificar coberturas del suelo en la zona de Armero Viejo mediante técnicas de aprendizaje automático.

---

## Región de Estudio

La región de interés corresponde a **Armero Viejo**, ubicado en el municipio de Armero-Guayabal, departamento del Tolima, Colombia.

El área incluye:

- Depósito de lahar consolidado.
- Vegetación secundaria.
- Bosque ripario.
- Pastos y cultivos.
- Zona urbana cercana.
- Cauce del río Lagunilla.

---

## Datos Utilizados

La imagen satelital utilizada corresponde a una escena **Sentinel-2B L2A**, obtenida desde Copernicus Data Space Ecosystem.

### Información principal de la imagen

| Campo | Valor |
|---|---|
| Misión | Sentinel-2B |
| Instrumento | MSI - Multispectral Instrument |
| Nivel de procesamiento | L2A |
| Fecha de adquisición | 23 de enero de 2024 |
| Tile | T18NWL |
| Sistema de referencia | EPSG:32618 |
| Resolución principal | 10 m y 20 m |

---

## Bandas Espectrales Utilizadas

Para la construcción del dataset se utilizaron diez bandas espectrales de Sentinel-2:

| Banda | Nombre | Uso principal |
|---|---|---|
| B02 | Azul | Color real, agua |
| B03 | Verde | Color real, vegetación |
| B04 | Rojo | Color real, suelo |
| B05 | Red Edge 1 | Estado vegetal |
| B06 | Red Edge 2 | Estado vegetal |
| B07 | Red Edge 3 | Estado vegetal |
| B08 | NIR | Vegetación, NDVI |
| B8A | NIR estrecho | Vegetación |
| B11 | SWIR 1 | Humedad, geología |
| B12 | SWIR 2 | Minerales, lahar |

---

## Clases de Cobertura

Se definieron cuatro clases principales para el entrenamiento del modelo:

| Código | Clase | Descripción |
|---|---|---|
| 1 | Vegetacion | Bosque y vegetación densa |
| 2 | Pastos | Pastizales y cultivos |
| 3 | Lahar | Depósito volcánico consolidado |
| 4 | Urbano | Zonas construidas |

---

## Dataset Generado

El dataset final fue exportado en formato `.tsv` con el nombre:

```text
dataset_armero.tsv

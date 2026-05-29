<div align="center">

# 🛰️ Geospatial Deep Learning Segmentation

### Semantic Segmentation of Buildings & Roads from Satellite Imagery

[![Python](https://img.shields.io/badge/Python-3.6%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.14-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![GDAL](https://img.shields.io/badge/GDAL-Geospatial-5CAD4A?style=flat-square)](https://gdal.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Geospatial AI](https://img.shields.io/badge/Domain-Geospatial%20AI-6A4C93?style=flat-square)]()

<br/>

*An end-to-end deep learning pipeline for extracting building footprints and road networks from high-resolution satellite imagery — from raw annotation data to production API deployment.*

</div>

---

## Overview

This project implements a complete semantic segmentation workflow for satellite imagery analysis. Using an extended U-Net architecture with a VGG-16 encoder, the system identifies and delineates **building footprints** and **road networks** from raw satellite images, producing pixel-level segmentation masks along with geospatially referenced GeoJSON and GeoTIFF outputs.

**Key capabilities:**

- Pixel-level semantic segmentation of buildings and roads
- End-to-end pipeline: raw satellite image → GeoJSON polygon output
- REST API for single-image inference
- Coordinate-accurate outputs via pixel-to-latitude/longitude conversion
- GeoTIFF export for integration with GIS tooling

**Real-world applications:** urban planning, infrastructure mapping, disaster response, smart city analytics, autonomous systems, and geospatial intelligence.

---

## Features

| Category | Details |
|---|---|
| **Segmentation** | U-Net with modified VGG-16 encoder for buildings; variant U-Net for roads |
| **Output Formats** | Binary masks, overlay PNGs, GeoJSON contours, GeoTIFF rasters |
| **Coordinate Conversion** | Pixel → WGS84 lat/lon with geotag support |
| **API Deployment** | Flask REST API for single-image inference |
| **Preprocessing** | CSV annotation ingestion, mask generation, HDF5 caching |
| **Geospatial Tooling** | GDAL integration, shapefile export via Java converter |
| **Batch Inference** | Script-based batch prediction for large image sets |

---

## Architecture

### Model Design

The segmentation network is based on an **extended U-Net** with an encoder-decoder structure:

```
Input Satellite Image (512×512 or 256×256)
        │
        ▼
┌─────────────────────────────┐
│     Encoder (VGG-16 base)   │  ← Modified with additional conv layers
│  Conv → Pool → Conv → Pool  │
│  (extracts spatial features)│
└─────────────┬───────────────┘
              │  Skip Connections
              ▼
┌─────────────────────────────┐
│     Bottleneck Layer        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Decoder (Custom upsampler)│  ← Tuned for semantic segmentation
│  UpConv → Concat → Conv     │
└─────────────┬───────────────┘
              │
              ▼
     Segmentation Mask Output
```

- **Buildings:** 512×512 input, standard U-Net decoder, 50 epochs
- **Roads:** 256×256 input, modified U-Net with extra convolution layers, 100 epochs

### Full Inference Pipeline

```
Satellite Image
      │
      ▼
  Image Crop / Tiling (split_image.py)
      │
      ▼
  Segmentation Inference (predict.py)
      │
      ├──→ Binary Mask (.png)
      │
      ├──→ Overlay Image (.png)
      │
      ├──→ GeoTIFF (.tif via geotiff.py)
      │
      └──→ GeoJSON Contours (pixel_to_lat_lon.py)
```

---

## Project Structure

```
geospatial-deep-learning-segmentation/
│
├── Training/                          # Training scripts and configs
│   ├── train_building.py              # U-Net training for building segmentation
│   └── train_roads.py                 # U-Net training for road segmentation
│
├── Pre-Processing/                    # Data preparation utilities
│   ├── data_to_xy.py                  # Convert lat/lon annotations → pixel coords
│   ├── gen_fp_masks.py                # Generate building footprint masks
│   └── gen_rd_masks.py                # Generate road segmentation masks
│
├── Clustering/                        # Spatial clustering utilities
│
├── Results/                           # Output masks, overlays, predictions
│
├── Api_Wrapper/                       # API helper modules
│
├── models.py                          # U-Net and VGG16 encoder definitions
├── data_loader.py                     # Dataset loading and augmentation
├── cache_train.py                     # Convert dataset → HDF5 cache
├── extra_functions.py                 # Helper: read_image, read_mask, read_image_test
├── params.py                          # Global hyperparameters and config
│
├── make_predictions_building.py       # Batch inference for buildings
├── predictions_soil.py                # Soil segmentation prediction
├── predictions_vegetation.py          # Vegetation segmentation prediction
├── vegetation_batchprocessing*.py     # Batch vegetation prediction
│
├── api_deploy.py                      # Flask REST API (single image)
├── API_Deploy_Pixel.py                # Pixel-level API deployment
│
├── geotiff.py                         # PNG → GeoTIFF converter
├── pixel_to_lat_lon.py                # Pixel coords → WGS84 + GeoJSON export
├── pixel_to_lat_long.py               # Reference implementation
├── geotagged.py                       # Geotagging utilities
│
├── split_image.py                     # Image tiling for large inputs
├── split.py                           # General split utilities
├── blend_it.py                        # Mask blending and overlay
├── ploting.py                         # Loss/accuracy visualization
├── count.py                           # Pixel count utilities
│
├── Ocean.py                           # Ocean segmentation module
├── Soil.py                            # Soil segmentation module
├── vegetation.py                      # Vegetation segmentation module
├── vegetation_crop.py                 # Vegetation crop analysis
│
├── requirements.txt                   # Python dependencies
├── sat_seg_env.yml                    # Conda environment definition
└── workspace.code-workspace          # VS Code workspace config
```

---

## Installation

### Option 1 — Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/jyothir-369/geospatial-deep-learning-segmentation.git
cd geospatial-deep-learning-segmentation

# Create and activate environment from YAML
conda env create --name sat_seg_env --file sat_seg_env.yml
conda activate sat_seg_env
```

### Option 2 — Manual Setup

```bash
# Create and activate a clean conda environment
conda create -n sat_seg_env python=3.6
conda activate sat_seg_env

# Install dependencies
pip install tensorflow==1.14.0 \
            flask \
            gdal \
            pandas \
            scikit-learn \
            opencv-python \
            matplotlib \
            numpy \
            keras
```

### Option 3 — Export Existing Environment

```bash
# Export your current environment
conda env export > sat_seg_env.yml

# Recreate it elsewhere
conda env create --name sat_seg_env --file sat_seg_env.yml
```

> **GPU Recommendation:** Training is significantly faster on CUDA-capable GPUs. See the [GPU section](#gpu-acceleration) for details. Ensure your CUDA and cuDNN versions are compatible with TensorFlow 1.14.

---

## Dataset

### Annotation Format

Annotation files are CSVs stored at `/mnt/vol1/Datasets/Satellite_Images/`.

| File | Purpose | Key Fields |
|---|---|---|
| `FP_Buildings.csv` | Building footprint annotations | `EdgeID`, `Grid No`, `Sqn_no`, `centroid_X`, `centroid_Y` |
| `RD_LATLON_WITH_GRID.csv` | Road network annotations | `EdgeID`, `Grid No`, `Sqn_no`, `centroid_X`, `centroid_Y` |

### Mask Generation Steps

```bash
# Step 1: Convert lat/lon coordinates to pixel coordinates
python data_to_xy.py

# Step 2a: Generate building footprint masks
python gen_fp_masks.py

# Step 2b: Generate road segmentation masks
python gen_rd_masks.py
```

### Preprocessing & HDF5 Caching

```bash
# Convert image/mask pairs to HDF5 format for efficient training
python cache_train.py
```

This outputs three objects into the `cache/` directory: `train`, `train_masks`, and `train_ids`.

**Helper functions in `extra_functions.py`:**

| Function | Description |
|---|---|
| `read_image()` | Loads input satellite images |
| `read_mask()` | Loads corresponding binary masks |
| `read_image_test()` | Loads images for inference (no mask required) |

---

## Training

### Configuration

| Target | Image Size | Epochs | Output Weights |
|---|---|---|---|
| Buildings | 512 × 512 | 50 | `45.h5` (city-level) · `49.h5` (state-level) |
| Roads | 256 × 256 | 100 | `83.h5` · `100.h5` |

> **Note:** `45.h5` is the primary building model used in production. State-level weights (`49.h5`) are available but not used in the default API.

### Run Training

```bash
# Train building segmentation model
python Training/train_building.py

# Train road segmentation model
python Training/train_roads.py
```

**Training artifacts produced:**

- Model weight snapshots saved as `.h5` files (per epoch)
- Loss and accuracy plots exported as PDFs

---

## Inference & Prediction

### Building Segmentation

```bash
python make_predictions_building.py \
    --images /path/to/test/images \
    --weights /path/to/weights/45.h5 \
    --output /path/to/output/folder
```

> Ensure test image dimensions match training dimensions: **512×512 for buildings**, **256×256 for roads**.

### Additional Segmentation Targets

```bash
# Vegetation prediction
python predictions_vegetation.py

# Soil prediction
python predictions_soil.py
```

### Outputs

Each prediction run produces:

- Binary segmentation mask (`.png`)
- Overlay image showing mask on original satellite image (`.png`)
- (Optional) GeoJSON contours via `pixel_to_lat_lon.py`

---

## API Deployment

### Start the Flask API

```bash
python api_deploy.py
```

The server exposes a REST endpoint accepting satellite image inputs and returning segmentation results.

### Supporting Modules

| Module | Role |
|---|---|
| `geotiff.py` | Converts output PNG masks to GeoTIFF format |
| `predict.py` | Handles image cropping, model inference, and mask stitching |
| `pixel_to_lat_lon.py` | Converts mask pixels to WGS84 coordinates; exports GeoJSON |

### Local Testing

```bash
# Use the test client scripts for local API validation
python git_api.py
python git_predict.py
```

> **Note:** Batch processing is not implemented in the API. For batch inference, use `make_predictions_building.py` directly.

---

## Geospatial Processing

### Pixel → Coordinate Conversion

The `pixel_to_lat_lon.py` module converts segmentation mask pixels to real-world WGS84 coordinates using the image's geotag metadata. It then traces contours and writes them to a GeoJSON file.

**Inputs:** Image folder with `.TAB` georeference files + corresponding mask folder  
**Output:** GeoJSON file containing polygon contours of detected features

```bash
python pixel_to_lat_lon.py \
    --images /path/to/geotagged/images \
    --masks /path/to/masks \
    --output /path/to/output/geojson
```

### Shapefile Export

GeoJSON outputs can be converted to ESRI Shapefiles using the included Java converter:

```bash
java -jar yourconverter.jar input.geojson output.shp
```

### Verifying Geotag Metadata

```python
import geoio
from geoio import GeoImage

geoimg = GeoImage("path/to/image.tif")
print(geoimg)
```

---

## Execution Pipeline

A complete end-to-end run follows this sequence:

```
┌─────────────────────────────────────────────────────────┐
│                 FULL EXECUTION PIPELINE                 │
└─────────────────────────────────────────────────────────┘

  [1] data_to_xy.py
      └── Convert annotation CSV lat/lon → pixel coordinates

  [2] gen_fp_masks.py  +  gen_rd_masks.py
      └── Generate binary masks for buildings and roads

  [3] cache_train.py
      └── Package image/mask pairs into HDF5 training cache

  [4] train_building.py  +  train_roads.py
      └── Train U-Net models; save weight checkpoints

  [5] make_predictions_building.py
      └── Run inference; produce masks and overlays

  [6] pixel_to_lat_lon.py
      └── Convert mask pixels to WGS84; export GeoJSON

  [7] api_deploy.py
      └── Serve the model via Flask REST API
```

---

## GPU Acceleration

Deep learning models in this project benefit substantially from GPU acceleration.

**Why GPUs outperform CPUs for this workload:**

- **Massive parallelism:** GPUs expose thousands of SIMD compute threads simultaneously, ideal for the matrix operations in convolutional neural networks.
- **High memory bandwidth:** Enables rapid loading of large satellite image tensors during training.
- **Reduced epoch time:** A training run that takes hours on CPU completes in a fraction of the time on GPU.
- **Inference speed:** Real-time or near-real-time prediction is feasible only with GPU acceleration.

**Setup checklist:**

1. Install CUDA 10.0 and cuDNN 7.x (compatible with TensorFlow 1.14)
2. Install the GPU build: `pip install tensorflow-gpu==1.14.0`
3. Verify GPU detection:

```python
import tensorflow as tf
print(tf.test.is_gpu_available())
```

---

## Applications

This system is applicable across a wide range of geospatial intelligence use cases:

- **Urban Planning** — Automated building footprint extraction for zoning and land-use mapping
- **Infrastructure Mapping** — Road network detection for logistics and connectivity analysis
- **Disaster Response** — Rapid damage assessment by comparing pre/post-event imagery
- **Smart Cities** — Input layer for digital twin platforms and urban simulation models
- **GIS Analytics** — Enriching spatial databases with AI-derived feature layers
- **Autonomous Systems** — Providing map priors for ground and aerial navigation systems

---

## Future Improvements

- [ ] Migrate to transformer-based architectures (SegFormer, Mask2Former) for improved accuracy
- [ ] Add cloud deployment support (AWS SageMaker, GCP Vertex AI)
- [ ] Implement real-time streaming inference for live satellite feeds
- [ ] Distributed training support across multi-GPU nodes
- [ ] Model quantization and ONNX export for edge deployment
- [ ] Batch REST API endpoint
- [ ] Automated dataset versioning and experiment tracking (MLflow / W&B)

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to your branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please ensure your code follows existing style conventions and include relevant documentation for new modules.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Jyothir / Dhruv Raghav**  
Geospatial AI · Deep Learning · Computer Vision

[![GitHub](https://img.shields.io/badge/GitHub-jyothir--369-181717?style=flat-square&logo=github)](https://github.com/jyothir-369/geospatial-deep-learning-segmentation)

---

<div align="center">

*Built for satellite intelligence — from raw imagery to geospatial insight.*

</div>

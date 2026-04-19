🌍 Satellite Image Segmentation (Buildings & Roads)

This project implements a deep learning-based semantic segmentation system for extracting buildings and road networks from high-resolution satellite imagery using an extended U-Net architecture.

It covers the complete ML pipeline:
data preprocessing → mask generation → training → prediction → geospatial API deployment

🚀 Key Features
🧠 Semantic segmentation using U-Net (VGG-16 encoder variant)
🏙️ Dual-class segmentation: Buildings + Roads
🗺️ Geospatial conversion (Pixel → Latitude/Longitude)
📦 H5 dataset caching for optimized training
⚡ GPU-accelerated training pipeline
🌐 Flask-based API deployment for inference
📍 GeoJSON output for mapping applications
🛰️ Satellite imagery preprocessing & mask generation
🏗️ Model Architecture
Encoder: Modified VGG-16 backbone with additional convolution layers
Decoder: Custom U-Net decoder for segmentation reconstruction
Task Type: Semantic Segmentation
Outputs: Pixel-wise classification masks (buildings & roads)
📊 Dataset & Annotation Format
Input Annotations:
FP_Buildings.csv → Building footprints
RD_LATLON_WITH_GRID.csv → Road grid annotations
Key Fields:
EdgeID
Grid No
Sqn_no
centroid_X
centroid_Y
⚙️ Data Pipeline
1. Coordinate Conversion

Convert geospatial coordinates → pixel space:

python data_to_xy.py
2. Mask Generation
python gen_fp_masks.py   # Building masks
python gen_rd_masks.py   # Road masks
3. Dataset Caching

Convert dataset into optimized .h5 format:

python cache_train.py

Outputs:

train images
train masks
train IDs
🧠 Training Pipeline
Train Models
python train_building.py
python train_roads.py
Training Configuration
🏙️ Buildings: 512 × 512 images → 50 epochs
🛣️ Roads: 256 × 256 images → 100 epochs
Outputs
.h5 trained weights
Loss & accuracy plots (PDF)
Best models:
Buildings: 45.h5, 49.h5
Roads: 83.h5, 100.h5
🔍 Inference / Prediction
python make_predictions_building.py
Features:
Batch prediction support
Mask overlay visualization
Output segmentation maps
🌐 API Deployment

Run segmentation as a REST API:

python api_deploy.py
Supporting Modules:
predict.py → segmentation inference pipeline
geotiff.py → PNG → GeoTIFF conversion
pixel_to_lat_lon.py → Pixel → GeoJSON conversion
git_predict.py → local testing utilities
📍 Geospatial Outputs
🗺️ GeoJSON generation from segmentation masks
📌 Latitude/Longitude mapping from pixel coordinates
🧭 Useful for GIS applications and mapping systems
🧩 Execution Flow
Convert annotations → pixel coordinates
Generate segmentation masks
Cache dataset into .h5 format
Train U-Net models
Run inference on satellite images
Export GeoJSON / overlays
Deploy via Flask API
⚡ Why GPU is Important
Parallel processing for CNN training
High throughput for image tensors
Faster convergence in deep learning models
Efficient handling of large satellite datasets
Supports large-scale geospatial computation
🧪 Tools & Technologies
Python
TensorFlow / Keras
OpenCV
NumPy, Pandas
GDAL (Geospatial processing)
Flask (API deployment)
Matplotlib (visualization)
🌍 Applications
Smart city mapping
Urban planning
Disaster monitoring
Infrastructure detection
GIS-based analytics
📌 Notes
Batch API is not included (can be extended via make_prediction.py)
All preprocessing scripts are modular and reusable
Designed for research + production-style workflows
# Lúa Khỏe AI - Rice Plant Disease Detection

A production-ready rice plant disease detection API built with **Python 3.12**, **FastAPI**, and **ONNX Runtime**.

## ✨ Key Features
- **Strategy Design Pattern**: Decoupled Inference and Storage architectures.
- **High Performance**: Optimized for fast CPU inference using ONNX.
- **Edge-Ready**: Uses MobileNetV3-Small (~3.4M parameters) for a small footprint.
- **Data Guard**: Automatically validates uploads to filter out junk/low-quality images.
- **Observability**: Rich logging with `loguru` including latency and confidence tracking.
- **Cloud-Ready**: Storage layer supports both Local and Cloud (S3) via configuration.
- **Dockerized**: Easy deployment with Docker and Docker Compose.

## 📁 Project Structure
- `api/`: FastAPI application, endpoints, and Pydantic schemas.
- `src/`: Core logic including AI strategies, storage drivers, and preprocessors.
- `models/`: Storage for exported AI models (.onnx).
- `data/`: Local storage for uploaded/log images.
- `tests/`: Integration tests and sample image downloaders.

## 🛠️ Getting Started

### 1. Requirements
Ensure you have Python 3.12+ installed.

### 2. Setup Environment
```powershell
.\setup_env.ps1
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Initialize Model & Data
```powershell
python src/export_onnx.py        # Generates initial ONNX model
python tests/download_test_images.py # Downloads sample images
```

### 4. Running the API
```powershell
# Using Native FastAPI
uvicorn api.main:app --reload

# Using Docker
docker-compose up --build
```

Access the API documentation at `http://localhost:8000/docs`.

## 📈 Disease Coverage
The current version targets:
- Healthy
- Bacterial Leaf Blight
- Brown Spot
- Leaf Blast
- Leaf Smut

## 💡 Training on Colab
The project includes `train_rice_diseases.ipynb` for easy fine-tuning on Google Colab when your custom dataset is ready.

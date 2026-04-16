# 🌾 Lúa Khỏe AI - Rice Plant Disease Detection

A production-ready rice plant disease detection API built with **Python 3.13**, **FastAPI**, and **ONNX Runtime**. Designed for high performance and modularity.

---

## ✨ Key Features
- **🎯 Strategy Design Pattern**: Decoupled Inference and Storage architectures for maximum flexibility.
- **🚀 High Performance**: Optimized for fast CPU inference using ONNX Runtime.
- **📱 Edge-Ready**: Uses MobileNetV3-Small (~3.4M parameters) for a small memory footprint.
- **🛡️ Data Guard**: Automatically validates uploads to filter out junk or low-quality images.
- **📊 Observability**: Rich logging with `loguru` including latency and confidence tracking.
- **☁️ Cloud-Ready**: Storage layer supports both Local and S3-compatible storage via configuration.
- **🐳 Dockerized**: Fully containerized for easy deployment with Docker and Docker Compose.

---

## 📁 Project Structure
- `api/` — FastAPI application, endpoints, and Pydantic schemas.
- `src/` — Core logic including AI strategies, storage drivers, and preprocessors.
- `models/` — Storage for exported AI models (`.onnx`).
- `data/` — Local storage for uploaded/log images and persistent data.
- `tests/` — Integration tests and sample image downloaders.

---

## 🛠️ Getting Started

### 1. Prerequisites
- **Python 3.12 or 3.13** (Tested on 3.13)
- **Make** (Optional but highly recommended for Windows/Linux)

### 2. Quick Start (Makefile - Recommended)
The fastest way to get the environment ready:

```powershell
make setup      # 1. Creates venv, installs dependencies & .env
make init       # 2. Exports ONNX model & downloads sample images
make dev        # 3. Starts the development server with --reload
```

> [!TIP]
> Use `make help` to see all available commands, including Docker and testing utilities.

### 3. Manual Setup
If you don't have `make` installed:

```powershell
# Create and activate environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies robustly
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run development server
python -m uvicorn api.main:app --reload
```

---

## 🚀 Deployment

### Using Docker
```powershell
make docker-up    # Builds and starts containers in detached mode
make logs         # Follow logs to monitor performance
```

Access the API documentation at: `http://localhost:8000/docs`

---

## 📈 Disease Coverage
The current model is trained to detect:
- ✅ **Healthy**
- ✅ **Bacterial Leaf Blight**
- ✅ **Brown Spot**
- ✅ **Leaf Blast**
- ✅ **Leaf Smut**

---

## 💡 Custom Training
The project includes `train_rice_diseases.ipynb` for easy fine-tuning. You can run this on Google Colab to retrain the model with your own dataset before exporting it back to this repository.

---
*Developed for the Lúa Khỏe Ecosystem.*

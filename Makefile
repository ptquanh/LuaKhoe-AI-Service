# Makefile for Lua Khoe AI (Windows Optimized)

# Variables
VENV = venv
VENV_PYTHON = $(VENV)\Scripts\python.exe
VENV_PIP = $(VENV_PYTHON) -m pip
# VENV_UVICORN no longer needed as we use -m uvicorn

.PHONY: all help setup install init run dev test clean docker-up docker-down freeze logs

all: help

help:
	@echo "======================================================================"
	@echo "Lua Khoe AI - Project Commands"
	@echo "======================================================================"
	@echo "  make setup        - Create venv, install deps, and create .env"
	@echo "  make install      - Alias for setup"
	@echo "  make init         - Export ONNX model and download test images"
	@echo "  make run          - Run the FastAPI application (respects config.py)"
	@echo "  make dev          - Run development server with --reload"
	@echo "  make test         - Run internal tests (Note: API must be running)"
	@echo "  make docker-up    - Build and start Docker containers"
	@echo "  make docker-down  - Stop Docker containers"
	@echo "  make logs         - View Docker logs"
	@echo "  make freeze       - Update requirements.txt from current venv"
	@echo "  make clean        - Remove virtual environment and cache files"
	@echo "======================================================================"

# Setup Environment
setup:
	@echo ">>> Creating virtual environment..."
	@if not exist "$(VENV)" python -m venv $(VENV)
	@echo ">>> Upgrading pip and installing dependencies..."
	@$(VENV_PYTHON) -m pip install --upgrade pip
	@$(VENV_PIP) install -r requirements.txt
	@echo ">>> Creating .env from template if not exists..."
	@if not exist ".env" copy .env.example .env
	@echo ">>> Ensuring data directories exist..."
	@if not exist "data\uploads" mkdir data\uploads
	@if not exist "models" mkdir models
	@echo ">>> Setup complete!"

install: setup

# Initialize Model & Data
init:
	@echo ">>> Exporting ONNX model..."
	@$(VENV_PYTHON) src/export_onnx.py
	@echo ">>> Downloading test images..."
	@$(VENV_PYTHON) tests/download_test_images.py
	@echo ">>> Initialization complete!"

# Run Application
run:
	@echo ">>> Starting FastAPI server (Production Mode)..."
	@$(VENV_PYTHON) src/main.py

dev:
	@echo ">>> Starting FastAPI server (Development Mode)..."
	@$(VENV_PYTHON) -m uvicorn src.main:app --reload --host $(shell $(VENV_PYTHON) -c "from config import settings; print(settings.HOST)") --port $(shell $(VENV_PYTHON) -c "from config import settings; print(settings.PORT)")

# Docker Commands
docker-up:
	@echo ">>> Building and starting Docker containers..."
	docker-compose up --build -d

docker-down:
	@echo ">>> Stopping Docker containers..."
	docker-compose down

logs:
	docker-compose logs -f

# Test
test:
	@echo ">>> Running tests..."
	@$(VENV_PYTHON) tests/test_api.py

# Maintenance
freeze:
	@echo ">>> Updating requirements.txt..."
	@$(VENV_PIP) freeze > requirements.txt

clean:
	@echo ">>> Cleaning up..."
	@if exist "$(VENV)" rd /s /q $(VENV)
	@if exist "__pycache__" rd /s /q __pycache__
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@echo ">>> Cleaned!"

# Makefile for Lua Khoe AI Inference Service (Windows)

VENV = venv
VENV_PYTHON = $(VENV)\Scripts\python.exe
VENV_PIP = $(VENV_PYTHON) -m pip

.PHONY: all help setup install run dev test clean docker-build docker-run

all: help

help:
	@echo "======================================================================"
	@echo "Lua Khoe AI Inference Service - Commands"
	@echo "======================================================================"
	@echo "  make setup        - Create venv and install dependencies"
	@echo "  make install      - Alias for setup"
	@echo "  make run          - Run FastAPI server (production mode)"
	@echo "  make dev          - Run development server with --reload"
	@echo "  make test         - Run tests"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make clean        - Remove venv and cache files"
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
	@echo ">>> Ensuring model directory exists..."
	@if not exist "models" mkdir models
	@echo ">>> Setup complete!"

install: setup

# Run Application
run:
	@echo ">>> Starting AI Inference Service..."
	@$(VENV_PYTHON) src/main.py

dev:
	@echo ">>> Starting AI Inference Service (dev mode)..."
	@$(VENV_PYTHON) -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Docker
docker-build:
	docker build -t luakhoe-ai .

docker-run:
	docker run -p 8000:8000 -v ./models:/app/models luakhoe-ai

# Test
test:
	@echo ">>> Running tests..."
	@$(VENV_PYTHON) -m pytest tests/ -v

# Cleanup
clean:
	@echo ">>> Cleaning up..."
	@if exist "$(VENV)" rd /s /q $(VENV)
	@if exist "__pycache__" rd /s /q __pycache__
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@echo ">>> Cleaned!"

# Check if Python is installed
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found. Please install Python 3.12." -ForegroundColor Red
    exit
}

# Create venv if not exists
if (!(Test-Path venv)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

# Activate venv and update pip
Write-Host "Upgrading pip and installing dependencies..." -ForegroundColor Cyan
& .\venv\Scripts\python.exe -m pip install --upgrade pip
& .\venv\Scripts\pip install -r requirements.txt

# Create .env from .env.example if not exists
if (!(Test-Path .env)) {
    Write-Host "Creating .env from template..." -ForegroundColor Yellow
    Copy-Item .env.example .env
}

Write-Host "Setup complete!" -ForegroundColor Green

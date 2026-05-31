"""
Environment Adjustment Module for Rice Disease Prediction.

Adjusts YOLO confidence scores based on:
  1. Real-time weather data (via Open-Meteo API — free, no API key)
  2. Field condition parameters (user-reported)

Rule-based weights sourced from:
  - IRRI Rice Knowledge Bank
  - UF/IFAS EDIS (2024)

Refactored from task2_env_module.py for production use.
"""

import requests
from typing import Any

from src.shared.utils.logger import logger

# ============================================================
from src.modules.predict.constants import (
    CLASS_NAMES_VI,
    CLASS_WEIGHTS,
    DISEASE_BACTERIAL_BLIGHT,
    DISEASE_BROWN_SPOT,
    DISEASE_RICE_BLAST,
    DISEASE_RICE_TUNGRO,
    DISEASE_SHEATH_BLIGHT,
    # Field conditions
    WATER_NORMAL, WATER_FLOODED, WATER_DROUGHT,
    GROWTH_SEEDLING, GROWTH_TILLERING, GROWTH_BOOTING, GROWTH_HEADING, GROWTH_RIPENING,
    DENSITY_MEDIUM, DENSITY_THICK, DENSITY_THIN
)


# ============================================================
# OPEN-METEO WEATHER API
# ============================================================

def get_weather(lat: float, lng: float, location_name: str = "Unknown") -> dict[str, Any]:
    """
    Fetch current weather from Open-Meteo API (free, no API key).

    Returns:
        dict with humidity, temperature, rainfall classification, wind classification, source.
    """
    default = {
        "humidity":    75.0,
        "temperature": 28.0,
        "rainfall":    "none",
        "wind":        "calm",
        "source":      "default",
    }

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lng,
            "current": ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_10m"],
        }

        response = requests.get(url, params=params, timeout=5)

        if response.status_code != 200:
            logger.warning(f"Open-Meteo API error ({response.status_code}) — using defaults")
            return default

        data = response.json()
        current = data.get("current", {})

        humidity = float(current.get("relative_humidity_2m", 75.0))
        temperature = float(current.get("temperature_2m", 28.0))
        rain_mm = float(current.get("rain", 0.0))
        wind_speed_kmh = float(current.get("wind_speed_10m", 0.0))

        # Classify rainfall
        if rain_mm == 0:
            rainfall = "none"
        elif rain_mm < 2.5:
            rainfall = "light"
        else:
            rainfall = "heavy"

        # Classify wind (km/h → same thresholds as original, converted from m/s)
        # Original: <3.3 m/s = calm, 3.3-8 m/s = moderate, >=8 m/s = strong
        # Converted: <12 km/h = calm, 12-29 km/h = moderate, >=29 km/h = strong
        if wind_speed_kmh < 12:
            wind = "calm"
        elif wind_speed_kmh < 29:
            wind = "moderate"
        else:
            wind = "strong"

        logger.info(
            f"Weather at {location_name} ({lat}, {lng}): {temperature}°C, "
            f"humidity {humidity}%, rain={rainfall}, wind={wind}"
        )

        return {
            "humidity":    humidity,
            "temperature": temperature,
            "rainfall":    rainfall,
            "wind":        wind,
            "source":      "api",
        }

    except Exception as e:
        logger.warning(f"Open-Meteo API connection error ({e}) — using defaults")
        return default


# ============================================================
# WEATHER WEIGHTS — IRRI Rice Knowledge Bank, UF/IFAS EDIS
# ============================================================

def compute_weather_weights(
    humidity: float,
    temperature: float,
    rainfall: str,
    wind: str,
) -> dict[str, float]:
    """Compute weather-based adjustment weights per disease class."""

    w = {k: 1.0 for k in CLASS_NAMES_VI}

    # Bacterial Blight — favors 25-34°C, humidity >70%, strong winds + heavy rain
    if humidity > 70:
        w[DISEASE_BACTERIAL_BLIGHT] *= 1.20
    if humidity > 85:
        w[DISEASE_BACTERIAL_BLIGHT] *= 1.20
    if 25 <= temperature <= 34:
        w[DISEASE_BACTERIAL_BLIGHT] *= 1.25
    if rainfall == "heavy":
        w[DISEASE_BACTERIAL_BLIGHT] *= 1.30
    elif rainfall == "light":
        w[DISEASE_BACTERIAL_BLIGHT] *= 1.10
    if wind == "strong":
        w[DISEASE_BACTERIAL_BLIGHT] *= 1.40
    elif wind == "moderate":
        w[DISEASE_BACTERIAL_BLIGHT] *= 1.15

    # Brown Spot — humidity 86-100%, temp 16-36°C, wet leaves
    if humidity > 86:
        w[DISEASE_BROWN_SPOT] *= 1.30
    if 16 <= temperature <= 36:
        w[DISEASE_BROWN_SPOT] *= 1.20
    if rainfall == "heavy":
        w[DISEASE_BROWN_SPOT] *= 1.20
    elif rainfall == "light":
        w[DISEASE_BROWN_SPOT] *= 1.10

    # Rice Blast — humidity >90%, temp 25-28°C optimal for spore germination
    if humidity > 90:
        w[DISEASE_RICE_BLAST] *= 1.45
    elif humidity > 85:
        w[DISEASE_RICE_BLAST] *= 1.20
    if 25 <= temperature <= 28:
        w[DISEASE_RICE_BLAST] *= 1.35
    elif 20 <= temperature <= 25:
        w[DISEASE_RICE_BLAST] *= 1.15
    elif temperature > 30:
        w[DISEASE_RICE_BLAST] *= 0.75
    if rainfall == "heavy":
        w[DISEASE_RICE_BLAST] *= 1.25

    # Rice Tungro — transmitted by leafhoppers, moderate wind helps spread
    if humidity > 80:
        w[DISEASE_RICE_TUNGRO] *= 1.15
    if 25 <= temperature <= 35:
        w[DISEASE_RICE_TUNGRO] *= 1.25
    if wind == "moderate":
        w[DISEASE_RICE_TUNGRO] *= 1.30
    elif wind == "strong":
        w[DISEASE_RICE_TUNGRO] *= 1.10
    if rainfall == "light":
        w[DISEASE_RICE_TUNGRO] *= 1.15
    elif rainfall == "heavy":
        w[DISEASE_RICE_TUNGRO] *= 0.90

    # Sheath Blight — temp 28-32°C, humidity 85-100%
    if humidity > 85:
        w[DISEASE_SHEATH_BLIGHT] *= 1.35
    if humidity > 95:
        w[DISEASE_SHEATH_BLIGHT] *= 1.15
    if 28 <= temperature <= 32:
        w[DISEASE_SHEATH_BLIGHT] *= 1.35
    elif temperature > 32:
        w[DISEASE_SHEATH_BLIGHT] *= 1.10
    if rainfall == "heavy":
        w[DISEASE_SHEATH_BLIGHT] *= 1.30
    elif rainfall == "light":
        w[DISEASE_SHEATH_BLIGHT] *= 1.10

    return w


# ============================================================
# FIELD CONDITION WEIGHTS — 6 user-reported parameters
# ============================================================

def compute_field_weights(
    water: str | None = None,
    growth: str | None = None,
    density: str | None = None,
    fog: bool | None = None,
    pesticide: bool | None = None,
) -> dict[str, float]:
    """
    Compute field condition adjustment weights per disease class.
    Only applies weights for fields the user has explicitly selected.
    Fields that are None or empty string are skipped entirely.
    """
    w = {k: 1.0 for k in CLASS_NAMES_VI}

    # Water status — skip if not provided
    if water and water != "":
        if water == WATER_FLOODED:
            w[DISEASE_SHEATH_BLIGHT]    *= 1.40
            w[DISEASE_BACTERIAL_BLIGHT] *= 1.20
            w[DISEASE_BROWN_SPOT]       *= 1.15
            w[DISEASE_RICE_BLAST]       *= 0.90
        elif water == WATER_DROUGHT:
            w[DISEASE_BROWN_SPOT]       *= 1.30
            w[DISEASE_SHEATH_BLIGHT]    *= 0.80
            w[DISEASE_BACTERIAL_BLIGHT] *= 0.85

    # Growth stage — skip if not provided
    if growth and growth != "":
        if growth == GROWTH_SEEDLING:
            w[DISEASE_RICE_BLAST]    *= 1.30
            w[DISEASE_BROWN_SPOT]    *= 1.20
        elif growth == GROWTH_TILLERING:
            w[DISEASE_SHEATH_BLIGHT] *= 1.20
            w[DISEASE_RICE_BLAST]    *= 1.25
            w[DISEASE_RICE_TUNGRO]   *= 1.20
        elif growth == GROWTH_BOOTING:
            w[DISEASE_SHEATH_BLIGHT]    *= 1.35
            w[DISEASE_BACTERIAL_BLIGHT] *= 1.25
            w[DISEASE_RICE_BLAST]       *= 1.20
        elif growth == GROWTH_HEADING:
            w[DISEASE_BACTERIAL_BLIGHT] *= 1.30
            w[DISEASE_SHEATH_BLIGHT]    *= 1.25
            w[DISEASE_BROWN_SPOT]       *= 1.20
        elif growth == GROWTH_RIPENING:
            w[DISEASE_BROWN_SPOT]       *= 1.15
            w[DISEASE_BACTERIAL_BLIGHT] *= 1.10

    # Sowing density — skip if not provided
    if density and density != "":
        if density == DENSITY_THICK:
            w[DISEASE_SHEATH_BLIGHT]    *= 1.40
            w[DISEASE_RICE_BLAST]       *= 1.20
            w[DISEASE_BACTERIAL_BLIGHT] *= 1.15
        elif density == DENSITY_THIN:
            w[DISEASE_SHEATH_BLIGHT]    *= 0.80
            w[DISEASE_RICE_BLAST]       *= 0.90

    # Fog — skip if None (user hasn't chosen)
    if fog is True:
        w[DISEASE_RICE_BLAST]    *= 1.45
        w[DISEASE_BROWN_SPOT]    *= 1.20
        w[DISEASE_SHEATH_BLIGHT] *= 1.15

    # Pesticide — skip if None (user hasn't chosen)
    if pesticide is True:
        w[DISEASE_BACTERIAL_BLIGHT] *= 0.80
        w[DISEASE_BROWN_SPOT]       *= 0.80
        w[DISEASE_RICE_BLAST]       *= 0.80
        w[DISEASE_RICE_TUNGRO]      *= 0.85
        w[DISEASE_SHEATH_BLIGHT]    *= 0.80

    return w


# ============================================================
# MAIN ADJUSTMENT FUNCTION
# ============================================================

def adjust_prediction(
    yolo_scores: dict[str, float],
    province: str | None = None,
    gps_lat: float | None = None,
    gps_lng: float | None = None,
    field_params: dict[str, Any] | None = None,
    weather: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Combine YOLO confidence scores with environmental factors.

    Args:
        yolo_scores:  {disease_name: confidence, ...} from YOLO
        province:     Vietnamese province name (for logging)
        gps_lat:      GPS Latitude
        gps_lng:      GPS Longitude
        field_params: {water, growth, density, fog, leafhopper, pesticide}
        weather:      Pre-fetched weather dictionary

    Returns:
        dict with disease_en, disease_vi, final_confidence, original_score,
             all_scores, weather, applied
    """
    field_params = field_params or {}

    # Step 1: Get weather
    if weather is not None:
        logger.info(f"Using weather data passed from backend orchestrator: {weather}")
    elif gps_lat is not None and gps_lng is not None:
        weather = get_weather(gps_lat, gps_lng, location_name=province or "Selected Location")
    else:
        logger.warning("No GPS coordinates or weather data provided — using default weather")
        weather = {
            "humidity": 75.0, "temperature": 28.0,
            "rainfall": "none", "wind": "calm", "source": "default",
        }

    # Step 2: Weather weights
    weather_weights = compute_weather_weights(
        humidity=weather["humidity"],
        temperature=weather["temperature"],
        rainfall=weather["rainfall"],
        wind=weather["wind"],
    )

    # Step 3: Field condition weights
    field_weights = compute_field_weights(
        water=field_params.get("water") or None,
        growth=field_params.get("growth") or None,
        density=field_params.get("density") or None,
        fog=field_params.get("fog"),
        pesticide=field_params.get("pesticide"),
    )

    # Step 4: Combine weights (Weather * Field * Base Class Weight)
    combined_weights = {
        disease: weather_weights[disease] * field_weights[disease] * CLASS_WEIGHTS.get(disease, 1.0)
        for disease in CLASS_NAMES_VI
    }

    # Step 5: Apply to YOLO scores and normalize
    adjusted = {
        disease: yolo_scores.get(disease, 0.0) * combined_weights[disease]
        for disease in CLASS_NAMES_VI
    }

    total = sum(adjusted.values())
    if total == 0:
        total = sum(combined_weights.values())
        normalized = {k: v / total for k, v in combined_weights.items()}
    else:
        normalized = {k: v / total for k, v in adjusted.items()}

    # Step 6: Best disease
    best_disease = max(normalized, key=normalized.get)
    best_score = normalized[best_disease]
    original = yolo_scores.get(best_disease, 0.0)

    logger.info(
        f"Env adjustment: {CLASS_NAMES_VI.get(best_disease, best_disease)} "
        f"{original:.1%} → {best_score:.1%}"
    )

    return {
        "disease_en":       best_disease,
        "disease_vi":       CLASS_NAMES_VI.get(best_disease, best_disease),
        "final_confidence": round(best_score, 4),
        "original_score":   round(original, 4),
        "all_scores":       {k: round(v, 4) for k, v in normalized.items()},
        "weather":          weather,
        "applied":          True,
    }

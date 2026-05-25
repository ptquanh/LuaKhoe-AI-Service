# src/modules/predict/constants.py

# Disease Keys (internal mapping)
DISEASE_BACTERIAL_BLIGHT = "Bacterial Blight"
DISEASE_BROWN_SPOT       = "Brown Spot"
DISEASE_RICE_BLAST       = "Rice Blast"
DISEASE_RICE_TUNGRO      = "Rice Tungro"
DISEASE_SHEATH_BLIGHT    = "Sheath Blight"
DISEASE_HEALTHY          = "Healthy"

# Field Condition Values (Vietnamese)
WATER_NORMAL   = "Bình thường"
WATER_FLOODED  = "Ngập úng"
WATER_DROUGHT  = "Khô hạn"

GROWTH_SEEDLING   = "Mạ"
GROWTH_TILLERING  = "Đẻ nhánh"
GROWTH_BOOTING    = "Làm đòng"
GROWTH_HEADING    = "Trỗ bông"
GROWTH_RIPENING   = "Chín"

DENSITY_MEDIUM = "Vừa"
DENSITY_THICK  = "Dày"
DENSITY_THIN   = "Thưa"

# Disease class names — matches YOLO model labels
CLASS_NAMES_VI: dict[str, str] = {
    DISEASE_BACTERIAL_BLIGHT: "Bạc lá",
    DISEASE_BROWN_SPOT:       "Đốm nâu",
    DISEASE_RICE_BLAST:       "Đạo ôn",
    DISEASE_RICE_TUNGRO:      "Vàng lùn lúa cỏ",
    DISEASE_SHEATH_BLIGHT:    "Khô vằn",
}

# Base weights from model training to address class imbalance
# These help balance the influence of different disease classes
CLASS_WEIGHTS: dict[str, float] = {
    DISEASE_BACTERIAL_BLIGHT: 1.46,
    DISEASE_BROWN_SPOT:       1.00,
    DISEASE_RICE_BLAST:       2.14,
    DISEASE_RICE_TUNGRO:      1.53,
    DISEASE_SHEATH_BLIGHT:    2.13,
}

from src.modules.predict.env_adjustment import adjust_prediction

# Mock YOLO scores
yolo_scores = {
    "Rice Blast": 0.85,
    "Brown Spot": 0.60,
    "Bacterial Blight": 0.40,
    "Rice Tungro": 0.20,
    "Sheath Blight": 0.15
}

print("=== SCENARIO 1: DEFAULT (No params) ===")
res1 = adjust_prediction(yolo_scores)
print(f"Best: {res1['disease_en']} ({res1['final_confidence']:.2f})")
print(f"Weather: {res1['weather']}")

print("\n=== SCENARIO 2: WITH PROVINCE (Cần Thơ) AND PARAMS ===")
# Tungro thrives with leafhoppers
field_params = {
    "water": "Ngập úng",
    "growth": "Đẻ nhánh",
    "density": "Dày",
    "fog": True,
    "leafhopper": True,  # High weight for Tungro
    "pesticide": False
}
res2 = adjust_prediction(yolo_scores, province="Cần Thơ", field_params=field_params)
print(f"Best: {res2['disease_en']} ({res2['final_confidence']:.2f})")
print(f"Weather: {res2['weather']}")
print(f"Original Blast: {res2['original_score']:.2f}, Tungro: {res2['all_scores'].get('Rice Tungro', 0):.2f}")
print("Test completed successfully.")

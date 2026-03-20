import requests
import os
import sys

# Add root to path for local imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_status():
    url = "http://localhost:8000/status"
    try:
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        print(response.json())
        assert response.status_code == 200
    except Exception as e:
        print(f"Error connecting to /status: {e}")

def test_predict():
    url = "http://localhost:8000/predict"
    test_image = "tests/test_images/healthy_rice.jpg"
    
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found. Skip prediction test.")
        return

    try:
        with open(test_image, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
            print(f"Prediction result: {response.status_code}")
            print(response.json())
            assert response.status_code == 200
    except Exception as e:
        print(f"Error connecting to /predict: {e}")

if __name__ == "__main__":
    print("Testing Lúa Khỏe AI API...")
    test_status()
    test_predict()

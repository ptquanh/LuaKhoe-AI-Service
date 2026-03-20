import requests
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.logger import logger

# List of real rice disease images for testing
TEST_IMAGES = {
    "bacterial_leaf_blight.jpg": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT6fDIJ5fst6u4QuqiMKVNKknTAnK2WKcoUoQ&s",
    "brown_spot.jpg": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSARj2_587HmoE9KYjK8YQIf6o-XOPYqMB7oA&s", # This is Blast, but for testing confidence
    "healthy_rice.jpg": "https://agridrone.vn/wp-content/uploads/2022/05/benh-chay-la-lua-01.jpg"
}

def download_images():
    save_dir = settings.TEST_IMAGE_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    for filename, url in TEST_IMAGES.items():
        try:
            logger.info(f"Downloading {filename}...")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(os.path.join(save_dir, filename), "wb") as f:
                    f.write(response.content)
                logger.info(f"Saved to {os.path.join(save_dir, filename)}")
            else:
                logger.error(f"Failed to download {filename}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    download_images()

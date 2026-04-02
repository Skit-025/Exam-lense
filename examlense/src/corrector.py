import os
import requests
from PIL import Image
from io import BytesIO
def correct_image_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error fetching image from URL: {e}")
        return None
from PIL import Image
import numpy as np
import cv2
import io

def read_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Convert raw bytes from upload into a PIL Image."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL image to numpy array for PaddleOCR."""
    return np.array(image)

def crop_plate(image: Image.Image, box: list) -> Image.Image:
    """Crop the detected plate region from the image."""
    x1, y1, x2, y2 = map(int, box)
    return image.crop((x1, y1, x2, y2))

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess the image for better OCR results."""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Apply adaptive thresholding
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return enhanced_image
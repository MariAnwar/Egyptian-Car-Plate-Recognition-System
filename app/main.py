from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

from app.model import run_pipeline
from app.utils import read_image_from_bytes

# ─── App definition ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Egyptian Car Plate Recognition API",
    description="""
    Upload a car image and this API will:
    - Detect the license plate using YOLOv8
    - Crop the plate region
    - Extract the plate text using PaddleOCR
    """,
    version="1.0.0"
)

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Check that the API is running."""
    return {"status": "running", "message": "Plate Recognition API is live."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a car image (jpg/png) and get the detected plate text back.
    """

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPG or PNG image."
        )

    # Read and convert image
    image_bytes = await file.read()
    image = read_image_from_bytes(image_bytes)

    # Run pipeline
    result = run_pipeline(image)

    return JSONResponse(content=result)
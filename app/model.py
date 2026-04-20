# from ultralytics import YOLO
# from paddleocr import PaddleOCR
# from PIL import Image
# from app.utils import pil_to_numpy, crop_plate, preprocess_image

# # ─── Load models once when the server starts ───────────────────────────────
# # This is important — you never want to reload the model on every request
# # because it's slow. Load once, reuse forever.

# print("Loading YOLO model...")
# yolo_model = YOLO("models/best.pt")

# print("Loading PaddleOCR model...")
# ocr_model = PaddleOCR(lang='ar')

# print("Models ready.")

# # ─── Main pipeline function ─────────────────────────────────────────────────

# def run_pipeline(image: Image.Image) -> dict:
#     """
#     Full pipeline:
#       1. Detect plate with YOLO
#       2. Crop the plate region
#       3. Preprocess the cropped image (optional but can improve OCR)
#       4. Extract text with PaddleOCR
#     Returns a dict with results.
#     """

#     # Step 1: YOLO detection
#     results = yolo_model(image)
#     boxes = results[0].boxes.xyxy.tolist()  # list of [x1, y1, x2, y2]

#     if not boxes:
#         return {
#             "status": "no_plate_found",
#             "plates": [],
#             "message": "No license plate detected in this image."
#         }

#     plates = []

#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box)

#         # Step 2: Crop the plate
#         cropped = crop_plate(image, box)
#         cropped_np = pil_to_numpy(cropped)

#         # Step 3: Preprocess the cropped image
#         processed = preprocess_image(cropped)

#         # Step 4: OCR on processed plate
#         ocr_result = ocr_model.ocr(processed, cls=True)

#         # Extract text safely
#         plate_text = ""
#         if ocr_result and ocr_result[0]:
#             plate_text = " ".join([
#                 line[1][0] for line in ocr_result[0] if line and line[1]
#             ])

#         plates.append({
#             "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
#             "plate_text": plate_text if plate_text else "Text not readable"
#         })

#     return {
#         "status": "success",
#         "plates_found": len(plates),
#         "plates": plates
#     }


from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image
from app.utils import pil_to_numpy, crop_plate, preprocess_image

print("Loading YOLO model...")
yolo_model = YOLO("models/best.pt")

print("Loading PaddleOCR model...")
ocr_model = PaddleOCR(lang='ar')

print("Models ready.")


def format_plate_text(ocr_result) -> str:
    """
    Takes raw PaddleOCR result and returns the plate text
    with each character separated by a space.
    e.g. "أبج1234" → "أ ب ج 1 2 3 4"
    """
    if not ocr_result or not ocr_result[0]:
        return ""

    inner_results = ocr_result[0]

    # Extract all text pieces from OCR lines
    texts = [res[1][0] for res in inner_results if res and res[1] and res[1][0]]

    if not texts:
        return ""

    # Separate each character in each text piece with a space
    separated_texts = [' '.join(text) for text in texts]

    # Join multiple lines with a space between them
    return '  '.join(separated_texts)


def run_pipeline(image: Image.Image) -> dict:

    # Step 1: YOLO detection
    results = yolo_model(image)
    boxes = results[0].boxes.xyxy.tolist()

    if not boxes:
        return {
            "status": "no_plate_found",
            "plates": [],
            "message": "No license plate detected in this image."
        }

    plates = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Step 2: Crop the plate
        cropped = crop_plate(image, box)
        cropped_np = pil_to_numpy(cropped)

        # Step 3: Preprocess the cropped image
        processed = preprocess_image(cropped)

        # Step 4: OCR
        ocr_result = ocr_model.ocr(cropped_np, cls=True)

        # Step 5: Format text — separated Arabic letters and numbers
        plate_text = format_plate_text(ocr_result)

        plates.append({
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "plate_text": plate_text if plate_text else "Text not readable"
        })

    return {
        "status": "success",
        "plates_found": len(plates),
        "plates": plates
    }